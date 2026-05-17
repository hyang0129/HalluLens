"""
generate_manifest.py — populate pending/ with one JSON cell per (task, model, split).

Usage:
    python scripts/dispatch/generate_manifest.py \
        --dispatch-root shared/icr_capture/_dispatch \
        --out-base-dir shared/icr_capture \
        [--tasks hotpotqa,mmlu,popqa,natural_questions,sciq,searchqa] \
        [--models meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen3-8B] \
        [--splits test,train] \
        [--n-samples N]

Re-runnable: cells whose output already exists (eval_results.json + full meta.jsonl)
are skipped. Cells already in pending/claimed/done/failed are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

# Logical split → actual HF split name per task.
# Tasks not listed here use the logical name literally ("test"/"train").
_TASK_SPLITS: dict[str, dict[str, str]] = {
    "hotpotqa":  {"test": "validation", "train": "train"},
    "mmlu":      {"test": "test",       "train": "auxiliary_train"},
    "searchqa":  {"test": "validation", "train": "train"},
}

_DEFAULT_TASKS = [
    "hotpotqa", "mmlu", "popqa", "natural_questions", "sciq", "searchqa",
]
_DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]
_DEFAULT_SPLITS = ["test", "train"]

# Canonical expected split sizes (rows) — used only for completion check.
# Maps (task, hf_split_name) -> expected count.
_EXPECTED_SIZES: dict[tuple[str, str], int] = {
    ("hotpotqa",        "validation"):      7405,
    ("hotpotqa",        "train"):           90447,
    ("mmlu",            "test"):            14079,
    ("mmlu",            "auxiliary_train"): 99800,
    ("popqa",           "test"):            2853,
    ("popqa",           "train"):           11414,
    ("natural_questions", "test"):          4155,
    ("natural_questions", "train"):         16617,
    ("sciq",            "test"):            1000,
    ("sciq",            "train"):           11679,
    ("searchqa",        "validation"):      13893,
    ("searchqa",        "train"):           99820,
}


def _resolve_split(task: str, logical: str) -> str:
    return _TASK_SPLITS.get(task, {}).get(logical, logical)


def _model_slug(model: str) -> str:
    return model.split("/")[-1]


def _cell_is_done(out_dir: Path, task: str, hf_split: str) -> bool:
    eval_path = out_dir / "eval_results.json"
    meta_path = out_dir / "meta.jsonl"
    if not eval_path.exists() or not meta_path.exists():
        return False
    expected = _EXPECTED_SIZES.get((task, hf_split))
    if expected is None:
        return False
    actual = sum(1 for _ in meta_path.open(encoding="utf-8"))
    return actual >= expected


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    fname = cell_id + ".json"
    for sub in ("pending", "done", "failed"):
        if (dispatch_root / sub / fname).exists():
            return True
    claimed = dispatch_root / "claimed"
    if claimed.exists():
        for wd in claimed.iterdir():
            if wd.is_dir() and (wd / fname).exists():
                return True
    return False


def _slice_ranges_for_dataset(
    expected_size: int | None,
    cap: int | None,
    out_base_dir: Path,
    base_cell_id: str,
) -> list[tuple[int | None, int | None]]:
    """Return list of (start, end) slices to emit for this dataset.

    Slice semantics:
      - cap is None OR expected_size is None OR expected_size <= cap →
        one cell with (None, None) = full dataset, no suffix in cell_id/out_dir.
      - expected_size > cap → emit cells covering [0, cap), [cap, 2*cap), …
        until expected_size is reached. Cells already-completed on disk
        (out_dir/_{start}-{end} contains eval_results.json) are skipped so a
        second invocation appends the next slice without re-queuing finished
        ones.
    """
    if cap is None or expected_size is None or expected_size <= cap:
        return [(None, None)]
    ranges: list[tuple[int | None, int | None]] = []
    start = 0
    while start < expected_size:
        end = min(start + cap, expected_size)
        sub_id = f"{base_cell_id}_{start}-{end}"
        sub_dir = out_base_dir / sub_id
        if (sub_dir / "eval_results.json").exists():
            start = end
            continue
        ranges.append((start, end))
        start = end
    return ranges


def generate_manifest(
    dispatch_root: Path,
    out_base_dir: Path,
    tasks: list[str],
    models: list[str],
    splits: list[str],
    n_samples: int | None,
    max_prompt_len: int = 512,
    max_response_len: int = 64,
    r_max: int = 64,
    top_k: int = 20,
    batch_size: int = 1,
    cap: int | None = None,
    shuffle_seed: int = 0,
) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0

    for task in tasks:
        for model in models:
            slug = _model_slug(model)
            for logical_split in splits:
                hf_split = _resolve_split(task, logical_split)
                base_cell_id = f"{task}_{logical_split}_{slug}"

                # Why: out_dir MUST include split — test/train have different N and
                # InferenceCaptureWriter pre-allocates memmap rows at construction.
                # Sharing one out_dir across splits would clobber on resume.
                base_out_dir = out_base_dir / base_cell_id
                expected_size = _EXPECTED_SIZES.get((task, hf_split))

                ranges = _slice_ranges_for_dataset(
                    expected_size, cap, out_base_dir, base_cell_id,
                )

                for (idx_start, idx_end) in ranges:
                    if idx_start is None and idx_end is None:
                        cell_id = base_cell_id
                        out_dir = base_out_dir
                    else:
                        cell_id = f"{base_cell_id}_{idx_start}-{idx_end}"
                        out_dir = out_base_dir / cell_id

                    cell_path = dispatch_root / "pending" / f"{cell_id}.json"

                    if _dispatch_has_cell(dispatch_root, cell_id):
                        continue

                    if (n_samples is None and idx_start is None
                            and _cell_is_done(out_dir, task, hf_split)):
                        continue

                    cell = {
                        "cell_id":         cell_id,
                        "task":            task,
                        "split":           hf_split,
                        "model":           model,
                        "out_dir":         str(out_dir).replace("\\", "/"),
                        "n_samples":       n_samples,
                        "max_prompt_len":  max_prompt_len,
                        "max_response_len": max_response_len,
                        "r_max":           r_max,
                        "top_k":           top_k,
                        "batch_size":      batch_size,
                        "index_start":     idx_start,
                        "index_end":       idx_end,
                        "shuffle_seed":    shuffle_seed,
                    }
                    cell_path.write_text(
                        json.dumps(cell, indent=2), encoding="utf-8"
                    )
                    written += 1
                    print(f"  queued: {cell_id}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate dispatch pending/ queue for ICR capture jobs."
    )
    parser.add_argument("--dispatch-root", required=True,
                        help="Path to <root>/_dispatch/ directory.")
    parser.add_argument("--out-base-dir", required=True,
                        help="Base directory for per-cell output (shared/icr_capture).")
    parser.add_argument("--tasks",
                        default=",".join(_DEFAULT_TASKS),
                        help="Comma-separated task names.")
    parser.add_argument("--models",
                        default=",".join(_DEFAULT_MODELS),
                        help="Comma-separated HuggingFace model IDs.")
    parser.add_argument("--splits",
                        default=",".join(_DEFAULT_SPLITS),
                        help="Comma-separated logical split names (test, train).")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Cap per cell (omit for full split).")
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-response-len", type=int, default=64,
                        help="Default 64 — matches r_max so we never generate "
                             "past the attention sub-block ICR scoring uses.")
    parser.add_argument("--r-max", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of samples per generate() call (default 1). "
                             "Pass 4 for Phase 1 HotpotQA grid.")
    parser.add_argument("--cap", type=int, default=None,
                        help="Per-cell sample cap (default: no cap). Splits whose "
                             "expected size exceeds the cap are shuffled "
                             "deterministically (seed=--shuffle-seed) and emitted as "
                             "multiple [start, end) slice cells of size up to cap, e.g. "
                             "0-50000, 50000-100000, .... Slices already complete on "
                             "disk are skipped, so re-running with a larger cap appends.")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for the deterministic shuffle when --cap is set. "
                             "Must stay constant across appendix runs of the same "
                             "dataset for the slices to remain non-overlapping.")
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    out_base_dir = Path(args.out_base_dir)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    total = generate_manifest(
        dispatch_root, out_base_dir, tasks, models, splits, args.n_samples,
        max_prompt_len=args.max_prompt_len,
        max_response_len=args.max_response_len,
        r_max=args.r_max,
        top_k=args.top_k,
        batch_size=args.batch_size,
        cap=args.cap,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"Done — {total} cells queued in {dispatch_root / 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
