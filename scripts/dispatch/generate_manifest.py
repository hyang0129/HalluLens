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

Chat-template re-capture: point --dispatch-root and --out-base-dir at a
separate tree so this never collides with legacy (non-templated) captures,
and pass --chat-template so every emitted cell carries "chat_template": true
through to capture_inference.py. --shard-size splits every dataset (test AND
train) into fixed-size shards emitted upfront, for draining with many
parallel workers off the shared queue:

    python scripts/dispatch/generate_manifest.py \
        --dispatch-root shared/icr_capture_chat/_dispatch \
        --out-base-dir shared/icr_capture_chat \
        --chat-template \
        --top-k 500 --max-samples 50000 --shard-size 5000 --batch-size 4 \
        [--tasks ...] [--models ...] [--splits ...]

(top_k defaults to 20 — the 500 above is just documenting the intended
invocation, not a new default.)

Re-runnable: cells whose output already exists (eval_results.json + full meta.jsonl)
are skipped. Cells already in pending/claimed/done/failed are not touched.

--cap vs --shard-size (mutually exclusive):
  --cap N       "give me the next slice of size N" — emits at most ONE
                incomplete slice per dataset per invocation; a
                headline-then-appendix workflow for sequential capture.
  --shard-size N  splits every dataset > N into ALL of its shards, emitted
                upfront in one call — for load-balancing a drain across many
                parallel dispatch workers rather than working through slices
                one at a time.

--max-samples N: composable with --shard-size (errors out with --cap). Caps
  the effective dataset size to min(expected_size, max_samples) BEFORE
  sharding, taking the front of the same deterministic shuffle --cap uses
  (seed=--shuffle-seed). E.g. hotpotqa train (90447) with --max-samples 50000
  --shard-size 5000 emits exactly 10 shards [0-5000) ... [45000-50000);
  popqa train (11414) with the same flags emits 3 shards [0-5000),
  [5000-10000), [10000-11414). A dataset already <= shard-size (before any
  capping) still emits the single unsuffixed cell. Only takes effect
  alongside --shard-size — without it, --max-samples has no effect on the
  default (whole-dataset, unsuffixed) or --cap paths.
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
    """Return at most ONE slice to emit for this dataset.

    Slice semantics:
      - cap is None OR expected_size is None OR expected_size <= cap →
        single cell (None, None) = full dataset, no suffix in cell_id/out_dir.
      - expected_size > cap → emit exactly one slice: the first
        [k*cap, min((k+1)*cap, expected_size)) whose out_dir does NOT yet
        contain eval_results.json. Earlier slices that ARE complete are
        skipped (so a re-invocation appends the next slice rather than
        re-queuing the headline grid).
      - All slices already done → return [] (no cell to emit).

    This makes --cap behave as "give me the next slice of size cap" rather
    than "emit every slice upfront", matching the headline-then-appendix
    workflow.
    """
    if cap is None or expected_size is None or expected_size <= cap:
        return [(None, None)]
    start = 0
    while start < expected_size:
        end = min(start + cap, expected_size)
        sub_id = f"{base_cell_id}_{start}-{end}"
        sub_dir = out_base_dir / sub_id
        if (sub_dir / "eval_results.json").exists():
            start = end
            continue
        return [(start, end)]
    return []


def _shard_ranges_for_dataset(
    expected_size: int | None,
    shard_size: int | None,
    max_samples: int | None = None,
) -> list[tuple[int | None, int | None]]:
    """Return ALL shard ranges for this dataset, emitted upfront.

    Unlike _slice_ranges_for_dataset (--cap: at most one incomplete slice per
    call, for a sequential headline-then-appendix workflow), --shard-size
    splits the WHOLE dataset into fixed-size shards in one shot — every
    (task, model, split) including test splits — so many parallel dispatch
    workers can drain the queue concurrently instead of waiting on a single
    "next slice" cell at a time.

    max_samples (--max-samples) caps the effective dataset size to
    min(expected_size, max_samples) BEFORE sharding — the front of the same
    deterministic shuffle --cap uses. This must stay distinct from the
    "under-threshold → unsuffixed cell" shortcut below: if max_samples
    actually truncates expected_size (effective_size < expected_size), the
    resulting single shard still needs an explicit (0, effective_size) range
    to enforce the cap downstream — collapsing it to the unsuffixed (None,
    None) form would mean "no index range", i.e. capture_inference.py
    generates the FULL untruncated dataset, silently ignoring --max-samples.

    Semantics:
      - shard_size is None OR expected_size is None → single cell (None, None).
      - effective_size := expected_size if max_samples is None else
        min(expected_size, max_samples).
      - effective_size <= shard_size AND effective_size == expected_size (no
        truncation occurred) → single cell (None, None) = full dataset, no
        suffix in cell_id/out_dir (matches --cap's / no-cap's under-threshold
        behavior — no pointless _0-1000 suffix for a dataset already smaller
        than the shard size).
      - effective_size <= shard_size AND effective_size < expected_size
        (max_samples truncated a dataset that would otherwise exceed
        shard_size) → single EXPLICIT shard (0, effective_size).
      - effective_size > shard_size → every [k*shard_size,
        min((k+1)*shard_size, effective_size)) shard, in order, regardless of
        completion state — completion filtering happens per-shard in
        generate_manifest(), not here.
    """
    if shard_size is None or expected_size is None:
        return [(None, None)]
    effective_size = expected_size if max_samples is None else min(expected_size, max_samples)
    truncated = effective_size < expected_size
    if effective_size <= shard_size:
        if not truncated:
            return [(None, None)]
        return [(0, effective_size)]
    ranges: list[tuple[int | None, int | None]] = []
    start = 0
    while start < effective_size:
        end = min(start + shard_size, effective_size)
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
    chat_template: bool = False,
    shard_size: int | None = None,
    max_samples: int | None = None,
) -> int:
    if cap is not None and shard_size is not None:
        raise ValueError("--cap and --shard-size are mutually exclusive")
    if cap is not None and max_samples is not None:
        raise ValueError("--cap and --max-samples are mutually exclusive")

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

                if shard_size is not None:
                    ranges = _shard_ranges_for_dataset(expected_size, shard_size, max_samples)
                else:
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

                    # Why n_samples is None gates both checks: with an explicit
                    # --n-samples smoketest cap, no out_dir can ever reach the
                    # full expected_size, so "done" is meaningless and must not
                    # suppress re-queuing.
                    if n_samples is None:
                        if idx_start is None and _cell_is_done(out_dir, task, hf_split):
                            continue
                        # Why a simpler eval_results.json-only check for a real
                        # shard/slice (idx_start is not None): _cell_is_done
                        # compares against the FULL dataset's expected size,
                        # which is wrong for a shard — a completed shard's
                        # meta.jsonl row count matches the shard size, not
                        # expected_size. --cap's ranges already pre-filter
                        # completed slices before returning, so this is a
                        # no-op there; --shard-size returns every shard
                        # upfront regardless of completion, so this is the
                        # only place that filters them.
                        if idx_start is not None and (out_dir / "eval_results.json").exists():
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
                        "chat_template":   chat_template,
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
                        help="Per-cell sample cap (default: no cap). For splits whose "
                             "expected size exceeds the cap, emits exactly ONE slice "
                             "cell of size up to cap from the deterministically shuffled "
                             "(seed=--shuffle-seed) dataset — the first slice whose "
                             "out_dir does NOT yet contain eval_results.json. Re-invoking "
                             "after the first 50k is captured appends [50000, 100000), "
                             "then [100000, ...) on subsequent runs. Datasets under cap "
                             "pass through unchanged.")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for the deterministic shuffle when --cap is set. "
                             "Must stay constant across appendix runs of the same "
                             "dataset for the slices to remain non-overlapping.")
    parser.add_argument("--shard-size", type=int, default=None,
                        help="Split every dataset (test AND train, including those "
                             "at or under any --cap threshold you'd otherwise use) into "
                             "ceil(expected_size / shard_size) shards of "
                             "[k*shard_size, min((k+1)*shard_size, expected_size)), ALL "
                             "emitted upfront into pending/ — for load-balancing a drain "
                             "across many parallel dispatch workers. Datasets whose "
                             "expected_size <= shard_size still get a single unsuffixed "
                             "cell (no pointless _0-N suffix). Mutually exclusive with "
                             "--cap. Re-invocation is idempotent: shards already done "
                             "(out_dir/eval_results.json exists) or already queued "
                             "anywhere in the dispatch tree are skipped.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the effective dataset size to min(expected_size, "
                             "max_samples) BEFORE sharding, taken from the front of the "
                             "deterministic shuffle (seed=--shuffle-seed) — same semantics "
                             "as --cap's slicing. Only takes effect together with "
                             "--shard-size (a dataset already <= shard-size still emits "
                             "the single unsuffixed cell); has no effect on the default or "
                             "--cap paths. Mutually exclusive with --cap.")
    parser.add_argument("--chat-template", action="store_true", default=False,
                        help="Set 'chat_template': true on every emitted cell, so worker.sh "
                             "passes --chat-template through to capture_inference.py. Point "
                             "--dispatch-root / --out-base-dir at a separate tree (e.g. "
                             "shared/icr_capture_chat) when using this — do not mix with "
                             "legacy non-templated captures.")
    args = parser.parse_args()

    if args.cap is not None and args.shard_size is not None:
        parser.error("--cap and --shard-size are mutually exclusive")
    if args.cap is not None and args.max_samples is not None:
        parser.error("--cap and --max-samples are mutually exclusive")

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
        chat_template=args.chat_template,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
    )
    print(f"Done — {total} cells queued in {dispatch_root / 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
