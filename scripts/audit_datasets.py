#!/usr/bin/env python3
"""Audit inference + activation datasets for a given model.

Usage:
    python scripts/audit_datasets.py --model Qwen/Qwen3-8B
    python scripts/audit_datasets.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr          # fast: sample count only
    python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr-size     # slow: also runs du on each store

The ``judge`` and ``unk`` columns report finalized LLM-judge coverage and
UNKNOWN verdicts from the canonical ``shared/icr_capture`` sidecar.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.label_audit.backfill_status import inspect_capture

# (output_dir_name, hf_split_label, expected_size)
# Canonical sizes from Qwen3-8B complete run (2026-04-30).
# Tolerance: 0% for test splits, 0.25% for train splits.
DATASETS = [
    ("hotpotqa",               "validation",      7_405),
    ("hotpotqa_train",         "train",          90_440),
    ("mmlu",                   "test",           10_421),
    ("mmlu_train",             "auxiliary_train",99_842),
    ("movies",                 "test",            7_856),
    ("natural_questions",      "validation",      4_155),
    ("natural_questions_train","train",          16_617),
    ("popqa",                  "test",            2_854),
    ("popqa_train",            "train",          11_413),
    ("sciq",                   "test",            1_000),
    ("sciq_train",             "train",          11_679),
    # Note: SearchQA dir names predate the #60 flip to HF-standard convention.
    # On-disk row counts reflect HF origin (unchanged); see configs/datasets/searchqa*.json
    # for the post-flip role mapping (151K dir → train, 43K dir → test).
    ("searchqa",               "train",         151_295),  # HF train split → used as train (post #60)
    ("searchqa_train",         "test",           43_228),  # HF test split  → used as test  (post #60)
    # Added directly through the icr_capture pipeline; legacy output/ generation
    # files may be absent even when their canonical captures are complete.
    ("simpleqa",               "test",              866),
    ("simpleqa_train",         "train",            3_460),
    ("triviaqa",               "test",            9_954),
    ("triviaqa_train",         "train",           11_000),
]

# Train splits tolerate 0.25% variance; test splits must be exact.
TRAIN_TOLERANCE = 0.0025
TEST_TOLERANCE  = 0.0


def model_slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with open(path) as f:
        return sum(1 for _ in f)


def dir_size_mb(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(["du", "-sm", str(path)], text=True)
        return float(out.split()[0])
    except Exception:
        return None


def memmap_cache_ready(zarr_path: Path) -> bool:
    """True iff zarr has a built memmap cache (any hash subdir with manifest.json)."""
    cache_dir = zarr_path / "_memmap_cache"
    if not cache_dir.exists():
        return False
    try:
        for sub in cache_dir.iterdir():
            if sub.is_dir() and (sub / "manifest.json").exists():
                return True
    except Exception:
        pass
    return False


def zarr_samples(zarr_path: Path) -> Optional[int]:
    try:
        import zarr
        store = zarr.open(str(zarr_path), mode="r")
        # first numeric array gives us sample count
        for key in store:
            arr = store[key]
            if hasattr(arr, "shape") and arr.ndim >= 1:
                return arr.shape[0]
    except Exception:
        pass
    return None


def _model_suffix(model_name: str) -> str:
    """Map model name to the experiment config suffix (e.g. 'Qwen3-8B' -> '_qwen3')."""
    low = model_name.lower()
    if "qwen" in low:
        return "_qwen3"
    if "smollm" in low:
        return "_smollm3"
    return ""  # Llama / default


def capture_dir_for(root: Path, dataset: str, model_name: str) -> Optional[Path]:
    """Resolve an audit dataset row to its canonical icr_capture directory.

    Most ``*_train`` output directories are train captures. SearchQA is the
    historical exception: its output directory names predate the split-role
    correction, so ``searchqa`` is train and ``searchqa_train`` is test.
    Large train captures may carry the ``_0-50000`` shard suffix.
    """
    if dataset == "movies":  # excluded from the icr_capture pipeline
        return None
    if dataset == "searchqa":
        base, role = "searchqa", "train"
    elif dataset == "searchqa_train":
        base, role = "searchqa", "test"
    else:
        base = dataset.removesuffix("_train")
        role = "train" if dataset.endswith("_train") else "test"

    capture_root = root / "shared" / "icr_capture"
    stem = f"{base}_{role}_{model_name}"
    candidates = [capture_root / stem]
    if role == "train":
        candidates.append(capture_root / f"{stem}_0-50000")
    return next((path for path in candidates if path.is_dir()), None)


def judge_coverage(capture_dir: Optional[Path]) -> Optional[dict]:
    """Return sidecar coverage, or None when no capture/sidecar is present."""
    if capture_dir is None or not (capture_dir / "judge_labels.jsonl").exists():
        return None
    return inspect_capture(capture_dir)


def _print_runs_section(root: Path, model_name: str) -> None:
    """Print training run completeness for all experiment configs matching this model."""
    from scripts.experiment_utils import (
        RunStatus,
        classify_run,
        enumerate_runs,
        load_experiment_config,
    )

    suffix = _model_suffix(model_name)
    pattern = f"baseline_comparison_*{suffix}.json"
    configs = sorted((root / "configs" / "experiments").glob(pattern))

    if not configs:
        print(f"  (no experiment configs matching '{pattern}')")
        return

    # Collect all runs across configs; group by dataset then method
    from collections import defaultdict
    method_order: list[str] = []
    dataset_order: list[str] = []
    # grid[dataset][method] -> (complete, total)
    grid: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)

    for cfg_path in configs:
        try:
            cfg = load_experiment_config(str(cfg_path), project_root=str(root))
        except Exception as e:
            print(f"  warning: could not load {cfg_path.name}: {e}")
            continue

        output_base = str(root / cfg.get("output_dir", "runs"))
        run_specs = enumerate_runs(cfg, output_base=output_base, project_root=str(root))

        for spec in run_specs:
            ds = spec.dataset_name
            m = spec.method_name
            if ds not in dataset_order:
                dataset_order.append(ds)
            if m not in method_order:
                method_order.append(m)
            status = classify_run(spec.run_dir)
            prev = grid[ds].get(m, (0, 0))
            grid[ds][m] = (prev[0] + (1 if status == RunStatus.COMPLETE else 0), prev[1] + 1)

    if not method_order:
        print("  (no runs found)")
        return

    # Print compact matrix: rows = methods, cols = datasets
    m_w = max(len(m) for m in method_order)
    ds_w = max(max(len(d) for d in dataset_order), 8)
    header = f"  {'method':<{m_w}}  " + "  ".join(d[:ds_w].ljust(ds_w) for d in dataset_order)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in method_order:
        row = f"  {method:<{m_w}}  "
        cells = []
        for ds in dataset_order:
            if ds in grid and method in grid[ds]:
                ok, tot = grid[ds][method]
                cells.append(f"{ok}/{tot}".ljust(ds_w))
            else:
                cells.append("-".ljust(ds_w))
        row += "  ".join(cells)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or bare model name")
    parser.add_argument("--root", default=".",
                        help="Repo root (default: current directory)")
    parser.add_argument("--zarr", action="store_true",
                        help="Show zarr sample count (fast — metadata only)")
    parser.add_argument("--zarr-size", action="store_true",
                        help="Also show zarr disk usage via du (slow on large stores)")
    parser.add_argument("--runs", action="store_true",
                        help="Also show training run completeness from experiment configs")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_name = args.model.split("/")[-1]
    slug = model_slug(model_name)

    show_zarr_n    = args.zarr or args.zarr_size
    show_zarr_size = args.zarr_size

    print(f"Model : {model_name}  (zarr slug: {slug})")
    print(f"Root  : {root}")
    print()

    col_w = 26
    header = (f"  {'dataset':<{col_w}} {'split':<16}  {'gen':>8}  {'eval':>4}"
              f"  {'zarr':>4}  {'mem':>4}  {'judge':>17}  {'unk':>6}")
    if show_zarr_n:
        header += f"  {'zarr_n':>8}"
    if show_zarr_size:
        header += f"  {'zarr_mb':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_complete = n_partial = n_missing = 0
    judge_finalized = judge_total = judge_unknown = 0
    judge_full = judge_partial = 0

    for dataset, split, expected in DATASETS:
        gen_path  = root / "output" / dataset / model_name / "generation.jsonl"
        eval_path = root / "output" / dataset / model_name / "eval_results.json"
        zarr_path = root / "shared" / f"{dataset}_{slug}" / "activations.zarr"

        lines    = count_lines(gen_path)
        has_eval = eval_path.exists()
        has_zarr = zarr_path.exists()
        coverage = judge_coverage(capture_dir_for(root, dataset, model_name))

        is_train  = dataset.endswith("_train")
        tolerance = TRAIN_TOLERANCE if is_train else TEST_TOLERANCE
        gen_ok    = lines is not None and abs(lines - expected) / expected <= tolerance

        # status icon
        if gen_ok and has_eval:
            icon = "✓"
            n_complete += 1
        elif lines is not None:
            icon = "~"
            n_partial += 1
        else:
            icon = "✗"
            n_missing += 1

        # gen count: flag if outside tolerance
        if lines is None:
            gen_str = "—"
        elif not gen_ok:
            gen_str = f"{lines:,}?"
        else:
            gen_str = f"{lines:,}"

        has_mem = memmap_cache_ready(zarr_path) if has_zarr else False
        row = (f"  {icon} {dataset:<{col_w}} {split:<16}  {gen_str:>8}"
               f"  {'✓' if has_eval else '✗':>4}  {'✓' if has_zarr else '✗':>4}"
               f"  {'✓' if has_mem else '✗':>4}")

        if coverage is None:
            row += f"  {'—':>17}  {'—':>6}"
        else:
            finalized = coverage["finalized"]
            total_judge = coverage["generation"]
            unknown = coverage["unknown"]
            row += f"  {f'{finalized:,}/{total_judge:,}':>17}  {unknown:>6,}"
            judge_finalized += finalized
            judge_total += total_judge
            judge_unknown += unknown
            if finalized == total_judge:
                judge_full += 1
            else:
                judge_partial += 1

        if show_zarr_n:
            zn = zarr_samples(zarr_path) if has_zarr else None
            row += f"  {str(zn) if zn else '—':>8}"
        if show_zarr_size:
            zm = dir_size_mb(zarr_path) if has_zarr else None
            row += f"  {f'{zm:.0f}' if zm else '—':>8}"

        print(row)

    total = len(DATASETS)
    print()
    print(f"Complete (gen+eval): {n_complete}/{total}")
    if n_partial:
        print(f"Needs eval:          {n_partial}")
    if n_missing:
        print(f"Missing gen:         {n_missing}")
    print(f"LLM judge:           {judge_finalized:,}/{judge_total:,} finalized"
          f" ({100 * judge_finalized / max(1, judge_total):.2f}%),"
          f" {judge_unknown:,} UNKNOWN")
    print(f"Judge sidecars:      {judge_full} complete, {judge_partial} partial")

    if args.runs:
        print()
        print("Training runs (complete/expected):")
        _print_runs_section(root, model_name)


if __name__ == "__main__":
    main()
