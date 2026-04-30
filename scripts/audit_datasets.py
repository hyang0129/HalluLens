#!/usr/bin/env python3
"""Audit inference + activation datasets for a given model.

Usage:
    python scripts/audit_datasets.py --model Qwen/Qwen3-8B
    python scripts/audit_datasets.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional

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
    ("searchqa",               "train",         151_295),  # HF "train" = searchqa test (Qwen3 convention)
    ("searchqa_train",         "test",           43_228),  # HF "test"  = searchqa train (Qwen3 convention)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or bare model name")
    parser.add_argument("--root", default=".",
                        help="Repo root (default: current directory)")
    parser.add_argument("--zarr", action="store_true",
                        help="Inspect zarr stores for sample count + disk size")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_name = args.model.split("/")[-1]
    slug = model_slug(model_name)

    print(f"Model : {model_name}  (zarr slug: {slug})")
    print(f"Root  : {root}")
    print()

    col_w = 26
    header = f"  {'dataset':<{col_w}} {'split':<16}  {'gen':>8}  {'eval':>4}  {'zarr':>4}"
    if args.zarr:
        header += f"  {'zarr_n':>8}  {'zarr_mb':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    n_complete = n_partial = n_missing = 0

    for dataset, split, expected in DATASETS:
        gen_path  = root / "output" / dataset / model_name / "generation.jsonl"
        eval_path = root / "output" / dataset / model_name / "eval_results.json"
        zarr_path = root / "shared" / f"{dataset}_{slug}" / "activations.zarr"

        lines    = count_lines(gen_path)
        has_eval = eval_path.exists()
        has_zarr = zarr_path.exists()

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

        row = (f"  {icon} {dataset:<{col_w}} {split:<16}  {gen_str:>8}"
               f"  {'✓' if has_eval else '✗':>4}  {'✓' if has_zarr else '✗':>4}")

        if args.zarr:
            zn = zarr_samples(zarr_path) if has_zarr else None
            zm = dir_size_mb(zarr_path)  if has_zarr else None
            row += f"  {str(zn) if zn else '—':>8}  {f'{zm:.0f}' if zm else '—':>8}"

        print(row)

    total = len(DATASETS)
    print()
    print(f"Complete (gen+eval): {n_complete}/{total}")
    if n_partial:
        print(f"Needs eval:          {n_partial}")
    if n_missing:
        print(f"Missing gen:         {n_missing}")


if __name__ == "__main__":
    main()
