#!/usr/bin/env python3
"""
Assemble all per-sample prediction files into a single tarball for SCP transfer.

Designed to run on a GPU node (not the login node) — the find/tar walk over
runs/baseline_comparison_*_memmap/ can be heavy.

Output: ~/LLM_research/HalluLens/shared/predictions_bundle.tar.gz
        (override with --output)

Tarball layout matches the local layout used by scripts/pull_predictions.py,
so extracting into results/preds/ on a developer machine drops files into
place directly:

    {dataset_canonical}/{model_slug}/{file}                       (sampling / p_true)
    {dataset_canonical}/{model_slug}/{method}/seed_N/{file}       (training)

dataset_canonical uses `natural_questions` (not `nq`); model_slug is the
HuggingFace model name (Llama-3.1-8B-Instruct or Qwen3-8B).

Usage:
    python scripts/assemble_predictions_tarball.py
    python scripts/assemble_predictions_tarball.py --dry-run
    python scripts/assemble_predictions_tarball.py --output /tmp/preds.tar.gz
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(os.path.expanduser("~/LLM_research/HalluLens"))
DEFAULT_OUTPUT = REPO_ROOT / "shared" / "predictions_bundle.tar.gz"

DATASETS_SAMPLING = ["hotpotqa", "natural_questions", "popqa", "sciq", "searchqa"]
DATASETS_PTRUE = DATASETS_SAMPLING + ["mmlu"]
MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]

# Per-sample files emitted by the sampling-baseline pipeline. selfcheck_samples
# and nli_matrix are intermediate but useful for ablation/calibration analysis.
SAMPLING_FILES = (
    "se_labels.jsonl",
    "selfcheck_scores.jsonl",
    "selfcheck_samples.jsonl",
    "nli_matrix.jsonl",
)
PTRUE_FILE = "ptrue.jsonl"

# Experiment short-name → canonical dataset directory name
DATASET_SHORT_TO_CANONICAL = {
    "hotpotqa": "hotpotqa",
    "nq":       "natural_questions",
    "mmlu":     "mmlu",
    "popqa":    "popqa",
    "sciq":     "sciq",
    "searchqa": "searchqa",
}

# Model tag in baseline_comparison_{ds}{tag}_memmap → slug
MODEL_TAG_TO_SLUG = {
    "":      "Llama-3.1-8B-Instruct",
    "qwen3": "Qwen3-8B",
}

TRAINING_FILES = ("predictions.csv", "eval_metrics.json")


def _maybe_add(tf, src: Path, arc: str, added: list, missed: list, dry: bool) -> None:
    if src.is_file():
        if not dry:
            tf.add(str(src), arcname=arc)
        added.append(arc)
    else:
        missed.append(arc)


def assemble(output: Path, dry_run: bool) -> tuple[int, int]:
    added: list[str] = []
    missed: list[str] = []

    if dry_run:
        tf = None
        print(f"[dry-run] would write tarball: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        tf = tarfile.open(output, mode="w:gz")
        print(f"writing tarball: {output}")

    print(f"repo root: {REPO_ROOT}")

    # 1) Sampling baselines
    print("\n== Sampling baselines ==")
    for ds in DATASETS_SAMPLING:
        for model in MODELS:
            for fname in SAMPLING_FILES:
                src = REPO_ROOT / "output" / "sampling_baselines" / ds / model / fname
                _maybe_add(tf, src, f"{ds}/{model}/{fname}", added, missed, dry_run)

    # 2) P(True)
    print("\n== P(True) ==")
    for ds in DATASETS_PTRUE:
        for model in MODELS:
            src = REPO_ROOT / "output" / "p_true" / ds / model / PTRUE_FILE
            _maybe_add(tf, src, f"{ds}/{model}/{PTRUE_FILE}", added, missed, dry_run)

    # 3) Training baselines — per-seed predictions.csv + eval_metrics.json
    print("\n== Training baselines (memmap) ==")
    for ds_short, ds_canonical in DATASET_SHORT_TO_CANONICAL.items():
        for model_tag, model_slug in MODEL_TAG_TO_SLUG.items():
            tag_part = f"_{model_tag}" if model_tag else ""
            inner_key = f"{ds_short}{tag_part}_memmap"
            exp_dir = REPO_ROOT / "runs" / f"baseline_comparison_{ds_short}{tag_part}_memmap" / inner_key
            if not exp_dir.is_dir():
                print(f"  [---] missing: {exp_dir}")
                continue

            for method_dir in sorted(exp_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    seed = seed_dir.name
                    for fname in TRAINING_FILES:
                        src = seed_dir / fname
                        arc = f"{ds_canonical}/{model_slug}/{method}/{seed}/{fname}"
                        _maybe_add(tf, src, arc, added, missed, dry_run)

    if tf is not None:
        tf.close()

    return len(added), len(missed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Tarball output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true", help="List files without writing")
    args = parser.parse_args()

    if not REPO_ROOT.is_dir():
        sys.exit(f"repo root not found: {REPO_ROOT}")

    n_added, n_missed = assemble(args.output, args.dry_run)

    print(f"\nAdded:   {n_added} files")
    print(f"Missing: {n_missed} expected files (not present in cluster — gaps)")
    if not args.dry_run and args.output.is_file():
        sz = args.output.stat().st_size
        print(f"Tarball: {args.output}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
