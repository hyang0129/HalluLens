#!/usr/bin/env python3
"""
Pull per-sample prediction confidence files from Empire AI to results/preds/.

Covers two source types:
  1. Sampling methods  — se_labels, selfcheck, p_true (per-sample JSONL)
  2. Training baselines — predictions.csv + eval_metrics.json from memmap runs

Output layout:
    results/preds/{dataset}/{model}/{file}                         (sampling)
    results/preds/{dataset}/{model}/{method}/seed_{n}/{file}       (training)

Usage:
    python scripts/pull_predictions.py             # pull everything
    python scripts/pull_predictions.py --dry-run   # preview only
    python scripts/pull_predictions.py --datasets hotpotqa sciq
    python scripts/pull_predictions.py --models Llama-3.1-8B-Instruct
    python scripts/pull_predictions.py --sampling-only
    python scripts/pull_predictions.py --training-only
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REMOTE_HOST = "empire-ai"
REMOTE_BASE = "~/LLM_research/HalluLens"
LOCAL_BASE = Path(__file__).parent.parent / "results" / "preds"

# Canonical dataset names used as local directory names
MEMMAP_DATASETS = [
    "hotpotqa",
    "natural_questions",
    "popqa",
    "sciq",
    "searchqa",
    "mmlu",
]

MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
]

# Per-sample confidence files for sampling methods
SAMPLING_FILES = [
    "se_labels.jsonl",         # semantic entropy variants
    "selfcheck_scores.jsonl",  # SelfCheckGPT (nli, ngram, bertscore)
    "selfcheck_samples.jsonl", # generated samples used by selfcheck
    "nli_matrix.jsonl",        # NLI pair probabilities (SE intermediate)
]
PTRUE_FILE = "ptrue.jsonl"

# Experiment short-name → canonical dataset name
# (keys match the fragment in baseline_comparison_{key}_memmap)
DATASET_SHORT_TO_CANONICAL = {
    "hotpotqa": "hotpotqa",
    "nq": "natural_questions",
    "mmlu": "mmlu",
    "popqa": "popqa",
    "sciq": "sciq",
    "searchqa": "searchqa",
}

# Model tag used in experiment/inner-key names → model slug
MODEL_TAG_TO_SLUG = {
    "":      "Llama-3.1-8B-Instruct",
    "qwen3": "Qwen3-8B",
}

# Files to pull from each training seed directory
TRAINING_FILES = ["predictions.csv", "eval_metrics.json"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remote_exists(remote_path: str) -> bool:
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"test -f {remote_path}"],
        capture_output=True,
    )
    return result.returncode == 0


def pull_file(remote_path: str, local_path: Path, dry_run: bool) -> bool:
    """scp a single file from remote. Returns True if pulled (or would be in dry-run)."""
    rel = local_path.relative_to(LOCAL_BASE.parent.parent)
    if not remote_exists(remote_path):
        print(f"  [---] {rel}")
        return False
    if dry_run:
        print(f"  [DRY] {rel}")
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["scp", "-q", f"{REMOTE_HOST}:{remote_path}", str(local_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  [OK ] {rel}  ({local_path.stat().st_size:,} B)")
        return True
    print(f"  [ERR] {rel}  {result.stderr.strip()}")
    return False


# ---------------------------------------------------------------------------
# Sampling methods (se, selfcheck, p_true)
# ---------------------------------------------------------------------------

def pull_sampling(datasets: list[str], models: list[str], dry_run: bool) -> tuple[int, int]:
    pulled = missing = 0
    for dataset in datasets:
        # sampling baselines don't exist for mmlu
        if dataset == "mmlu":
            continue
        print(f"\n=== {dataset} (sampling) ===")
        for model in models:
            print(f"  -- {model}")
            dst = LOCAL_BASE / dataset / model
            for fname in SAMPLING_FILES:
                remote = f"{REMOTE_BASE}/output/sampling_baselines/{dataset}/{model}/{fname}"
                ok = pull_file(remote, dst / fname, dry_run)
                pulled += ok
                missing += not ok
            remote = f"{REMOTE_BASE}/output/p_true/{dataset}/{model}/{PTRUE_FILE}"
            ok = pull_file(remote, dst / PTRUE_FILE, dry_run)
            pulled += ok
            missing += not ok
    return pulled, missing


# ---------------------------------------------------------------------------
# Training baselines (memmap runs)
# ---------------------------------------------------------------------------

def _build_training_specs(datasets: list[str], models: list[str]) -> list[tuple[str, str, str, str]]:
    """
    Return list of (exp_name, inner_key, canonical_dataset, model_slug) for
    baseline_comparison experiments matching the requested datasets/models.
    """
    specs = []
    for ds_short, ds_canonical in DATASET_SHORT_TO_CANONICAL.items():
        if ds_canonical not in datasets:
            continue
        for model_tag, model_slug in MODEL_TAG_TO_SLUG.items():
            if model_slug not in models:
                continue
            tag_part = f"_{model_tag}" if model_tag else ""
            exp_name  = f"baseline_comparison_{ds_short}{tag_part}_memmap"
            inner_key = f"{ds_short}{tag_part}_memmap"
            specs.append((exp_name, inner_key, ds_canonical, model_slug))
    return specs


def pull_training(datasets: list[str], models: list[str], dry_run: bool) -> tuple[int, int]:
    """
    Pull predictions.csv + eval_metrics.json from all memmap training runs via
    a single tar-pipe per experiment (one SSH round-trip per experiment).
    """
    specs = _build_training_specs(datasets, models)
    pulled = missing = 0

    for exp_name, inner_key, ds_canonical, model_slug in specs:
        remote_exp = f"{REMOTE_BASE}/runs/{exp_name}/{inner_key}"
        print(f"\n=== {ds_canonical} / {model_slug}  [{exp_name}] ===")

        # Ask remote to find all target files and report them
        find_cmd = (
            f"find {remote_exp} -type f \\( -name 'predictions.csv' -o -name 'eval_metrics.json' \\) "
            f"2>/dev/null | sort"
        )
        result = subprocess.run(
            ["ssh", REMOTE_HOST, find_cmd],
            capture_output=True, text=True,
        )
        remote_files = [l.strip() for l in result.stdout.splitlines() if l.strip()]

        if not remote_files:
            print(f"  [---] no files found under {remote_exp}")
            missing += 1
            continue

        if dry_run:
            for rf in remote_files:
                # Convert remote absolute path to local relative path
                rel_from_inner = Path(rf).relative_to(
                    Path(REMOTE_BASE.replace("~", "/mnt/home/hyang1")) / "runs" / exp_name / inner_key
                )
                local = LOCAL_BASE / ds_canonical / model_slug / rel_from_inner
                rel = local.relative_to(LOCAL_BASE.parent.parent)
                print(f"  [DRY] {rel}")
                pulled += 1
            continue

        # Stream tar from remote, extract to temp dir, then move files into place
        with tempfile.TemporaryDirectory() as tmp:
            file_list = "\n".join(remote_files)
            tar_cmd = f"echo '{file_list}' | xargs tar -czf - 2>/dev/null"
            scp_proc = subprocess.run(
                ["ssh", REMOTE_HOST, tar_cmd],
                capture_output=True,
            )
            if scp_proc.returncode != 0 or not scp_proc.stdout:
                print(f"  [ERR] tar failed for {exp_name}")
                missing += 1
                continue

            # Write tar bytes and extract
            tar_bytes = scp_proc.stdout
            import io
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                tf.extractall(tmp)

            # Move each extracted file to its target location
            for rf in remote_files:
                # Strip the leading "/" from absolute remote paths
                stripped = rf.lstrip("/")
                extracted = Path(tmp) / stripped
                if not extracted.exists():
                    print(f"  [---] {Path(rf).name} (extract failed)")
                    missing += 1
                    continue

                # Build local path: results/preds/{dataset}/{model}/{method}/seed_N/{file}
                # Remote path ends with: .../{inner_key}/{method}/seed_N/{file}
                inner_parts = Path(rf).parts
                try:
                    idx = next(
                        i for i, p in enumerate(inner_parts) if p == inner_key
                    )
                    rel_from_inner = Path(*inner_parts[idx + 1:])
                except StopIteration:
                    print(f"  [ERR] can't parse path: {rf}")
                    missing += 1
                    continue

                local = LOCAL_BASE / ds_canonical / model_slug / rel_from_inner
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extracted, local)
                rel = local.relative_to(LOCAL_BASE.parent.parent)
                print(f"  [OK ] {rel}  ({local.stat().st_size:,} B)")
                pulled += 1

    return pulled, missing


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview transfers without copying")
    parser.add_argument("--datasets", nargs="+", default=MEMMAP_DATASETS, metavar="D",
                        help="Datasets to pull (default: all memmap datasets)")
    parser.add_argument("--models", nargs="+", default=MODELS, metavar="M",
                        help="Model slugs to pull (default: all)")
    parser.add_argument("--sampling-only", action="store_true",
                        help="Pull only sampling method files")
    parser.add_argument("--training-only", action="store_true",
                        help="Pull only training baseline files")
    args = parser.parse_args()

    do_sampling = not args.training_only
    do_training = not args.sampling_only

    total_pulled = total_missing = 0

    if do_sampling:
        p, m = pull_sampling(args.datasets, args.models, args.dry_run)
        total_pulled += p
        total_missing += m

    if do_training:
        p, m = pull_training(args.datasets, args.models, args.dry_run)
        total_pulled += p
        total_missing += m

    tag = "[dry-run] " if args.dry_run else ""
    print(f"\nDone. {tag}Pulled: {total_pulled}  Missing/skipped: {total_missing}")


if __name__ == "__main__":
    main()
