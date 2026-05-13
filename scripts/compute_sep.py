"""Phase 5: Fit SEP-SE and SEP-binary probes.

SEP-binary works on all 6 datasets including MMLU (no sampling needed).
SEP-SE requires se_labels.jsonl (train split) and skips MMLU.

Usage:
    # All datasets, both models (runs after SE labels are available)
    python scripts/compute_sep.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 22

    python scripts/compute_sep.py \\
        --model Qwen/Qwen3-8B \\
        --layer 24       # from layer sweep

    # Single dataset
    python scripts/compute_sep.py \\
        --dataset hotpotqa \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 22

    # MMLU SEP-binary (no SE, can run any time after main inference)
    python scripts/compute_sep.py \\
        --dataset mmlu \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 22 \\
        --no-sep-se
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    DATASETS,
    SAMPLING_DATASETS,
    model_name,
    se_labels_path,
    sep_results_path,
    subset_index_path,
)
from tasks.sampling_baselines.sep import run_sep


def load_subset_indices(dataset: str, mid: str) -> Optional[list]:
    idx_path = subset_index_path(dataset, mid)
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        return json.load(f)["question_ids"]


def main():
    parser = argparse.ArgumentParser(description="Fit SEP probes.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to process (default: all)",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--layer", required=True, type=int, help="Layer index for feature extraction"
    )
    parser.add_argument(
        "--no-sep-se",
        action="store_true",
        help="Skip SEP-SE (e.g. for MMLU or when SE labels not yet available)",
    )
    args = parser.parse_args()

    datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]

    print(f"SEP probes | {model_name(args.model)} | layer={args.layer}")

    for ds in datasets:
        print(f"\n--- {ds} ---")
        out_path = sep_results_path(ds, args.model)

        if out_path.exists():
            print(f"  Already exists — skipping. Delete to rerun: {out_path}")
            continue

        # SEP-SE: needs train subset + SE labels; skip MMLU and when --no-sep-se
        run_se = not args.no_sep_se and ds in SAMPLING_DATASETS
        train_subset = None
        se_train_path = None

        if run_se:
            train_subset = load_subset_indices(ds, args.model)
            se_train = se_labels_path(ds, args.model, "train")
            if not se_train.exists():
                print(
                    f"  WARNING: SE labels not found ({se_train}) — skipping SEP-SE for {ds}."
                )
                run_se = False
            else:
                se_train_path = str(se_train)

        if not run_se:
            train_subset = None
            se_train_path = None

        try:
            run_sep(
                dataset=ds,
                model_id=args.model,
                layer_idx=args.layer,
                train_subset_indices=train_subset,
                se_labels_train_path=se_train_path,
                output_path=str(out_path),
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    print("\nSEP done.")


if __name__ == "__main__":
    main()
