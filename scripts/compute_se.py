"""Phase 3: Compute Semantic Entropy from NLI matrices.

Outputs se_labels.jsonl with discrete_se and length_normalized_se per question.
CPU-only; runs after Phase 2.

Usage:
    python scripts/compute_se.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    model_name,
    nli_matrix_path,
    se_labels_path,
    selfcheck_samples_path,
)
from tasks.sampling_baselines.se import score_files


def main():
    parser = argparse.ArgumentParser(description="Compute Semantic Entropy labels.")
    parser.add_argument("--dataset", required=True, choices=SAMPLING_DATASETS)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID used for generation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Bidirectional NLI threshold")
    args = parser.parse_args()

    samples = selfcheck_samples_path(args.dataset, args.model, args.split)
    nli = nli_matrix_path(args.dataset, args.model, args.split)
    out = se_labels_path(args.dataset, args.model, args.split)

    for p, label in [(samples, "samples"), (nli, "nli_matrix")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    print(f"SE | {args.dataset}/{args.split} | {model_name(args.model)} | threshold={args.threshold}")
    score_files(str(samples), str(nli), str(out), threshold=args.threshold)


if __name__ == "__main__":
    main()
