"""Phase 4: Compute SelfCheckGPT scores (NLI, BERTScore, n-gram).

NLI score reuses the precomputed Phase 2 matrix — no GPU needed.
BERTScore and n-gram are CPU-only.

Outputs selfcheck_scores.jsonl with {nli, bertscore, ngram} per question.

Usage:
    python scripts/compute_selfcheck.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct

    # Skip BERTScore (slow) for a quick run:
    python scripts/compute_selfcheck.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --no-bertscore
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    model_name,
    nli_matrix_path,
    selfcheck_samples_path,
    selfcheck_scores_path,
)
from tasks.sampling_baselines.selfcheck import score_files


def main():
    parser = argparse.ArgumentParser(description="Compute SelfCheckGPT scores.")
    parser.add_argument("--dataset", required=True, choices=SAMPLING_DATASETS)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID used for generation")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore variant")
    parser.add_argument("--no-ngram", action="store_true", help="Skip n-gram variant")
    parser.add_argument(
        "--bertscore-batch-size", type=int, default=64, help="Pairs per BERTScore call"
    )
    args = parser.parse_args()

    samples = selfcheck_samples_path(args.dataset, args.model, args.split)
    nli = nli_matrix_path(args.dataset, args.model, args.split)
    out = selfcheck_scores_path(args.dataset, args.model, args.split)

    for p, label in [(samples, "samples"), (nli, "nli_matrix")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    print(
        f"SelfCheckGPT | {args.dataset}/{args.split} | {model_name(args.model)}\n"
        f"NLI=True | BERTScore={not args.no_bertscore} | n-gram={not args.no_ngram}"
    )
    score_files(
        str(samples),
        str(nli),
        str(out),
        run_bertscore=not args.no_bertscore,
        run_ngram=not args.no_ngram,
        bertscore_batch_size=args.bertscore_batch_size,
    )


if __name__ == "__main__":
    main()
