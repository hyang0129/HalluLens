"""Phase 2: Compute (K+1)x(K+1) NLI matrices from selfcheck_samples.jsonl.

Outputs nli_matrix.jsonl alongside the samples file.
GPU-accelerated; uses cross-encoder/nli-deberta-v3-base.

Usage:
    python scripts/compute_nli_matrix.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct

    python scripts/compute_nli_matrix.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --batch-size 512
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.nli_scorer import NLI_MODEL_ID, NLIScorer
from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    model_name,
    nli_matrix_path,
    selfcheck_samples_path,
)


def main():
    parser = argparse.ArgumentParser(description="Compute NLI matrices from stochastic samples.")
    parser.add_argument("--dataset", required=True, choices=SAMPLING_DATASETS)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID used for generation")
    parser.add_argument("--nli-model", default=NLI_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=256, help="NLI pairs per forward pass")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if not set)")
    args = parser.parse_args()

    samples_path = selfcheck_samples_path(args.dataset, args.model, args.split)
    out_path = nli_matrix_path(args.dataset, args.model, args.split)

    if not samples_path.exists():
        print(f"ERROR: samples file not found: {samples_path}")
        print("Run scripts/run_sampling_pass.py first.")
        sys.exit(1)

    print(
        f"NLI matrix | {args.dataset}/{args.split} | {model_name(args.model)}\n"
        f"NLI model: {args.nli_model} | batch_size: {args.batch_size}"
    )

    scorer = NLIScorer(
        model_id=args.nli_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    scorer.score_file(str(samples_path), str(out_path))


if __name__ == "__main__":
    main()
