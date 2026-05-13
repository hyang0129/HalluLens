"""Phase 1: K=10 stochastic sampling pass.

Generates K samples per question at T=1.0 for the scoped (dataset, split, model) cells.
Writes selfcheck_samples.jsonl to output/sampling_baselines/{dataset[_train]}/{model_name}/.

Usage:
    # Test split (all rows)
    python scripts/run_sampling_pass.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct

    # Train split (auto-loads subset index from compute_subsets.py output)
    python scripts/run_sampling_pass.py \\
        --dataset hotpotqa --split train \\
        --model meta-llama/Llama-3.1-8B-Instruct

    # Smoke test (50 rows only, validates alignment)
    python scripts/run_sampling_pass.py \\
        --dataset hotpotqa --split test \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --smoketest
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    generation_jsonl,
    model_name,
    searchqa_test_cap_path,
    selfcheck_samples_path,
    subset_index_path,
)
from tasks.sampling_baselines.sampler import SamplingPass

SMOKETEST_ROWS = 50


def load_row_indices(dataset: str, mid: str, split: str) -> list | None:
    """Return row indices to process, or None (= all rows)."""
    if split == "test":
        if dataset == "searchqa":
            cap_path = searchqa_test_cap_path(mid)
            if not cap_path.exists():
                raise FileNotFoundError(
                    f"SearchQA test cap index not found: {cap_path}\n"
                    "Run scripts/compute_subsets.py first."
                )
            with open(cap_path) as f:
                return json.load(f)["question_ids"]
        return None  # all test rows for other datasets

    # train split: load stratified subset
    idx_path = subset_index_path(dataset, mid)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Subset index not found: {idx_path}\n"
            "Run scripts/compute_subsets.py first."
        )
    with open(idx_path) as f:
        return json.load(f)["question_ids"]


def main():
    parser = argparse.ArgumentParser(description="K-sample stochastic generation pass.")
    parser.add_argument("--dataset", required=True, choices=SAMPLING_DATASETS)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Process only first 50 rows, then validate alignment.",
    )
    args = parser.parse_args()

    gen_path = generation_jsonl(args.dataset, args.model, args.split)
    if not gen_path.exists():
        print(f"ERROR: generation.jsonl not found: {gen_path}")
        sys.exit(1)

    out_path = str(selfcheck_samples_path(args.dataset, args.model, args.split))

    row_indices = load_row_indices(args.dataset, args.model, args.split)

    if args.smoketest:
        # Override to first 50 rows (or first 50 of the subset)
        if row_indices is not None:
            row_indices = row_indices[:SMOKETEST_ROWS]
        else:
            row_indices = list(range(SMOKETEST_ROWS))
        print(f"SMOKETEST: processing {len(row_indices)} rows only.")

    sampler = SamplingPass(
        model_name=args.model,
        K=args.K,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print(
        f"Dataset: {args.dataset} | Split: {args.split} | Model: {model_name(args.model)}\n"
        f"K={args.K} | T={args.temperature} | max_tokens={args.max_tokens} | seed={args.seed}"
    )

    sampler.run(str(gen_path), out_path, row_indices=row_indices)

    if args.smoketest:
        print("Validating alignment...")
        sampler.validate_alignment(str(gen_path), out_path, n=min(SMOKETEST_ROWS, 50))


if __name__ == "__main__":
    main()
