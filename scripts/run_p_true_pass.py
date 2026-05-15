"""P(true) self-evaluation pass — per (dataset, model) cell.

Scores each row in generation.jsonl with P(true) (Kadavath et al. 2022).
Writes ptrue.jsonl to output/p_true/{dataset}/{model_name}/.
Resumable: rows already present in ptrue.jsonl are skipped.

Usage:
    # Full test split (all rows — no cap, even for searchqa)
    python scripts/run_p_true_pass.py \\
        --dataset hotpotqa --model meta-llama/Llama-3.1-8B-Instruct

    # Smoke test (first 50 rows, then validates resumability)
    python scripts/run_p_true_pass.py \\
        --dataset sciq --model meta-llama/Llama-3.1-8B-Instruct \\
        --smoketest
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.p_true.paths import DATASETS, ptrue_scores_path, resolve_split_paths
from tasks.p_true.scorer import PTrueScorer
from tasks.sampling_baselines.paths import model_name

SMOKETEST_ROWS = 50


def load_labels(eval_path) -> list:
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_json not found: {eval_path}")
    with open(eval_path) as f:
        data = json.load(f)
    return data["halu_test_res"]


def main():
    parser = argparse.ArgumentParser(description="P(true) scoring pass.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Process first 50 rows only, then verify resumability.",
    )
    args = parser.parse_args()

    paths = resolve_split_paths(args.dataset, args.model, args.split)
    gen_path = paths["generation_jsonl"]
    eval_path = paths["eval_json"]
    if not gen_path.exists():
        print(f"ERROR: generation.jsonl not found: {gen_path}")
        sys.exit(1)

    labels = load_labels(eval_path)
    out_path = ptrue_scores_path(args.dataset, args.model, args.split)

    row_indices = None
    if args.smoketest:
        row_indices = list(range(SMOKETEST_ROWS))
        print(f"SMOKETEST: processing first {SMOKETEST_ROWS} rows only.")

    print(
        f"Dataset: {args.dataset} | Split: {args.split} | "
        f"Model: {model_name(args.model)} | batch_size={args.batch_size}"
    )

    scorer = PTrueScorer(model_name=args.model, batch_size=args.batch_size)
    scorer.run(str(gen_path), str(out_path), row_indices=row_indices, labels=labels)

    if args.smoketest:
        # Verify resumability: re-running should be a no-op
        print("\nSmoketest: verifying resumability (re-run should skip all rows)...")
        scorer2 = PTrueScorer(model_name=args.model, batch_size=args.batch_size)
        scorer2._model = scorer._model
        scorer2._tokenizer = scorer._tokenizer
        scorer2._tok_a = scorer._tok_a
        scorer2._tok_b = scorer._tok_b
        scorer2.run(str(gen_path), str(out_path), row_indices=row_indices, labels=labels)

        # Spot-check 5 rows
        print("\nSmoketest spot-check (5 rows):")
        with open(out_path) as f:
            rows = [json.loads(line) for line in f][:5]
        for r in rows:
            print(
                f"  row_idx={r['row_idx']:5d} | p_true={r['p_true']:.4f} | "
                f"p_true_reversed={r['p_true_reversed']:.4f} | "
                f"halu_label={r['halu_label']}"
            )
        print(f"\nSmoketest passed. Output: {out_path}")


if __name__ == "__main__":
    main()
