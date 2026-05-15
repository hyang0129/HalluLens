"""P(true) driver: load model once, score all datasets.

Saves ~30s × 5 model-load overhead compared to launching run_p_true_pass.py
once per dataset. Preferred for single-node serial execution.

Usage:
    python scripts/run_p_true_for_model.py \\
        --model meta-llama/Llama-3.1-8B-Instruct

    python scripts/run_p_true_for_model.py \\
        --model Qwen/Qwen3-8B \\
        --datasets hotpotqa,nq,sciq
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.p_true.paths import DATASETS, ptrue_scores_path, resolve_split_paths
from tasks.p_true.scorer import PTrueScorer
from tasks.sampling_baselines.paths import model_name


def load_labels(eval_path) -> list:
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_json not found: {eval_path}")
    with open(eval_path) as f:
        data = json.load(f)
    return data["halu_test_res"]


def main():
    parser = argparse.ArgumentParser(description="P(true) — load model once, score all datasets.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names (default: all 6).",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    invalid = [d for d in datasets if d not in DATASETS]
    if invalid:
        print(f"ERROR: unknown datasets: {invalid}")
        sys.exit(1)

    print(f"Model: {model_name(args.model)}")
    print(f"Datasets: {datasets}")
    print(f"Split: {args.split} | batch_size={args.batch_size}\n")

    scorer = PTrueScorer(model_name=args.model, batch_size=args.batch_size)

    for ds in datasets:
        try:
            paths = resolve_split_paths(ds, args.model, args.split)
        except (FileNotFoundError, KeyError) as e:
            print(f"SKIP {ds}: {e}")
            continue
        gen_path = paths["generation_jsonl"]
        eval_path = paths["eval_json"]
        if not gen_path.exists():
            print(f"SKIP {ds}: generation.jsonl not found ({gen_path})")
            continue

        try:
            labels = load_labels(eval_path)
        except FileNotFoundError as e:
            print(f"SKIP {ds}: {e}")
            continue

        out_path = ptrue_scores_path(ds, args.model, args.split)
        print(f"=== {ds} ===")
        scorer.run(str(gen_path), str(out_path), row_indices=None, labels=labels)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
