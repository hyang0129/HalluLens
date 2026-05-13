"""Phase 0: Generate stratified train subset indices and SearchQA test cap indices.

Outputs:
  output/sep_subset_{dataset}_{model_name}_seed42.json   for each (dataset, model)
  output/searchqa_test_cap_{model_name}_seed42.json       for each model

Run before any GPU work. Fast, CPU-only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    DATASETS,
    MODELS,
    SAMPLING_DATASETS,
    eval_results_json,
    generation_jsonl,
    model_name,
    searchqa_test_cap_path,
    subset_index_path,
)

SUBSET_SIZE = 5000
SEARCHQA_TEST_CAP = 10_000
SEED = 42


def compute_train_subset(dataset: str, mid: str) -> dict:
    eval_path = eval_results_json(dataset, mid, "train")
    if not eval_path.exists():
        print(f"  SKIP {dataset}/{model_name(mid)}: eval_results not found at {eval_path}")
        return None

    with open(eval_path) as f:
        data = json.load(f)
    labels = np.array(data["halu_test_res"], dtype=int)
    n = len(labels)
    size = min(SUBSET_SIZE, n)

    if size < n:
        # Stratified subset
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=SEED)
        subset_idx, _ = next(splitter.split(np.zeros(n), labels))
        subset_idx = sorted(subset_idx.tolist())
    else:
        subset_idx = list(range(n))

    hallu_count = int(labels[subset_idx].sum())
    print(
        f"  {dataset}/{model_name(mid)}: n_total={n}, subset={len(subset_idx)}, "
        f"hallu={hallu_count}/{len(subset_idx)} "
        f"({'capped' if size < n else 'full'})"
    )

    return {
        "dataset": dataset,
        "model": mid,
        "question_ids": subset_idx,
        "seed": SEED,
        "n_total": n,
        "subset_size": len(subset_idx),
        "hallu_count": hallu_count,
        "stratified": size < n,
    }


def compute_searchqa_test_cap(mid: str) -> dict:
    gen_path = generation_jsonl("searchqa", mid, "test")
    if not gen_path.exists():
        print(f"  SKIP searchqa test cap for {model_name(mid)}: {gen_path} not found")
        return None

    df = pd.read_json(gen_path, lines=True)
    n = len(df)
    cap = min(SEARCHQA_TEST_CAP, n)

    rng = np.random.default_rng(SEED)
    cap_idx = sorted(rng.choice(n, size=cap, replace=False).tolist())

    print(f"  searchqa test cap / {model_name(mid)}: n_total={n}, cap={cap}")

    return {
        "model": mid,
        "question_ids": cap_idx,
        "seed": SEED,
        "n_total": n,
        "cap_size": cap,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate subset and cap index files.")
    parser.add_argument(
        "--model",
        nargs="+",
        default=MODELS,
        help="HuggingFace model IDs to process (default: all)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=SAMPLING_DATASETS,
        help="Datasets to compute train subsets for (default: all sampling datasets)",
    )
    args = parser.parse_args()

    Path("output").mkdir(exist_ok=True)

    for mid in args.model:
        print(f"\n=== {model_name(mid)} ===")

        # Train subsets
        for ds in args.dataset:
            out_path = subset_index_path(ds, mid)
            if out_path.exists():
                print(f"  {ds}: already exists — skipping")
                continue
            result = compute_train_subset(ds, mid)
            if result is not None:
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  -> {out_path}")

        # SearchQA test cap
        if "searchqa" in args.dataset:
            cap_path = searchqa_test_cap_path(mid)
            if cap_path.exists():
                print(f"  searchqa test cap: already exists — skipping")
            else:
                result = compute_searchqa_test_cap(mid)
                if result is not None:
                    with open(cap_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"  -> {cap_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
