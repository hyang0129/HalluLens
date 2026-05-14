"""Audit token-length distribution of NLI inputs (CPU, no GPU needed).

Loads the DeBERTa-v2-xlarge-mnli tokenizer (~2 MB, no model weights) and
tokenizes the *greedy generations* across all 5 sampling datasets × both
models. The K=10 stochastic samples are produced with max_new_tokens=64 from
the same prompts, so greedy length is a reasonable proxy for the per-sample
length distribution. Each NLI pair is (sample_i, sample_j), so the relevant
input length is roughly 2 × per-sample tokens + 3 special tokens.

Output:
  - Per-cell percentile table (p50/p90/p95/p99/max)
  - Pooled overall stats
  - Recommended max_length cap = ceil((2 * overall_max + 3) * margin) rounded
    to the next power of 2 or 32 boundary for tidiness.

Usage:
    python scripts/profile_nli_input_lengths.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    generation_jsonl,
    model_name,
)

PERCENTILES = (50, 90, 95, 99)
NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"
MAX_ROWS_PER_CELL = 5000


def load_texts(dataset: str, model_id: str, split: str, n_max: int) -> list:
    path = generation_jsonl(dataset, model_id, split)
    if not path.exists():
        return []
    texts = []
    with open(path) as f:
        for line in f:
            t = (json.loads(line).get("generation") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= n_max:
                break
    return texts


def round_up_to(n: int, granularity: int = 32) -> int:
    return ((n + granularity - 1) // granularity) * granularity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=NLI_MODEL)
    parser.add_argument(
        "--models",
        default="meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen3-8B",
    )
    parser.add_argument("--datasets", default=",".join(SAMPLING_DATASETS))
    parser.add_argument("--splits", default="test,train",
                        help="Comma-separated splits to audit.")
    parser.add_argument("--margin", type=float, default=1.5,
                        help="Multiplicative safety margin over observed 2× max.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer} (CPU)")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print()
    print(f"{'split':>6} {'dataset':>10} {'model':>26} {'n':>6} "
          f"{'p50':>5} {'p90':>5} {'p95':>5} {'p99':>5} {'max':>5}")
    print("-" * 86)

    per_cell = []
    all_lens = []
    for split in splits:
        for ds in datasets:
            for mid in models:
                texts = load_texts(ds, mid, split, MAX_ROWS_PER_CELL)
                if not texts:
                    print(f"{split:>6} {ds:>10} {model_name(mid):>26} "
                          f"{'-':>6} (missing)")
                    continue
                lens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
                arr = np.array(lens)
                stats = {
                    "split": split, "dataset": ds, "model": model_name(mid),
                    "n": len(lens),
                    "p50": int(np.percentile(arr, 50)),
                    "p90": int(np.percentile(arr, 90)),
                    "p95": int(np.percentile(arr, 95)),
                    "p99": int(np.percentile(arr, 99)),
                    "max": int(arr.max()),
                }
                per_cell.append(stats)
                all_lens.extend(lens)
                print(f"{split:>6} {ds:>10} {model_name(mid):>26} "
                      f"{stats['n']:>6} {stats['p50']:>5} {stats['p90']:>5} "
                      f"{stats['p95']:>5} {stats['p99']:>5} {stats['max']:>5}")

    if not all_lens:
        print("\nNo data — aborting.")
        return

    pooled = np.array(all_lens)
    print()
    print("=== Pooled across all cells ===")
    print(f"  n             = {len(pooled)}")
    for p in PERCENTILES:
        print(f"  p{p:>2}           = {int(np.percentile(pooled, p))}")
    print(f"  max           = {int(pooled.max())}")

    pair_max = 2 * int(pooled.max()) + 3  # premise + hypothesis + [CLS][SEP][SEP]
    suggested = round_up_to(int(np.ceil(pair_max * args.margin)), granularity=32)
    print(f"\nNLI pair length upper bound (2*max + 3 specials): {pair_max}")
    print(f"Recommended max_length (×{args.margin} margin, rounded to 32): {suggested}")
    print(f"  (current production value: 512; reference paper uses 512)")

    summary = {
        "tokenizer": args.tokenizer,
        "max_rows_per_cell": MAX_ROWS_PER_CELL,
        "per_cell": per_cell,
        "pooled": {
            "n": len(pooled),
            **{f"p{p}": int(np.percentile(pooled, p)) for p in PERCENTILES},
            "max": int(pooled.max()),
        },
        "pair_max_tokens": pair_max,
        "margin": args.margin,
        "suggested_max_length": suggested,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
