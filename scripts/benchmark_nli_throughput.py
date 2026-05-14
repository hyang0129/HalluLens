"""NLI throughput batch-size sweep (fp32, paper-faithful).

Measures rows/sec + peak VRAM at increasing batch sizes for the production
NLI scorer config. Per size: 2 warmup + 5 measured batches. Stops on OOM.

dtype stays fp32 to match the jlko/semantic_uncertainty reference (which loads
via from_pretrained without dtype kwarg).

Usage:
    python scripts/benchmark_nli_throughput.py \
        --batch-sizes 256,512,1024,2048
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import generation_jsonl

K = 10
PAIRS_PER_ROW = (K + 1) * (K + 1) - (K + 1)  # 110
WARMUP_BATCHES = 2
MEASURED_BATCHES = 5


def load_texts(dataset: str, model_id: str, split: str, n: int) -> list:
    path = generation_jsonl(dataset, model_id, split)
    if not path.exists():
        raise FileNotFoundError(f"No generation.jsonl at {path}")
    texts = []
    with open(path) as f:
        for line in f:
            t = (json.loads(line).get("generation") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= n:
                break
    while len(texts) < n:
        texts.extend(texts)
    return texts[:n]


def build_pairs(texts: list, n_rows: int) -> List[Tuple[str, str]]:
    pairs = []
    pool = len(texts)
    for r in range(n_rows):
        row_texts = [texts[(r * (K + 1) + j) % pool] for j in range(K + 1)]
        for i in range(K + 1):
            for j in range(K + 1):
                if i != j:
                    pairs.append((row_texts[i], row_texts[j]))
    return pairs


def run_batch(tok, model, batch: list, max_length: int):
    premises = [p for p, _ in batch]
    hypotheses = [h for _, h in batch]
    enc = tok(premises, hypotheses, padding=True, truncation=True,
              max_length=max_length, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**enc).logits


def benchmark_size(tok, model, pairs, batch_size: int, max_length: int) -> dict:
    needed = (WARMUP_BATCHES + MEASURED_BATCHES) * batch_size
    pool = (pairs * ((needed // len(pairs)) + 1))[:needed]
    for w in range(WARMUP_BATCHES):
        run_batch(tok, model, pool[w * batch_size : (w + 1) * batch_size], max_length)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(MEASURED_BATCHES):
        s = WARMUP_BATCHES + i
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_batch(tok, model, pool[s * batch_size : (s + 1) * batch_size], max_length)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mean_wall = sum(times) / len(times)
    pairs_per_sec = batch_size / mean_wall
    return {
        "batch_size": batch_size,
        "max_length": max_length,
        "mean_wall_sec": round(mean_wall, 4),
        "pairs_per_sec": round(pairs_per_sec, 1),
        "sec_per_row": round(PAIRS_PER_ROW / pairs_per_sec, 4),
        "peak_alloc_gb": round(peak, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--gen-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-sizes", default="256,512,1024,2048")
    parser.add_argument("--max-length", type=int, default=512,
                        help="NLI tokenizer max_length (paper default 512).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    batch_sizes = sorted(int(b) for b in args.batch_sizes.split(",") if b.strip())

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"Loading NLI model: {args.model} (fp32)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to("cuda").eval()

    biggest = max(batch_sizes)
    n_rows = (WARMUP_BATCHES + MEASURED_BATCHES) * biggest // PAIRS_PER_ROW + 2
    n_texts = n_rows * (K + 1)
    print(f"Loading {n_texts} answer strings from {args.dataset}/{args.split}...")
    texts = load_texts(args.dataset, args.gen_model, args.split, n_texts)
    pairs = build_pairs(texts, n_rows)
    print(f"  Built {len(pairs)} pairs across {n_rows} synthetic rows.")

    results = []
    print()
    print(f"{'batch':>6} {'maxL':>5} {'mean wall':>10} {'pairs/s':>10} "
          f"{'sec/row':>10} {'peak GB':>10}")
    print("-" * 60)
    for bs in batch_sizes:
        gc.collect(); torch.cuda.empty_cache()
        try:
            r = benchmark_size(tok, model, pairs, bs, args.max_length)
            results.append(r)
            print(f"{r['batch_size']:>6} {r['max_length']:>5} "
                  f"{r['mean_wall_sec']:>10.4f} {r['pairs_per_sec']:>10.1f} "
                  f"{r['sec_per_row']:>10.4f} {r['peak_alloc_gb']:>10.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} {args.max_length:>5} OOM — stopping")
            results.append({"batch_size": bs, "max_length": args.max_length, "oom": True})
            break

    out = {
        "model": args.model,
        "dtype": "fp32",
        "max_length": args.max_length,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
