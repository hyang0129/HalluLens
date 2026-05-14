"""Benchmark Phase 1 sampling throughput vs. batch size.

For each batch size: 2 warmup batches + 5 measured batches. Reports mean
rows/sec, mean wall per batch, and peak VRAM across the measured batches.
Stops on CUDA OOM. Uses real hotpotqa prompts from generation.jsonl to keep
prompt-length distribution realistic.

Usage:
    python scripts/benchmark_sampling_batch.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --batch-sizes 16,32,64,128,256
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import generation_jsonl, model_name

WARMUP_BATCHES = 2
MEASURED_BATCHES = 5
MAX_TOKENS = 64
TEMPERATURE = 1.0
SEED = 42


def load_prompts(dataset: str, model_id: str, split: str, n: int) -> list:
    path = generation_jsonl(dataset, model_id, split)
    if not path.exists():
        raise FileNotFoundError(f"No generation.jsonl at {path}")
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    # Cycle if we still need more (smoke fixtures rarely smaller than 100, but be safe)
    while len(prompts) < n:
        prompts.extend(prompts)
    return prompts[:n]


def run_one_batch(model, tokenizer, prompts: list) -> None:
    """Match SamplingPass._infer_batch generation kwargs exactly."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    torch.manual_seed(SEED)
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            output_hidden_states=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )


def benchmark_batch_size(model, tokenizer, prompts: list, batch_size: int) -> dict:
    """Run warmup + measured batches at one batch_size. Returns metrics dict.

    Raises torch.cuda.OutOfMemoryError on OOM.
    """
    needed = (WARMUP_BATCHES + MEASURED_BATCHES) * batch_size
    pool = (prompts * ((needed // len(prompts)) + 1))[:needed]

    # Warmup
    for w in range(WARMUP_BATCHES):
        batch = pool[w * batch_size : (w + 1) * batch_size]
        run_one_batch(model, tokenizer, batch)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for i in range(MEASURED_BATCHES):
        start = WARMUP_BATCHES + i
        batch = pool[start * batch_size : (start + 1) * batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one_batch(model, tokenizer, batch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    mean_wall = sum(times) / len(times)
    rows_per_sec = batch_size / mean_wall

    return {
        "batch_size": batch_size,
        "n_warmup": WARMUP_BATCHES,
        "n_measured": MEASURED_BATCHES,
        "batch_walls_sec": [round(t, 3) for t in times],
        "mean_wall_sec": round(mean_wall, 3),
        "rows_per_sec": round(rows_per_sec, 2),
        "peak_alloc_gb": round(peak_alloc_gb, 2),
        "peak_reserved_gb": round(peak_reserved_gb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark sampling batch size vs throughput.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--batch-sizes",
        default="16,32,64,128,256",
        help="Comma-separated batch sizes to try (ascending).",
    )
    parser.add_argument("--out", default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    assert batch_sizes == sorted(batch_sizes), "Batch sizes must be ascending"

    # Defer model load until after CLI parse
    from activation_logging.server import get_model_and_tokenizer

    print(f"Loading model: {args.model}")
    model, tokenizer = get_model_and_tokenizer(args.model, None)
    model.eval()

    print(f"Loading prompts from {args.dataset}/{args.split}")
    max_bs = max(batch_sizes)
    needed = (WARMUP_BATCHES + MEASURED_BATCHES) * max_bs
    prompts = load_prompts(args.dataset, args.model, args.split, needed)
    print(f"  Loaded {len(prompts)} prompts (mean tokens: "
          f"{sum(len(tokenizer.encode(p)) for p in prompts[:50]) // 50})")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
                          if torch.cuda.is_available() else None,
        "rows": [],
    }

    print()
    print(f"{'batch':>6} {'mean wall':>10} {'rows/sec':>10} {'peak GB':>10} {'rsv GB':>10}")
    print("-" * 50)

    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            r = benchmark_batch_size(model, tokenizer, prompts, bs)
            results["rows"].append(r)
            print(f"{r['batch_size']:>6} {r['mean_wall_sec']:>10.3f} "
                  f"{r['rows_per_sec']:>10.2f} {r['peak_alloc_gb']:>10.2f} "
                  f"{r['peak_reserved_gb']:>10.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} OOM — stopping")
            results["rows"].append({"batch_size": bs, "oom": True})
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"{bs:>6} ERROR: {type(e).__name__}: {e}")
            results["rows"].append({"batch_size": bs, "error": str(e)})
            break

    print()
    print("=" * 50)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote: {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
