"""
Pure inference throughput benchmark for Qwen3-8B.
No activation logging, no disk writes. Just raw HF/PyTorch inference.
Tests batch sizes 8, 16, 32 over 50 batches each.
"""

import time
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 64
N_BATCHES = 50
BATCH_SIZES = [8, 16, 32]

SAMPLE_PROMPTS = [
    "Answer the question concisely.\n\nQuestion: What is the capital of France?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who wrote Hamlet?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What year did World War II end?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the chemical symbol for gold?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: How many bones are in the human body?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What planet is closest to the sun?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the speed of light?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who painted the Mona Lisa?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the largest ocean?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the boiling point of water in Celsius?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: How many continents are there?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the tallest mountain?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who invented the telephone?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the square root of 144?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What language is spoken in Brazil?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the largest country by area?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who developed the theory of relativity?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the currency of Japan?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: How many elements are in the periodic table?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the nearest star to Earth?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who was the first US president?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the largest planet in the solar system?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is DNA an abbreviation for?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: In what year did the Berlin Wall fall?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the longest river in the world?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: How many teeth does an adult human have?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What gas do plants absorb from the atmosphere?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the atomic number of carbon?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: Who wrote Pride and Prejudice?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the smallest country in the world?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What is the chemical formula for water?\n\nAnswer:",
    "Answer the question concisely.\n\nQuestion: What instrument measures atmospheric pressure?\n\nAnswer:",
]


def run_benchmark(model, tokenizer, batch_size, n_batches, device):
    print(f"\n{'='*60}")
    print(f"  Batch size: {batch_size}  |  Batches: {n_batches}  |  Max tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*60}")

    # Warmup: 2 batches not counted
    for _ in range(2):
        prompts = (SAMPLE_PROMPTS * ((batch_size // len(SAMPLE_PROMPTS)) + 1))[:batch_size]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    print(f"  Warmup done.")

    batch_times = []
    total_output_tokens = 0

    for i in range(n_batches):
        prompts = (SAMPLE_PROMPTS * ((batch_size // len(SAMPLE_PROMPTS)) + 1))[:batch_size]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_tokens = out.shape[1] - input_len
        total_output_tokens += gen_tokens * batch_size
        batch_times.append(elapsed)

        if (i + 1) % 10 == 0:
            recent = batch_times[-10:]
            print(f"  Batch {i+1:3d}/{n_batches}  "
                  f"last10_avg={np.mean(recent):.2f}s  "
                  f"samp/s={batch_size/np.mean(recent):.2f}  "
                  f"tok/s={MAX_NEW_TOKENS*batch_size/np.mean(recent):.0f}")

    arr = np.array(batch_times)
    # exclude first 5 batches (possible JIT/cache effects)
    steady = arr[5:]

    print(f"\n  Results (batches 6-{n_batches}, steady state):")
    print(f"    Mean batch time:    {np.mean(steady):.3f}s  (std={np.std(steady):.3f}s)")
    print(f"    Samples/sec:        {batch_size / np.mean(steady):.2f}")
    print(f"    Output tokens/sec:  {MAX_NEW_TOKENS * batch_size / np.mean(steady):.0f}")
    print(f"    GPU mem allocated:  {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return {
        "batch_size": batch_size,
        "mean_batch_s": float(np.mean(steady)),
        "std_batch_s": float(np.std(steady)),
        "samples_per_sec": batch_size / float(np.mean(steady)),
        "output_tokens_per_sec": MAX_NEW_TOKENS * batch_size / float(np.mean(steady)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--n-batches", type=int, default=N_BATCHES)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading model: {args.model}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                  device_map="auto")
    model.eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s  |  "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    results = []
    for bs in args.batch_sizes:
        r = run_benchmark(model, tokenizer, bs, args.n_batches, device)
        results.append(r)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'BS':>4}  {'Mean(s)':>8}  {'Samp/s':>8}  {'Tok/s':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['batch_size']:>4}  {r['mean_batch_s']:>8.3f}  "
              f"{r['samples_per_sec']:>8.2f}  {r['output_tokens_per_sec']:>8.0f}")


if __name__ == "__main__":
    main()
