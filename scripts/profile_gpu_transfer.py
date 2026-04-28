"""
GPU→CPU transfer benchmark: per-layer loop vs stacked single transfer.

Simulates the hidden_states tuple that HFTransformersAdapter receives from
model.generate(output_hidden_states=True):
  - tuple of n_layers tensors, each shape [batch_size, seq_len, hidden_dim] on GPU

Tests three strategies:
  A) per_layer_loop   — current approach: iterate layers, .cpu() each separately
  B) stack_transfer   — stack all layers on GPU first, single .cpu() call
  C) pinned_nonblock  — stack + pre-allocated pinned buffer + non_blocking=True

No model or tokenizer needed — tensors are created directly on GPU.

Run on the GPU node:
  python scripts/profile_gpu_transfer.py
"""

import time
import torch
import argparse

# Match Qwen3-8B config used in profiling
N_LAYERS   = 37
HIDDEN_DIM = 4096
SEQ_LEN    = 64
DTYPE      = torch.float16

N_WARMUP = 5
N_TRIALS = 50
BATCH_SIZES = [8, 16, 32, 64]


def make_gpu_hidden_states(batch_size: int, device: torch.device) -> tuple:
    """Simulate the hidden_states tuple from model.generate(output_hidden_states=True).
    Each element: [batch_size, seq_len, hidden_dim] on GPU."""
    return tuple(
        torch.randn(batch_size, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE, device=device)
        for _ in range(N_LAYERS)
    )


# ── Strategy A: per-layer blocking .cpu() (current code) ─────────────────────

def transfer_per_layer_loop(hidden_states: tuple) -> list:
    """Mirrors zarr_activations_logger.py lines 760-772."""
    result = []
    for layer_tensor in hidden_states:
        arr = layer_tensor.cpu().numpy()
        result.append(arr)
    return result


# ── Strategy B: stack on GPU, single .cpu() ──────────────────────────────────

def transfer_stack_single(hidden_states: tuple):
    """Stack all layers into one tensor, then one PCIe transfer."""
    stacked = torch.stack(list(hidden_states), dim=0)  # [n_layers, bs, seq, hidden]
    return stacked.cpu().numpy()


# ── Strategy C: stack + pinned buffer + non_blocking ─────────────────────────

class PinnedTransfer:
    """Pre-allocates a page-locked CPU buffer; reuses it across calls."""

    def __init__(self, batch_size: int):
        self.buffer = torch.empty(
            (N_LAYERS, batch_size, SEQ_LEN, HIDDEN_DIM),
            dtype=DTYPE,
            pin_memory=True,
        )

    def transfer(self, hidden_states: tuple):
        stacked = torch.stack(list(hidden_states), dim=0)  # [n_layers, bs, seq, hidden]
        self.buffer.copy_(stacked, non_blocking=True)
        torch.cuda.synchronize()
        return self.buffer.numpy()


# ── Benchmark harness ─────────────────────────────────────────────────────────

def bench(fn, hidden_states, n_warmup, n_trials):
    for _ in range(n_warmup):
        fn(hidden_states)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        fn(hidden_states)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batch_time = elapsed / n_trials
    return batch_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found — run this on the GPU node.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {gpu_name}  ({vram_gb:.0f} GB)")
    print(f"Config: {N_LAYERS} layers × SEQ={SEQ_LEN} × HIDDEN={HIDDEN_DIM} × {DTYPE}")
    print(f"Trials: {args.n_warmup} warmup + {args.n_trials} measured\n")

    header = f"{'Strategy':<20} {'BS':>4} {'Batch(s)':>10} {'Samp/s':>10} {'MB/batch':>10} {'GB/s':>8}"
    print(header)
    print("-" * len(header))

    for bs in args.batch_sizes:
        hidden_states = make_gpu_hidden_states(bs, device)
        bytes_per_batch = N_LAYERS * bs * SEQ_LEN * HIDDEN_DIM * 2  # float16
        mb_per_batch = bytes_per_batch / 1e6

        # Strategy A
        t = bench(lambda hs=hidden_states: transfer_per_layer_loop(hs),
                  hidden_states, args.n_warmup, args.n_trials)
        print(f"{'per_layer_loop':<20} {bs:>4} {t:>10.4f} {bs/t:>10.1f} {mb_per_batch:>10.0f} {mb_per_batch/1e3/t:>8.2f}")

        # Strategy B
        t = bench(lambda hs=hidden_states: transfer_stack_single(hs),
                  hidden_states, args.n_warmup, args.n_trials)
        print(f"{'stack_single':<20} {bs:>4} {t:>10.4f} {bs/t:>10.1f} {mb_per_batch:>10.0f} {mb_per_batch/1e3/t:>8.2f}")

        # Strategy C
        pinned = PinnedTransfer(bs)
        t = bench(lambda hs=hidden_states, p=pinned: p.transfer(hs),
                  hidden_states, args.n_warmup, args.n_trials)
        print(f"{'pinned_nonblock':<20} {bs:>4} {t:>10.4f} {bs/t:>10.1f} {mb_per_batch:>10.0f} {mb_per_batch/1e3/t:>8.2f}")

        print()

    print("Notes:")
    print("  Batch(s)  — wall time for one batch transfer (lower is better)")
    print("  Samp/s    — samples transferred per second")
    print("  GB/s      — effective PCIe bandwidth utilized")


if __name__ == "__main__":
    main()
