"""
Disk throughput benchmark for zarr activation writes.
Simulates the data that HFTransformersAdapter produces per sample:
  - response_activations: [n_layers, seq_len, hidden_dim] float16
  - prompt_activations:   [n_layers, seq_len, hidden_dim] float16
  - logprobs arrays (small)

Tests two write strategies:
  A) Zarr sample-by-sample (current approach in run_inference_batched)
  B) Single flat numpy file per batch (baseline comparison)

Run for batch sizes 8, 16, 32.  N_TRIALS batches per config.
"""

import time
import os
import sys
import tempfile
import shutil
import numpy as np
import zarr
import argparse

# Qwen3-8B config
N_LAYERS   = 37
HIDDEN_DIM = 4096
SEQ_LEN    = 64     # max_new_tokens
DTYPE      = np.float16

N_TRIALS   = 20
BATCH_SIZES = [8, 16, 32]

ZARR_PATH_BASE = "/tmp/profile_zarr_bench"


def make_fake_activations(batch_size):
    """Generate random activation tensors matching what HFTransformersAdapter produces."""
    return {
        "response_activations": np.random.randn(batch_size, N_LAYERS, SEQ_LEN, HIDDEN_DIM).astype(DTYPE),
        "prompt_activations":   np.random.randn(batch_size, N_LAYERS, SEQ_LEN, HIDDEN_DIM).astype(DTYPE),
        "logprobs":             np.random.randn(batch_size, SEQ_LEN).astype(DTYPE),
    }


# ---------------------------------------------------------------------------
# Strategy A: Zarr store (one sample per chunk row, matching ZarrActivationsLogger)
# ---------------------------------------------------------------------------

def bench_zarr(batch_size, n_trials, zarr_path):
    """Write activations sample-by-sample into a zarr store (current approach)."""
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    expected_samples = batch_size * n_trials
    store = zarr.open(zarr_path, mode="w")
    resp_arr = store.create_array(
        "response_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(1, 1, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    prom_arr = store.create_array(
        "prompt_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(1, 1, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    logp_arr = store.create_array(
        "logprobs",
        shape=(expected_samples, SEQ_LEN),
        chunks=(1, SEQ_LEN),
        dtype=DTYPE,
    )

    batch_times = []
    sample_idx = 0

    for trial in range(n_trials):
        data = make_fake_activations(batch_size)

        t0 = time.perf_counter()
        for i in range(batch_size):
            resp_arr[sample_idx] = data["response_activations"][i]
            prom_arr[sample_idx] = data["prompt_activations"][i]
            logp_arr[sample_idx] = data["logprobs"][i]
            sample_idx += 1
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

    arr = np.array(batch_times)
    bytes_per_batch = batch_size * (
        N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 * 2  # resp + prompt, float16=2 bytes
        + SEQ_LEN * 2                              # logprobs
    )
    mb_per_batch = bytes_per_batch / 1e6

    return {
        "strategy": "zarr_per_sample",
        "batch_size": batch_size,
        "mean_s": float(np.mean(arr)),
        "std_s":  float(np.std(arr)),
        "samples_per_sec": batch_size / float(np.mean(arr)),
        "mb_per_sec": mb_per_batch / float(np.mean(arr)),
        "mb_per_batch": mb_per_batch,
    }


# ---------------------------------------------------------------------------
# Strategy B: Single .npy file per batch (whole batch written at once)
# ---------------------------------------------------------------------------

def bench_npy_batch(batch_size, n_trials, tmp_dir):
    """Write activations as a single numpy array per batch."""
    batch_times = []

    for trial in range(n_trials):
        data = make_fake_activations(batch_size)
        path = os.path.join(tmp_dir, f"batch_{trial:04d}.npy")

        t0 = time.perf_counter()
        # pack everything into one array to simulate a flat batch write
        combined = np.concatenate([
            data["response_activations"].reshape(batch_size, -1),
            data["prompt_activations"].reshape(batch_size, -1),
            data["logprobs"],
        ], axis=1)
        np.save(path, combined)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

    arr = np.array(batch_times)
    bytes_per_batch = batch_size * (
        N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 * 2 + SEQ_LEN * 2
    )
    mb_per_batch = bytes_per_batch / 1e6

    return {
        "strategy": "npy_batch",
        "batch_size": batch_size,
        "mean_s": float(np.mean(arr)),
        "std_s":  float(np.std(arr)),
        "samples_per_sec": batch_size / float(np.mean(arr)),
        "mb_per_sec": mb_per_batch / float(np.mean(arr)),
        "mb_per_batch": mb_per_batch,
    }


# ---------------------------------------------------------------------------
# Strategy C: Zarr with batch-row writes (write the whole batch at once)
# ---------------------------------------------------------------------------

def bench_zarr_batch(batch_size, n_trials, zarr_path):
    """Write activations batch-at-a-time into zarr (slice assignment)."""
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    expected_samples = batch_size * n_trials
    store = zarr.open(zarr_path, mode="w")
    resp_arr = store.create_array(
        "response_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(batch_size, 1, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    prom_arr = store.create_array(
        "prompt_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(batch_size, 1, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    logp_arr = store.create_array(
        "logprobs",
        shape=(expected_samples, SEQ_LEN),
        chunks=(batch_size, SEQ_LEN),
        dtype=DTYPE,
    )

    batch_times = []
    sample_idx = 0

    for trial in range(n_trials):
        data = make_fake_activations(batch_size)

        t0 = time.perf_counter()
        sl = slice(sample_idx, sample_idx + batch_size)
        resp_arr[sl] = data["response_activations"]
        prom_arr[sl] = data["prompt_activations"]
        logp_arr[sl] = data["logprobs"]
        sample_idx += batch_size
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

    arr = np.array(batch_times)
    bytes_per_batch = batch_size * (
        N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 * 2 + SEQ_LEN * 2
    )
    mb_per_batch = bytes_per_batch / 1e6

    return {
        "strategy": "zarr_batch_slice",
        "batch_size": batch_size,
        "mean_s": float(np.mean(arr)),
        "std_s":  float(np.std(arr)),
        "samples_per_sec": batch_size / float(np.mean(arr)),
        "mb_per_sec": mb_per_batch / float(np.mean(arr)),
        "mb_per_batch": mb_per_batch,
    }


# ---------------------------------------------------------------------------
# Strategy D: Zarr with (BS, N_LAYERS, SEQ, HIDDEN) chunk — full block per batch
# ---------------------------------------------------------------------------

def bench_zarr_block(batch_size, n_trials, zarr_path):
    """Write activations with chunk shape (BS, N_LAYERS, SEQ, HIDDEN) — 2 writes per batch."""
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    expected_samples = batch_size * n_trials
    store = zarr.open(zarr_path, mode="w")
    resp_arr = store.create_array(
        "response_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(batch_size, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    prom_arr = store.create_array(
        "prompt_activations",
        shape=(expected_samples, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        chunks=(batch_size, N_LAYERS, SEQ_LEN, HIDDEN_DIM),
        dtype=DTYPE,
    )
    logp_arr = store.create_array(
        "logprobs",
        shape=(expected_samples, SEQ_LEN),
        chunks=(batch_size, SEQ_LEN),
        dtype=DTYPE,
    )

    batch_times = []
    sample_idx = 0

    for trial in range(n_trials):
        data = make_fake_activations(batch_size)

        t0 = time.perf_counter()
        sl = slice(sample_idx, sample_idx + batch_size)
        resp_arr[sl] = data["response_activations"]
        prom_arr[sl] = data["prompt_activations"]
        logp_arr[sl] = data["logprobs"]
        sample_idx += batch_size
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)

    arr = np.array(batch_times)
    bytes_per_batch = batch_size * (
        N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 * 2 + SEQ_LEN * 2
    )
    mb_per_batch = bytes_per_batch / 1e6

    chunk_mb = batch_size * N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 / 1e6

    return {
        "strategy": "zarr_block",
        "batch_size": batch_size,
        "mean_s": float(np.mean(arr)),
        "std_s":  float(np.std(arr)),
        "samples_per_sec": batch_size / float(np.mean(arr)),
        "mb_per_sec": mb_per_batch / float(np.mean(arr)),
        "mb_per_batch": mb_per_batch,
        "chunk_mb": chunk_mb,
        "writes_per_batch": 2,  # one chunk per activation array
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output-dir", default=ZARR_PATH_BASE)
    args = parser.parse_args()

    tmp_dir = os.path.join(args.output_dir, "npy_batches")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Disk throughput benchmark")
    print(f"  Layers={N_LAYERS}, seq_len={SEQ_LEN}, hidden={HIDDEN_DIM}, dtype=float16")
    print(f"  MB per sample: {N_LAYERS * SEQ_LEN * HIDDEN_DIM * 2 * 2 / 1e6:.1f}")
    print(f"  Trials per config: {args.n_trials}")
    print(f"  Output dir: {args.output_dir}")

    results = []

    for bs in args.batch_sizes:
        print(f"\n--- batch_size={bs} ---")

        zarr_path_a = os.path.join(args.output_dir, f"zarr_per_sample_bs{bs}")
        r = bench_zarr(bs, args.n_trials, zarr_path_a)
        results.append(r)
        print(f"  zarr_per_sample:  {r['mean_s']:.3f}s/batch  "
              f"{r['samples_per_sec']:.2f} samp/s  {r['mb_per_sec']:.0f} MB/s")

        zarr_path_c = os.path.join(args.output_dir, f"zarr_batch_slice_bs{bs}")
        r = bench_zarr_batch(bs, args.n_trials, zarr_path_c)
        results.append(r)
        print(f"  zarr_batch_slice: {r['mean_s']:.3f}s/batch  "
              f"{r['samples_per_sec']:.2f} samp/s  {r['mb_per_sec']:.0f} MB/s")

        r = bench_npy_batch(bs, args.n_trials, tmp_dir)
        results.append(r)
        print(f"  npy_batch:        {r['mean_s']:.3f}s/batch  "
              f"{r['samples_per_sec']:.2f} samp/s  {r['mb_per_sec']:.0f} MB/s")

        zarr_path_d = os.path.join(args.output_dir, f"zarr_block_bs{bs}")
        r = bench_zarr_block(bs, args.n_trials, zarr_path_d)
        results.append(r)
        print(f"  zarr_block:       {r['mean_s']:.3f}s/batch  "
              f"{r['samples_per_sec']:.2f} samp/s  {r['mb_per_sec']:.0f} MB/s  "
              f"chunk={r['chunk_mb']:.0f}MB  writes/batch={r['writes_per_batch']}")

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Strategy':<20} {'BS':>4}  {'Mean(s)':>8}  {'Samp/s':>8}  {'MB/s':>8}")
    print(f"  {'-'*20} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['strategy']:<20} {r['batch_size']:>4}  {r['mean_s']:>8.3f}  "
              f"{r['samples_per_sec']:>8.2f}  {r['mb_per_sec']:>8.0f}")

    # Cleanup
    shutil.rmtree(args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
