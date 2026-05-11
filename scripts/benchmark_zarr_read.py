"""
Benchmark zarr preload read strategies on NFS.

Four strategies:
  1. oindex-dforder  : current code — batches of 256 in df (JSONL) order
  2. oindex-sorted   : same batches but sort zarr indices first
  3. chunk-scan      : zarr-chunk-aligned sequential slices
  4. sorted-memmap   : sorted oindex + local /tmp memmap output (avoids RAM pressure)

Each strategy runs, prints its result, then frees memory before the next one
starts — safe to use with --n-samples on large datasets.

Usage:
    python scripts/benchmark_zarr_read.py \
        --zarr shared/hotpotqa_train_llama_3_1_8b_instruct/activations.zarr \
        --n-samples 4096
"""

import argparse
import os
import tempfile
import time
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def open_response_activations(zarr_path: str):
    """Open response_activations from either v2 or v3 zarr layout."""
    store = zarr.open_group(zarr_path, mode="r")
    for key in ("response_activations", "arrays/response_activations"):
        if key in store:
            return store[key]
    raise KeyError(f"response_activations not found in {zarr_path}. Keys: {list(store.keys())}")


def make_indices(n_zarr: int, n_samples: int, shuffle: bool) -> np.ndarray:
    """Return n_samples zarr row indices, optionally shuffled."""
    indices = np.arange(min(n_samples, n_zarr), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
    return indices


def run_oindex_dforder(arr, zr_indices, relevant_layers, T_read, batch_size=256):
    """Current approach: oindex in df order."""
    N = len(zr_indices)
    L = len(relevant_layers)
    H = arr.shape[-1]
    cache = np.zeros((N, L, T_read, H), dtype=arr.dtype)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = zr_indices[start:end].tolist()
        chunk = np.asarray(arr.oindex[batch_idx, relevant_layers, :T_read, :])
        cache[start:end] = chunk
    return cache


def run_oindex_sorted(arr, zr_indices, relevant_layers, T_read, batch_size=256):
    """Sort zarr indices first, then oindex in sorted order."""
    N = len(zr_indices)
    L = len(relevant_layers)
    H = arr.shape[-1]

    order = np.argsort(zr_indices)
    sorted_zr = zr_indices[order]
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(N)

    tmp = np.zeros((N, L, T_read, H), dtype=arr.dtype)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = sorted_zr[start:end].tolist()
        chunk = np.asarray(arr.oindex[batch_idx, relevant_layers, :T_read, :])
        tmp[start:end] = chunk

    cache = tmp[inv_order]
    del tmp
    return cache


def run_sorted_memmap(arr, zr_indices, relevant_layers, T_read, batch_size=256):
    """Sorted oindex reads + /tmp memmap output.

    Reads zarr in sorted order, scatters results into a local /tmp memmap so
    the output never lives fully in RAM.  Returns (memmap_array, tmp_path) —
    caller is responsible for del + os.unlink when done.
    """
    N = len(zr_indices)
    L = len(relevant_layers)
    H = arr.shape[-1]

    order = np.argsort(zr_indices)
    sorted_zr = zr_indices[order]

    tmp_f = tempfile.NamedTemporaryFile(dir='/tmp', delete=False, suffix='.mm')
    tmp_f.close()
    out_mm = np.memmap(tmp_f.name, dtype=arr.dtype, mode='w+',
                       shape=(N, L, T_read, H))
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = sorted_zr[start:end].tolist()
        chunk = np.asarray(arr.oindex[batch_idx, relevant_layers, :T_read, :])
        out_mm[order[start:end]] = chunk
    out_mm.flush()
    return out_mm, tmp_f.name


def run_chunk_scan(arr, zr_indices, relevant_layers, T_read):
    """Zarr-chunk-aligned sequential scan.

    Iterates the zarr in chunk-size increments so each read is a basic slice
    instead of an oindex over arbitrary rows.
    """
    N = len(zr_indices)
    L = len(relevant_layers)
    H = arr.shape[-1]
    cache = np.zeros((N, L, T_read, H), dtype=arr.dtype)

    chunk_size = arr.chunks[0]
    n_zarr = arr.shape[0]

    zr_to_cache = {}
    for cache_pos, zr_idx in enumerate(zr_indices):
        zr_to_cache.setdefault(int(zr_idx), []).append(cache_pos)

    for chunk_start in range(0, n_zarr, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_zarr)
        wanted_zr = [z for z in range(chunk_start, chunk_end) if z in zr_to_cache]
        if not wanted_zr:
            continue
        data = np.asarray(arr[chunk_start:chunk_end, relevant_layers, :T_read, :])
        for zr_row in wanted_zr:
            local = zr_row - chunk_start
            for cache_pos in zr_to_cache[zr_row]:
                cache[cache_pos] = data[local]

    return cache


def _allclose_chunked(ref_mm, candidate, chunk=1024):
    """Compare two arrays in row-chunks to avoid double-RAM spike."""
    N = ref_mm.shape[0]
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        if not np.allclose(ref_mm[start:end].astype(np.float32),
                           candidate[start:end].astype(np.float32)):
            return False
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="shared/hotpotqa_train_llama_3_1_8b_instruct/activations.zarr")
    parser.add_argument("--n-samples", type=int, default=4096,
                        help="Number of samples to benchmark (default 4096)")
    parser.add_argument("--layers", default="0,8,16,24,32",
                        help="Comma-separated layer indices to read")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle zarr indices to simulate worst-case df ordering")
    parser.add_argument("--verify", action="store_true",
                        help="Check that all strategies produce identical output")
    args = parser.parse_args()

    arr = open_response_activations(args.zarr)
    print(f"zarr shape : {arr.shape}")
    print(f"zarr chunks: {arr.chunks}")
    print(f"zarr dtype : {arr.dtype}")
    print()

    relevant_layers = [int(x) for x in args.layers.split(",")]
    T_read = arr.shape[2]
    n_samples = min(args.n_samples, arr.shape[0])

    zr_indices = make_indices(arr.shape[0], n_samples, shuffle=args.shuffle)
    mode = "shuffled (worst-case)" if args.shuffle else "sequential (best-case)"
    print(f"Testing {n_samples} samples, {len(relevant_layers)} layers, {T_read} tokens")
    print(f"Index order: {mode}")
    print(f"Chunk size (sample dim): {arr.chunks[0]}")
    print()

    timings = {}
    ref_mm = None
    ref_mm_path = None

    # --- strategy 1: oindex-dforder ---
    print("Strategy 1: oindex-dforder (current code) ...")
    t0 = time.perf_counter()
    c = run_oindex_dforder(arr, zr_indices, relevant_layers, T_read)
    t = time.perf_counter() - t0
    timings["oindex-dforder"] = t
    print(f"  {t:.1f}s  →  {n_samples / t:.1f} samples/sec\n")

    if args.verify:
        ref_f = tempfile.NamedTemporaryFile(dir='/tmp', delete=False, suffix='.ref.mm')
        ref_f.close()
        ref_mm_path = ref_f.name
        ref_mm = np.memmap(ref_mm_path, dtype=c.dtype, mode='w+', shape=c.shape)
        ref_mm[:] = c
        ref_mm.flush()
    del c

    # --- strategy 2: oindex-sorted ---
    print("Strategy 2: oindex-sorted ...")
    t0 = time.perf_counter()
    c = run_oindex_sorted(arr, zr_indices, relevant_layers, T_read)
    t = time.perf_counter() - t0
    timings["oindex-sorted"] = t
    print(f"  {t:.1f}s  →  {n_samples / t:.1f} samples/sec")

    if args.verify:
        ok = _allclose_chunked(ref_mm, c)
        print(f"  verify vs oindex-dforder: {'PASS' if ok else 'FAIL'}")
    del c
    print()

    # --- strategy 3: chunk-scan ---
    print("Strategy 3: chunk-scan ...")
    t0 = time.perf_counter()
    c = run_chunk_scan(arr, zr_indices, relevant_layers, T_read)
    t = time.perf_counter() - t0
    timings["chunk-scan"] = t
    print(f"  {t:.1f}s  →  {n_samples / t:.1f} samples/sec")

    if args.verify:
        ok = _allclose_chunked(ref_mm, c)
        print(f"  verify vs oindex-dforder: {'PASS' if ok else 'FAIL'}")
    del c
    print()

    # --- strategy 4: sorted-memmap ---
    print("Strategy 4: sorted-memmap (sorted oindex + /tmp memmap output) ...")
    t0 = time.perf_counter()
    mm, mm_path = run_sorted_memmap(arr, zr_indices, relevant_layers, T_read)
    t = time.perf_counter() - t0
    timings["sorted-memmap"] = t
    print(f"  {t:.1f}s  →  {n_samples / t:.1f} samples/sec")

    if args.verify:
        ok = _allclose_chunked(ref_mm, mm)
        print(f"  verify vs oindex-dforder: {'PASS' if ok else 'FAIL'}")
    del mm
    os.unlink(mm_path)
    print()

    # --- clean up reference memmap ---
    if ref_mm is not None:
        del ref_mm
        os.unlink(ref_mm_path)

    # --- summary ---
    baseline = timings["oindex-dforder"]
    print("=" * 58)
    print(f"{'Strategy':<24} {'Time (s)':>10} {'samp/s':>10} {'speedup':>10}")
    print("-" * 58)
    for name, t in timings.items():
        print(f"{name:<24} {t:>10.1f} {n_samples/t:>10.1f} {baseline/t:>10.2f}x")
    print("=" * 58)


if __name__ == "__main__":
    main()
