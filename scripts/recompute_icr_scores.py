"""Recompute icr_scores.npy from existing capture memmaps.

The capture worker accumulates per-sample ICR scores in an in-memory buffer
and only persists them to icr_scores.npy at finalize() (see
activation_logging/inference_capture_writer.py:410). If the worker crashes
mid-run, or if its inline ICR computation was disabled, the raw memmaps
(response_attention.npy + response_activations.npy) are still durable on
disk — we can reconstruct icr_scores.npy from those.

This script:
  - Reads meta.jsonl to get the list of committed sample_indices
  - For each sample × layer, slices the relevant memmap regions lazily
    (np.memmap pages in only what we access)
  - Calls activation_research.icr_score.compute_icr_score
  - Writes icr_scores.npy in sample_index order (matches worker finalize layout)

Runs on CPU only. Per-sample working set is ~17 MB (one sample's slice of
response_activations across L+1 layers). With 8 multiprocessing workers,
wall time for a 50k-sample cell is dominated by lustre read throughput.

Usage:
    # In-place: overwrite the cell's icr_scores.npy
    python scripts/recompute_icr_scores.py shared/icr_capture/<cell>

    # Out-of-place: write to a different path
    python scripts/recompute_icr_scores.py shared/icr_capture/<cell> \\
        --output /tmp/icr_scores_recomputed.npy

    # Limit samples (smoketest)
    python scripts/recompute_icr_scores.py shared/icr_capture/<cell> --limit 100

    # Multi-process (recommended for large cells)
    python scripts/recompute_icr_scores.py shared/icr_capture/<cell> --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# Import the canonical CPU implementation; share the exact formula the worker
# uses inline so recomputed scores are bit-equivalent to what finalize() writes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_research.icr_score import compute_icr_score


# ---------------------------------------------------------------------------
# Worker process state — set once via initializer to avoid sending memmaps
# across the multiprocessing pickle boundary on every call.
# ---------------------------------------------------------------------------

_MM_ATTN: np.memmap | None = None
_MM_ACT: np.memmap | None = None
_R_MAX: int = 0
_NUM_LAYERS: int = 0
_RESPONSE_LENS: np.ndarray | None = None
_PROMPT_LENS: np.ndarray | None = None


def _init_worker(
    cell_path: str,
    n_alloc: int,
    num_layers: int,
    r_max: int,
    max_response_len: int,
    hidden_dim: int,
) -> None:
    global _MM_ATTN, _MM_ACT, _R_MAX, _NUM_LAYERS, _RESPONSE_LENS, _PROMPT_LENS
    cell = Path(cell_path)
    _MM_ATTN = np.memmap(
        cell / "response_attention.npy",
        dtype=np.float16,
        mode="r",
        shape=(n_alloc, num_layers, r_max, r_max),
    )
    # Why max_response_len (S) not r_max (R): the writer allocates
    # response_activations at (N, L+1, S, H) where S can exceed R (e.g. sciq
    # smoketest has S=256, R=64). The ICR formula only needs the first R
    # positions per layer; we slice those below.
    _MM_ACT = np.memmap(
        cell / "response_activations.npy",
        dtype=np.float16,
        mode="r",
        shape=(n_alloc, num_layers + 1, max_response_len, hidden_dim),
    )
    _R_MAX = r_max
    _NUM_LAYERS = num_layers
    # response_len.npy and prompt_len.npy are raw int32 memmaps (no .npy header).
    _RESPONSE_LENS = np.memmap(
        cell / "response_len.npy", dtype=np.int32, mode="r", shape=(n_alloc,)
    )
    # prompt_len.npy: same layout as response_len.npy.  Used to compute the
    # correct effective top-k: k = int(top_p * (prompt_len + response_len)),
    # matching the upstream ICR Probe code (icr_score.py:226 in XavierZhang2002/ICR_Probe).
    _PROMPT_LENS = np.memmap(
        cell / "prompt_len.npy", dtype=np.int32, mode="r", shape=(n_alloc,)
    )


def _compute_one(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """Compute the per-layer ICR vector for one sample.

    Returns (sample_index, scores) where scores has shape (num_layers,) fp32.
    """
    sample_idx, _row = args
    assert _MM_ATTN is not None and _MM_ACT is not None and _RESPONSE_LENS is not None
    assert _PROMPT_LENS is not None

    # Clamp to r_max: attention is only captured for the first R response
    # positions; any tokens beyond are implicitly truncated. This matches the
    # worker's inline behavior (attention memmap is RxR regardless of how long
    # the model actually generated).
    rlen = min(int(_RESPONSE_LENS[sample_idx]), _R_MAX)
    plen = int(_PROMPT_LENS[sample_idx])
    if rlen <= 0:
        return sample_idx, np.zeros(_NUM_LAYERS, dtype=np.float32)

    # Slice the per-sample blocks once (pages in ~17 MB per sample).
    # response_attention: (L, R, R) for this sample
    attn_block = np.asarray(_MM_ATTN[sample_idx], dtype=np.float32)  # (L, R, R)
    # response_activations: (L+1, R, H) for this sample, truncated to R positions
    act_block = np.asarray(
        _MM_ACT[sample_idx, :, : _R_MAX, :], dtype=np.float32
    )  # (L+1, R, H)

    scores = np.zeros(_NUM_LAYERS, dtype=np.float32)
    for l in range(_NUM_LAYERS):
        h_in = act_block[l]       # h^{l-1} at response positions (R, H)
        h_out = act_block[l + 1]  # h^l at response positions      (R, H)
        delta_h = h_out - h_in
        scores[l] = compute_icr_score(
            response_attn=attn_block[l],
            h_block_input=h_in,
            delta_h=delta_h,
            response_len=rlen,
            prompt_len=plen,
        )
    return sample_idx, scores


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path, help="capture directory")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output icr_scores.npy path (default: <cell>/icr_scores.npy, in-place)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only compute scores for the first N committed samples (smoketest)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="multiprocessing pool size (default 1 = single-process)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="multiprocessing chunksize (default 64); larger = less IPC overhead",
    )
    args = ap.parse_args()

    cell = args.cell
    if not cell.is_dir():
        print(f"ERROR: not a directory: {cell}", file=sys.stderr)
        return 1

    cfg = json.loads((cell / "config.json").read_text())
    num_layers = int(cfg["num_layers"])
    hidden_dim = int(cfg["hidden_dim"])
    r_max = int(cfg["r_max"])
    max_response_len = int(cfg["max_response_len"])
    n_alloc = int(cfg["n_samples"])

    meta_lines = (cell / "meta.jsonl").read_text().splitlines()
    meta = [json.loads(l) for l in meta_lines if l.strip()]
    # Stack rows in sample_index order, matching finalize()'s layout.
    meta.sort(key=lambda r: r["sample_index"])
    sample_indices = [int(r["sample_index"]) for r in meta]
    if args.limit is not None:
        sample_indices = sample_indices[: args.limit]
    n = len(sample_indices)

    if n == 0:
        print(f"ERROR: no committed samples in {cell}/meta.jsonl", file=sys.stderr)
        return 1

    print(
        f"Recomputing ICR scores for {n} samples × {num_layers} layers "
        f"from {cell}",
        file=sys.stderr,
    )
    print(
        f"  n_alloc={n_alloc} r_max={r_max} max_response_len={max_response_len} "
        f"hidden_dim={hidden_dim} workers={args.workers}",
        file=sys.stderr,
    )

    out = np.zeros((n, num_layers), dtype=np.float32)
    work = list(zip(sample_indices, range(n)))

    t0 = time.perf_counter()
    if args.workers <= 1:
        _init_worker(
            str(cell), n_alloc, num_layers, r_max, max_response_len, hidden_dim
        )
        for i, item in enumerate(work):
            _, scores = _compute_one(item)
            out[i] = scores
            if (i + 1) % 100 == 0 or i + 1 == n:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1:6d}/{n}] {rate:.1f} samp/s  eta={eta:.0f}s",
                    file=sys.stderr,
                )
    else:
        idx_to_row = {idx: row for row, idx in enumerate(sample_indices)}
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                str(cell), n_alloc, num_layers, r_max, max_response_len, hidden_dim,
            ),
        ) as pool:
            done = 0
            for i, scores in pool.imap_unordered(
                _compute_one_pair, work, chunksize=args.chunk
            ):
                out[idx_to_row[i]] = scores
                done += 1
                if done % 200 == 0 or done == n:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done:6d}/{n}] {rate:.1f} samp/s  eta={eta:.0f}s",
                        file=sys.stderr,
                    )

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({n / elapsed:.1f} samp/s)", file=sys.stderr)

    out_path = args.output if args.output is not None else (cell / "icr_scores.npy")
    # Write atomically: .tmp then rename, so a concurrent reader never sees a
    # half-written file. This matters because the capture worker may also be
    # writing icr_scores.npy at the same time on finalize.
    # Why open() + np.save(fh): np.save(str_path) appends ".npy" if missing,
    # which breaks our .tmp suffix; passing a file object bypasses that.
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, out)
    tmp.replace(out_path)
    print(f"Wrote {out_path}  shape={out.shape}  dtype={out.dtype}", file=sys.stderr)
    return 0


def _compute_one_pair(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """Pool-friendly wrapper: returns (sample_index, scores) for index lookup."""
    return _compute_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
