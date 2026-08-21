#!/usr/bin/env python3
"""Profile (and optionally run) the streaming pre-generation baseline pass.

Issue #161, baselines 2 (difference-of-means) and 7 (first-token entropy).

Why one script for both: they read the *same* capture and differ only in which
cached array they touch, so paying for the pass once covers both.

The key access pattern is a SINGLE streaming pass that accumulates every layer
at once.  Reading ``response_activations[i, :, 0, :]`` pulls all L+1 layer
slices for one sample (~270 KB at L=32) in one neighbourhood, so a full
all-layer sweep costs one pass over ~95 GB rather than L separate strided
passes over the same data.  Class accumulators are ``(L+1, hidden)`` float64 --
about 2 MB -- so this is I/O bound and needs no large memory.

Doubles as the `grace`-partition feasibility check: it prints the host, the
visible mount, and whether the capture dir is readable before doing any work.

Usage
-----
    # profile on a subset (default 2000 rows), no results written
    python scripts/profile_pregen_baselines.py --capture-dir <dir> --limit 2000

    # full pass over one capture, writing accumulators
    python scripts/profile_pregen_baselines.py --capture-dir <dir> \
        --split test --out-dir output/pregen_profile
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preflight(capture_dir: Path) -> dict:
    """Report where we are and whether the data is actually visible.

    This is the grace/betagg filesystem check: alpha-cluster nodes mount
    /mnt/home over NFS, and whether the beta segment does was never confirmed.
    """
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "capture_dir": str(capture_dir),
        "capture_dir_visible": capture_dir.exists(),
    }
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["mem_total_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    _log(f"host={info['host']} cpus={info['cpu_count']} "
         f"mem={info.get('mem_total_gb', '?')}GB partition={info['slurm_partition']}")
    if not info["capture_dir_visible"]:
        raise SystemExit(
            f"FILESYSTEM CHECK FAILED: {capture_dir} not visible from "
            f"{info['host']}. This node cannot see the captures."
        )
    _log(f"capture dir visible: {capture_dir}")
    return info


def stream_pass(capture_dir: Path, split: str, label_source: str,
                split_seed: int, limit: int | None) -> dict:
    """One pass over the split, accumulating all layers simultaneously."""
    from activation_research.memmap_activation_parser import MemmapActivationParser

    cfg = json.loads((capture_dir / "config.json").read_text())
    n_samples = int(cfg["n_samples"])
    num_layers = int(cfg["num_layers"])
    hidden = int(cfg["hidden_dim"])
    r_max = int(cfg["max_response_len"])
    top_k = int(cfg.get("response_logprobs_top_k", 20))
    L1 = num_layers + 1  # layer axis includes the embedding row at index 0

    strategy = "none" if split == "test" else "three_way"
    parser = MemmapActivationParser(
        capture_dir, split_strategy=strategy,
        random_seed=split_seed, label_source=label_source,
    )
    df = parser.df
    rows = df[df["split"] == split] if "split" in df.columns else df
    idx = rows["sample_index"].to_numpy()
    labels = rows["halu"].to_numpy().astype(np.int64)
    if limit is not None:
        idx, labels = idx[:limit], labels[:limit]
    _log(f"{split}: {len(idx)} rows, halu_frac={labels.mean():.3f}, "
         f"layers={L1} hidden={hidden}")

    # Header-less raw memmaps -- shapes come from config.json, not np.load.
    resp = np.memmap(capture_dir / "response_activations.npy", dtype=np.float16,
                     mode="r", shape=(n_samples, L1, r_max, hidden))
    topk = np.memmap(capture_dir / "response_topk_logprobs.npy", dtype=np.float32,
                     mode="r", shape=(n_samples, r_max, top_k))

    sums = np.zeros((2, L1, hidden), dtype=np.float64)   # class -> layer -> mean
    counts = np.zeros(2, dtype=np.int64)
    ent_trunc = np.empty(len(idx), dtype=np.float64)
    top1 = np.empty(len(idx), dtype=np.float64)
    resid = np.empty(len(idx), dtype=np.float64)

    bytes_read = 0
    t0 = time.time()
    for n, (i, y) in enumerate(zip(idx, labels)):
        block = np.asarray(resp[i, :, 0, :], dtype=np.float32)  # (L1, hidden)
        sums[y] += block
        counts[y] += 1
        bytes_read += block.size * 2

        lp = np.asarray(topk[i, 0, :], dtype=np.float64)        # pre-generation
        p = np.exp(lp)
        m = max(0.0, 1.0 - p.sum())
        ent_trunc[n] = float(-(p * lp).sum())
        top1[n] = float(p[0])
        resid[n] = m
        bytes_read += lp.size * 4

        if n and n % 2000 == 0:
            el = time.time() - t0
            _log(f"  {n}/{len(idx)} rows  {el:.1f}s  "
                 f"{n/el:.0f} rows/s  {bytes_read/el/1e6:.0f} MB/s")

    elapsed = time.time() - t0
    means = sums / np.maximum(counts, 1)[:, None, None]
    w = means[0] - means[1]  # truthful - hallucinated, per layer

    return {
        "n_rows": int(len(idx)),
        "elapsed_s": round(elapsed, 2),
        "rows_per_s": round(len(idx) / elapsed, 1),
        "mb_per_s": round(bytes_read / elapsed / 1e6, 1),
        "gb_read": round(bytes_read / 1e9, 3),
        "bytes_per_row": int(bytes_read / max(len(idx), 1)),
        "class_counts": counts.tolist(),
        "w_norm_per_layer": np.linalg.norm(w, axis=1).round(4).tolist(),
        "_arrays": {"w": w, "ent_trunc": ent_trunc, "top1": top1,
                    "resid": resid, "labels": labels},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--label-source", default="substring")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=2000,
                    help="rows to read; omit/-1 for the full split")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    limit = None if args.limit is not None and args.limit < 0 else args.limit
    info = preflight(args.capture_dir)
    res = stream_pass(args.capture_dir, args.split, args.label_source,
                      args.split_seed, limit)
    arrays = res.pop("_arrays")

    _log(f"DONE {res['n_rows']} rows in {res['elapsed_s']}s "
         f"({res['rows_per_s']} rows/s, {res['mb_per_s']} MB/s)")

    # Extrapolate to the whole corpus: 351,484 rows across 20 capture dirs.
    total_rows = 351_484
    est_h = total_rows / res["rows_per_s"] / 3600
    est_gb = total_rows * res["bytes_per_row"] / 1e9
    _log(f"EXTRAPOLATION: {total_rows} rows -> {est_h:.2f} h single-process, "
         f"{est_gb:.0f} GB read. Shards independently by (dataset, model).")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.capture_dir.name
        np.savez_compressed(args.out_dir / f"{stem}_{args.split}_pregen.npz",
                            w=arrays["w"], ent_trunc=arrays["ent_trunc"],
                            top1=arrays["top1"], resid=arrays["resid"],
                            labels=arrays["labels"])
        (args.out_dir / f"{stem}_{args.split}_profile.json").write_text(
            json.dumps({"preflight": info, "profile": res,
                        "extrapolation_hours": round(est_h, 3),
                        "extrapolation_gb": round(est_gb, 1)}, indent=2))
        _log(f"wrote {args.out_dir}/{stem}_{args.split}_*")


if __name__ == "__main__":
    main()
