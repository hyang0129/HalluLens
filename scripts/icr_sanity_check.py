"""Check 1 + Check 2 from notes/icr_probe_sanity.md for a single capture cell.

Reads icr_scores.npy + meta.jsonl from a capture directory and reports:
  - score sanity (NaN/Inf, range, per-layer variance)
  - per-layer single-feature AUROC against the hallucinated label
  - direction-free AUROC = max(auroc, 1 - auroc)

Usage:
    python scripts/icr_sanity_check.py shared/icr_capture/<cell>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path, help="capture directory")
    args = ap.parse_args()

    cell = args.cell
    scores_path = cell / "icr_scores.npy"
    meta_path = cell / "meta.jsonl"

    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found", file=sys.stderr)
        return 1
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found", file=sys.stderr)
        return 1

    scores = np.load(scores_path)
    meta = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
    # icr_scores.npy is written at finalize() with samples STACKED in sample_index
    # order (see inference_capture_writer.finalize). If the worker is still running
    # and finalize ran at an earlier checkpoint, meta.jsonl may have more rows than
    # icr_scores.npy. Take the lowest-N sample_indices that appear in BOTH.
    meta_sorted = sorted(meta, key=lambda r: r["sample_index"])
    n_scored = scores.shape[0]
    if n_scored < len(meta):
        print(
            f"NOTE: scores has {n_scored} rows but meta has {len(meta)} — "
            f"icr_scores.npy is a stale snapshot. Aligning to first {n_scored} "
            f"sample_indices (sample_index order).",
            file=sys.stderr,
        )
    aligned = meta_sorted[:n_scored]
    labels = np.array([int(bool(r["hallucinated"])) for r in aligned], dtype=np.int64)
    s = scores  # (n_scored, L); rows correspond to aligned[i] in order.

    print(f"=== {cell.name} ===")
    print(f"scores.shape:       {scores.shape}")
    print(f"scores.dtype:       {scores.dtype}")
    print(f"meta rows:          {len(meta)}")
    print(f"aligned rows used:  {len(aligned)}")
    print(f"label balance:      pos={labels.mean():.3f}  (n_pos={int(labels.sum())}/{len(labels)})")

    print()
    print("--- Check 1: score sanity ---")
    any_nan = bool(np.isnan(s).any())
    any_inf = bool(np.isinf(s).any())
    print(f"any NaN:            {any_nan}")
    print(f"any Inf:            {any_inf}")
    print(f"global min/max:     {s.min():.6f} / {s.max():.6f}  (JSD bound: ~0.693)")
    per_layer_mean = s.mean(axis=0)
    per_layer_std = s.std(axis=0)
    print(f"per-layer mean:     min={per_layer_mean.min():.4f}  max={per_layer_mean.max():.4f}")
    print(f"per-layer std:      min={per_layer_std.min():.6f}  max={per_layer_std.max():.6f}")
    n_varying = int((per_layer_std > 1e-4).sum())
    print(f"layers with std>1e-4: {n_varying}/{s.shape[1]}")

    check1_pass = (
        not any_nan
        and not any_inf
        and s.min() >= -1e-6
        and s.max() <= 1.0
        and per_layer_mean.std() > 1e-6
        and n_varying >= int(0.8 * s.shape[1])
    )
    print(f"Check 1 PASS:       {check1_pass}")

    print()
    print("--- Check 2: per-layer AUROC ---")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("sklearn not available — install scikit-learn", file=sys.stderr)
        return 2

    if labels.min() == labels.max():
        print("WARN: only one label class present, cannot compute AUROC", file=sys.stderr)
        return 1

    aurocs = np.array([roc_auc_score(labels, s[:, l]) for l in range(s.shape[1])])
    df_aurocs = np.maximum(aurocs, 1.0 - aurocs)  # direction-free

    best_layer = int(np.argmax(df_aurocs))
    print(f"raw AUROC mean:     {aurocs.mean():.4f}")
    print(f"raw AUROC range:    {aurocs.min():.4f} – {aurocs.max():.4f}")
    print(f"direction-free max: {df_aurocs.max():.4f} at layer {best_layer}")
    print(f"direction-free mean: {df_aurocs.mean():.4f}")
    n_above_55 = int((df_aurocs > 0.55).sum())
    print(f"layers > 0.55 (df): {n_above_55}/{len(df_aurocs)}")

    check2_pass = (df_aurocs.max() > 0.55) and (n_above_55 >= 0.25 * len(df_aurocs))
    print(f"Check 2 PASS:       {check2_pass}")

    print()
    print("--- per-layer detail (raw AUROC, direction-free AUROC) ---")
    for l in range(s.shape[1]):
        marker = " <-- best" if l == best_layer else ""
        print(f"  layer {l:2d}: raw={aurocs[l]:.4f}  df={df_aurocs[l]:.4f}{marker}")

    return 0 if (check1_pass and check2_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
