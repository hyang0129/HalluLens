"""Aggregate extended metrics across seeds.

Reads ``eval_metrics_extended.json`` files produced by
``compute_extended_metrics.py`` and emits a per-(method, dataset) summary
with mean +/- std over seeds and a seed-resampled 95% bootstrap CI.

Usage:
    python scripts/aggregate_extended_metrics.py --runs-dir runs
    python scripts/aggregate_extended_metrics.py --runs-dir runs --output results/extended.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("auroc", "auprc", "fpr_at_95_tpr", "ece")


def collect(roots: list[Path], filename: str) -> pd.DataFrame:
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            print(f"[skip] {root} does not exist", file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(root):
            if filename not in files:
                continue
            run_dir = Path(dirpath)
            sidecar = run_dir / filename
            with open(sidecar) as f:
                rec = json.load(f)
            # Backfill identity fields from eval_metrics.json if absent.
            ev_path = run_dir / "eval_metrics.json"
            if ev_path.exists():
                with open(ev_path) as f:
                    ev = json.load(f)
                for k in ("method", "dataset", "model_id", "seed", "split_seed"):
                    rec.setdefault(k, ev.get(k))
            rec["run_dir"] = str(run_dir)
            rows.append(rec)
    return pd.DataFrame(rows)


def _seed_bootstrap_ci(values: np.ndarray, *, n_resamples: int, seed: int) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate(df: pd.DataFrame, *, n_resamples: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    # Archived runs have seed=None per #57 — keep them in the raw dump but
    # drop from seed-aggregated CIs (they would inflate sample count without
    # contributing seed-variation signal).
    seeded = df[df["seed"].notna()].copy()
    group_cols = [c for c in ("dataset", "method") if c in seeded.columns]
    out_rows: list[dict] = []
    for keys, group in seeded.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n_seeds"] = int(len(group))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                base[f"{metric}_mean"] = None
                base[f"{metric}_std"] = None
                base[f"{metric}_ci95_lo"] = None
                base[f"{metric}_ci95_hi"] = None
                continue
            base[f"{metric}_mean"] = float(vals.mean())
            base[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci = _seed_bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
            base[f"{metric}_ci95_lo"] = ci[0] if ci else None
            base[f"{metric}_ci95_hi"] = ci[1] if ci else None
        out_rows.append(base)
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", nargs="+", default=["runs"])
    parser.add_argument(
        "--input-name",
        default="eval_metrics_extended.json",
        help="Sidecar filename to read (default: eval_metrics_extended.json)",
    )
    parser.add_argument("--bootstrap-b", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", default="results/extended_summary.csv")
    args = parser.parse_args()

    roots = [Path(r) for r in args.runs_dir]
    raw = collect(roots, args.input_name)
    if raw.empty:
        print("No extended-metrics records found.")
        return 1

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.output.replace(".csv", "_raw.csv")
    raw.to_csv(raw_path, index=False)
    print(f"Raw: {raw_path} ({len(raw)} runs)")

    summary = aggregate(raw, n_resamples=args.bootstrap_b, seed=args.bootstrap_seed)
    summary.to_csv(args.output, index=False)
    print(f"Summary: {args.output} ({len(summary)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
