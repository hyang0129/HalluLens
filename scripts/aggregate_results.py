"""Aggregate experiment results with mean +/- 95% CI across seeds.

Usage:
    python scripts/aggregate_results.py --runs-dir runs/baseline_comparison_hotpotqa
    python scripts/aggregate_results.py --runs-dir runs/ --output results/summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def collect_results(runs_dir: str) -> list[dict]:
    """Walk runs directory and collect all eval_metrics.json files."""
    records = []
    for root, dirs, files in os.walk(runs_dir):
        if "eval_metrics.json" in files:
            with open(os.path.join(root, "eval_metrics.json")) as f:
                record = json.load(f)
            record["run_dir"] = root
            records.append(record)
    return records


def compute_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Compute mean and 95% CI for a set of values."""
    import scipy.stats as st

    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0
    mean = np.mean(values)
    se = st.sem(values)
    ci = se * st.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(ci)


def aggregate(records: list[dict]) -> pd.DataFrame:
    """Group by (dataset, method, metric) and compute mean +/- CI."""
    df = pd.DataFrame(records)

    # Identify metric columns (any numeric column that isn't metadata)
    metadata_cols = {"method", "dataset", "seed", "split_seed", "run_dir",
                     "n_train", "n_test", "n_examples", "probe_layer"}
    auroc_cols = [
        c for c in df.columns
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for (dataset, method), group in df.groupby(["dataset", "method"]):
        for col in auroc_cols:
            values = group[col].dropna().values
            if len(values) == 0:
                continue
            mean, ci = compute_ci(values)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": col,
                    "mean": mean,
                    "ci_95": ci,
                    "n_seeds": len(values),
                    "formatted": (
                        f"{mean:.4f} +/- {ci:.4f}" if ci > 0 else f"{mean:.4f}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--runs-dir", required=True, help="Root directory of runs")
    parser.add_argument(
        "--output", default="results/summary.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    records = collect_results(args.runs_dir)
    if not records:
        print("No results found!")
        return

    # Raw results
    raw_df = pd.DataFrame(records)
    raw_path = args.output.replace(".csv", "_raw.csv")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw results: {raw_path} ({len(raw_df)} runs)")

    # Aggregated summary
    summary = aggregate(records)
    summary.to_csv(args.output, index=False)
    print(f"Summary: {args.output}")

    # Markdown table
    md_path = args.output.replace(".csv", "_table.md")
    unique_datasets = summary["dataset"].unique()
    with open(md_path, "w") as f:
        f.write(
            "| Method | Metric | " + " | ".join(unique_datasets) + " |\n"
        )
        f.write(
            "|--------|--------|"
            + "|".join(["--------"] * len(unique_datasets))
            + "|\n"
        )
        for (method, metric), grp in summary.groupby(["method", "metric"]):
            row = f"| {method} | {metric} |"
            for ds in unique_datasets:
                match = grp[grp["dataset"] == ds]
                row += f" {match.iloc[0]['formatted'] if len(match) > 0 else 'N/A'} |"
            f.write(row + "\n")
    print(f"Table: {md_path}")

    # Print to stdout
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
