"""paper/figures_src/render_transfer.py

Reads paper/data/baseline_comparison.csv and renders a bar chart comparing
ICR-probe vs SelfCheck AUROC across datasets.

Outputs:
  paper/generated/figures/transfer_llama_linear.pdf
  paper/generated/figures/transfer_llama_linear.numbers.csv

The sidecar lists every number that appears on the figure (bar heights,
annotations, axis limits). Figure captions in prose cite from the sidecar:

    \\result{fig.transfer_llama_linear}{diagonal_mean}[2]

Pure pandas + matplotlib; no dependency on runs/ or zarr stores.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("pdf")  # Non-interactive PDF backend — no display required
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIGURE_NAME = "transfer_llama_linear"
SOURCE_CSV = "baseline_comparison.csv"

# Datasets to show in the figure (display order)
DATASETS_ORDER = [
    "hotpotqa",
    "mmlu",
    "natural_questions",
    "popqa",
    "sciq",
    "searchqa",
]
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "mmlu": "MMLU",
    "natural_questions": "NatQ",
    "popqa": "PopQA",
    "sciq": "SciQ",
    "searchqa": "SearchQA",
}


def write_sidecar_header(fh, generator: str, source_data: str) -> None:
    """Write the sidecar comment header."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fh.write(f"# generator: {generator}\n")
    fh.write(f"# source_data: {source_data}\n")
    fh.write(f"# generated: {now}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render baseline comparison figure")
    parser.add_argument(
        "--paper-dir",
        default="paper",
        help="Path to the paper/ directory (default: paper/)",
    )
    args = parser.parse_args(argv)

    paper_dir = Path(args.paper_dir)
    data_csv = paper_dir / "data" / SOURCE_CSV
    out_dir = paper_dir / "generated" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / f"{FIGURE_NAME}.pdf"
    out_sidecar = out_dir / f"{FIGURE_NAME}.numbers.csv"

    if not data_csv.exists():
        print(f"ERROR: {data_csv} not found.", file=sys.stderr)
        return 1

    # --- Load data ---
    df = pd.read_csv(data_csv, comment="#")

    # Filter to AUROC rows only
    auroc_df = df[df["metric"] == "auroc"].copy()

    # Filter to datasets we want to show
    auroc_df = auroc_df[auroc_df["dataset"].isin(DATASETS_ORDER)].copy()

    # Pivot: rows = dataset, columns = method
    pivot = auroc_df.pivot_table(
        index="dataset", columns="method", values="mean", aggfunc="mean"
    )

    # Reindex to our display order (only datasets present in data)
    datasets_present = [d for d in DATASETS_ORDER if d in pivot.index]
    pivot = pivot.reindex(datasets_present)

    methods = list(pivot.columns)

    # --- Compute summary numbers for sidecar ---
    icr_col = "icr_probe" if "icr_probe" in methods else methods[0]
    ref_col = "selfcheck" if "selfcheck" in methods else (methods[1] if len(methods) > 1 else None)

    icr_values = pivot[icr_col].dropna().values
    diagonal_mean = float(icr_values.mean()) if len(icr_values) > 0 else 0.0
    icr_max = float(icr_values.max()) if len(icr_values) > 0 else 0.0
    icr_min = float(icr_values.min()) if len(icr_values) > 0 else 0.0

    ref_values = pivot[ref_col].dropna().values if ref_col else []
    off_diag_max = float(max(ref_values)) if len(ref_values) > 0 else 0.0

    # --- Build the figure ---
    n_datasets = len(datasets_present)
    n_methods = len(methods)
    bar_width = 0.35
    x = list(range(n_datasets))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    colors = ["#2166ac", "#d6604d"]
    method_labels = {"icr_probe": "ICR-Probe", "selfcheck": "SelfCheck"}

    bar_handles = []
    for i, method in enumerate(methods):
        offsets = [xi + i * bar_width - bar_width * (n_methods - 1) / 2 for xi in x]
        vals = [pivot[method].get(d, float("nan")) for d in datasets_present]
        color = colors[i % len(colors)]
        bars = ax.bar(
            offsets,
            vals,
            bar_width * 0.9,
            label=method_labels.get(method, method),
            color=color,
            alpha=0.85,
        )
        bar_handles.append(bars)

    # Annotation: diagonal_mean line
    ax.axhline(diagonal_mean, color="#2166ac", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(
        n_datasets - 0.05,
        diagonal_mean + 0.005,
        f"{diagonal_mean:.2f}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#2166ac",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS.get(d, d) for d in datasets_present], fontsize=8
    )
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_title("Baseline comparison: AUROC across datasets", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  Written figure: {out_pdf}")

    # --- Write sidecar ---
    y_axis_min = 0.5
    y_axis_max = 1.0

    sidecar_rows = [
        ("diagonal_mean", diagonal_mean, "annotation"),
        ("icr_auroc_max", icr_max, "annotation"),
        ("icr_auroc_min", icr_min, "annotation"),
        ("off_diag_max", off_diag_max, "annotation"),
        ("y_axis_min", y_axis_min, "axis"),
        ("y_axis_max", y_axis_max, "axis"),
    ]
    # Per-dataset values
    for d in datasets_present:
        safe_d = d.replace("_", "")
        if icr_col in pivot.columns and d in pivot.index:
            sidecar_rows.append(
                (f"{safe_d}_icr_auroc", float(pivot.loc[d, icr_col]), "data")
            )
        if ref_col and ref_col in pivot.columns and d in pivot.index:
            sidecar_rows.append(
                (f"{safe_d}_ref_auroc", float(pivot.loc[d, ref_col]), "data")
            )

    with open(out_sidecar, "w") as fh:
        write_sidecar_header(
            fh,
            generator="paper/figures_src/render_transfer.py",
            source_data=str(data_csv),
        )
        fh.write("label,value,role\n")
        for label, value, role in sidecar_rows:
            fh.write(f"{label},{value:.6g},{role}\n")

    print(f"  Written sidecar: {out_sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
