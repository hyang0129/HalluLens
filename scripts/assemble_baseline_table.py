"""Assemble the compute-matched AUROC figure (and optional P(true) bootstrap CIs).

AUROC numbers come from `output/results_table/results_table.json` — produced
by `scripts/results_table.py`. This script no longer re-computes AUROC; it
only renders the matplotlib figure and (optionally) bootstrap CIs from the
raw P(true) score files.

Usage:
    # Make the figure (reads results_table.json):
    python scripts/results_table.py
    python scripts/assemble_baseline_table.py

    # Custom output dir + bootstrap CIs for P(true):
    python scripts/assemble_baseline_table.py \\
        --output-dir output/baseline_results \\
        --with-ci
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.p_true.paths import ptrue_scores_path
from tasks.sampling_baselines.paths import (
    DATASETS,
    MODELS,
    SAMPLING_DATASETS,
    eval_results_json,
    model_name,
)

DEFAULT_RESULTS_JSON = PROJECT_ROOT / "output" / "results_table" / "results_table.json"


# ---------------------------------------------------------------------------
# Results-table loader: turn long-form cells into a {(dataset, model): {col: auroc}} view
# ---------------------------------------------------------------------------

# Mapping from results_table method names to the figure's column names.
_FIGURE_COLUMNS = {
    ("sampling", "se_length_normalized"):  "se_length_normalized",
    ("sampling", "selfcheck_nli"):         "selfcheck_nli",
    ("sep", "sep"):                        "sep_se_auroc",
}


def load_figure_table(results_json: Path) -> list[dict]:
    """Pivot the long-form cells into one row per (dataset, model)."""
    with open(results_json) as f:
        payload = json.load(f)

    pivot: dict[tuple[str, str], dict[str, float]] = {}
    for cell in payload["cells"]:
        col = _FIGURE_COLUMNS.get((cell["kind"], cell["key"]["method"]))
        if col is None or cell["status"] != "complete":
            continue
        key = (cell["key"]["dataset"], cell["key"]["model"])
        metric_key = "sep_se_auroc" if cell["kind"] == "sep" else "auroc"
        v = cell["metrics"].get(metric_key)
        if v is None:
            continue
        pivot.setdefault(key, {})[col] = v

    rows = []
    for (ds, m), cols in pivot.items():
        rows.append({"dataset": ds, "model": m, **cols})
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_compute_matched_figure(table: list[dict], output_path: Path) -> None:
    """One panel per dataset; x=forward-pass count, y=AUROC."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure.")
        return

    all_datasets = [d for d in SAMPLING_DATASETS] + ["mmlu"]

    n_cols = 3
    n_rows = (len(all_datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    method_style = {
        "SEP-SE (K=1)":            dict(K=1,  marker="s", color="tab:purple", col="sep_se_auroc"),
        "SelfCheckGPT-NLI (K=10)": dict(K=10, marker="o", color="tab:blue",   col="selfcheck_nli"),
        "SE length-norm (K=10)":   dict(K=10, marker="^", color="tab:green",  col="se_length_normalized"),
    }

    for ax_idx, ds in enumerate(all_datasets):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]
        ds_rows = [r for r in table if r["dataset"] == ds]

        for label, style in method_style.items():
            for row_data in ds_rows:
                val = row_data.get(style["col"])
                if val is None or not np.isfinite(val):
                    continue
                ax.scatter(
                    style["K"], val,
                    label=f"{label} ({row_data['model']})",
                    marker=style["marker"], color=style["color"], s=80, zorder=3,
                )

        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Forward passes at test time")
        ax.set_ylabel("AUROC")
        ax.set_xlim(0, 12)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=7, loc="lower right")

    # Hide unused panels
    for ax_idx in range(len(all_datasets), n_rows * n_cols):
        r, c = divmod(ax_idx, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Compute-matched AUROC comparison", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bootstrap CIs (P(true) only — uses raw scores, not the results table)
# ---------------------------------------------------------------------------

def _safe_auroc(scores, labels) -> Optional[float]:
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    if len(s) < 10 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def _bootstrap_auroc(scores, labels, n: int = 1000, seed: int = 42):
    """Return (lo, hi) 95% bootstrap CI on AUROC, or None if not computable."""
    base = _safe_auroc(scores, labels)
    if base is None:
        return None
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], s[idx]))
        except Exception:
            pass
    if len(aucs) < 50:
        return None
    aucs = np.array(aucs)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _write_ptrue_bootstrap_ci(models, datasets, out_dir: Path) -> None:
    print("\nComputing P(true) bootstrap CIs (1000 resamples)...")
    ci_results = []
    for mid in models:
        labels_cache = {}
        for ds in datasets:
            pt_path = ptrue_scores_path(ds, mid, "test")
            if not pt_path.exists():
                continue
            if ds not in labels_cache:
                eval_path = eval_results_json(ds, mid, "test")
                if not eval_path.exists():
                    continue
                with open(eval_path) as f:
                    labels_cache[ds] = np.array(json.load(f)["halu_test_res"], dtype=int)
            labels_all = labels_cache[ds]

            scores_fwd, lbls, scores_rev = [], [], []
            with open(pt_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    row_idx = rec.get("row_idx")
                    if row_idx is None or row_idx >= len(labels_all):
                        continue
                    lbl = int(labels_all[row_idx])
                    if (v := rec.get("p_true")) is not None and np.isfinite(v):
                        scores_fwd.append(1.0 - v)
                        lbls.append(lbl)
                    if (v := rec.get("p_true_reversed")) is not None and np.isfinite(v):
                        scores_rev.append(1.0 - v)

            ci_fwd = _bootstrap_auroc(scores_fwd, lbls)
            ci_rev = _bootstrap_auroc(scores_rev, lbls)
            ci_results.append({
                "dataset": ds,
                "model": model_name(mid),
                "p_true_auroc_fwd": _safe_auroc(scores_fwd, lbls),
                "p_true_ci_fwd": ci_fwd,
                "p_true_auroc_rev": _safe_auroc(scores_rev, lbls),
                "p_true_ci_rev": ci_rev,
                "n_rows": len(scores_fwd),
            })
            print(f"  {ds}/{model_name(mid)}: fwd={ci_fwd}, rev={ci_rev}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ci_path = out_dir / "p_true_bootstrap.json"
    with open(ci_path, "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"Bootstrap CIs saved → {ci_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help=f"Path to results_table.json (default: {DEFAULT_RESULTS_JSON}).",
    )
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models for P(true) bootstrap CI (default: all).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                        help="Datasets for P(true) bootstrap CI.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/baseline_results"))
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--with-ci", action="store_true",
                        help="Compute P(true) bootstrap 95%% CIs (1000 resamples × cells).")
    args = parser.parse_args()

    if not args.results_json.exists():
        sys.exit(
            f"results_table.json not found at {args.results_json}. "
            f"Run: python scripts/results_table.py"
        )

    table = load_figure_table(args.results_json)
    print(f"Loaded {len(table)} (dataset, model) cells from {args.results_json}")

    if not args.no_figure:
        make_compute_matched_figure(table, args.output_dir / "compute_matched_auroc.pdf")

    if args.with_ci:
        _write_ptrue_bootstrap_ci(args.models, args.datasets, args.output_dir)


if __name__ == "__main__":
    main()
