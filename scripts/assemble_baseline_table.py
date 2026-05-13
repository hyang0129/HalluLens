"""Phase 6: Assemble AUROC table and compute-matched figure.

Reads all score files, aligns by row_idx, computes AUROC vs binary hallu label,
outputs CSV + PDF figure.

Usage:
    python scripts/assemble_baseline_table.py \\
        --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B \\
        --output-dir output/baseline_results

    # Llama only
    python scripts/assemble_baseline_table.py \\
        --models meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    DATASETS,
    MODELS,
    SAMPLING_DATASETS,
    eval_results_json,
    model_name,
    nli_matrix_path,
    se_labels_path,
    selfcheck_samples_path,
    selfcheck_scores_path,
    sep_results_path,
    searchqa_test_cap_path,
)


# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------

def load_jsonl_by_row(path: str) -> dict:
    """Load jsonl into {row_idx: record} dict."""
    by_row = {}
    p = Path(path)
    if not p.exists():
        return by_row
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_row[rec["row_idx"]] = rec
            except Exception:
                pass
    return by_row


def load_hallu_labels(dataset: str, model_id: str, split: str = "test") -> Optional[np.ndarray]:
    eval_path = eval_results_json(dataset, model_id, split)
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        data = json.load(f)
    return np.array(data["halu_test_res"], dtype=int)


def load_cap_indices(dataset: str, model_id: str) -> Optional[set]:
    if dataset != "searchqa":
        return None
    cap_path = searchqa_test_cap_path(model_id)
    if not cap_path.exists():
        return None
    with open(cap_path) as f:
        return set(json.load(f)["question_ids"])


def safe_auroc(scores: list, labels: list) -> Optional[float]:
    if len(scores) < 10:
        return None
    y = np.array(labels, dtype=int)
    s = np.array(scores, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], s[mask]))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-dataset scorer
# ---------------------------------------------------------------------------

def compute_dataset_aurocs(dataset: str, model_id: str) -> dict:
    """Return dict of {method: auroc} for one (dataset, model) cell."""
    result = {"dataset": dataset, "model": model_name(model_id)}

    labels_all = load_hallu_labels(dataset, model_id, "test")
    if labels_all is None:
        return result

    cap_indices = load_cap_indices(dataset, model_id)

    def get_aligned(by_row: dict, score_key: str):
        """Extract (scores, labels) aligned by row_idx, respecting cap."""
        scores, lbls = [], []
        for row_idx, rec in sorted(by_row.items()):
            if cap_indices is not None and row_idx not in cap_indices:
                continue
            if row_idx >= len(labels_all):
                continue
            val = rec.get(score_key)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            scores.append(val)
            lbls.append(labels_all[row_idx])
        return scores, lbls

    # SE (length-normalized)
    if dataset in SAMPLING_DATASETS:
        se_by_row = load_jsonl_by_row(str(se_labels_path(dataset, model_id, "test")))
        scores, lbls = get_aligned(se_by_row, "length_normalized_se")
        result["se_length_normalized"] = safe_auroc(scores, lbls)

        scores, lbls = get_aligned(se_by_row, "discrete_se")
        result["se_discrete"] = safe_auroc(scores, lbls)

        # SelfCheckGPT
        sc_by_row = load_jsonl_by_row(str(selfcheck_scores_path(dataset, model_id, "test")))
        for key in ("nli", "bertscore", "ngram"):
            scores, lbls = get_aligned(sc_by_row, key)
            result[f"selfcheck_{key}"] = safe_auroc(scores, lbls)

    # SEP
    sep_path = sep_results_path(dataset, model_id)
    if sep_path.exists():
        with open(sep_path) as f:
            sep_data = json.load(f)
        result["sep_binary_auroc"] = sep_data.get("sep_binary_auroc")
        result["sep_se_auroc"] = sep_data.get("sep_se_auroc")
        result["sep_layer"] = sep_data.get("layer")
        result["sep_binary_n_train"] = sep_data.get("train_size_sep_binary")
        result["sep_se_n_train"] = sep_data.get("train_size_sep_se")

    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_compute_matched_figure(table: pd.DataFrame, output_path: str) -> None:
    """One panel per dataset; x=forward-pass count, y=AUROC."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure.")
        return

    free_form = [d for d in SAMPLING_DATASETS if d != "mmlu"]
    all_datasets = free_form + ["mmlu"]

    n_cols = 3
    n_rows = (len(all_datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    method_style = {
        "Our contrastive (K=1)": dict(K=1, marker="*", color="tab:red", linestyle="none"),
        "SEP-binary (K=1)": dict(K=1, marker="D", color="tab:orange", linestyle="none"),
        "SEP-SE (K=1)": dict(K=1, marker="s", color="tab:purple", linestyle="none"),
        "SelfCheckGPT-NLI (K=10)": dict(K=10, marker="o", color="tab:blue", linestyle="none"),
        "SE length-norm (K=10)": dict(K=10, marker="^", color="tab:green", linestyle="none"),
    }

    col_map = {
        "SEP-binary (K=1)": "sep_binary_auroc",
        "SEP-SE (K=1)": "sep_se_auroc",
        "SelfCheckGPT-NLI (K=10)": "selfcheck_nli",
        "SE length-norm (K=10)": "se_length_normalized",
    }

    for ax_idx, ds in enumerate(all_datasets):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]
        ds_rows = table[table["dataset"] == ds]

        for label, style in method_style.items():
            if label == "Our contrastive (K=1)":
                continue  # placeholder — no data yet in this table
            col_key = col_map.get(label)
            if col_key is None or col_key not in ds_rows.columns:
                continue
            for _, row_data in ds_rows.iterrows():
                val = row_data.get(col_key)
                if val is not None and np.isfinite(val):
                    ax.scatter(
                        style["K"],
                        val,
                        label=f"{label} ({model_name(row_data['model'])})",
                        marker=style["marker"],
                        color=style["color"],
                        s=80,
                        zorder=3,
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assemble AUROC table and figure.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--output-dir", default="output/baseline_results")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mid in args.models:
        for ds in args.datasets:
            print(f"  {ds} / {model_name(mid)}")
            r = compute_dataset_aurocs(ds, mid)
            rows.append(r)

    table = pd.DataFrame(rows)

    csv_path = out_dir / "baseline_auroc_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nTable saved → {csv_path}")

    # Pretty-print main table columns
    main_cols = [
        "dataset", "model",
        "se_length_normalized", "selfcheck_nli",
        "sep_se_auroc", "sep_binary_auroc",
    ]
    print_cols = [c for c in main_cols if c in table.columns]
    print(table[print_cols].to_string(index=False))

    if not args.no_figure:
        make_compute_matched_figure(table, str(out_dir / "compute_matched_auroc.pdf"))


if __name__ == "__main__":
    main()
