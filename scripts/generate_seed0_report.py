#!/usr/bin/env python3
"""Generate results/seed0_results.md from seed=0 run outputs.

Usage:
    python scripts/generate_seed0_report.py [--out results/seed0_results.md]
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = sorted(ROOT.glob("configs/experiments/baseline_comparison_*.json"))

DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "nq_test_hallu_cor": "NQ",
    "mmlu": "MMLU",
    "movies": "Movies",
    "popqa": "PopQA",
    "sciq": "SciQ",
    "searchqa": "SearchQA",
}

# Display order
DATASET_ORDER = ["hotpotqa", "nq_test_hallu_cor", "mmlu", "movies", "popqa", "sciq", "searchqa"]


def load_json(path: Path):
    try:
        text = path.read_text()
        # json.loads doesn't handle NaN; replace bare NaN with null first
        import re
        text = re.sub(r'\bNaN\b', 'null', text)
        return json.loads(text)
    except Exception:
        return None


def fmt(val, decimals=3):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), decimals)


def get_metrics(runs_dir: Path, dataset: str, seed: int = 0):
    """Return dict of method -> metric value(s) for the given dataset/seed."""
    result = {}

    # Non-learned: logprob_baseline
    m = load_json(runs_dir / dataset / "logprob_baseline" / "eval_metrics.json")
    if m:
        result["logprob"] = fmt(m.get("mean_logprob_auroc"))
        result["n_test"] = m.get("n_test")

    # Non-learned: token_entropy — pick best non-None auroc
    m = load_json(runs_dir / dataset / "token_entropy" / "eval_metrics.json")
    if m:
        for key in ("mean_entropy_auroc", "min_logprob_auroc", "mean_logprob_auroc"):
            v = fmt(m.get(key))
            if v is not None:
                result["token_entropy"] = v
                break
        if result.get("n_test") is None:
            result["n_test"] = m.get("n_test")

    seed_dir = f"seed_{seed}"

    # linear_probe
    m = load_json(runs_dir / dataset / "linear_probe" / seed_dir / "eval_metrics.json")
    if m:
        result["linear_probe"] = fmt(m.get("auroc"))
        if result.get("n_test") is None:
            result["n_test"] = m.get("n_test")

    # multi_layer_linear_probe
    m = load_json(runs_dir / dataset / "multi_layer_linear_probe" / seed_dir / "eval_metrics.json")
    if m:
        result["multi_layer"] = fmt(m.get("auroc"))

    # contrastive
    m = load_json(runs_dir / dataset / "contrastive_logprob_recon" / seed_dir / "eval_metrics.json")
    if m:
        result["cosine"] = fmt(m.get("cosine_auroc"))
        result["mahal"] = fmt(m.get("mahalanobis_auroc"))
        result["knn"] = fmt(m.get("knn_auroc"))

    return result


def bold_best(row_vals: list):
    """Given a list of (label, value_or_None), return list with best value bolded."""
    numeric = [(i, v) for i, (_, v) in enumerate(row_vals) if v is not None]
    if not numeric:
        return [lbl if v is None else f"{v:.3f}" for lbl, v in row_vals]
    best_i, best_v = max(numeric, key=lambda x: x[1])
    out = []
    for i, (lbl, v) in enumerate(row_vals):
        if v is None:
            out.append("—")
        elif i == best_i:
            out.append(f"**{v:.3f}**")
        else:
            out.append(f"{v:.3f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/seed0_results.md")
    args = parser.parse_args()

    seed = args.seed
    out_path = ROOT / args.out

    # Collect metrics per dataset
    data = {}
    for exp_path in EXPERIMENTS:
        cfg = load_json(exp_path)
        if not cfg:
            continue
        dataset = cfg["dataset"]
        exp_name = cfg["experiment_name"]
        runs_dir = ROOT / cfg.get("output_dir", "runs") / exp_name
        data[dataset] = get_metrics(runs_dir, dataset, seed)

    # Build table rows in display order
    cols = [
        ("Logprob", "logprob"),
        ("Entropy", "token_entropy"),
        ("Lin. Probe", "linear_probe"),
        ("Multi-Layer", "multi_layer"),
        ("Contr. Cosine", "cosine"),
        ("Contr. Mahal.", "mahal"),
        ("Contr. KNN", "knn"),
    ]

    header = "| Dataset | n_test | " + " | ".join(c[0] for c in cols) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in cols) + " |"

    rows = []
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS.get(dataset, dataset)
        m = data.get(dataset, {})
        n_test = m.get("n_test")
        n_str = f"{n_test:,}" if n_test else "—"

        row_vals = [(c[0], m.get(c[1])) for c in cols]
        cells = bold_best(row_vals)
        rows.append(f"| {label} | {n_str} | " + " | ".join(cells) + " |")

    lines = [
        f"# Seed {seed} Results — Hallucination Detection AUROC",
        "",
        "**Model:** Llama-3.1-8B-Instruct  ",
        f"**Seed:** {seed} | **Split seed:** 42 | **Evaluation:** held-out test split",
        "",
        "---",
        "",
        "## Results",
        "",
        header,
        sep,
        *rows,
        "",
        "Bold = best per dataset. `—` = not yet complete or failed. Contr. = Contrastive+Logprob recon (SimCLR + logprob aux loss); scorer variant in parentheses.",
        "",
        "---",
        "",
        "## Method Details",
        "",
        "| Method | Type | Key metric |",
        "|--------|------|------------|",
        "| Logprob Baseline | non-learned | `mean_logprob_auroc` |",
        "| Token Entropy | non-learned | `mean_entropy_auroc` (best non-NaN) |",
        "| Linear Probe | learned, layer 22 | `auroc` |",
        "| Multi-Layer Linear Probe | learned, layers 14–29 | `auroc` |",
        "| Contrastive+Logprob (Cosine) | learned | contrastive (SimCLR) + logprob recon aux loss; cosine distance scorer |",
        "| Contrastive+Logprob (Mahal.) | learned | same model; Mahalanobis distance scorer |",
        "| Contrastive+Logprob (KNN) | learned | same model; k-NN scorer (calibrated k) |",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written to {out_path}")

    # Also print table to stdout
    print()
    print(header)
    print(sep)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
