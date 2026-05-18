#!/usr/bin/env python3
"""Draft main paper headline table — mean AUROC per method per dataset.

Reads results/results_table.json (snapshot from scripts/results_table.py) and
emits a markdown table for each model with methods as rows and datasets as
columns. Aggregates over training seeds. Designed for paper-section drafting,
not for live monitoring — re-run scripts/results_table.py and snapshot first.

Usage:
    python scripts/draft_headline_table.py                       # stdout
    python scripts/draft_headline_table.py -o results/headline.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "results" / "results_table.json"

# Family #1 (headline) methods. Order = paper table order: competitors first,
# our method last. The metric_key column picks the headline scorer for each
# method (see PAPER_ROADMAP §6).
HEADLINE_METHODS = [
    # (method_id_in_json, display_label, kind, metric_key)
    ("logprob_baseline",          "Logprob (mean)",        "training", "mean_logprob_auroc"),
    ("token_entropy",             "Token entropy",         "training", "mean_entropy_auroc"),
    ("p_true",                    "P(true)",               "p_true",   "p_true_auroc_best"),
    ("linear_probe",              "Linear probe",          "training", "auroc"),
    ("saplma",                    "SAPLMA",                "training", "auroc"),
    ("llmsknow_probe",            "LLMsKnow probe",        "training", "auroc"),
    ("icr_probe",                 "ICR probe",             "training", "auroc"),
    ("contrastive_logprob_recon", "**Ours (KNN)**",        "training", "knn_auroc"),
]

# Headline datasets (excludes *_memmap dupes from the issue #79 re-ingestion).
HEADLINE_DATASETS = ["hotpotqa", "nq", "mmlu", "popqa", "sciq", "searchqa"]
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "nq":       "NQ",
    "mmlu":     "MMLU",
    "popqa":    "PopQA",
    "sciq":     "SciQ",
    "searchqa": "SearchQA",
}
MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]


def collect(cells: list[dict]) -> dict:
    """key = (model, method_id, dataset) -> list of (seed, auroc)"""
    bucket: dict[tuple, list[tuple]] = defaultdict(list)
    method_keys = {m[0]: (m[2], m[3]) for m in HEADLINE_METHODS}
    for c in cells:
        if c["status"] != "complete":
            continue
        k = c["key"]
        if k["dataset"] not in HEADLINE_DATASETS:
            continue
        if k["model"] not in MODELS:
            continue
        method = k["method"]
        if method not in method_keys:
            continue
        want_kind, metric_key = method_keys[method]
        if c["kind"] != want_kind:
            continue
        v = c["metrics"].get(metric_key)
        if not isinstance(v, (int, float)):
            continue
        bucket[(k["model"], method, k["dataset"])].append((k["seed"], float(v)))
    return bucket


def render_cell(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    mu = statistics.mean(values)
    sd = statistics.pstdev(values)
    return f"{mu:.3f} ± {sd:.3f} (n={len(values)})"


def render_table(model: str, bucket: dict) -> str:
    out = []
    out.append(f"### {model}")
    out.append("")
    header = ["Method"] + [DATASET_LABELS[d] for d in HEADLINE_DATASETS]
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for method_id, label, _kind, _metric in HEADLINE_METHODS:
        row = [label]
        for ds in HEADLINE_DATASETS:
            vals = [v for _, v in bucket.get((model, method_id, ds), [])]
            row.append(render_cell(vals))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    ap.add_argument("-o", "--out", type=Path,
                    help="Write to file (else stdout).")
    args = ap.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)
    bucket = collect(data["cells"])

    sections = [
        "# Draft headline table — mean AUROC across seeds",
        "",
        f"Source: `{args.results_json.relative_to(PROJECT_ROOT) if args.results_json.is_absolute() else args.results_json}`  ",
        f"Snapshot generated at: {data.get('generated_at','?')}  ",
        f"Git: {data.get('git', {}).get('commit','?')[:8]} on `{data.get('git', {}).get('branch','?')}`  ",
        "",
        "Methods: family #1 (headline) per `results/README.md`. Metric is "
        "per-seed test AUROC, aggregated as `mean ± popstdev (n=seeds)` for "
        "trained methods; single-run methods (logprob, entropy, P(true)) "
        "report the scalar. Empty cells = no complete runs in the table.",
        "",
    ]
    for model in MODELS:
        sections.append(render_table(model, bucket))
        sections.append("")
    sections.append(
        "Scorer choices: `contrastive_logprob_recon` → `knn_auroc` (headline "
        "per PAPER_ROADMAP §6); `logprob_baseline` → `mean_logprob_auroc`; "
        "`token_entropy` → `mean_entropy_auroc`; `p_true` → `p_true_auroc_best`. "
        "Other scorers available in `results/results_table.csv`."
    )
    text = "\n".join(sections) + "\n"

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
