#!/usr/bin/env python3
"""Draft main paper headline table — mean AUROC per method per dataset.

Reads results/results_table.json (snapshot from scripts/results_table.py) and
emits a sectioned markdown table (Baseline + Sampling) for each model. Re-run
scripts/results_table.py first to refresh the snapshot.

Usage:
    python scripts/draft_headline_table.py                       # stdout
    python scripts/draft_headline_table.py -o results/draft_headline_table.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / "results" / "results_table.json"

# (method_id_in_json, display_label, kind, metric_key)
BASELINE_METHODS = [
    ("logprob_baseline",          "LogProb (seq)",                "training", "seq_logprob_auroc"),
    ("token_entropy",             "Token Entropy",                "training", "mean_entropy_auroc"),
    ("p_true",                    "P(true)",                      "p_true",   "p_true_auroc_best"),
    ("linear_probe",              "Linear Probe",                 "training", "auroc"),
    ("saplma",                    "SAPLMA",                       "training", "auroc"),
    ("llmsknow_probe",            "LLMsKnow Probe",               "training", "auroc"),
    ("icr_probe",                 "ICR Probe",                    "training", "auroc"),
    ("act_vit",                   "ACT-ViT",                      "training", "auroc"),
    ("contrastive_logprob_recon", "**Contrastive+Recon (ours)**", "training", "knn_auroc"),
]

SAMPLING_METHODS = [
    ("se_length_normalized", "SE (length-norm)",    "sampling", "auroc"),
    ("se_semantic_entropy",  "SE (semantic)",        "sampling", "auroc"),
    ("selfcheck_nli",        "SelfCheckGPT-NLI",    "sampling", "auroc"),
    ("selfcheck_bertscore",  "SelfCheckGPT-BERT",   "sampling", "auroc"),
    ("selfcheck_ngram",      "SelfCheckGPT-ngram",  "sampling", "auroc"),
]

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
    all_methods = BASELINE_METHODS + SAMPLING_METHODS
    method_keys = {m[0]: (m[2], m[3]) for m in all_methods}
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
    n_complete = len(values)
    if n_complete < 5:
        return f"{mu:.3f}\\*({n_complete}/5)"
    return f"{mu:.3f}"


def render_section(title: str, methods: list, model: str, bucket: dict) -> list[str]:
    out = []
    out.append(f"### {title} — AUROC")
    out.append("")
    header = ["Method"] + [DATASET_LABELS[d] for d in HEADLINE_DATASETS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for method_id, label, _kind, _metric in methods:
        row = [label]
        for ds in HEADLINE_DATASETS:
            vals = [v for _, v in bucket.get((model, method_id, ds), [])]
            row.append(render_cell(vals))
        out.append("| " + " | ".join(row) + " |")
    return out


def render_model_block(model: str, bucket: dict) -> list[str]:
    out = []
    out.append(f"## {model}")
    out.append("")
    out.extend(render_section("Baseline (memmap trained)", BASELINE_METHODS, model, bucket))
    out.append("")
    out.extend(render_section("Sampling", SAMPLING_METHODS, model, bucket))
    out.append("")
    out.append("---")
    out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    ap.add_argument("-o", "--out", type=Path,
                    help="Write to file (else stdout).")
    args = ap.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)
    bucket = collect(data["cells"])

    today = data.get("generated_at", "?")[:10]
    lines = [
        "# Draft Headline AUROC Table",
        "",
        "Canonical numbers for §5 main results. "
        f"Source: `results_table.csv` (memmap + sampling categories only).",
        "",
        "\\* = incomplete: N/5 seeds complete. All other trained cells are 5/5 seeds.",
        "",
        "MMLU has no sampling results (multiple-choice; sampling-based methods not applicable).",
        "",
        "---",
        "",
    ]
    for model in MODELS:
        lines.extend(render_model_block(model, bucket))

    lines += [
        "### Scorer notes",
        "",
        "Trained methods: `contrastive_logprob_recon` → `knn_auroc`; "
        "`logprob_baseline` → `seq_logprob_auroc`; "
        "`token_entropy` → `mean_entropy_auroc`; `p_true` → `p_true_auroc_best`; "
        "all probe/sampling methods → `auroc`. "
        "Means reported across 5 seeds; stdev available in `results_table.csv`.",
        "",
        "*Generated from `results_table.csv` via `scripts/draft_headline_table.py`. Re-run to refresh.*  ",
        f"*Last updated: {today}*",
    ]

    text = "\n".join(lines) + "\n"

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
