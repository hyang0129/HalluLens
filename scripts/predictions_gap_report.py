#!/usr/bin/env python3
"""
Check Empire AI for gaps in per-sample prediction files and write
results/predictions_gap_report.md.

Coverage checked:
  Training baselines  — predictions.csv per (dataset, model, method, seed)
                        across all baseline_comparison_*_memmap experiments
  Sampling baselines  — se_labels.jsonl, selfcheck_scores.jsonl, ptrue.jsonl
                        per (dataset, model)

contrastive_logprob_recon is excluded from predictions.csv gaps because it
produces only distance-based scores (eval_metrics.json) by design.
"""

import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REMOTE_HOST = "empire-ai"
REMOTE_BASE = "~/LLM_research/HalluLens"
RESULTS_DIR = Path(__file__).parent.parent / "results"

EXPECTED_SEEDS = 5

# (experiment_short, model_tag) → (exp_dir_name, inner_key, model_slug, canonical_dataset)
TRAINING_SPECS = [
    ("hotpotqa",       "",      "baseline_comparison_hotpotqa_memmap",          "hotpotqa_memmap",          "Llama-3.1-8B-Instruct", "hotpotqa"),
    ("hotpotqa",       "qwen3", "baseline_comparison_hotpotqa_qwen3_memmap",     "hotpotqa_qwen3_memmap",    "Qwen3-8B",              "hotpotqa"),
    ("nq",             "",      "baseline_comparison_nq_memmap",                 "nq_memmap",                "Llama-3.1-8B-Instruct", "natural_questions"),
    ("nq",             "qwen3", "baseline_comparison_nq_qwen3_memmap",           "nq_qwen3_memmap",          "Qwen3-8B",              "natural_questions"),
    ("mmlu",           "",      "baseline_comparison_mmlu_memmap",               "mmlu_memmap",              "Llama-3.1-8B-Instruct", "mmlu"),
    ("mmlu",           "qwen3", "baseline_comparison_mmlu_qwen3_memmap",         "mmlu_qwen3_memmap",        "Qwen3-8B",              "mmlu"),
    ("popqa",          "",      "baseline_comparison_popqa_memmap",              "popqa_memmap",             "Llama-3.1-8B-Instruct", "popqa"),
    ("popqa",          "qwen3", "baseline_comparison_popqa_qwen3_memmap",        "popqa_qwen3_memmap",       "Qwen3-8B",              "popqa"),
    ("sciq",           "",      "baseline_comparison_sciq_memmap",               "sciq_memmap",              "Llama-3.1-8B-Instruct", "sciq"),
    ("sciq",           "qwen3", "baseline_comparison_sciq_qwen3_memmap",         "sciq_qwen3_memmap",        "Qwen3-8B",              "sciq"),
    ("searchqa",       "",      "baseline_comparison_searchqa_memmap",           "searchqa_memmap",          "Llama-3.1-8B-Instruct", "searchqa"),
    ("searchqa",       "qwen3", "baseline_comparison_searchqa_qwen3_memmap",     "searchqa_qwen3_memmap",    "Qwen3-8B",              "searchqa"),
]

SAMPLING_DATASETS = ["hotpotqa", "natural_questions", "popqa", "sciq", "searchqa"]
SAMPLING_MODELS   = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]
SAMPLING_FILES    = ["se_labels.jsonl", "selfcheck_scores.jsonl", "ptrue.jsonl"]

# Methods that are expected to produce predictions.csv
PREDICTIONS_EXCLUDED: set[str] = set()


def ssh(cmd: str) -> str:
    result = subprocess.run(
        ["ssh", REMOTE_HOST, cmd],
        capture_output=True, text=True,
    )
    return result.stdout


def gather_training_data() -> dict:
    """
    Returns nested dict:
        data[exp_name][method] = {"preds": N, "metrics": M}
    One SSH call per experiment.
    """
    data = {}
    for *_, exp_name, inner_key, model_slug, canonical in TRAINING_SPECS:
        base = f"{REMOTE_BASE}/runs/{exp_name}/{inner_key}"
        cmd = (
            f"for m in $(ls {base}/ 2>/dev/null); do "
            f"  p=$(ls {base}/$m/seed_*/predictions.csv 2>/dev/null | wc -l); "
            f"  q=$(ls {base}/$m/seed_*/eval_metrics.json 2>/dev/null | wc -l); "
            f"  echo \"$m $p $q\"; "
            f"done"
        )
        out = ssh(cmd)
        data[exp_name] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3:
                method, preds, metrics = parts
                data[exp_name][method] = {"preds": int(preds), "metrics": int(metrics)}
    return data


def gather_sampling_data() -> dict:
    """
    Returns dict:
        data[(dataset, model, file)] = True/False
    """
    lines = []
    for ds in SAMPLING_DATASETS:
        for model in SAMPLING_MODELS:
            for fname in SAMPLING_FILES:
                if fname == "ptrue.jsonl":
                    path = f"{REMOTE_BASE}/output/p_true/{ds}/{model}/{fname}"
                else:
                    path = f"{REMOTE_BASE}/output/sampling_baselines/{ds}/{model}/{fname}"
                lines.append(f"test -f {path} && echo OK {ds} {model} {fname} || echo MISS {ds} {model} {fname}")
    cmd = "; ".join(lines)
    out = ssh(cmd)
    data = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            status, ds, model, fname = parts
            data[(ds, model, fname)] = (status == "OK")
    return data


def build_report(training: dict, sampling: dict) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Predictions Gap Report",
        f"",
        f"Generated: {now}",
        f"",
        f"Checks per-sample prediction files on Empire AI across all memmap baseline experiments.",
        f"Note: `contrastive_logprob_recon` uses distance-based scores (cosine/mahalanobis/knn) stored in `eval_metrics.json` — it does not write `predictions.csv`. This is tracked as a gap.",
        f"",
    ]

    # ── Training baselines ──────────────────────────────────────────────────
    lines += ["## Training Baselines (memmap)", ""]

    # Collect all gaps
    complete_rows = []
    gap_rows = []
    stub_rows = []  # method dir exists but no seeds at all

    for *_, exp_name, inner_key, model_slug, canonical in TRAINING_SPECS:
        exp_data = training.get(exp_name, {})
        if not exp_data:
            gap_rows.append(f"| {canonical} | {model_slug} | *(experiment directory missing)* | — | 0/{EXPECTED_SEEDS} |")
            continue

        for method, counts in sorted(exp_data.items()):
            if method in PREDICTIONS_EXCLUDED:
                continue
            preds  = counts["preds"]
            metrics = counts["metrics"]

            if metrics == 0 and preds == 0:
                # Stub directory — method dir exists but no seed runs at all
                stub_rows.append(f"| {canonical} | {model_slug} | {method} | stub (no seeds) | 0/{EXPECTED_SEEDS} |")
            elif preds == EXPECTED_SEEDS:
                complete_rows.append((canonical, model_slug, method, preds))
            else:
                gap_rows.append(f"| {canonical} | {model_slug} | {method} | {preds}/{EXPECTED_SEEDS} seeds have predictions.csv | {preds}/{EXPECTED_SEEDS} |")

        # Methods expected but entirely absent
        all_present = set(exp_data.keys())
        # icr_probe: only flag absent if it IS present in at least one experiment for this model
        # logprob/token_entropy: flag absent elsewhere too if applicable
        # (absence reporting is handled via stub/gap rows above for present-but-empty dirs)

    # Summarise complete
    lines += [
        f"### Complete ({len(complete_rows)} method × dataset × model cells)",
        "",
        "| Dataset | Model | Method | Seeds |",
        "|---------|-------|--------|-------|",
    ]
    for canonical, model_slug, method, preds in sorted(complete_rows):
        lines.append(f"| {canonical} | {model_slug} | {method} | {preds}/{EXPECTED_SEEDS} ✓ |")

    lines += [""]

    if stub_rows:
        lines += [
            f"### Stub runs — method dir exists but no seeds executed ({len(stub_rows)} cells)",
            "",
            "| Dataset | Model | Method | Status | Seeds |",
            "|---------|-------|--------|--------|-------|",
        ] + stub_rows + [""]

    if gap_rows:
        lines += [
            f"### Incomplete — partial seed coverage ({len(gap_rows)} cells)",
            "",
            "| Dataset | Model | Method | Gap | Seeds with predictions.csv |",
            "|---------|-------|--------|-----|---------------------------|",
        ] + gap_rows + [""]
    else:
        lines += ["### Incomplete\n\nNone — all non-stub methods have full seed coverage.\n"]

    # Methods completely absent from certain datasets
    lines += ["### Methods absent from specific datasets", ""]
    absence_notes = []
    for *_, exp_name, inner_key, model_slug, canonical in TRAINING_SPECS:
        exp_data = training.get(exp_name, {})
        present = set(exp_data.keys())
        # icr_probe: expected everywhere but only present in nq, mmlu, popqa
        if "icr_probe" not in present:
            absence_notes.append(f"| {canonical} | {model_slug} | icr_probe | not present in experiment dir |")
    if absence_notes:
        lines += [
            "| Dataset | Model | Method | Note |",
            "|---------|-------|--------|------|",
        ] + sorted(set(absence_notes)) + [""]
    else:
        lines += ["None.\n"]

    # ── Sampling baselines ──────────────────────────────────────────────────
    lines += ["## Sampling Baselines", ""]

    sampling_gaps = []
    sampling_ok   = []
    for ds in SAMPLING_DATASETS:
        for model in SAMPLING_MODELS:
            for fname in SAMPLING_FILES:
                ok = sampling.get((ds, model, fname), False)
                if ok:
                    sampling_ok.append((ds, model, fname))
                else:
                    sampling_gaps.append(f"| {ds} | {model} | {fname} | missing |")

    lines += [
        f"Complete: {len(sampling_ok)}/{len(SAMPLING_DATASETS)*len(SAMPLING_MODELS)*len(SAMPLING_FILES)} files present.",
        "",
    ]
    if sampling_gaps:
        lines += [
            "| Dataset | Model | File | Status |",
            "|---------|-------|------|--------|",
        ] + sampling_gaps + [""]
    else:
        lines += ["All sampling files present. No gaps.\n"]

    # ── Summary ──────────────────────────────────────────────────────────────
    total_gaps = len(gap_rows) + len(stub_rows) + len(absence_notes) + len(sampling_gaps)
    lines += [
        "## Summary",
        "",
        f"| Category | Gaps |",
        f"|----------|------|",
        f"| Training: partial seeds | {len(gap_rows)} |",
        f"| Training: stub runs (no seeds) | {len(stub_rows)} |",
        f"| Training: method absent from dataset | {len(sorted(set(absence_notes)))} |",
        f"| Sampling: missing files | {len(sampling_gaps)} |",
        f"| **Total gaps** | **{total_gaps}** |",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    print("Gathering training baseline data from Empire AI...")
    training = gather_training_data()
    print("Gathering sampling baseline data from Empire AI...")
    sampling = gather_sampling_data()
    print("Building report...")

    report = build_report(training, sampling)

    out_path = RESULTS_DIR / "predictions_gap_report.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
