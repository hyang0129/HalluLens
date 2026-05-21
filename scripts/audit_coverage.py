#!/usr/bin/env python3
"""Render a human-readable coverage report from results_table.json.

Coverage and metrics live together in `results_table.json` (produced by
`scripts/results_table.py`). This script is a thin formatter — it does not
walk the filesystem. Run `results_table.py` first if the JSON is stale.

Complementary tool: `scripts/predictions_gap_report.py` SSHes to Empire AI
and checks for actual per-sample files (predictions.csv, sampling JSONL, and
§6 transfer-matrix cell JSONs). Run it separately to see what is physically
present on the cluster, including transfer matrix coverage gaps.

Usage:
    python scripts/results_table.py            # refresh source-of-truth
    python scripts/audit_coverage.py           # print markdown to stdout
    python scripts/audit_coverage.py --out reports/coverage.md
    python scripts/predictions_gap_report.py   # check actual files on cluster
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = PROJECT_ROOT / "output" / "results_table" / "results_table.json"


def _md_training_section(cells: list[dict]) -> str:
    # Group by experiment config (inferred from cell.paths.config).
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        if c["kind"] != "training":
            continue
        cfg = c["paths"].get("config", "?")
        by_cfg[cfg].append(c)

    lines = [
        "## Training-based baselines (experiment configs)",
        "",
        "| Config | Complete / Total | Failed | Running | Pending |",
        "|---|---:|---:|---:|---:|",
    ]
    overall = {"total": 0, "complete": 0, "failed": 0, "running": 0, "pending": 0}
    for cfg, runs in sorted(by_cfg.items()):
        total = len(runs)
        counts = defaultdict(int)
        for r in runs:
            counts[r["status"]] += 1
        exp_name = Path(cfg).stem
        lines.append(
            f"| {exp_name} | {counts['complete']} / {total} | "
            f"{counts['failed']} | {counts['running']} | {counts['pending']} |"
        )
        overall["total"] += total
        for k in ("complete", "failed", "running", "pending"):
            overall[k] += counts[k]
    lines += [
        "",
        f"**Overall training runs:** {overall['complete']} / {overall['total']} complete, "
        f"{overall['failed']} failed, {overall['running']} running, {overall['pending']} pending.",
        "",
    ]
    return "\n".join(lines)


def _md_sampling_section(cells: list[dict]) -> str:
    lines = [
        "## Sampling-baseline artefacts (SE + SelfCheckGPT)",
        "",
        "Status is per scoring variant. The pipeline runs on a subset of"
        " the test split (full for hotpotqa/nq/popqa/sciq; 10K cap for"
        " searchqa). Expected = subset size; actual = rows in the JSONL.",
        "",
        "| Dataset | Model | Method | Expected | Actual | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for c in cells:
        if c["kind"] != "sampling":
            continue
        k = c["key"]
        exp = c["expected_rows"] if c["expected_rows"] is not None else "-"
        act = c["actual_rows"] if c["actual_rows"] is not None else "-"
        lines.append(
            f"| {k['dataset']} | {k['model']} | {k['method']} | "
            f"{exp} | {act} | {c['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_sep_section(cells: list[dict]) -> str:
    lines = ["## SEP probe", "", "| Dataset | Model | Status |", "|---|---|---|"]
    for c in cells:
        if c["kind"] != "sep":
            continue
        k = c["key"]
        lines.append(f"| {k['dataset']} | {k['model']} | {c['status']} |")
    lines.append("")
    return "\n".join(lines)


def _md_ptrue_section(cells: list[dict]) -> str:
    lines = [
        "## P(true) baseline",
        "",
        "| Dataset | Model | Gen rows | P(true) rows | Status |",
        "|---|---|---:|---:|---|",
    ]
    for c in cells:
        if c["kind"] != "p_true":
            continue
        k = c["key"]
        exp = c["expected_rows"] if c["expected_rows"] is not None else "-"
        act = c["actual_rows"] if c["actual_rows"] is not None else "-"
        lines.append(
            f"| {k['dataset']} | {k['model']} | {exp} | {act} | {c['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _md_training_breakdown(cells: list[dict]) -> str:
    sym = {"running": "~", "failed": "x", "pending": "."}

    # model -> dataset -> method -> {status: [seeds]}
    by_model: dict[str, dict[str, dict[str, dict[str, list]]]] = {}
    for c in cells:
        if c["kind"] != "training" or c["status"] == "complete":
            continue
        m = by_model.setdefault(c["key"]["model"], {})
        d = m.setdefault(c["key"]["dataset"], {})
        method_bucket = d.setdefault(c["key"]["method"], {})
        method_bucket.setdefault(c["status"], []).append(c["key"]["seed"])

    lines = [
        "## Training baselines — missing/incomplete breakdown by model",
        "",
        "Each entry lists the (method, seeds) combos that are not yet"
        " complete. Status: `~` running, `x` failed, `.` pending.",
        "",
    ]

    all_models = sorted({c["key"]["model"] for c in cells if c["kind"] == "training"})
    model_order = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "SmolLM3"]
    seen = [m for m in model_order if m in all_models] + sorted(
        m for m in all_models if m not in model_order
    )
    if not by_model:
        lines.append("All training runs complete across every model. ✓")
        lines.append("")
        return "\n".join(lines)

    for m in seen:
        lines.append(f"### {m}")
        if m not in by_model:
            lines.append("")
            lines.append("All complete. ✓")
            lines.append("")
            continue
        for ds in sorted(by_model[m]):
            lines.append(f"- **{ds}**")
            for method in sorted(by_model[m][ds]):
                buckets = by_model[m][ds][method]
                parts = []
                for st in ("running", "failed", "pending"):
                    seeds = sorted(s for s in buckets.get(st, []) if s is not None)
                    no_seed = [s for s in buckets.get(st, []) if s is None]
                    if seeds:
                        parts.append(f"{sym[st]} seeds {{{','.join(map(str, seeds))}}}")
                    if no_seed:
                        parts.append(f"{sym[st]} (no-seed)")
                lines.append(f"    - `{method}`: {'; '.join(parts)}")
        lines.append("")
    return "\n".join(lines)


def build_report(payload: dict) -> str:
    cells = payload["cells"]
    ts = payload.get("generated_at", "unknown")
    git = payload.get("git", {})
    header = [
        f"# Experiment coverage audit",
        "",
        f"Generated: {ts}  ·  branch `{git.get('branch', '?')}` "
        f"@ `{git.get('commit', '?')[:10]}`",
        "",
        "Coverage view rendered from `output/results_table/results_table.json`."
        " A cell is OK when its canonical artefact is present (eval_metrics.json"
        " for training runs; full-length JSONL for sampling artefacts and P(true);"
        " sep_results.json for SEP).",
        "",
    ]
    return "\n".join([
        *header,
        _md_training_section(cells),
        _md_sampling_section(cells),
        _md_sep_section(cells),
        _md_ptrue_section(cells),
        _md_training_breakdown(cells),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_JSON,
        help=f"Path to results_table.json (default: {DEFAULT_JSON}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional markdown output path (default: print to stdout).",
    )
    args = parser.parse_args()

    if not args.results_json.exists():
        sys.exit(
            f"results_table.json not found at {args.results_json}. "
            f"Run: python scripts/results_table.py"
        )

    with open(args.results_json) as f:
        payload = json.load(f)

    out = build_report(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
