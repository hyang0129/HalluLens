#!/usr/bin/env python3
"""
Pivot results/results_table.csv (the canonical "what we have" view, produced by
scripts/results_table.py) and results/transfer_matrix_table.csv into a per-cell
gap report. For each cell, also check whether the per-sample prediction file
has been pulled down to results/preds/ (see scripts/pull_predictions.py).

No SSH / no remote calls — the input CSVs are already synced down by
results_table.py. Re-run that first if numbers feel stale.

Two coverage axes per cell:
  Summary AUROC — the headline number is present in results_table.csv (status=complete).
                  Implies upstream training + eval finished as of the last
                  results_table.py run.
  Predictions   — the per-sample prediction file is present under results/preds/
                  (synced from the cluster by scripts/pull_predictions.py).

Sections:
  §1 Training baselines (kind=training)
  §2 Ablations         (kind=ablation)
  §3 Sampling baselines (kind=sampling)
  §4 P(True)           (kind=p_true)
  §5 SEP               (kind=sep)
  §6 Transfer matrix   (from transfer_matrix_table.csv)
  Summary

The headline `contrastive_logprob_recon` emits per-sample distance scores
(one-column predictions.csv with score_halu = kNN distance). Its ablation
variants (b0..b9, c0..c3, cd1..cd3, d2a, d2b) skip per-sample output and
report eval_metrics.json only — those cells show `n/a (distance)` in the
Predictions column.
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_TABLE_CSV = RESULTS_DIR / "results_table.csv"
TRANSFER_TABLE_CSV = RESULTS_DIR / "transfer_matrix_table.csv"
LOCAL_PREDS_DIR = RESULTS_DIR / "preds"

EXPECTED_SEEDS = 5

MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]

# Per-sample files expected under results/preds/ for each cell kind
TRAINING_PREDS_FILES = ("predictions.csv", "eval_metrics.json")
SAMPLING_PREDS_FILES = {
    "se_semantic_entropy":  "se_labels.jsonl",
    "se_length_normalized": "se_labels.jsonl",
    "se_discrete":          "se_labels.jsonl",
    "selfcheck_nli":        "selfcheck_scores.jsonl",
    "selfcheck_ngram":      "selfcheck_scores.jsonl",
    "selfcheck_bertscore":  "selfcheck_scores.jsonl",
}
PTRUE_PREDS_FILE = "ptrue.jsonl"

# Methods that emit only eval_metrics.json, no per-sample predictions.csv.
# Note: the headline `contrastive_logprob_recon` itself DOES emit predictions.csv
# (one column: score_halu, the kNN distance score). Only its ablation variants
# (contrastive_logprob_recon_{b0..b9,c0..c3,cd1..cd3,d2a,d2b}) skip per-sample
# scores and report eval_metrics.json only.
DISTANCE_BASED_METHODS: set[str] = set()
DISTANCE_BASED_PREFIXES = ("contrastive_logprob_recon_",)

# Transfer-matrix grid (mirrors constants in results_table.py)
TRANSFER_MODEL_SLUGS = ["llama", "qwen3"]
TRANSFER_MODEL_DISPLAY = {"llama": "Llama-3.1-8B-Instruct", "qwen3": "Qwen3-8B"}
TRANSFER_DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
TRANSFER_METHODS = [
    "contrastive_logprob_recon",
    "saplma",
    "llmsknow_probe",
    "act_vit",
]
TRANSFER_EXPECTED_SEEDS = list(range(5))

# Higher = "better" status; collapse multiple metric rows per cell to the best.
_STATUS_RANK = {"complete": 4, "running": 3, "partial": 2, "pending": 1, "missing": 0, "": 0}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _canonical_ds(ds: str) -> str:
    """Normalize a results_table dataset label to the local-preds directory name."""
    ds = ds.removesuffix("_qwen3_memmap").removesuffix("_memmap")
    if ds == "nq":
        return "natural_questions"
    return ds


def _is_distance_based(method: str) -> bool:
    if method in DISTANCE_BASED_METHODS:
        return True
    return any(method.startswith(p) for p in DISTANCE_BASED_PREFIXES)


def load_results_table() -> list[dict]:
    if not RESULTS_TABLE_CSV.exists():
        sys.exit(f"missing {RESULTS_TABLE_CSV} — run scripts/results_table.py first")
    with open(RESULTS_TABLE_CSV) as f:
        return list(csv.DictReader(f))


def load_transfer_table() -> list[dict]:
    if not TRANSFER_TABLE_CSV.exists():
        return []
    with open(TRANSFER_TABLE_CSV) as f:
        return list(csv.DictReader(f))


def load_local_preds_index() -> dict:
    """
    Walk results/preds/ and return a dict keyed by (canonical_ds, model, method, seed)
    whose value is the set of filenames present at that key. Layouts (from
    scripts/pull_predictions.py):

        {ds}/{model}/{file}                         → (ds, model, None, None)
        {ds}/{model}/{method}/seed_{n}/{file}       → (ds, model, method, seed)
    """
    idx: dict = defaultdict(set)
    if not LOCAL_PREDS_DIR.exists():
        return idx
    for p in LOCAL_PREDS_DIR.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(LOCAL_PREDS_DIR).parts
        if len(rel) == 3:
            ds, model, fname = rel
            idx[(ds, model, None, None)].add(fname)
        elif len(rel) == 5 and rel[3].startswith("seed_"):
            ds, model, method, seed_dir, fname = rel
            try:
                seed = int(seed_dir.removeprefix("seed_"))
            except ValueError:
                continue
            idx[(ds, model, method, seed)].add(fname)
    return idx


# ---------------------------------------------------------------------------
# Per-cell status collapse
# ---------------------------------------------------------------------------

def _best_status(rows: list[dict]) -> str:
    """Collapse multiple metric rows for one cell to the highest-rank status."""
    return max((r.get("status", "") for r in rows), key=lambda s: _STATUS_RANK.get(s, 0), default="missing")


def _group_by_cell(rows: list[dict], kind: str, seeded: bool) -> dict:
    """
    Group results_table rows of a given kind into cells. Returns:
        seeded=True  → dict[(canon_ds, model, method)] → dict[seed → best_status]
        seeded=False → dict[(canon_ds, model, method)] → best_status
    """
    if seeded:
        seed_map: dict = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r.get("kind") != kind:
                continue
            try:
                seed = int(r["seed"]) if r.get("seed") not in (None, "") else None
            except ValueError:
                continue
            if seed is None:
                continue
            cell = (_canonical_ds(r["dataset"]), r["model"], r["method"])
            seed_map[cell][seed].append(r)
        return {cell: {sd: _best_status(rs) for sd, rs in seeds.items()} for cell, seeds in seed_map.items()}

    cell_rows: dict = defaultdict(list)
    for r in rows:
        if r.get("kind") != kind:
            continue
        cell = (_canonical_ds(r["dataset"]), r["model"], r["method"])
        cell_rows[cell].append(r)
    return {cell: _best_status(rs) for cell, rs in cell_rows.items()}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _seed_pred_status(preds: dict, ds: str, model: str, method: str, seed: int) -> str:
    """Return a short marker for local-preds presence at (ds, model, method, seed)."""
    if _is_distance_based(method):
        files = preds.get((ds, model, method, seed), set())
        return "n/a (distance)" if "eval_metrics.json" not in files else "metrics only"
    files = preds.get((ds, model, method, seed), set())
    have = [f for f in TRAINING_PREDS_FILES if f in files]
    if len(have) == len(TRAINING_PREDS_FILES):
        return "full"
    if have:
        return "partial (" + ",".join(have) + ")"
    return "missing"


def _section_seeded(
    title: str,
    cells: dict,
    preds: dict,
    *,
    expected_seeds: int = EXPECTED_SEEDS,
) -> tuple[list[str], int, int]:
    """
    Render a section for a seeded kind (training/ablation).
    Returns (lines, n_summary_gaps, n_predictions_gaps).

    Cells are split into three buckets:
      - complete:        Summary AUROC seeds == expected AND Predictions seeds == expected (or distance)
      - Summary AUROC gap: not enough seeds with status=complete in results_table.csv
      - Predictions gap:   Summary AUROC complete, per-sample preds not yet pulled to results/preds/
    """
    lines = [f"## {title}", ""]
    complete_rows = []
    summary_gap_rows = []
    preds_gap_rows = []

    for cell in sorted(cells.keys()):
        ds, model, method = cell
        seed_statuses = cells[cell]
        n_complete = sum(1 for s in seed_statuses.values() if s == "complete")
        summary_str = f"{n_complete}/{expected_seeds}"

        is_distance = _is_distance_based(method)
        if is_distance:
            preds_str = "n/a (distance)"
            n_preds_full = expected_seeds  # treat as satisfied for bucketing
        else:
            n_preds_full = sum(
                1 for sd in range(expected_seeds)
                if all(f in preds.get((ds, model, method, sd), set()) for f in TRAINING_PREDS_FILES)
            )
            preds_str = f"{n_preds_full}/{expected_seeds}"

        row = f"| {ds} | {model} | {method} | {summary_str} | {preds_str} |"
        summary_ok = n_complete == expected_seeds
        preds_ok = is_distance or n_preds_full == expected_seeds

        if summary_ok and preds_ok:
            complete_rows.append(row)
        elif not summary_ok:
            summary_gap_rows.append(row)
        else:
            preds_gap_rows.append(row)

    header = (
        "| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |\n"
        "|---------|-------|--------|---------------------|-------------------|"
    )

    lines += [
        f"### Complete ({len(complete_rows)})",
        "",
        header,
    ] + complete_rows + [""]

    if summary_gap_rows:
        lines += [
            f"### Missing Summary AUROC — need more training/eval ({len(summary_gap_rows)})",
            "",
            header,
        ] + summary_gap_rows + [""]
    else:
        lines += ["### Missing Summary AUROC\n\nNone — every cell has a Summary AUROC in results_table.csv.\n"]

    if preds_gap_rows:
        lines += [
            f"### Missing Predictions — run `scripts/pull_predictions.py` ({len(preds_gap_rows)})",
            "",
            header,
        ] + preds_gap_rows + [""]
    else:
        lines += ["### Missing Predictions\n\nNone — per-sample preds pulled for every Summary-AUROC-complete cell.\n"]

    return lines, len(summary_gap_rows), len(preds_gap_rows)


def _section_unseeded(
    title: str,
    cells: dict,
    preds: dict,
    *,
    expected_grid: list[tuple[str, str, str]] | None,
    preds_file_fn,
) -> tuple[list[str], int, int]:
    """
    Render a section for a non-seeded kind (sampling / p_true / sep).
    Returns (lines, n_summary_gaps, n_predictions_gaps).

    expected_grid: explicit list of (ds, model, method) tuples that should exist.
                   If None, use the cells we observed.
    preds_file_fn: callable (ds, model, method) → filename | None to check
                   under results/preds/{ds}/{model}/. Returning None means
                   no per-sample file is tracked for this cell kind.
    """
    lines = [f"## {title}", ""]
    expected = expected_grid if expected_grid is not None else sorted(cells.keys())

    complete_rows = []
    summary_gap_rows = []
    preds_gap_rows = []

    for ds, model, method in expected:
        summary = cells.get((ds, model, method), "missing")
        fname = preds_file_fn(ds, model, method)
        if fname is None:
            preds_str = "n/a"
            preds_ok = True
        else:
            files = preds.get((ds, model, None, None), set())
            preds_str = "present" if fname in files else "missing"
            preds_ok = preds_str == "present"

        row = f"| {ds} | {model} | {method} | {summary} | {preds_str} |"
        summary_ok = summary == "complete"

        if summary_ok and preds_ok:
            complete_rows.append(row)
        elif not summary_ok:
            summary_gap_rows.append(row)
        else:
            preds_gap_rows.append(row)

    header = (
        "| Dataset | Model | Method | Summary AUROC | Predictions file |\n"
        "|---------|-------|--------|---------------|------------------|"
    )

    lines += [
        f"### Complete ({len(complete_rows)})",
        "",
        header,
    ] + complete_rows + [""]

    if summary_gap_rows:
        lines += [
            f"### Missing Summary AUROC — need more training/eval ({len(summary_gap_rows)})",
            "",
            header,
        ] + summary_gap_rows + [""]
    else:
        lines += ["### Missing Summary AUROC\n\nNone — every cell has a Summary AUROC in results_table.csv.\n"]

    if preds_gap_rows:
        lines += [
            f"### Missing Predictions — run `scripts/pull_predictions.py` ({len(preds_gap_rows)})",
            "",
            header,
        ] + preds_gap_rows + [""]
    else:
        lines += ["### Missing Predictions\n\nNone — per-sample preds pulled for every Summary-AUROC-complete cell.\n"]

    return lines, len(summary_gap_rows), len(preds_gap_rows)


def _build_transfer_section(transfer_rows: list[dict]) -> tuple[list[str], int]:
    """§6 transfer matrix — pivoted from results/transfer_matrix_table.csv."""
    lines = ["## §6 Transfer Matrix", ""]

    present: set = set()
    for row in transfer_rows:
        if row.get("metric_name") != "auroc":
            continue
        try:
            seed = int(row["seed"])
        except (ValueError, KeyError, TypeError):
            continue
        present.add((row["model"], row["source_dataset"], row["target_dataset"], row["method"], seed))

    n_tgt = len(TRANSFER_DATASETS)
    n_seeds = len(TRANSFER_EXPECTED_SEEDS)
    cells_per_method_model = n_tgt * n_tgt * n_seeds  # 180
    total_expected = len(TRANSFER_MODEL_SLUGS) * len(TRANSFER_METHODS) * cells_per_method_model
    total_present = len(present)
    total_missing = total_expected - total_present

    lines += [
        f"Expected grid: {len(TRANSFER_MODEL_SLUGS)} models × {len(TRANSFER_METHODS)} methods"
        f" × {n_tgt} src × {n_tgt} tgt × {n_seeds} seeds = **{total_expected} cells**.",
        "",
        "Sources are gated on baseline checkpoints existing — missing source-dataset cells may"
        " reflect incomplete training rather than a missing transfer evaluation.",
        "",
        "### Coverage by method × model",
        "",
        f"| Method | Llama ({cells_per_method_model}) | Qwen3 ({cells_per_method_model}) |",
        "|--------|---------|---------|",
    ]
    method_gaps = {}
    for method in TRANSFER_METHODS:
        parts = [f"| {method}"]
        for slug in TRANSFER_MODEL_SLUGS:
            model_display = TRANSFER_MODEL_DISPLAY[slug]
            n = sum(
                1 for (mdl, src, tgt, m, sd) in present
                if mdl == model_display and m == method
            )
            mark = " ✓" if n == cells_per_method_model else ""
            parts.append(f"{n}/{cells_per_method_model}{mark}")
            method_gaps[(slug, method)] = cells_per_method_model - n
        lines.append(" | ".join(parts) + " |")
    lines.append("")

    # Missing-cell detail only for incomplete methods
    detail = []
    for slug in TRANSFER_MODEL_SLUGS:
        model_display = TRANSFER_MODEL_DISPLAY[slug]
        for method in TRANSFER_METHODS:
            if method_gaps.get((slug, method), 0) == 0:
                continue
            for src in TRANSFER_DATASETS:
                for tgt in TRANSFER_DATASETS:
                    for seed in TRANSFER_EXPECTED_SEEDS:
                        if (model_display, src, tgt, method, seed) not in present:
                            detail.append(f"| {model_display} | {method} | {src} | {tgt} | {seed} |")

    if detail:
        lines += [
            f"### Missing cells ({len(detail)} of {total_expected})",
            "",
            "| Model | Method | Source | Target | Seed |",
            "|-------|--------|--------|--------|------|",
        ] + detail + [""]
    else:
        lines += ["### Missing cells\n\nNone — full transfer grid is complete.\n"]

    return lines, total_missing


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(rows: list[dict], transfer_rows: list[dict], preds: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Predictions Gap Report",
        "",
        f"Generated: {now}",
        "",
        f"Source of truth: `{RESULTS_TABLE_CSV.relative_to(RESULTS_DIR.parent)}` and"
        f" `{TRANSFER_TABLE_CSV.relative_to(RESULTS_DIR.parent)}` (produced by"
        " `scripts/results_table.py` — re-run that first if numbers feel stale)."
        f" Local cache: `{LOCAL_PREDS_DIR.relative_to(RESULTS_DIR.parent)}` (synced"
        " by `scripts/pull_predictions.py`).",
        "",
        "Each cell is rated on two axes:",
        "",
        "- **Summary AUROC** — the headline metric is present in `results_table.csv`"
        " (i.e. upstream training + eval finished as of the last `results_table.py` run).",
        "- **Predictions** — the per-sample prediction file is in `results/preds/`,"
        " available for offline downstream analysis.",
        "",
        "The headline `contrastive_logprob_recon` emits per-sample distance"
        " scores in `predictions.csv` (one column: `score_halu`, the kNN"
        " distance). Its ablation variants (`*_b0`..`*_b9`, `*_c0`..`*_c3`,"
        " `*_cd1`..`*_cd3`, `*_d2a`, `*_d2b`) skip per-sample output and report"
        " `eval_metrics.json` only — those cells are marked `n/a (distance)`.",
        "",
    ]

    training_cells = _group_by_cell(rows, kind="training", seeded=True)
    sec, n_tr_summary, n_tr_preds = _section_seeded(
        "§1 Training Baselines (kind=training)", training_cells, preds)
    lines += sec

    ablation_cells = _group_by_cell(rows, kind="ablation", seeded=True)
    sec, n_ab_summary, n_ab_preds = _section_seeded(
        "§2 Ablations (kind=ablation)", ablation_cells, preds)
    lines += sec

    sampling_cells = _group_by_cell(rows, kind="sampling", seeded=False)
    sampling_grid = [
        (_canonical_ds(ds), model, method)
        for ds in ("hotpotqa", "nq", "popqa", "sciq", "searchqa")
        for model in MODELS
        for method in SAMPLING_PREDS_FILES.keys()
    ]
    sec, n_sa_summary, n_sa_preds = _section_unseeded(
        "§3 Sampling Baselines (kind=sampling)",
        sampling_cells,
        preds,
        expected_grid=sampling_grid,
        preds_file_fn=lambda ds, model, method: SAMPLING_PREDS_FILES.get(method),
    )
    lines += sec

    ptrue_cells = _group_by_cell(rows, kind="p_true", seeded=False)
    ptrue_grid = [
        (_canonical_ds(ds), model, "p_true")
        for ds in ("hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa")
        for model in MODELS
    ]
    sec, n_pt_summary, n_pt_preds = _section_unseeded(
        "§4 P(True) (kind=p_true)",
        ptrue_cells,
        preds,
        expected_grid=ptrue_grid,
        preds_file_fn=lambda ds, model, method: PTRUE_PREDS_FILE,
    )
    lines += sec

    sep_cells = _group_by_cell(rows, kind="sep", seeded=False)
    sep_grid = [
        (_canonical_ds(ds), model, "sep")
        for ds in ("hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa")
        for model in MODELS
    ]
    sec, n_sp_summary, n_sp_preds = _section_unseeded(
        "§5 SEP (kind=sep)",
        sep_cells,
        preds,
        expected_grid=sep_grid,
        preds_file_fn=lambda ds, model, method: None,  # no per-sample preds tracked
    )
    lines += sec

    sec, n_tx_summary = _build_transfer_section(transfer_rows)
    lines += sec

    total_summary = n_tr_summary + n_ab_summary + n_sa_summary + n_pt_summary + n_sp_summary + n_tx_summary
    total_preds = n_tr_preds + n_ab_preds + n_sa_preds + n_pt_preds + n_sp_preds
    lines += [
        "## Summary",
        "",
        "**Missing Summary AUROC** = `results_table.csv` had no `status=complete` row"
        " for this cell as of the last `results_table.py` run (more training/eval"
        " needed upstream, or re-run the table if it's stale).",
        "",
        "**Missing Predictions** = the cell has a Summary AUROC but its per-sample"
        " prediction file isn't in `results/preds/` yet (run `scripts/pull_predictions.py`).",
        "",
        "| Section | Missing Summary AUROC | Missing Predictions |",
        "|---------|-----------------------|---------------------|",
        f"| §1 Training | {n_tr_summary} | {n_tr_preds} |",
        f"| §2 Ablations | {n_ab_summary} | {n_ab_preds} |",
        f"| §3 Sampling | {n_sa_summary} | {n_sa_preds} |",
        f"| §4 P(True) | {n_pt_summary} | {n_pt_preds} |",
        f"| §5 SEP | {n_sp_summary} | {n_sp_preds} |",
        f"| §6 Transfer matrix | {n_tx_summary} | — |",
        f"| **Total** | **{total_summary}** | **{total_preds}** |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"Reading {RESULTS_TABLE_CSV.relative_to(RESULTS_DIR.parent)} ...")
    rows = load_results_table()
    print(f"Reading {TRANSFER_TABLE_CSV.relative_to(RESULTS_DIR.parent)} ...")
    transfer_rows = load_transfer_table()
    print(f"Indexing local preds under {LOCAL_PREDS_DIR.relative_to(RESULTS_DIR.parent)} ...")
    preds = load_local_preds_index()
    print(f"Building report from {len(rows)} results-table rows, {len(transfer_rows)} transfer rows,"
          f" {sum(len(v) for v in preds.values())} local pred files ...")

    report = build_report(rows, transfer_rows, preds)
    out_path = RESULTS_DIR / "predictions_gap_report.md"
    out_path.write_text(report)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
