#!/usr/bin/env python3
"""Assemble a unified, long-form results table across every baseline.

Walks all of:
  - Training experiments (configs/experiments/baseline_comparison_*.json)
  - Sampling baselines  (SE variants + SelfCheckGPT)
  - SEP probe           (output/sampling_baselines/sep/*)
  - P(true)             (output/p_true/*)

Emits TWO files in a single run (always both, never just one):

  results_table.json — structured, agent-friendly. One entry per cell:
    {
      "schema_version": 1,
      "generated_at": "...",
      "git": {"branch": "...", "commit": "..."},
      "cells": [
        {
          "key":    {"dataset": "...", "model": "...", "method": "...", "seed": 0},
          "kind":   "training" | "sampling" | "sep" | "p_true",
          "status": "complete" | "missing" | "failed" | "running" | "pending" | "partial",
          "metrics":  {"knn_auroc": 0.85, ...},
          "expected_rows": int|null,
          "actual_rows":   int|null,
          "paths":    {"name": "relative/path", ...}
        },
        ...
      ]
    }

  results_table.csv — long form. One row per (cell × metric).
    Columns: kind, dataset, model, method, seed, status, metric_name,
             metric_value, expected_rows, actual_rows, path

Cells with status != complete still appear (with empty metric/value) so the
table doubles as the coverage view — `audit_coverage.py` is now redundant
modulo presentation.

Usage:
    python scripts/results_table.py              # writes output/results_table/*
    python scripts/results_table.py --out-dir DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import (  # noqa: E402
    RunStatus,
    classify_run,
    enumerate_runs,
    load_experiment_config,
)
from tasks.sampling_baselines.paths import (  # noqa: E402
    SAMPLING_DATASETS,
    eval_results_json,
    generation_jsonl,
    model_name,
    nli_matrix_path,
    se_labels_path,
    searchqa_test_cap_path,
    selfcheck_samples_path,
    selfcheck_scores_path,
    sep_results_path,
    subset_index_path,
)
from tasks.p_true.paths import (  # noqa: E402
    DATASETS as PTRUE_DATASETS,
    MODELS as PTRUE_MODELS,
    ptrue_scores_path,
)

SCHEMA_VERSION = 1

SAMPLING_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]

# Optional sklearn (preferred); fall back to numpy Mann–Whitney AUROC.
try:
    from sklearn.metrics import roc_auc_score as _sklearn_auroc
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------

def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return -1
    with open(path) as f:
        return sum(1 for _ in f)


def _auroc(scores: Iterable[float], labels: Iterable[int]) -> Optional[float]:
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    if len(s) < 10 or len(np.unique(y)) < 2:
        return None
    if _HAS_SKLEARN:
        try:
            return float(_sklearn_auroc(y, s))
        except Exception:
            return None
    # Mann–Whitney AUROC with average ranks for ties.
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    # Average ranks for ties
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg = 0.5 * ((i + 1) + j)  # ranks are 1-indexed
        ranks[order[i:j]] = avg
        i = j
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        branch = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    return {"branch": branch, "commit": commit}


# ---------------------------------------------------------------------------
# Cell collectors — each returns a list of cell dicts ready for serialization.
# ---------------------------------------------------------------------------

# Per-method metric keys to surface from eval_metrics.json. Each method may
# have its own set; anything not listed here is dropped (keeps the table from
# accidentally including run_size / hparams / etc.).
_TRAINING_METRIC_KEYS = {
    "contrastive_logprob_recon": ("cosine_auroc", "mahalanobis_auroc", "knn_auroc"),
    "contrastive":               ("cosine_auroc", "mahalanobis_auroc", "knn_auroc"),
    "linear_probe":               ("auroc",),
    "saplma":                     ("auroc",),
    "saplma_logprob_recon":       ("auroc",),
    "llmsknow_probe":             ("auroc", "sweep_best_dev_auroc"),
    "multi_layer_linear_probe":   ("auroc",),
    "icr_probe":                  ("auroc",),
    "token_entropy":              ("mean_entropy_auroc", "mean_logprob_auroc",
                                   "min_logprob_auroc"),
    "logprob_baseline":           ("mean_logprob_auroc", "perplexity_auroc",
                                   "seq_logprob_auroc"),
}

# Extended metrics from eval_metrics_extended.json (issue #59). Universally
# applicable across methods — sidecar handles the per-method ECE skip itself
# (writes null for non-probabilistic methods, which is dropped below).
_EXTENDED_METRIC_KEYS = (
    "auprc",
    "fpr_at_95_tpr",
    "ece",
)
# CI fields are 2-tuples on disk; flatten to two scalar columns for the
# long-form CSV (which expects scalar metric_value).
_EXTENDED_CI_KEYS = (
    "auroc_ci95",
    "auprc_ci95",
    "fpr_at_95_tpr_ci95",
)


def _merge_extended_metrics(run_dir: Path, metrics: dict[str, float]) -> None:
    """Merge eval_metrics_extended.json sidecar into ``metrics`` if present.

    Missing sidecar is silent — the corresponding columns will simply be
    blank in the long-form CSV (effectively NaN). Run
    ``scripts/compute_extended_metrics.py`` to populate.
    """
    sidecar = run_dir / "eval_metrics_extended.json"
    if not sidecar.exists():
        return
    with open(sidecar) as f:
        ext = json.load(f)
    for k in _EXTENDED_METRIC_KEYS:
        v = ext.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            metrics[k] = float(v)
    for k in _EXTENDED_CI_KEYS:
        ci = ext.get(k)
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = ci
            if isinstance(lo, (int, float)) and np.isfinite(lo):
                metrics[f"{k}_lo"] = float(lo)
            if isinstance(hi, (int, float)) and np.isfinite(hi):
                metrics[f"{k}_hi"] = float(hi)


def _model_from_config_name(cfg_name: str) -> str:
    stem = cfg_name.removesuffix(".json")
    if stem.endswith("_qwen3"):
        return "Qwen3-8B"
    if stem.endswith("_smollm3"):
        return "SmolLM3"
    return "Llama-3.1-8B-Instruct"


def _dataset_from_config_name(cfg_name: str, model_label: str) -> str:
    stem = cfg_name.removesuffix(".json").removeprefix("baseline_comparison_")
    if model_label == "Qwen3-8B":
        stem = stem.removesuffix("_qwen3")
    elif model_label == "SmolLM3":
        stem = stem.removesuffix("_smollm3")
    return stem


def collect_training_cells(*, include_smollm3: bool = False) -> list[dict]:
    """One cell per (config, dataset, method, seed).

    SmolLM3 configs are excluded by default per PAPER_ROADMAP §5 (third-model
    expansion is out of scope for this submission). Pass ``include_smollm3=True``
    (or ``--include-smollm3`` on the CLI) to re-enable them.
    """
    cells: list[dict] = []
    cfg_dir = PROJECT_ROOT / "configs" / "experiments"
    for cfg_path in sorted(cfg_dir.glob("baseline_comparison_*.json")):
        cfg = load_experiment_config(str(cfg_path), project_root=str(PROJECT_ROOT))
        model_label = _model_from_config_name(cfg_path.name)
        if model_label == "SmolLM3" and not include_smollm3:
            continue
        dataset_label = _dataset_from_config_name(cfg_path.name, model_label)

        specs = enumerate_runs(
            cfg,
            output_base=cfg.get("output_dir", "runs"),
            project_root=str(PROJECT_ROOT),
        )
        for spec in specs:
            run_dir = Path(spec.run_dir)
            status = classify_run(str(run_dir))
            metrics: dict[str, float] = {}
            if status == RunStatus.COMPLETE:
                with open(run_dir / "eval_metrics.json") as f:
                    eval_data = json.load(f)
                allowed = _TRAINING_METRIC_KEYS.get(spec.method_name, ())
                for k in allowed:
                    v = eval_data.get(k)
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        metrics[k] = float(v)
                _merge_extended_metrics(run_dir, metrics)

            cells.append({
                "key": {
                    "dataset": dataset_label,
                    "model": model_label,
                    "method": spec.method_name,
                    "seed": spec.seed,
                },
                "kind": "training",
                "status": status.value,
                "metrics": metrics,
                "expected_rows": None,
                "actual_rows": None,
                "paths": {
                    "run_dir":      _rel(run_dir),
                    "eval_metrics": _rel(run_dir / "eval_metrics.json"),
                    "eval_metrics_extended": (
                        _rel(run_dir / "eval_metrics_extended.json")
                        if (run_dir / "eval_metrics_extended.json").exists()
                        else None
                    ),
                    "config":       _rel(cfg_path),
                },
            })
    return cells


def _expected_sampling_rows(ds: str, mid: str, split: str, gen_n: int) -> int:
    if split == "train":
        idx_path = subset_index_path(ds, mid)
        if idx_path.exists():
            with open(idx_path) as f:
                return len(json.load(f)["question_ids"])
        return 5000
    if ds == "searchqa":
        cap_path = searchqa_test_cap_path(mid)
        if cap_path.exists():
            with open(cap_path) as f:
                return len(json.load(f)["question_ids"])
        return 10000
    return gen_n


def _load_jsonl_by_row(path: Path) -> dict[int, dict]:
    by_row: dict[int, dict] = {}
    if not path.exists():
        return by_row
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_row[rec["row_idx"]] = rec
            except Exception:
                pass
    return by_row


def _load_hallu_labels(ds: str, mid: str, split: str = "test") -> Optional[np.ndarray]:
    p = eval_results_json(ds, mid, split)
    if not p.exists():
        return None
    with open(p) as f:
        return np.array(json.load(f)["halu_test_res"], dtype=int)


def _load_cap(ds: str, mid: str) -> Optional[set[int]]:
    if ds != "searchqa":
        return None
    p = searchqa_test_cap_path(mid)
    if not p.exists():
        return None
    with open(p) as f:
        return set(json.load(f)["question_ids"])


def _aligned(by_row: dict, key: str, labels: np.ndarray, cap: Optional[set[int]]):
    scores, lbls = [], []
    for row_idx, rec in sorted(by_row.items()):
        if cap is not None and row_idx not in cap:
            continue
        if row_idx >= len(labels):
            continue
        v = rec.get(key)
        if v is None:
            continue
        if isinstance(v, float) and not np.isfinite(v):
            continue
        scores.append(v)
        lbls.append(int(labels[row_idx]))
    return scores, lbls


def collect_sampling_cells() -> list[dict]:
    """SE + SelfCheckGPT.

    Each (dataset, model) generates several "method" rows, one per scoring
    variant (semantic_entropy, length_normalized_se, discrete_se,
    selfcheck_nli, selfcheck_ngram, selfcheck_bertscore). The test-split
    JSONL is what defines completeness; AUROC is computed against
    halu_test_res from eval_results_for_training.json.
    """
    cells: list[dict] = []

    variants = [
        # (method_key,            file_fn,               score_key)
        ("se_semantic_entropy",   se_labels_path,        "semantic_entropy"),
        ("se_length_normalized",  se_labels_path,        "length_normalized_se"),
        ("se_discrete",           se_labels_path,        "discrete_se"),
        ("selfcheck_nli",         selfcheck_scores_path, "nli"),
        ("selfcheck_ngram",       selfcheck_scores_path, "ngram"),
        ("selfcheck_bertscore",   selfcheck_scores_path, "bertscore"),
    ]

    for ds in SAMPLING_DATASETS:
        for mid in SAMPLING_MODELS:
            gen_n = _count_lines(generation_jsonl(ds, mid, "test"))
            expected = _expected_sampling_rows(ds, mid, "test", gen_n)
            labels = _load_hallu_labels(ds, mid, "test")
            cap = _load_cap(ds, mid)

            for method_key, path_fn, score_key in variants:
                p = path_fn(ds, mid, "test")
                actual = _count_lines(p)

                if gen_n < 0:
                    status = "missing"  # no generation
                elif actual < 0:
                    status = "missing"
                elif actual < expected:
                    status = "partial"
                else:
                    status = "complete"

                metrics: dict[str, float] = {}
                if status == "complete" and labels is not None:
                    by_row = _load_jsonl_by_row(p)
                    # bertscore lives in selfcheck_scores.jsonl but is gated
                    # by --no-bertscore at generation time, so most rows lack
                    # the key. We still try; _auroc returns None if too sparse.
                    scores, lbls = _aligned(by_row, score_key, labels, cap)
                    auroc = _auroc(scores, lbls)
                    if auroc is not None:
                        metrics["auroc"] = auroc
                        metrics["n_scored"] = float(len(scores))

                cells.append({
                    "key": {
                        "dataset": ds,
                        "model": model_name(mid),
                        "method": method_key,
                        "seed": None,
                    },
                    "kind": "sampling",
                    "status": status,
                    "metrics": metrics,
                    "expected_rows": expected if gen_n >= 0 else None,
                    "actual_rows": actual if actual >= 0 else None,
                    "paths": {
                        "scores":     _rel(p),
                        "generation": _rel(generation_jsonl(ds, mid, "test")),
                        "labels":     _rel(eval_results_json(ds, mid, "test")),
                    },
                })
    return cells


def collect_sep_cells() -> list[dict]:
    cells: list[dict] = []
    sep_datasets = list(SAMPLING_DATASETS) + ["mmlu"]
    for ds in sep_datasets:
        for mid in SAMPLING_MODELS:
            p = sep_results_path(ds, mid)
            metrics: dict[str, float] = {}
            status = "missing"
            if p.exists():
                status = "complete"
                with open(p) as f:
                    data = json.load(f)
                for k in ("sep_se_auroc", "sep_binary_auroc"):
                    v = data.get(k)
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        metrics[k] = float(v)
                for k in ("layer", "train_size_sep_se"):
                    v = data.get(k)
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
            cells.append({
                "key": {
                    "dataset": ds,
                    "model": model_name(mid),
                    "method": "sep",
                    "seed": None,
                },
                "kind": "sep",
                "status": status,
                "metrics": metrics,
                "expected_rows": None,
                "actual_rows": None,
                "paths": {"sep_results": _rel(p)},
            })
    return cells


def collect_p_true_cells() -> list[dict]:
    cells: list[dict] = []
    for mid in PTRUE_MODELS:
        for ds in PTRUE_DATASETS:
            p = ptrue_scores_path(ds, mid, "test")
            gen_n = _count_lines(generation_jsonl(ds, mid, "test"))
            actual = _count_lines(p)

            if gen_n < 0:
                status = "missing"
            elif actual < 0:
                status = "missing"
            elif actual < gen_n:
                status = "partial"
            else:
                status = "complete"

            metrics: dict[str, float] = {}
            if status == "complete":
                labels = _load_hallu_labels(ds, mid, "test")
                if labels is not None:
                    by_row = _load_jsonl_by_row(p)
                    fwd_scores, fwd_lbls = [], []
                    rev_scores, rev_lbls = [], []
                    for row_idx, rec in sorted(by_row.items()):
                        if row_idx >= len(labels):
                            continue
                        lbl = int(labels[row_idx])
                        if (v := rec.get("p_true")) is not None and np.isfinite(v):
                            fwd_scores.append(1.0 - v)  # high score = halu
                            fwd_lbls.append(lbl)
                        if (v := rec.get("p_true_reversed")) is not None and np.isfinite(v):
                            rev_scores.append(1.0 - v)
                            rev_lbls.append(lbl)
                    a_fwd = _auroc(fwd_scores, fwd_lbls)
                    a_rev = _auroc(rev_scores, rev_lbls)
                    if a_fwd is not None:
                        metrics["p_true_auroc_fwd"] = a_fwd
                    if a_rev is not None:
                        metrics["p_true_auroc_rev"] = a_rev
                    if a_fwd is not None or a_rev is not None:
                        metrics["p_true_auroc_best"] = max(
                            v for v in (a_fwd, a_rev) if v is not None
                        )

            cells.append({
                "key": {
                    "dataset": ds,
                    "model": model_name(mid),
                    "method": "p_true",
                    "seed": None,
                },
                "kind": "p_true",
                "status": status,
                "metrics": metrics,
                "expected_rows": gen_n if gen_n >= 0 else None,
                "actual_rows": actual if actual >= 0 else None,
                "paths": {
                    "scores":     _rel(p),
                    "generation": _rel(generation_jsonl(ds, mid, "test")),
                    "labels":     _rel(eval_results_json(ds, mid, "test")),
                },
            })
    return cells


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "kind", "dataset", "model", "method", "seed", "status",
    "metric_name", "metric_value",
    "expected_rows", "actual_rows", "path",
]


def cells_to_long_rows(cells: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for c in cells:
        key = c["key"]
        primary_path = next(iter(c["paths"].values()), "")
        base = {
            "kind":          c["kind"],
            "dataset":       key["dataset"],
            "model":         key["model"],
            "method":        key["method"],
            "seed":          key["seed"] if key["seed"] is not None else "",
            "status":        c["status"],
            "expected_rows": c["expected_rows"] if c["expected_rows"] is not None else "",
            "actual_rows":   c["actual_rows"] if c["actual_rows"] is not None else "",
            "path":          primary_path,
        }
        if c["metrics"]:
            for metric_name, value in c["metrics"].items():
                rows.append({**base, "metric_name": metric_name, "metric_value": value})
        else:
            # Cells with no metrics still emit a single row (preserves coverage view).
            rows.append({**base, "metric_name": "", "metric_value": ""})
    return rows


def write_outputs(cells: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results_table.json"
    csv_path  = out_dir / "results_table.csv"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_info(),
        "cells": cells,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    rows = cells_to_long_rows(cells)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return json_path, csv_path


def _summary(cells: list[dict]) -> str:
    by_kind: dict[str, dict[str, int]] = {}
    for c in cells:
        d = by_kind.setdefault(c["kind"], {})
        d[c["status"]] = d.get(c["status"], 0) + 1
        d["total"] = d.get("total", 0) + 1
    lines = ["Summary:"]
    for kind in ("training", "sampling", "sep", "p_true"):
        d = by_kind.get(kind, {})
        if not d:
            continue
        total = d.get("total", 0)
        complete = d.get("complete", 0)
        other = ", ".join(f"{k}={v}" for k, v in sorted(d.items())
                          if k not in ("total", "complete"))
        lines.append(f"  {kind:<10} {complete}/{total} complete  ({other})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "results_table",
        help="Directory to write results_table.json + results_table.csv "
             "(default: output/results_table/).",
    )
    parser.add_argument(
        "--include-smollm3",
        action="store_true",
        help="Include SmolLM3 training cells. Excluded by default since "
             "SmolLM3 is out of scope per PAPER_ROADMAP §5; flip on to inspect "
             "the cached SmolLM3 configs without losing the rest of the table.",
    )
    args = parser.parse_args()

    cells: list[dict] = []
    cells += collect_training_cells(include_smollm3=args.include_smollm3)
    cells += collect_sampling_cells()
    cells += collect_sep_cells()
    cells += collect_p_true_cells()

    json_path, csv_path = write_outputs(cells, args.out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print()
    print(_summary(cells))
    if not _HAS_SKLEARN:
        print()
        print("(note: sklearn not available — AUROC computed with numpy "
              "Mann–Whitney implementation.)")


if __name__ == "__main__":
    main()
