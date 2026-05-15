"""Compute extended evaluation metrics (AUPRC, FPR@95, ECE, bootstrap CIs).

Pure CPU post-hoc analysis. Walks runs/ and runs_archive/, reads each
``predictions.csv``, and writes an ``eval_metrics_extended.json`` sidecar
next to the existing ``eval_metrics.json``. Does not modify the existing
file (additive schema).

Usage:
    python scripts/compute_extended_metrics.py --runs-dir runs
    python scripts/compute_extended_metrics.py --runs-dir runs runs_archive
    python scripts/compute_extended_metrics.py --runs-dir runs --force
    python scripts/compute_extended_metrics.py --runs-dir runs --bootstrap-b 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Methods that emit calibrated probabilities — ECE is meaningful for these.
# Anything not in this set gets ECE=null (raw scores / margins are not probs).
PROBABILISTIC_METHODS: frozenset[str] = frozenset(
    {
        "linear_probe",
        "saplma",
        "llmsknow_probe",
        "multi_layer_linear_probe",
    }
)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float | None:
    """Return FPR at the threshold where TPR first reaches ``target_tpr``.

    Returns None when the achievable max TPR is below the target (degenerate
    ranking), so callers can serialise the cell as null.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if tpr.max() < target_tpr:
        return None
    # roc_curve returns thresholds in descending order, so tpr is monotone
    # non-decreasing — np.interp is well defined.
    return float(np.interp(target_tpr, tpr, fpr))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width binning over [0, 1].

    Per Guo et al. 2017. ``y_prob`` is interpreted as P(label == 1).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_prob)
    ece = 0.0
    # bin index in [0, n_bins-1]; right edge inclusive for the last bin
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def _stratified_bootstrap_indices(
    y_true: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> np.ndarray:
    """Yield resampling indices that preserve at least one example per class.

    Returns shape (n_resamples, n) array. We resample each class separately
    (with replacement) at its observed marginal — guarantees both classes are
    present in every resample, which is required for AUROC/AUPRC/FPR@95.
    """
    n = len(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n_pos = len(pos_idx)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Bootstrap requires both classes to be present.")
    # Sample with replacement within each class, preserving the per-class count.
    pos_samples = rng.integers(0, n_pos, size=(n_resamples, n_pos))
    neg_samples = rng.integers(0, n_neg, size=(n_resamples, n_neg))
    out = np.empty((n_resamples, n), dtype=np.int64)
    out[:, :n_pos] = pos_idx[pos_samples]
    out[:, n_pos:] = neg_idx[neg_samples]
    return out


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    """2.5th/97.5th percentile bootstrap CI for an arbitrary metric.

    ``metric_fn(y_true, y_score) -> float | None``. Resamples that yield None
    (e.g. FPR@95 on a degenerate split) are dropped; if too few survive, the
    CI itself is None.
    """
    if len(y_true) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx_matrix = _stratified_bootstrap_indices(y_true, rng, n_resamples)
    vals: list[float] = []
    for i in range(n_resamples):
        idx = idx_matrix[i]
        v = metric_fn(y_true[idx], y_score[idx])
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    if len(vals) < max(20, n_resamples // 50):
        # Not enough valid resamples — refuse to fabricate a CI.
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-run computation
# ---------------------------------------------------------------------------


def _infer_method(run_dir: Path, eval_metrics: dict) -> str | None:
    """Best-effort recovery of the method name for ECE eligibility."""
    method = eval_metrics.get("method")
    if method:
        return method
    # Path convention: runs/<exp>/<dataset>/<method>/(seed_N|.)
    parts = run_dir.parts
    for candidate in reversed(parts):
        if candidate.startswith("seed_"):
            continue
        if candidate in PROBABILISTIC_METHODS:
            return candidate
    return None


def compute_for_run(
    run_dir: Path,
    *,
    bootstrap_b: int,
    bootstrap_seed: int,
) -> dict:
    """Compute the extended-metrics record for a single run directory."""
    pred_path = run_dir / "predictions.csv"
    eval_path = run_dir / "eval_metrics.json"

    df = pd.read_csv(pred_path)
    missing = {"score_halu", "label_halu"} - set(df.columns)
    if missing:
        raise ValueError(f"{pred_path}: missing columns {sorted(missing)}")
    df = df.dropna(subset=["score_halu", "label_halu"])
    y_score = df["score_halu"].to_numpy(dtype=np.float64)
    y_true = df["label_halu"].to_numpy(dtype=np.int64)
    n_test = int(len(y_true))
    n_pos = int(y_true.sum())

    eval_metrics: dict = {}
    if eval_path.exists():
        with open(eval_path) as f:
            eval_metrics = json.load(f)

    out: dict = {
        "auroc": None,
        "auprc": None,
        "fpr_at_95_tpr": None,
        "auroc_ci95": None,
        "auprc_ci95": None,
        "fpr_at_95_tpr_ci95": None,
        "ece": None,
        "bootstrap_b": int(bootstrap_b),
        "bootstrap_seed": int(bootstrap_seed),
        "n_test": n_test,
        "n_pos": n_pos,
    }

    # Need both classes for any of the ranking metrics.
    if n_pos == 0 or n_pos == n_test:
        out["note"] = "single-class predictions; ranking metrics undefined"
        return out

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    fpr95 = _fpr_at_tpr(y_true, y_score, 0.95)

    out["auroc"] = auroc
    out["auprc"] = auprc
    out["fpr_at_95_tpr"] = fpr95

    # AUROC sanity-check vs persisted value (do not overwrite, just record).
    persisted = eval_metrics.get("auroc")
    if persisted is not None:
        out["auroc_persisted"] = float(persisted)
        out["auroc_match"] = bool(abs(float(persisted) - auroc) < 1e-4)

    # Bootstrap CIs. Each metric uses an independent RNG stream derived from
    # the same seed so a single seed change re-rolls all three identically.
    out["auroc_ci95"] = list(
        _bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: float(roc_auc_score(yt, ys)),
            n_resamples=bootstrap_b, seed=bootstrap_seed,
        ) or []
    ) or None
    out["auprc_ci95"] = list(
        _bootstrap_ci(
            y_true, y_score,
            lambda yt, ys: float(average_precision_score(yt, ys)),
            n_resamples=bootstrap_b, seed=bootstrap_seed + 1,
        ) or []
    ) or None
    if fpr95 is not None:
        out["fpr_at_95_tpr_ci95"] = list(
            _bootstrap_ci(
                y_true, y_score,
                lambda yt, ys: _fpr_at_tpr(yt, ys, 0.95),
                n_resamples=bootstrap_b, seed=bootstrap_seed + 2,
            ) or []
        ) or None

    # ECE only for methods that emit calibrated probabilities.
    method = _infer_method(run_dir, eval_metrics)
    if method in PROBABILISTIC_METHODS:
        # Probability inputs must lie in [0, 1]; clip for numerical safety.
        probs = np.clip(y_score, 0.0, 1.0)
        out["ece"] = _ece(y_true, probs, n_bins=15)
        out["ece_n_bins"] = 15
    out["method"] = method
    if "seed" in eval_metrics:
        out["seed"] = eval_metrics["seed"]

    # Diagnostic: low-positive-count cells make FPR@95 a discrete step.
    if n_pos < 20:
        out["low_positives_warning"] = True

    return out


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def find_run_dirs(roots: Iterable[Path]) -> list[Path]:
    runs: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[skip] {root} does not exist", file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(root):
            if "predictions.csv" in files:
                runs.append(Path(dirpath))
    runs.sort()
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        nargs="+",
        default=["runs"],
        help="One or more roots to walk (e.g. runs runs_archive)",
    )
    parser.add_argument("--bootstrap-b", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if eval_metrics_extended.json already exists",
    )
    parser.add_argument(
        "--output-name",
        default="eval_metrics_extended.json",
        help="Sidecar filename (default: eval_metrics_extended.json)",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.runs_dir]
    run_dirs = find_run_dirs(roots)
    if not run_dirs:
        print("No predictions.csv files found.")
        return 1

    print(f"Found {len(run_dirs)} runs.")
    n_written = 0
    n_skipped = 0
    n_failed = 0
    n_mismatch = 0
    for run_dir in run_dirs:
        sidecar = run_dir / args.output_name
        if sidecar.exists() and not args.force:
            n_skipped += 1
            continue
        try:
            record = compute_for_run(
                run_dir,
                bootstrap_b=args.bootstrap_b,
                bootstrap_seed=args.bootstrap_seed,
            )
        except Exception as exc:  # noqa: BLE001 — surface per-run errors, keep going
            print(f"[fail] {run_dir}: {exc}", file=sys.stderr)
            n_failed += 1
            continue
        with open(sidecar, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        n_written += 1
        if record.get("auroc_match") is False:
            n_mismatch += 1
            print(
                f"[warn] AUROC mismatch in {run_dir}: "
                f"persisted={record.get('auroc_persisted')} recomputed={record.get('auroc')}",
                file=sys.stderr,
            )
    print(
        f"Wrote {n_written}, skipped {n_skipped} (already present), "
        f"failed {n_failed}, AUROC mismatches {n_mismatch}."
    )
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
