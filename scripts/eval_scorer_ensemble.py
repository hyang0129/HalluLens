#!/usr/bin/env python
"""Post-hoc KNN + Mahalanobis scorer ensemble over dumped contrastive embeddings.

Zero GPU, zero re-inference. Reads ``<run>/embeddings/{train,test}_z.npy`` +
``{train,test}_labels.npy`` (written when a run sets ``dump_embeddings: true``),
recomputes per-sample KNN and Mahalanobis OOD scores via the *same* library
functions the eval uses, and evaluates an a-priori rank-average ensemble against
each single scorer.

Orientation safety: for each cell we auto-detect ``(outlier_class,
train_label_filter)`` by matching the recomputed ``knn_auroc`` /
``mahalanobis_auroc`` to the persisted values in the sibling
``eval_metrics.json``. A cell whose recomputed AUROC does not match the
persisted value within tolerance is flagged and excluded from aggregates, so we
never trust a mis-oriented cell.

The headline ensemble rule is **parameter-free**: equal-weight average of the
two scorers' percentile ranks (scale-free, nothing tuned on test). We *also*
report, per cell, the test-optimal blend weight as an explicit UPPER BOUND (not
a usable result) so we can see how much headroom a train-selected weight could
capture in a follow-up.

Issue #127 (flipped-convention focus). Companion to scripts/compute_extended_metrics.py.

Usage (run from repo root, e.g. on the Empire AI login node):
    python scripts/eval_scorer_ensemble.py --runs-dir runs --filter b5
    python scripts/eval_scorer_ensemble.py --runs-dir runs --filter b5 --out /tmp/ensemble_flipped.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activation_research.metrics import knn_ood_stats, mahalanobis_ood_stats  # noqa: E402

# KNN params mirror configs/methods/contrastive_logprob_recon_b5.json, but with
# calibrate_k disabled: this is a fast proof and k=50 is the method's default k.
# The recomputed knn may differ slightly from the persisted (calibrated) value;
# we report that gap per cell so we know where calibration mattered. The native
# Phase-2 metric will use calibrate_k for the final paper number.
KNN_PARAMS = dict(
    k=50,
    metric="euclidean",
    calibrate_k=False,
    max_train_size=20000,  # stratified subsample for a fast proof; big balanced
)                          # trains get capped, simpleqa/popqa are below it anyway
KNN_GAP_TOL = 0.05  # recomputed knn@50 vs persisted-calibrated: calibration tolerance
MAHA_TOL = 0.01     # Mahalanobis has no calibration -> must match persisted tightly


def load_records(emb_dir: Path, split: str):
    z = np.load(emb_dir / f"{split}_z.npy")  # (N, K, D) fp16
    y = np.load(emb_dir / f"{split}_labels.npy")  # (N,) int8
    recs = []
    for i in range(z.shape[0]):
        recs.append({"z_views": torch.from_numpy(z[i].astype(np.float32)), "halu": int(y[i])})
    return recs


def percentile_ranks(x: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] percentile ranks (higher score -> higher rank)."""
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / (n - 1)


def detect_knn(train, test, persisted):
    """Find (outlier_class, filter) whose recomputed knn_auroc matches persisted."""
    best = None
    for oc in (1, 0):
        for filt in ("id_only", "all"):
            try:
                s = knn_ood_stats(train, test, outlier_class=oc, train_label_filter=filt,
                                  include_per_sample=True, **KNN_PARAMS)
            except Exception:
                continue
            d = abs(s["knn_auroc"] - persisted)
            cand = (d, oc, filt, s["knn_scores"], np.asarray(s["knn_labels"]), s["knn_auroc"])
            if best is None or d < best[0]:
                best = cand
    return best


def score_maha(train, test, oc, persisted):
    """Maha per-sample at fixed outlier_class; pick filter matching persisted."""
    best = None
    for filt in ("id_only", "all"):
        try:
            s = mahalanobis_ood_stats(train, test, outlier_class=oc,
                                      train_label_filter=filt, include_per_sample=True)
        except Exception:
            continue
        d = abs(s["mahalanobis_auroc"] - persisted)
        cand = (d, filt, s["mahalanobis_scores"], np.asarray(s["mahalanobis_labels"]), s["mahalanobis_auroc"])
        if best is None or d < best[0]:
            best = cand
    return best


def parse_cell(emb_dir: Path):
    """Return (dataset, model, seed, method) from an embeddings dir path."""
    # runs/<exp>/<dataset_cfg>/<method>/<seed>/embeddings
    parts = emb_dir.parts
    seed = parts[-2]
    method = parts[-3]
    dataset = parts[-4]
    model = "Qwen3" if "qwen3" in dataset else "Llama"
    ds = dataset.replace("_qwen3_memmap", "").replace("_memmap", "")
    return ds, model, seed, method


parse_cell.cache = None  # cached (outlier_class, knn_filter, maha_filter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--filter", default="b5", help="substring the method dir must contain")
    ap.add_argument("--out", default="/tmp/ensemble_results.csv")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    emb_dirs = sorted(p.parent for p in runs.rglob("embeddings/test_z.npy")
                      if args.filter in str(p))
    print(f"found {len(emb_dirs)} embedding dirs matching filter={args.filter!r}\n", flush=True)

    rows = []
    for emb_dir in emb_dirs:
        ds, model, seed, method = parse_cell(emb_dir)
        run_dir = emb_dir.parent
        em_path = run_dir / "eval_metrics.json"
        if not em_path.exists():
            print(f"SKIP {ds}/{model}/{seed}: no eval_metrics.json", flush=True)
            continue
        em = json.loads(em_path.read_text())
        p_knn = em.get("knn_auroc")
        p_maha = em.get("mahalanobis_auroc")
        p_cos = em.get("cosine_auroc")
        if p_knn is None or p_maha is None:
            print(f"SKIP {ds}/{model}/{seed}: missing persisted aurocs", flush=True)
            continue

        try:
            train = load_records(emb_dir, "train")
            test = load_records(emb_dir, "test")
        except Exception as e:
            print(f"SKIP {ds}/{model}/{seed}: load failed ({e})", flush=True)
            continue

        # Orientation is constant across cells of one convention; cache it and
        # only fall back to the full 4-way search if the cached combo mismatches.
        knn_sc = labels = knn_auroc = maha_sc = maha_labels = maha_auroc = None
        oc = kfilt = mfilt = None
        if parse_cell.cache is not None:
            c_oc, c_kfilt, c_mfilt = parse_cell.cache
            s = knn_ood_stats(train, test, outlier_class=c_oc, train_label_filter=c_kfilt,
                              include_per_sample=True, **KNN_PARAMS)
            if abs(s["knn_auroc"] - p_knn) < KNN_GAP_TOL:
                oc, kfilt = c_oc, c_kfilt
                knn_sc, labels, knn_auroc, d_knn = s["knn_scores"], np.asarray(s["knn_labels"]), s["knn_auroc"], abs(s["knn_auroc"] - p_knn)
                sm = mahalanobis_ood_stats(train, test, outlier_class=oc,
                                           train_label_filter=c_mfilt, include_per_sample=True)
                mfilt, maha_sc, maha_labels, maha_auroc = c_mfilt, sm["mahalanobis_scores"], np.asarray(sm["mahalanobis_labels"]), sm["mahalanobis_auroc"]
                d_maha = abs(maha_auroc - p_maha)
        if knn_sc is None:
            d_knn, oc, kfilt, knn_sc, labels, knn_auroc = detect_knn(train, test, p_knn)
            d_maha, mfilt, maha_sc, maha_labels, maha_auroc = score_maha(train, test, oc, p_maha)
        if d_knn < KNN_GAP_TOL and d_maha < MAHA_TOL:
            parse_cell.cache = (oc, kfilt, mfilt)

        ok = (d_knn < KNN_GAP_TOL) and (d_maha < MAHA_TOL) and np.array_equal(labels, maha_labels)
        flag = "" if ok else "  <-- SELF-CHECK FAILED"

        rk = percentile_ranks(np.asarray(knn_sc, dtype=np.float64))
        rm = percentile_ranks(np.asarray(maha_sc, dtype=np.float64))
        ens_eqw = roc_auc_score(labels, 0.5 * rk + 0.5 * rm)

        # Test-optimal weight: UPPER BOUND only, not a usable result.
        ws = np.linspace(0, 1, 21)
        aurocs = [roc_auc_score(labels, w * rk + (1 - w) * rm) for w in ws]
        j = int(np.argmax(aurocs))
        ens_oracle, w_oracle = aurocs[j], ws[j]

        rows.append(dict(dataset=ds, model=model, seed=seed, n_test=len(test),
                         knn=knn_auroc, p_knn=p_knn, knn_gap=d_knn,
                         maha=maha_auroc, cosine=(p_cos if p_cos is not None else float("nan")),
                         ens_eqw=ens_eqw, d_eqw=ens_eqw - knn_auroc, d_eqw_p=ens_eqw - p_knn,
                         ens_oracle=ens_oracle, w_oracle=w_oracle, d_oracle=ens_oracle - knn_auroc,
                         oc=oc, kfilt=kfilt, mfilt=mfilt, ok=ok))
        print(f"{ds:10s} {model:6s} {seed:7s} "
              f"knn={knn_auroc:.4f}(p={p_knn:.4f}) maha={maha_auroc:.4f} "
              f"ens_eqw={ens_eqw:.4f} Δvsknn{ens_eqw-knn_auroc:+.4f} Δvsp{ens_eqw-p_knn:+.4f} "
              f"oracle{ens_oracle:.4f}@w={w_oracle:.2f}({ens_oracle-knn_auroc:+.4f}){flag}",
              flush=True)

    if not rows:
        print("no rows", flush=True)
        return

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)

    # Aggregates (valid cells only)
    valid = [r for r in rows if r["ok"]]
    print(f"\n===== AGGREGATES (valid cells: {len(valid)}/{len(rows)}) =====", flush=True)

    def summarize(label, subset):
        if not subset:
            return
        d = np.array([r["d_eqw_p"] for r in subset])  # vs persisted (calibrated) knn = the real bar
        do = np.array([r["d_oracle"] for r in subset])
        wins = int((d > 0.002).sum()); ties = int((np.abs(d) <= 0.002).sum()); losses = int((d < -0.002).sum())
        print(f"{label:22s} n={len(subset):2d}  "
              f"mean p_knn={np.mean([r['p_knn'] for r in subset]):.4f}  "
              f"mean ens_eqw={np.mean([r['ens_eqw'] for r in subset]):.4f}  "
              f"meanΔ(ens-pknn)={d.mean():+.4f}  W/T/L={wins}/{ties}/{losses}  "
              f"meanΔoracle(UB)={do.mean():+.4f}", flush=True)

    summarize("ALL", valid)
    summarize("Llama", [r for r in valid if r["model"] == "Llama"])
    summarize("Qwen3", [r for r in valid if r["model"] == "Qwen3"])
    summarize("simpleqa (both)", [r for r in valid if r["dataset"] == "simpleqa"])
    summarize("simpleqa Llama", [r for r in valid if r["dataset"] == "simpleqa" and r["model"] == "Llama"])
    summarize("simpleqa Qwen3", [r for r in valid if r["dataset"] == "simpleqa" and r["model"] == "Qwen3"])


if __name__ == "__main__":
    main()
