#!/usr/bin/env python
"""Per-sample complementarity + ensemble analysis: contrastive_logprob_recon (clr)
vs act_vit. Eval-only — reads each run's predictions.csv (example_id, score_halu,
label_halu), aligns by example_id per cell, and computes:

  - single-method AUROC (mean +/- std over seeds)
  - Spearman correlation of the two methods' per-sample scores
  - cross-method ensemble (z-avg and rank-avg) AUROC vs both methods, with a
    DeLong test for AUROC(ensemble) > AUROC(best single) per seed
  - same-family control: act_vit+act_vit and clr+clr ensembles (different seeds),
    to show cross-method gain exceeds generic ensembling.

mmlu is excluded (per project scope).
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

CLR = "contrastive_logprob_recon"
AV = "act_vit"


# ---------- AUROC + fast DeLong (Sun & Xu) for correlated ROCs ----------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, m):
    # preds_sorted_transposed: (k, n) with positives first (m of them)
    n = preds_sorted_transposed.shape[1] - m
    k = preds_sorted_transposed.shape[0]
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(scores_a, scores_b, labels):
    """Two-sided p-value for AUROC(a) == AUROC(b) on the same samples.
    Returns (auc_a, auc_b, p)."""
    labels = np.asarray(labels)
    order = (-labels).argsort(kind="mergesort")  # positives (1) first
    label_1_count = int(labels.sum())
    preds = np.vstack((np.asarray(scores_a)[order], np.asarray(scores_b)[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def auroc(scores, labels):
    a, _, _ = delong_test(scores, scores, labels)
    return a


def orient(scores, labels):
    """Flip so higher score => hallucination (label 1)."""
    return scores if auroc(scores, labels) >= 0.5 else -scores


def zavg(a, b):
    za = (a - a.mean()) / (a.std() + 1e-9)
    zb = (b - b.mean()) / (b.std() + 1e-9)
    return za + zb


def ravg(a, b):
    ra = stats.rankdata(a) / len(a)
    rb = stats.rankdata(b) / len(b)
    return ra + rb


def stack_oof(a, b, y, folds=5):
    """Leakage-free learned ensemble: out-of-fold logistic-regression stack of the
    two methods' scores. Returns OOF predicted prob (used as the ensemble score)."""
    X = np.column_stack([stats.zscore(a), stats.zscore(b)])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


# ---------- data loading ----------
def load_pred(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if "split" in r and r["split"] and r["split"] != "test":
                continue
            out[int(r["example_id"])] = (float(r["score_halu"]), int(r["label_halu"]))
    return out


def cell_name(bench_dir):
    # runs/baseline_comparison_<name>/<name>/...
    nm = os.path.basename(bench_dir).replace("baseline_comparison_", "")
    return nm


def discover(root):
    cells = {}
    for bench in sorted(glob.glob(os.path.join(root, "baseline_comparison_*memmap*"))):
        nm = cell_name(bench)
        if "mmlu" in nm:
            continue
        inner = glob.glob(os.path.join(bench, "*"))
        inner = [d for d in inner if os.path.isdir(d)]
        if not inner:
            continue
        base = inner[0]
        seeds = {AV: {}, CLR: {}}
        for meth in (AV, CLR):
            for sd in sorted(glob.glob(os.path.join(base, meth, "seed_*"))):
                p = os.path.join(sd, "predictions.csv")
                if os.path.exists(p):
                    m = re.search(r"seed_(\d+)", sd)
                    seeds[meth][int(m.group(1))] = p
        if seeds[AV] and seeds[CLR]:
            cells[nm] = seeds
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    a = ap.parse_args()

    cells = discover(a.root)
    print(f"# cells with both methods (mmlu excluded): {len(cells)}")
    hdr = ("cell", "av", "clr", "spear", "ens_z", "stack", "stk_gain", "p_med", "sig/n",
           "ctl_av", "ctl_clr")
    print("{:24} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>6} {:>6} {:>6} {:>6}".format(*hdr))

    agg = defaultdict(list)
    for nm, seeds in cells.items():
        common_seeds = sorted(set(seeds[AV]) & set(seeds[CLR]))
        av_aucs, clr_aucs, ens_z_aucs, stk_aucs, spears, dgains, pvals = ([] for _ in range(7))
        for s in common_seeds:
            av = load_pred(seeds[AV][s]); clr = load_pred(seeds[CLR][s])
            ids = sorted(set(av) & set(clr))
            y = np.array([av[i][1] for i in ids])
            sa = orient(np.array([av[i][0] for i in ids], float), y)
            sc = orient(np.array([clr[i][0] for i in ids], float), y)
            a_auc = auroc(sa, y); c_auc = auroc(sc, y)
            ez = zavg(sa, sc); st = stack_oof(sa, sc, y)
            ez_auc = auroc(ez, y); st_auc = auroc(st, y)
            best = sa if a_auc >= c_auc else sc
            # DeLong: does the learned stack beat the best single method?
            _, _, p = delong_test(st, best, y)
            av_aucs.append(a_auc); clr_aucs.append(c_auc)
            ens_z_aucs.append(ez_auc); stk_aucs.append(st_auc)
            spears.append(stats.spearmanr(sa, sc).statistic)
            dgains.append(st_auc - max(a_auc, c_auc)); pvals.append(p)

        # same-family control: ensemble distinct seed pairs within a method
        def ctl(method):
            ss = sorted(seeds[method]); gains = []
            preds = {s: load_pred(seeds[method][s]) for s in ss}
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    ids = sorted(set(preds[ss[i]]) & set(preds[ss[j]]))
                    y = np.array([preds[ss[i]][k][1] for k in ids])
                    s1 = orient(np.array([preds[ss[i]][k][0] for k in ids], float), y)
                    s2 = orient(np.array([preds[ss[j]][k][0] for k in ids], float), y)
                    base = max(auroc(s1, y), auroc(s2, y))
                    gains.append(auroc(zavg(s1, s2), y) - base)
            return float(np.mean(gains)) if gains else float("nan")

        ctl_av = ctl(AV); ctl_clr = ctl(CLR)
        n = len(common_seeds)
        sig = sum(1 for k in range(n) if pvals[k] < 0.05 and dgains[k] > 0)
        row = (nm[:24], np.mean(av_aucs), np.mean(clr_aucs), np.mean(spears),
               np.mean(ens_z_aucs), np.mean(stk_aucs), np.mean(dgains),
               float(np.median(pvals)), f"{sig}/{n}", ctl_av, ctl_clr)
        print("{:24} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:+8.3f} {:6.3f} {:>6} {:+6.3f} {:+6.3f}".format(*row))
        agg["av"].append(np.mean(av_aucs)); agg["clr"].append(np.mean(clr_aucs))
        agg["ens_z"].append(np.mean(ens_z_aucs)); agg["stk"].append(np.mean(stk_aucs))
        agg["spear"].append(np.mean(spears))
        agg["dgain"].append(np.mean(dgains)); agg["ctl_av"].append(ctl_av); agg["ctl_clr"].append(ctl_clr)
        agg["sig"].append(sig); agg["n"].append(n)

    print("-" * 100)
    print(f"# MEAN over {len(cells)} cells:")
    print(f"#   act_vit={np.mean(agg['av']):.3f}  clr={np.mean(agg['clr']):.3f}  "
          f"ensemble_z={np.mean(agg['ens_z']):.3f}  stack={np.mean(agg['stk']):.3f}")
    print(f"#   STACK gain over best-single (per cell) = {np.mean(agg['dgain']):+.3f}  "
          f"(spearman corr = {np.mean(agg['spear']):.3f})")
    print(f"#   same-family control gain (z-avg): av+av={np.nanmean(agg['ctl_av']):+.3f}  "
          f"clr+clr={np.nanmean(agg['ctl_clr']):+.3f}")
    print(f"#   cells where stack significantly (DeLong p<.05, majority seeds) beats best single: "
          f"{sum(1 for i in range(len(agg['sig'])) if agg['sig'][i] > agg['n'][i]/2)}/{len(cells)}")


if __name__ == "__main__":
    main()
