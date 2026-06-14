#!/usr/bin/env python
"""Is clr(b5) > act_vit's +0.014 mean significant? Proper paired + hierarchical tests.

Reuses ensemble_analysis's per-sample loaders. For each cell uses the b5
(flipped) contrastive. Reports:
  1. per-cell mean delta (clr - av) + per-SAMPLE DeLong p (paired, median over seeds)
  2. across-cell tests on the 14 paired deltas: one-sample t, Wilcoxon, sign test
  3. linear mixed-effects model auroc ~ method + (1|cell) on seed-level rows
"""
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from scripts.ensemble_analysis import discover, load_pred, orient, auroc, delong_test, AV, CLR

cells = discover("runs", exclude=("mmlu",))
rows = []          # seed-level: (cell, seed, av_auroc, clr_auroc)
percell = []       # (cell, av, clr, delta, delong_p_med, n_seed_clr_sig, n_seed_av_sig)
for nm, seeds in sorted(cells.items()):
    clr_seeds = seeds["clr_b5"] if seeds["clr_b5"] else seeds[CLR]
    common = sorted(set(seeds[AV]) & set(clr_seeds))
    avs, clrs, ps = [], [], []
    clr_sig = av_sig = 0
    for s in common:
        av = load_pred(seeds[AV][s]); clr = load_pred(clr_seeds[s])
        ids = sorted(set(av) & set(clr))
        y = np.array([av[i][1] for i in ids])
        sa = orient(np.array([av[i][0] for i in ids], float), y)
        sc = orient(np.array([clr[i][0] for i in ids], float), y)
        a = auroc(sa, y); c = auroc(sc, y)
        _, _, p = delong_test(sc, sa, y)            # paired, per-sample
        avs.append(a); clrs.append(c); ps.append(p)
        rows.append((nm, s, a, c))
        if p < 0.05 and c > a: clr_sig += 1
        if p < 0.05 and a > c: av_sig += 1
    percell.append((nm, np.mean(avs), np.mean(clrs), np.mean(clrs) - np.mean(avs),
                    float(np.median(ps)), clr_sig, av_sig, len(common)))

print(f"{'cell':24} {'av':>6} {'clr':>6} {'delta':>7} {'delongP':>8} {'clr_sig':>8} {'av_sig':>7}")
for nm, a, c, d, p, cs, as_, n in percell:
    print(f"{nm[:24]:24} {a:6.3f} {c:6.3f} {d:+7.3f} {p:8.3f} {cs:>4}/{n:<3} {as_:>3}/{n:<3}")

deltas = np.array([pc[3] for pc in percell])
print("\n--- across-cell tests on the 14 paired per-cell deltas (clr - av) ---")
print(f"mean delta = {deltas.mean():+.4f}   sd = {deltas.std(ddof=1):.4f}   "
      f"n_cells = {len(deltas)}")
t, pt = stats.ttest_1samp(deltas, 0.0)
print(f"paired t-test (H0: mean delta = 0):  t = {t:+.3f}, p = {pt:.3f}")
try:
    w, pw = stats.wilcoxon(deltas)
    print(f"Wilcoxon signed-rank:                W = {w:.1f}, p = {pw:.3f}")
except Exception as e:
    print(f"Wilcoxon: {e}")
n_pos = int((deltas > 0).sum()); n = len(deltas)
pb = stats.binomtest(n_pos, n, 0.5).pvalue
print(f"sign test: clr wins {n_pos}/{n} cells, binomial p = {pb:.3f}")

# leave-one-out: how much does popqa drive the mean?
for drop in ("popqa", "simpleqa"):
    keep = np.array([pc[3] for pc in percell if drop not in pc[0]])
    print(f"mean delta excluding {drop:9} = {keep.mean():+.4f}  (n={len(keep)})")

print("\n--- linear mixed-effects: auroc ~ method + (1|cell), seed-level rows ---")
try:
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame(
        [(nm, s, a, 0) for nm, s, a, c in rows] + [(nm, s, c, 1) for nm, s, a, c in rows],
        columns=["cell", "seed", "auroc", "is_clr"])
    m = smf.mixedlm("auroc ~ is_clr", df, groups=df["cell"]).fit()
    b = m.params["is_clr"]; se = m.bse["is_clr"]; p = m.pvalues["is_clr"]
    print(f"method effect (clr vs av): {b:+.4f}  SE {se:.4f}  z={b/se:+.2f}  p={p:.3f}")
    print(f"  -> 95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]")
except Exception as e:
    print(f"(statsmodels unavailable: {e})")
