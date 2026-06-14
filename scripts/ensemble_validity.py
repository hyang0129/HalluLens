#!/usr/bin/env python
"""Ensemble validity test (issue #137): is the cross-method (av+clr) ensemble benefit
genuine complementary signal, or just generic same-method ensembling (variance reduction)?

Eval-only — reuses existing predictions.csv via ensemble_analysis loaders. Uses the b5
(flipped) contrastive. mmlu excluded.

Two tests:
  T1 (matched size-2): mean z-avg AUROC over all 2-member ensembles of each type:
       av+av (C(5,2)), clr+clr (C(5,2)), av+clr (same-seed pairs).
  T2 (saturation): k-seed same-method ensemble mean AUROC for k=1..5 (av and clr);
       the k=5 value is the variance-reduction *asymptote*. The cross-method size-2
       ensemble must beat that asymptote (not just the k=1 baseline) for Leg B to hold.

Same-method ensembling across seeds is computed on the example_id INTERSECTION of the
combined seeds (their train/test splits differ); n_eval is reported so shrinkage is visible.
Cross-method same-seed pairs share a split -> full overlap.
"""
import itertools
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, ".")
from scripts.ensemble_analysis import discover, load_pred, orient, auroc, stack_oof, delong_test, AV, CLR

TIE_CELLS = {"hotpotqa", "nq", "searchqa", "triviaqa"}  # where Leg B lives


def ens_auroc(maps):
    """z-avg ensemble AUROC over the id-intersection of the given prediction maps."""
    ids = set(maps[0])
    for m in maps[1:]:
        ids &= set(m)
    ids = sorted(ids)
    if len(ids) < 50:
        return np.nan, len(ids)
    y = np.array([maps[0][i][1] for i in ids])
    Z = np.zeros(len(ids))
    for m in maps:
        s = orient(np.array([m[i][0] for i in ids], float), y)
        Z += (s - s.mean()) / (s.std() + 1e-9)
    return auroc(Z, y), len(ids)


def mean_kcombo(mp, k):
    """mean z-avg AUROC over all C(n,k) k-subsets of one method's seeds."""
    ss = sorted(mp)
    if len(ss) < k:
        return np.nan
    vals = [ens_auroc([mp[s] for s in combo])[0] for combo in itertools.combinations(ss, k)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def main():
    cells = discover("runs", exclude=("mmlu",))
    print(f"# cells: {len(cells)}  (mmlu excluded; clr=b5)")
    hdr = ("cell", "av1", "av2", "av5", "clr1", "clr2", "clr5", "cross2",
           "vs_match", "vs_asymp")
    print("{:20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}".format(*hdr))

    agg = {k: [] for k in ("av1", "av2", "av5", "clr1", "clr2", "clr5",
                           "cross2", "vs_match", "vs_asymp")}
    tie = {"vs_match": [], "vs_asymp": []}
    for nm, seeds in sorted(cells.items()):
        clrseeds = seeds["clr_b5"] if seeds["clr_b5"] else seeds[CLR]
        av = {s: load_pred(p) for s, p in seeds[AV].items()}
        clr = {s: load_pred(p) for s, p in clrseeds.items()}
        avk = {k: mean_kcombo(av, k) for k in (1, 2, 5)}
        clrk = {k: mean_kcombo(clr, k) for k in (1, 2, 5)}
        common = sorted(set(seeds[AV]) & set(clrseeds))
        cross2 = float(np.mean([ens_auroc([av[s], clr[s]])[0] for s in common]))
        # learned-stack cross (the paper's ensemble): per-seed OOF logistic, averaged
        cs = []
        for s in common:
            ids = sorted(set(av[s]) & set(clr[s]))
            y = np.array([av[s][i][1] for i in ids])
            a = orient(np.array([av[s][i][0] for i in ids], float), y)
            c = orient(np.array([clr[s][i][0] for i in ids], float), y)
            cs.append(auroc(stack_oof(a, c, y), y))
        cross_stk = float(np.mean(cs))
        vs_match = cross2 - max(avk[2], clrk[2])      # cross vs best same-method size-2
        vs_asymp = cross2 - max(avk[5], clrk[5])      # cross z-avg vs same-method asymptote
        vs_asymp_stk = cross_stk - max(avk[5], clrk[5])  # cross STACK vs same-method asymptote
        agg.setdefault("cross_stk", []).append(cross_stk)
        agg.setdefault("vs_asymp_stk", []).append(vs_asymp_stk)
        base0 = nm.replace("_qwen3", "").replace("_memmap", "")
        if base0 in TIE_CELLS:
            tie.setdefault("vs_asymp_stk", []).append(vs_asymp_stk)
        row = (nm[:20], avk[1], avk[2], avk[5], clrk[1], clrk[2], clrk[5],
               cross2, vs_match, vs_asymp)
        print("{:20} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:+8.3f} {:+8.3f}".format(*row))
        for key, val in zip(agg, (avk[1], avk[2], avk[5], clrk[1], clrk[2], clrk[5],
                                  cross2, vs_match, vs_asymp)):
            agg[key].append(val)
        base = nm.replace("_qwen3", "").replace("_memmap", "")
        if base in TIE_CELLS:
            tie["vs_match"].append(vs_match)
            tie["vs_asymp"].append(vs_asymp)

    print("-" * 96)
    print("# MEAN over all cells:")
    print(f"#   same-method asymptote (k=5): av={np.nanmean(agg['av5']):.3f}  clr={np.nanmean(agg['clr5']):.3f}")
    print(f"#   cross size-2 (av+clr) = {np.nanmean(agg['cross2']):.3f}")
    print(f"#   cross2 - best same-method SIZE-2  = {np.nanmean(agg['vs_match']):+.4f}")
    print(f"#   cross2 (z-avg) - best same-method ASYMPTOTE(k5) = {np.nanmean(agg['vs_asymp']):+.4f}")
    print(f"#   cross STACK = {np.nanmean(agg['cross_stk']):.3f}")
    print(f"#   cross STACK - best same-method ASYMPTOTE(k5) = {np.nanmean(agg['vs_asymp_stk']):+.4f}   <-- the load-bearing number")
    print(f"# MEAN over TIE cells ({len(tie['vs_match'])}): "
          f"vs_match(z)={np.nanmean(tie['vs_match']):+.4f}  vs_asymp(z)={np.nanmean(tie['vs_asymp']):+.4f}  "
          f"vs_asymp(stack)={np.nanmean(tie['vs_asymp_stk']):+.4f}")
    n_pos = sum(1 for v in agg['vs_asymp'] if v > 0)
    n_pos_s = sum(1 for v in agg['vs_asymp_stk'] if v > 0)
    print(f"# cells where cross > same-method asymptote: z-avg {n_pos}/{len(agg['vs_asymp'])}  |  stack {n_pos_s}/{len(agg['vs_asymp_stk'])}")
    print("\n# Interpretation: vs_asymp > 0 => cross-method beats what MORE same-method seeds")
    print("#   could ever give (variance-reduction ceiling) => genuine complementary signal.")
    print("#   vs_asymp <= 0 => Leg B is just ensembling; revisit framing (epic #136).")


if __name__ == "__main__":
    main()
