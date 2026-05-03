"""KNN hyperparameter sweep over contrastive embedding dumps.

Reads {dump_dir}/{train,val,test}.npz produced by dump_contrastive_embeddings.py,
sweeps over layer aggregation strategies, layer subsets, reference sets, k values,
and distance metrics, writes a long-form CSV plus a markdown report.

Usage:
    python scripts/run_knn_sweep.py \
        --dump-dir shared/knn_eval_dumps/qwen3_nq_seed0 \
        --output-dir results/knn_sweep_qwen3_nq \
        --tag qwen3_nq
"""

import argparse
import json
import os
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dump(dump_dir):
    splits = {}
    for name in ("train", "val", "test"):
        path = os.path.join(dump_dir, f"{name}.npz")
        d = np.load(path, allow_pickle=True)
        splits[name] = {
            "z": d["z_per_layer"].astype(np.float32),   # (N, L, D)
            "halu": d["halu"].astype(np.int32),          # (N,)
            "prompt_hash": d["prompt_hash"],              # (N,)
            "layer_indices": d["layer_indices"],          # (L,)
        }
    with open(os.path.join(dump_dir, "meta.json")) as f:
        meta = json.load(f)
    return splits, meta


def check_disjointness(splits):
    """Return dict of overlap counts between split pairs."""
    overlap = {}
    for a, b in combinations(splits.keys(), 2):
        ha = set(splits[a]["prompt_hash"].tolist())
        hb = set(splits[b]["prompt_hash"].tolist())
        overlap[f"{a}/{b}"] = len(ha & hb)
    return overlap


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def _l2_normalize(x):
    """L2-normalize along last axis."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def aggregate(z, strategy, layer_indices):
    """Collapse (N, L, D) → (N, D') using the given strategy and layer subset.

    layer_indices: list of int indices into the L dimension (not model layer IDs).
    """
    sub = z[:, layer_indices, :]                  # (N, n, D)
    if strategy == "mean":
        return sub.mean(axis=1)                   # (N, D)
    if strategy == "concat":
        return sub.reshape(len(z), -1)            # (N, n*D)
    if strategy == "concat_normalized":
        sub_n = _l2_normalize(sub)
        return sub_n.reshape(len(z), -1)          # (N, n*D)
    if strategy == "single":
        assert len(layer_indices) == 1
        return sub[:, 0, :]                       # (N, D)
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Subset enumeration
# ---------------------------------------------------------------------------

def build_subsets(L, layer_ids):
    """Return a list of (subset_name, [idx_into_L]) entries."""
    subsets = []

    # All single layers
    for i in range(L):
        subsets.append((f"single_l{layer_ids[i]}", [i], "single"))

    # Size sweep: {1, 2, 4, 8, 16}
    for n in [1, 2, 4, 8, 16]:
        if n > L:
            continue
        # Contiguous head
        idxs_head = list(range(n))
        layers_head = [layer_ids[i] for i in idxs_head]
        name_head = f"head{n}_l{layers_head[0]}-{layers_head[-1]}"
        # Contiguous tail
        idxs_tail = list(range(L - n, L))
        layers_tail = [layer_ids[i] for i in idxs_tail]
        name_tail = f"tail{n}_l{layers_tail[0]}-{layers_tail[-1]}"
        # Evenly spaced
        idxs_even = sorted(set(np.round(np.linspace(0, L - 1, n)).astype(int).tolist()))
        while len(idxs_even) < n:
            idxs_even = sorted(set(idxs_even + [idxs_even[-1] + 1]))
        idxs_even = idxs_even[:n]
        layers_even = [layer_ids[i] for i in idxs_even]
        name_even = f"even{n}_l{layers_even[0]}-{layers_even[-1]}"

        for strat in ["mean", "concat", "concat_normalized"]:
            if n == 1 and strat != "mean":
                continue  # single-layer concat == single-layer mean
            subsets.append((f"{strat}_{name_head}", idxs_head, strat))
            subsets.append((f"{strat}_{name_tail}", idxs_tail, strat))
            if idxs_even not in [idxs_head, idxs_tail]:
                subsets.append((f"{strat}_{name_even}", idxs_even, strat))

        # Per-layer distance/auroc mean strategies
        if n > 1:
            for strat in ["per_layer_dist_mean", "per_layer_auroc_mean"]:
                subsets.append((f"{strat}_{name_even}", idxs_even, strat))

    # Original eval-layer pair [22, 26] if present
    if 22 in layer_ids and 26 in layer_ids:
        i22 = layer_ids.index(22)
        i26 = layer_ids.index(26)
        for strat in ["mean", "concat", "concat_normalized"]:
            subsets.append((f"{strat}_original_22_26", [i22, i26], strat))

    return subsets


# ---------------------------------------------------------------------------
# KNN scoring
# ---------------------------------------------------------------------------

MAX_K = 127
K_VALUES = [1, 3, 5, 9, 15, 31, 63, 127]


def safe_auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def knn_sweep_one(ref_z, test_z, test_labels, metric):
    """Build one KNN index and sweep over all k values.

    Returns dict: k -> auroc
    """
    n_neighbors = min(MAX_K, len(ref_z))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    nn.fit(ref_z)
    dists, _ = nn.kneighbors(test_z)           # (N_test, n_neighbors)
    results = {}
    for k in K_VALUES:
        kk = min(k, n_neighbors)
        scores = dists[:, :kk].mean(axis=1)
        results[kk] = safe_auroc(test_labels, scores)
    return results


def per_layer_sweep(ref_z_full, test_z_full, test_labels, layer_indices, metric, mode):
    """Average KNN distances or AUROCs across layers.

    ref_z_full: (N_ref, L, D)
    test_z_full: (N_test, L, D)
    mode: 'per_layer_dist_mean' or 'per_layer_auroc_mean'
    """
    per_layer_dists = []
    per_layer_aurocs = defaultdict(list)

    for li in layer_indices:
        ref = ref_z_full[:, li, :]
        tst = test_z_full[:, li, :]
        n_neighbors = min(MAX_K, len(ref))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        nn.fit(ref)
        dists, _ = nn.kneighbors(tst)            # (N_test, n_neighbors)
        per_layer_dists.append(dists)            # for dist_mean
        for k in K_VALUES:
            kk = min(k, n_neighbors)
            scores = dists[:, :kk].mean(axis=1)
            per_layer_aurocs[kk].append(safe_auroc(test_labels, scores))

    results = {}
    if mode == "per_layer_dist_mean":
        stacked = np.mean(per_layer_dists, axis=0)  # (N_test, n_neighbors)
        for k in K_VALUES:
            kk = min(k, min(MAX_K, len(ref_z_full)))
            scores = stacked[:, :kk].mean(axis=1)
            results[kk] = safe_auroc(test_labels, scores)
    else:  # per_layer_auroc_mean
        for kk, auroc_list in per_layer_aurocs.items():
            results[kk] = float(np.nanmean(auroc_list))
    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(splits, meta):
    layer_ids = list(meta["layers"])   # e.g. [14,15,...,29]
    L = len(layer_ids)

    train_z = splits["train"]["z"]     # (N_tr, L, D)
    val_z = splits["val"]["z"]
    test_z = splits["test"]["z"]
    train_labels = splits["train"]["halu"]
    val_labels = splits["val"]["halu"]
    test_labels = splits["test"]["halu"]

    train_val_z = np.concatenate([train_z, val_z], axis=0)
    train_val_labels = np.concatenate([train_labels, val_labels])

    ref_sets = {
        "train": (train_z, train_labels),
        "val": (val_z, val_labels),
        "train_val": (train_val_z, train_val_labels),
    }

    subsets = build_subsets(L, layer_ids)
    distances = ["euclidean", "cosine"]

    rows = []
    total = len(subsets) * len(ref_sets) * len(distances)
    done = 0
    t0 = time.time()

    for (subset_name, layer_indices, strat) in subsets:
        is_per_layer = strat.startswith("per_layer")

        for ref_name, (ref_z_full, _) in ref_sets.items():
            for metric in distances:
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(
                    f"[{done}/{total}] {subset_name} ref={ref_name} metric={metric}  "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

                if is_per_layer:
                    k_aurocs = per_layer_sweep(
                        ref_z_full, test_z, test_labels, layer_indices, metric, strat
                    )
                else:
                    ref_agg = aggregate(ref_z_full, strat, layer_indices)
                    test_agg = aggregate(test_z, strat, layer_indices)
                    k_aurocs = knn_sweep_one(ref_agg, test_agg, test_labels, metric)

                for k, auroc in k_aurocs.items():
                    rows.append(
                        {
                            "subset_name": subset_name,
                            "strategy": strat,
                            "n_layers": len(layer_indices),
                            "layer_indices": str(layer_indices),
                            "layer_ids": str([layer_ids[i] for i in layer_indices]),
                            "ref_set": ref_name,
                            "metric": metric,
                            "k": k,
                            "auroc": auroc,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def best_row(df, group_cols, value_col="auroc"):
    return df.loc[df.groupby(group_cols)[value_col].idxmax()]


def fmt(x):
    return f"{x:.4f}" if not np.isnan(x) else "N/A"


def generate_report(df, splits, meta, overlap, tag, elapsed_s):
    lines = []
    a = lines.append

    a(f"# KNN Sweep Report — {tag}")
    a(f"\n**Generated:** {pd.Timestamp.now().isoformat(timespec='seconds')}")
    a(f"**Dump dir:** `{meta.get('run_dir', '?')}`")
    a(f"**Elapsed:** {elapsed_s:.0f}s\n")

    # Dataset stats
    a("## Dataset")
    a("| Split | N | Halu | Clean | Halu rate |")
    a("|-------|---|------|-------|-----------|")
    for name, sp in splits.items():
        n = len(sp["halu"])
        nh = sp["halu"].sum()
        nc = n - nh
        a(f"| {name} | {n} | {nh} | {nc} | {nh/n:.1%} |")
    a("")

    a("### Split overlap (prompt_hash)")
    for pair, count in overlap.items():
        status = "OK" if count == 0 else f"**WARNING: {count} shared hashes**"
        a(f"- {pair}: {status}")
    a("")

    a(f"**Layers dumped:** {meta['layers']}")
    a(f"**Original eval layers:** 22, 26")
    a("")

    # Overall best
    best = df.loc[df["auroc"].idxmax()]
    a("## Best overall")
    a(f"- **AUROC:** {fmt(best['auroc'])}")
    a(f"- Strategy: `{best['strategy']}`  Subset: `{best['subset_name']}`")
    a(f"- Reference set: `{best['ref_set']}`  k={best['k']}  metric=`{best['metric']}`")
    a("")

    # Original baseline for comparison
    orig = df[
        (df["subset_name"].str.contains("original_22_26"))
        & (df["strategy"] == "mean")
        & (df["ref_set"] == "train")
        & (df["metric"] == "euclidean")
    ]
    if len(orig):
        best_orig = orig.loc[orig["auroc"].idxmax()]
        a(f"**Original config** (mean [22,26], train ref, euclidean):")
        a(f"- Best AUROC: {fmt(best_orig['auroc'])} at k={best_orig['k']}")
        a("")

    # Best per strategy
    a("## Best AUROC by aggregation strategy")
    a("| Strategy | Best AUROC | Subset | Ref | k | Metric |")
    a("|----------|-----------|--------|-----|---|--------|")
    for strat, grp in df.groupby("strategy"):
        r = grp.loc[grp["auroc"].idxmax()]
        a(f"| {strat} | {fmt(r['auroc'])} | {r['subset_name']} | {r['ref_set']} | {r['k']} | {r['metric']} |")
    a("")

    # Single-layer AUROC table (best k, ref=val, metric=euclidean)
    a("## Single-layer AUROC (best k, val ref, euclidean)")
    single = df[(df["strategy"] == "single") & (df["ref_set"] == "val") & (df["metric"] == "euclidean")]
    if len(single):
        layer_best = single.groupby("subset_name")["auroc"].max().reset_index()
        layer_best["layer_id"] = layer_best["subset_name"].str.replace("single_l", "").astype(int)
        layer_best = layer_best.sort_values("layer_id")
        a("| Layer | Best AUROC |")
        a("|-------|-----------|")
        for _, row in layer_best.iterrows():
            marker = " ← original" if row["layer_id"] in [22, 26] else ""
            a(f"| {int(row['layer_id'])} | {fmt(row['auroc'])}{marker} |")
    a("")

    # Effect of layer count (mean strategy, evenly-spaced, val ref, euclidean)
    a("## Effect of layer count (mean, evenly-spaced, val ref, euclidean)")
    count_df = df[
        (df["strategy"] == "mean")
        & (df["subset_name"].str.startswith("mean_even"))
        & (df["ref_set"] == "val")
        & (df["metric"] == "euclidean")
    ]
    if len(count_df):
        cnt_best = count_df.groupby("n_layers")["auroc"].max().reset_index().sort_values("n_layers")
        a("| N layers | Best AUROC |")
        a("|---------|-----------|")
        for _, row in cnt_best.iterrows():
            a(f"| {int(row['n_layers'])} | {fmt(row['auroc'])} |")
    a("")

    # Reference set effect (best aggregation, best k, euclidean)
    a("## Reference set effect (best strategy, euclidean, best k)")
    best_strat = df.loc[df["auroc"].idxmax()]["strategy"]
    ref_df = df[(df["strategy"] == best_strat) & (df["metric"] == "euclidean")]
    ref_comp = ref_df.groupby("ref_set")["auroc"].max().reset_index()
    a("| Ref set | Best AUROC |")
    a("|---------|-----------|")
    for _, row in ref_comp.sort_values("auroc", ascending=False).iterrows():
        a(f"| {row['ref_set']} | {fmt(row['auroc'])} |")
    a("")

    # Train vs val memorization check (original config)
    a("## Memorization check: train vs val reference (mean [22,26], euclidean)")
    orig_both = df[
        df["subset_name"].str.contains("original_22_26")
        & (df["strategy"] == "mean")
        & (df["metric"] == "euclidean")
    ]
    if len(orig_both):
        mem = orig_both.groupby("ref_set")["auroc"].max().reset_index()
        a("| Ref set | Best AUROC |")
        a("|---------|-----------|")
        for _, row in mem.sort_values("auroc", ascending=False).iterrows():
            a(f"| {row['ref_set']} | {fmt(row['auroc'])} |")
        train_auroc = mem[mem["ref_set"] == "train"]["auroc"].values
        val_auroc = mem[mem["ref_set"] == "val"]["auroc"].values
        if len(train_auroc) and len(val_auroc):
            diff = float(train_auroc[0]) - float(val_auroc[0])
            a(f"\nTrain − val AUROC gap: **{diff:+.4f}**")
            if abs(diff) > 0.01:
                a(f"→ {'Positive gap: train reference inflates AUROC (memorization bias)' if diff > 0 else 'Negative gap: val reference outperforms train (unexpected)'}")
    a("")

    # k sensitivity
    a("## k sensitivity (best strategy, val ref, euclidean)")
    best_sub = df.loc[df["auroc"].idxmax()]["subset_name"]
    best_strat2 = df.loc[df["auroc"].idxmax()]["strategy"]
    k_df = df[
        (df["subset_name"] == best_sub)
        & (df["strategy"] == best_strat2)
        & (df["ref_set"] == "val")
        & (df["metric"] == "euclidean")
    ].sort_values("k")
    if len(k_df):
        a(f"Subset: `{best_sub}`")
        a("| k | AUROC |")
        a("|---|-------|")
        for _, row in k_df.iterrows():
            a(f"| {int(row['k'])} | {fmt(row['auroc'])} |")
    a("")

    # Distance metric comparison
    a("## Distance metric comparison (best strategy, val ref, best k)")
    metric_df = df[
        (df["strategy"] == best_strat2)
        & (df["ref_set"] == "val")
    ]
    metric_comp = metric_df.groupby("metric")["auroc"].max().reset_index()
    a("| Metric | Best AUROC |")
    a("|--------|-----------|")
    for _, row in metric_comp.sort_values("auroc", ascending=False).iterrows():
        a(f"| {row['metric']} | {fmt(row['auroc'])} |")
    a("")

    # Recommendations
    a("## Recommendations")
    best_all = df.loc[df["auroc"].idxmax()]
    a(f"1. **Best layer subset:** `{best_all['subset_name']}` ({best_all['n_layers']} layers)")
    a(f"2. **Best aggregation:** `{best_all['strategy']}`")
    a(f"3. **Best reference set:** `{best_all['ref_set']}`")
    a(f"4. **Best k:** {best_all['k']}")
    a(f"5. **Best distance metric:** `{best_all['metric']}`")
    a(f"6. **Best AUROC:** {fmt(best_all['auroc'])}")
    a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default=None,
                        help="Short label for CSV/report filenames (default: basename of dump-dir)")
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.dump_dir.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dumps from {args.dump_dir} ...", flush=True)
    splits, meta = load_dump(args.dump_dir)

    overlap = check_disjointness(splits)
    print("Split overlap:", overlap, flush=True)

    t0 = time.time()
    df = run_sweep(splits, meta)
    elapsed = time.time() - t0

    csv_path = os.path.join(args.output_dir, f"{tag}_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}  ({len(df)} rows)", flush=True)

    report = generate_report(df, splits, meta, overlap, tag, elapsed)
    md_path = os.path.join(args.output_dir, f"{tag}_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Saved report: {md_path}", flush=True)


if __name__ == "__main__":
    main()
