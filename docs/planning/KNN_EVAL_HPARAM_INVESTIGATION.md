# KNN Evaluation Hyperparameter Investigation

**Status:** planning
**Model:** Qwen3-8B
**Starting dataset:** NQ (natural_questions)

## Motivation

The current KNN OOD baseline collapses each sample's contrastive embeddings into a single vector by **mean-pooling across `--eval-layers` (default `[22, 26]`)** before running KNN. This bakes in three undocumented choices:

1. **Layer aggregation = mean of 2 layers** — discards layer-specific signal; downstream KNN sees one `(D,)` vector per sample.
2. **Reference index = the encoder's *training* split** (ID-only). Test samples are scored against examples the encoder has already optimized over, so distances likely understate true OOD difficulty.
3. **`k` is calibrated only when `--calibrate-k` is set**; otherwise it's a fixed default.

Before adding new methods (multilayer KNN, stacked-embedding KNN), we want to characterize the existing knob's sensitivity and resolve the train/val confound. The right way to do this cheaply is to **run GPU inference once**, dump every embedding we'd ever want, and explore knobs offline in numpy/sklearn.

## Open questions to answer

1. **Layer count:** how does KNN AUROC scale as we mean-pool 1, 2, 4, 8, 16 layers? Is `[22, 26]` near optimal or arbitrary?
2. **Layer choice at fixed count:** at K=2 layers, does the choice of which two layers matter more than the number?
3. **Aggregation across layers:** mean vs. concat vs. per-layer-then-average-AUROC vs. per-layer-then-average-distance. Which wins?
4. **Reference set:** does scoring against a held-out *val* split (encoder never saw) give different AUROC than scoring against *train*? How big is the optimism bias?
5. **`k`:** sensitivity sweep over `k ∈ {1, 3, 5, 9, 15, 31, 63, 127}`. Is the current default near optimal?
6. **Distance metric:** L2 vs. cosine vs. dot. (Embeddings are not L2-normalized by default; this matters.)
7. **Reference set size:** does KNN saturate, or is more data still helping at our current scale?
8. **Per-layer normalization before concat:** does L2-normalizing each layer's embedding before stacking change the picture?

## Plan of work

### Phase 1 — One-shot GPU inference dump

**Goal:** run the trained Qwen3 contrastive model on NQ once, save every per-layer embedding for train + val + test. After this, all sweeps are CPU/numpy.

**New script:** `scripts/dump_contrastive_embeddings.py`

Inputs:
- `--checkpoint` — trained contrastive model weights
- `--inference-json`, `--eval-json`, `--activations-path` — same as training
- `--layers` — full layer range to dump (e.g. `14-29`, all 16 layers in our typical training band)
- `--splits train,val,test` — emit all three (val = held-out portion of the train split, never seen by the encoder)
- `--output-dir shared/knn_eval_dumps/qwen3_nq/`

For each split, write:
```
{output_dir}/{split}.npz
  z_per_layer    : (N, L, D)  float16   — encoder output at each layer in --layers
  halu           : (N,)       int8      — hallucination label
  prompt_hash    : (N,)       <U32      — for joining back to source data
  layer_indices  : (L,)       int32     — which model layers these correspond to
  meta.json                              — checkpoint id, encoder config, eval-layers used at training time
```

Notes:
- Run encoder **once per layer per sample** (not the current "sample 2 of N views" path). We want every layer, not random subsets.
- L2-normalize per-layer embedding before saving? **No** — save raw, normalize at analysis time. Cheaper to redo.
- float16 is enough for distance comparisons; halves disk.
- **Train/val split:** verified the actual experiment uses `split_strategy="two_way"`
  (`baseline_comparison_nq_qwen3.json` does not override the default in
  `run_experiment.py`). Under two_way the train zarr is partitioned into 80% `train`
  (encoder trained on these) and 20% `test` (unused by the actual run — the trainer
  uses the held-out test zarr as `val_dataset`). The dump script maps that unused
  20% partition → our `val.npz` so it functions as a clean reference set the encoder
  never saw. The separate test zarr (loaded with `split_strategy="none"`) → `test.npz`.
  Three_way is supported in the script but not used here, since its `val` slice
  would lie *inside* the encoder's training set and defeat the purpose.

### Phase 2 — CPU analysis notebook

**New notebook:** `notebooks/knn_eval_sweep.ipynb` (master) → working copy at repo root.

Loads the three `.npz` files. All sweeps below are sklearn / numpy / faiss (CPU is fine at our scale).

Sweep matrix:
- **Layer aggregation strategies** (applied to per-layer embeddings):
  - `single(layer_i)` — pick one layer
  - `mean(subset)` — current behavior, parameterized by subset
  - `concat(subset)` — stacked-embedding KNN
  - `concat_normalized(subset)` — L2-normalize each layer first, then concat
  - `per_layer_distance_mean(subset)` — KNN per layer, average distances
  - `per_layer_auroc_mean(subset)` — KNN per layer, average AUROCs (sanity baseline)
- **Layer subsets:** all single layers; all pairs at fixed budget; size sweep `{1, 2, 4, 8, 16}` using both contiguous and evenly-spaced choices
- **Reference set:** train-only (current), val-only (clean), train+val combined
- **k:** `{1, 3, 5, 9, 15, 31, 63, 127}`
- **Distance:** `{l2, cosine}`

Output: a long-form CSV `results/knn_sweep_qwen3_nq.csv` with one row per `(aggregation, subset, reference_set, k, distance)` → AUROC. Plus a few summary plots.

### Phase 3 — Interpretation + decision

After the sweep, decide:
- Should the default `--eval-layers` change?
- Should KNN switch from mean-pool to one of the alternatives by default?
- Should the KNN reference set move to val (or train+val) to remove the encoder-memorization confound?
- Is a `knn_ood_stats_multilayer` function worth landing, and which mode(s) should it support?

## Scope guardrails

- **Qwen3 + NQ only for now.** Once we know which knobs matter, we re-run the dump script for Llama3 and other datasets without re-doing the analysis design.
- **No changes to training code in this investigation.** Only adds a dump script and an analysis notebook.
- **No new metric in `metrics.py` until Phase 3 conclusions are in.** We don't want to land a `knn_ood_stats_multilayer` API based on guesswork.

## Risks / things to watch

- **Train/test contamination:** NQ has known overlap issues across splits in some preprocessing pipelines. Verify split disjointness by `prompt_hash` before trusting numbers.
- **Encoder memorization of train:** this is the whole point of adding the val reference set. If train-vs-val AUROC differ a lot, the current eval has been over-reporting.
- **Float16 loss in distance computation:** convert to float32 inside the sweep before computing distances. Store as fp16 to save disk.
- **Disk:** rough estimate per split — `N * L * D * 2 bytes`. For N=10k, L=16, D=512 → ~160 MB. Cheap.

## Deliverables checklist

- [ ] `scripts/dump_contrastive_embeddings.py` — one-shot embedding dumper
- [ ] `shared/knn_eval_dumps/qwen3_nq/{train,val,test}.npz`
- [ ] `notebooks/knn_eval_sweep.ipynb` (master) + working copy
- [ ] `results/knn_sweep_qwen3_nq.csv`
- [ ] Short writeup at the bottom of this doc summarizing findings and recommended defaults
