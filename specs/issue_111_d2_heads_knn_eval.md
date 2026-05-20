# Issue #111 — Per-head + ensemble KNN re-eval for D2 shared-trunk checkpoints

## Context

PR #103 (issue #102) trained `SharedTrunkProjectionHeadCompressor` models on SciQ and NQ splits with three methods (C0, D2a, D2b). The D2 variants have a shared trunk + two projection heads (Head A trained with `ignore_label=1`, Head B with `ignore_label=0`), but the existing eval only uses the trunk as the eval surface — the heads are discarded.

This adds a re-evaluation script that loads existing D2 checkpoints and reports KNN AUROC on Head A, Head B, and a length-normalized ensemble of both, **without retraining**.

**Experimental scope: D2a only** for now (D2b is still training). If results justify it, the per-head or ensemble eval may be promoted to a headline model class in a follow-up issue — out of scope here.

## Deliverable

One new file:

- `scripts/eval_d2_heads_knn.py` — single-file re-eval script.

No changes to training code, no changes to existing `eval_metrics.json` files, no new model classes, no new dataset configs.

## Algorithm

For each run directory containing `config.json` + `artifacts/final_weights.pt`:

1. **Rebuild model** from `config.json["method_params"]["model_params"]`:
   - `SharedTrunkProjectionHeadCompressor(input_dim=..., trunk_dim=..., head_dim=..., head_hidden_dim=..., recon_seq_len=..., recon_hidden_dim=..., recon_lambda=..., input_dropout=..., logprob_var_threshold=...)`
   - Load `final_weights.pt` state dict.
2. **Rebuild train/test split** using the run's `split_seed` (read from `config.json["split_seed"]`).
3. **Extract embeddings** by calling `model.forward_with_heads(x)` over both splits. This returns `(trunk_z, z_A, z_B, _logprob_pred)`. Store three (N, D) tensors per split: `train_trunk`, `train_A`, `train_B`, `test_trunk`, `test_A`, `test_B`.
4. **For each rep in `{trunk, A, B}`:**
   - Build a KNN index on train embeddings. `train_label_filter="all"` (locked). `k=50` and `metric="euclidean"` by default — read from existing `eval_metrics.json` to confirm match.
   - **Test distance:** `d_test(x) = mean(top-k nearest train distances)`.
   - **Train distance (for ensemble normalization):** query `k+1` nearest, drop self at distance 0 (leave-one-out). Same `mean(top-k)` aggregation.
   - **AUROC:** `flip_auroc=False` matches existing — score = distance; positive class = OOD (halu=1) since OOD samples are expected to have higher distance.
5. **Ensemble:**
   ```
   scale_A = mean(train_nn_distances_A)   # scalar
   scale_B = mean(train_nn_distances_B)
   d_ensemble(x) = (d_A_test(x) / scale_A + d_B_test(x) / scale_B) / 2
   ```
   Compute AUROC on `d_ensemble`. The `/scale` makes each head's average train distance equal to 1.0 so neither dominates due to scale differences. The `/2` is cosmetic.

## Sanity check (REQUIRED — script aborts on failure)

Recompute trunk KNN AUROC via this script and compare against the existing `eval_metrics.json["knn_auroc"]` for the same run. Must match within `1e-6`. If mismatch:
- Abort with a clear error showing both values.
- Do not write `eval_metrics_heads.json` for that run.

This guards against silent drift between the production eval pipeline and the re-eval pipeline.

## Output: `eval_metrics_heads.json`

Sidecar written next to existing `eval_metrics.json`:

```json
{
  "method": "contrastive_logprob_recon_d2a",
  "dataset": "sciq_memmap",
  "seed": 0,
  "split_seed": 42,
  "n_train": 10448,
  "n_test": 1000,
  "knn_k": 50,
  "knn_metric": "euclidean",
  "knn_train_label_filter": "all",
  "trunk":   {"auroc": 0.683, "mean_id": ..., "std_id": ..., "mean_ood": ..., "std_ood": ..., "train_nn_mean": ...},
  "head_a":  {"auroc": ...,   "mean_id": ..., "std_id": ..., "mean_ood": ..., "std_ood": ..., "train_nn_mean": ...},
  "head_b":  {"auroc": ...,   "mean_id": ..., "std_id": ..., "mean_ood": ..., "std_ood": ..., "train_nn_mean": ...},
  "ensemble_a_b_normalized": {"auroc": ..., "mean_id": ..., "std_id": ..., "mean_ood": ..., "std_ood": ...}
}
```

## CLI

```bash
# Single run (primary use case for D2a)
python scripts/eval_d2_heads_knn.py \
    --run-dir runs/sharedtrunk_grid_sciq_llama_memmap/sciq_memmap/contrastive_logprob_recon_d2a/seed_0

# Batch over all D2a method × seed dirs in a grid (D2b can be added later)
python scripts/eval_d2_heads_knn.py \
    --grid sharedtrunk_grid_sciq_llama_memmap \
    --methods contrastive_logprob_recon_d2a
```

Required flags:
- One of `--run-dir <path>` or `--grid <name>` (mutually exclusive).
- `--methods <csv>` (optional, default: process every method dir found that has a D2 checkpoint).

Optional flags:
- `--knn-k <int>` (default: read from existing `eval_metrics.json`)
- `--knn-metric <str>` (default: read from existing `eval_metrics.json`)
- `--output-name <str>` (default: `eval_metrics_heads.json`)
- `--device <str>` (default: `cuda` if available, else `cpu`)
- `--batch-size <int>` (default: 256 — for embedding extraction)

## Reuse existing helpers (REQUIRED — do not reimplement)

Reuse whatever KNN AUROC helper lives in `activation_research/` and is called by `run_experiment.py` for the existing `knn_auroc` metric. Grep for `knn_auroc` and `knn_mean_id` in `activation_research/` to locate it. Importing and calling this helper three times (once per rep) avoids drift from the production eval.

If the helper does not return both `train_nn_mean` and per-sample test distances (needed for ensemble normalization), wrap/extend it minimally — but do not duplicate the index-build + AUROC math.

## Files in scope

- `scripts/eval_d2_heads_knn.py` (new)

## Files out of scope (DO NOT touch)

- `activation_research/model.py`, `activation_research/training.py`
- `scripts/run_experiment.py`
- Any existing `eval_metrics.json` files (sidecar only)
- D1 (`SharedTrunkSplitOutputCompressor`) — dropped from grids
- C0 runs (no heads)
- New training, new datasets, new model classes, new configs
- Any change to PR #103 / issue #102 outputs

## Acceptance criteria

- [ ] Script runs cleanly on `runs/sharedtrunk_grid_sciq_llama_memmap/sciq_memmap/contrastive_logprob_recon_d2a/seed_0` and writes `eval_metrics_heads.json` containing all four AUROC values (trunk, head_a, head_b, ensemble_a_b_normalized).
- [ ] `trunk.auroc` from re-eval matches existing `eval_metrics.json["knn_auroc"]` to within `1e-6`; script aborts on mismatch.
- [ ] `--grid <name> --methods contrastive_logprob_recon_d2a` iterates over all matching seed dirs under that grid and writes one sidecar per seed.
- [ ] Reuses the existing KNN AUROC helper from `activation_research/` — confirmed via import statement, not a reimplementation.
- [ ] Script logs all four AUROC values per run on completion.
- [ ] No files outside `scripts/eval_d2_heads_knn.py` are modified.

## Implementation notes

- For `train_nn_distances`, query `k+1` nearest neighbours and drop the self-match (distance 0). Standard leave-one-out.
- Model reconstruction: don't hardcode `head_hidden_dim=head_dim` — read each from config.
- For dataset rebuild, use `MemmapContrastiveDataset` with `num_views=1` (no augmentation needed for eval). Honor `relevant_layers`, `include_response_logprobs`, `pad_length`, `random_seed` from the original training config.
- Embedding extraction can be batched on GPU; KNN itself is CPU.
- AUROC must use the same Mann-Whitney implementation the existing eval uses (numpy fallback is fine — `sklearn` is not installed on the cluster). Reusing the helper handles this.

## Smoketest

After implementation, run on the existing D2a/SciQ/Llama checkpoint:

```bash
python scripts/eval_d2_heads_knn.py \
    --run-dir runs/sharedtrunk_grid_sciq_llama_memmap/sciq_memmap/contrastive_logprob_recon_d2a/seed_0
```

Expected: `trunk.auroc ≈ 0.6827` (matches existing `knn_auroc`), file written, four AUROC values logged.
