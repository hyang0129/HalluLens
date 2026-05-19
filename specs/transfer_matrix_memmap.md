# Spec — Cross-dataset transfer matrix on 50k memmap checkpoints (issue #89)

Supersedes [`specs/issue_62_transfer_matrix.md`](issue_62_transfer_matrix.md).
The transfer matrix in #62 / PR #63 was built against zarr-trained baselines.
This spec ports the protocol to the **50k icr_capture memmap checkpoints**
produced by [#79](https://github.com/hyang0129/HalluLens/issues/79), so every
method in the §6 *Transfer* row of the EMNLP paper is fit on the same train
set as the ICR probe ([#70](https://github.com/hyang0129/HalluLens/issues/70)).

## Why this is a port, not a rewrite

PR #63's logic — load source checkpoint → forward target test → AUROC — is
correct. What changes:

| Aspect | #62 / PR #63 (zarr) | #89 (memmap) |
|--------|---------------------|--------------|
| Parser | `ActivationParser` (zarr/json/lmdb) | `MemmapActivationParser` (icr_capture dirs) |
| Source-train data path | `<src>_train.json[activations_path]` | `configs/datasets/<src>[_qwen3]_memmap.json[icr_capture.train_dir]` |
| Target-test data path | `<tgt>_test.json[activations_path]` | `configs/datasets/<tgt>[_qwen3]_memmap.json[icr_capture.test_dir]` |
| Source-train split semantics | `split_strategy="none"` over pre-split train zarr | `split_strategy="none"` over the **whole 50k** train capture (Mahalanobis/KNN reference) |
| Run dirs scanned | `runs/baseline_comparison_<src>/...` | `runs/baseline_comparison_<src>[_qwen3]_memmap/<src>[_qwen3]_memmap/...` (inner dir name = `dataset` field of the experiment config; `_qwen3` is present on both halves for the qwen3 variant) |
| Method roster | `linear_probe`, `saplma`, `contrastive_logprob_recon` | `saplma`, `contrastive_logprob_recon`, `llmsknow_probe` (drop `linear_probe`; add `llmsknow_probe`) |
| Checkpoint files | `contrastive_last.pt`, `linear_probe_last.pt` | same — `contrastive_last.pt` for contrastive, `linear_probe_last.pt` for saplma. `llmsknow_probe` has no checkpoint on disk; refit on source train at transfer time (see below). |

## Methods × scorers (3)

| Method | Source artifact (resolved under `seed_dir/artifacts/`) | Transfer-time scoring |
|--------|---------------------------------------------------------|------------------------|
| `contrastive_logprob_recon` | `contrastive_last.pt` (fallback: `final_weights.pt`) | Embed `src_train` (whole 50k) and `tgt_test` with the contrastive model on `target_layers` (defaults to `[22, 26]` per method cfg). Mahalanobis OOD on source-train embeddings → AUROC (headline); KNN-OOD AUROC as secondary. |
| `saplma` | `linear_probe_last.pt` (fallback: `final_weights.pt`) | Forward-pass `tgt_test` at `probe_layer` resolved from `eval_metrics.json[selected_layer]` → `seed_dir/config.json[method.data.probe_layer]` → 22 (`saplma.json` canonical default; not 26). Sigmoid → AUROC. |
| `llmsknow_probe` | `seed_dir/artifacts/sweep_summary.json` — provides `best_layer`, `best_token_pos`. **The fitted sklearn probe is not persisted** by `run_llmsknow_probe`. | Re-fit a `LogisticRegression` on the source-train `(N, H)` column at the cached `(best_layer, best_token_pos)`, then score the target-test column at the same `(layer, token_pos)`. Single forward, no sweep. |

### Why refit `llmsknow_probe` instead of persisting it from #79

1. The `run_llmsknow_probe` code in `scripts/run_experiment.py` (lines ~1583-1684 as of `96cfb9f`) writes `sweep_auroc_matrix.npy` + `sweep_summary.json` but **not** the fitted probe.
2. Adding a save path inside this issue creates a hidden dep on re-running #79 to materialize pickles; the issue says the smoketest must work as soon as a single seed lands.
3. The probe is a single sklearn `LogisticRegression` over a `(N_train, H)` column (~50k × 4096 floats). Refit is seconds.
4. Determinism: refit uses the same `seed=training_seed`, `C=1.0`, `max_iter=100` (or whatever is in the source `config.json[method.sweep]`), so the diagonal sanity check still reproduces the in-distribution AUROC within float tolerance.

### Source-train slice (contrastive Mahalanobis/KNN reference)

`MemmapActivationParser(train_dir, split_strategy="none", random_seed=<discovered>)` — return **all** 50k rows, not the per-seed 90% subset. Rationale (same as #62): the transfer claim is *one labeled corpus → score arbitrary new corpus*; using the 45k train subset of a particular seed would couple the transfer number to the training fold.

The `random_seed` parameter is accepted but is a no-op for `split_strategy="none"` — pass `0` (or the run's split_seed; doesn't matter).

## Files

### New

- `activation_research/transfer_eval_memmap.py` — analogue of #63's `activation_research/transfer_eval.py`. Public surface:
  ```python
  load_checkpoint_model(method, checkpoint_path, dataset_cfg) -> nn.Module
  get_embeddings_contrastive(model, capture_dir, relevant_layers, device, ...) -> list[dict]
  get_scores_probe(model, capture_dir, probe_layer, device, ...) -> (scores, labels)
  evaluate_transfer_cell(source_run_dir, source_dataset_cfg, target_dataset_cfg, method, relevant_layers, probe_layer, device) -> dict
  discover_runs(runs_root, method) -> list[dict]   # scans runs/baseline_comparison_*_memmap/<ds>_memmap/<method>/seed_*/
  ```

- `scripts/eval_transfer_matrix_memmap.py` — CLI mirror of `scripts/eval_transfer_matrix.py`. Flags: `--source-datasets`, `--target-datasets`, `--methods`, `--model-slugs {llama,qwen3}`, `--seeds`, `--device cpu`, `--num-workers`, `--relevant-layers 14-29`, `--runs-dir runs`, `--configs-dir configs`, `--output-dir runs/transfer_matrix_memmap`, `--resume`.

### Deleted (in the same PR)

- `activation_research/transfer_eval.py` (the zarr version from PR #63 — has not been merged; deleted to keep one transfer-matrix script in tree)
- `scripts/eval_transfer_matrix.py`

These two files only exist on `feat/issue-62-transfer-matrix` (and any branches forked from it). They will not be on `main` when #89 lands, so the PR for #89 does **not** need to delete them — it just must not introduce them. Verified against `git show main:activation_research/transfer_eval.py` (absent). **No deletion lines in the diff.**

### Spec / housekeeping

- This file (`specs/transfer_matrix_memmap.md`).
- `specs/issue_62_transfer_matrix.md` — left untouched as historical reference.

## Output layout

```
runs/transfer_matrix_memmap/
  llama/
    <src>__<tgt>__<method>__<seed>.json          # one JSON per cell
    ...
  qwen3/
    ...
  transfer_matrix.csv          # long: model_slug, method, seed, source, target, auroc, mahalanobis_auroc, knn_auroc, n_src_train, n_tgt_test, status
  transfer_matrix_mean.csv     # mean over seeds per (model, method, source, target)
  transfer_matrix_ci.csv       # mean ± 1.96·std/√n
```

Each per-cell JSON also stores `{source_dataset, target_dataset, method, model_slug, seed, experiment_name, status, auroc, ...}`. Schema kept identical to PR #63 so downstream notebooks need no changes.

Headline scalars per (model, method) — written to a single `transfer_matrix_summary.json`:
- off-diagonal mean AUROC ± 95% CI
- worst-pair AUROC (min off-diag)
- in-distribution reference (diagonal mean)

Heatmaps `results/figures/transfer_memmap_<model>_<method>.png` are **out of scope for this PR** — wired in a follow-up; the CSVs are sufficient for the smoketest and for #79 to backfill.

## Diagonal sanity check (acceptance gate)

For each completed seed, the diagonal cell `(src=tgt, model, method, seed)` must reproduce the in-distribution number persisted at training time, within ±0.02 AUROC.

| Method | In-dist source of truth | Transfer cell field |
|--------|------------------------|---------------------|
| `saplma` | `seed_dir/eval_metrics.json[auroc]` | `auroc` |
| `contrastive_logprob_recon` | `seed_dir/eval_metrics.json[mahalanobis_auroc]` | `mahalanobis_auroc` (headline) |
| `llmsknow_probe` | `seed_dir/eval_metrics.json[auroc]` | `auroc` |

If any diagonal misses by more than ±0.02, the pipeline is wrong — most likely a data-loader mismatch between training-time and transfer-time (relevant_layers, pad_length, response_logprobs inclusion, …). Audit the constructor args passed to `get_dataset` and align with `run_saplma` / `run_contrastive_logprob_recon` / `run_llmsknow_probe` in `scripts/run_experiment.py`.

## Smoketest (do this first; do not wait on full #79)

Cell: `source=hotpotqa`, `target=hotpotqa` (diagonal), `model=llama`, `seed=0`.
Run each of the three methods. Each must pass the ±0.02 diagonal check above
against the persisted `eval_metrics.json` in
`runs/baseline_comparison_hotpotqa_memmap/hotpotqa_memmap/<method>/seed_0/`.

Then run one off-diagonal cell `(hotpotqa → mmlu, llama, saplma, seed=0)` end-to-end to exercise the cross-dataset path.

Once the 3-method × 2-cell smoketest passes, the rest of the grid backfills via `--resume` as #79 lands more checkpoints.

## Acceptance criteria

- [ ] `scripts/eval_transfer_matrix_memmap.py` runs a single cell from CLI against memmap checkpoints.
- [ ] Diagonal cells reproduce the in-dist `eval_metrics.json[auroc]` / `[mahalanobis_auroc]` within ±0.02.
- [ ] `--resume` skips cells where the per-cell JSON already exists; continues partial runs.
- [ ] `runs/transfer_matrix_memmap/transfer_matrix.csv` is written for all completed cells.
- [ ] Smoketest (3 methods × {diagonal, one off-diagonal} × Llama × seed 0) passes before #79 fully finishes.
- [ ] #62 closed as superseded; #63 closed without merge.

## Out of scope (per issue #89)

- `linear_probe`, `multi_layer_linear_probe`, `saplma_logprob_recon`, `contrastive`, `token_entropy`, `logprob_baseline`. (These are in `configs/experiments/baseline_comparison_*_memmap.json` for *training* but not for the transfer table.)
- Cross-model transfer (Llama-trained probe scored on Qwen activations).
- Target-train fine-tuning / calibration.
- Per-example `predictions.csv` for transfer cells (summary AUROCs only).
- Bootstrap CIs from #59 (separate follow-up after #65 merges).
- Heatmap figures.

## Compute

CPU-only. Each cell is one forward pass over the target test split (~10k–50k items). 1,080 cells × seconds-per-cell ≈ single-digit hours on alphacpu. No GPU, no SLURM, no dispatch approval.
