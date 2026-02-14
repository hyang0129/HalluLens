# K-View Contrastive Training Migration (Strict K-View, No Legacy Pair Format)

## Purpose

This document defines the **required code and API changes** to migrate HalluLens contrastive training from hard-coded 2-view batches (`layer1_*`, `layer2_*`) to a strict **K-view** design (`K >= 2`, e.g., 4 or 8).

Assumptions for this migration:

- We will update the whole training pipeline together.
- We will **not** support legacy two-view batch structures after migration.
- We keep supervised contrastive loss (`SupConLoss`) as the core objective.
- Activation storage remains Zarr-based (already enforced elsewhere).

---

## Executive Summary

Current training/evaluation code is deeply pair-centric. The following patterns are hard-coded and must be replaced:

- Two activation tensors per sample (`layer1_activations`, `layer2_activations`)
- Two layer indices (`layer1_idx`, `layer2_idx`)
- Pair metrics (`pairing_accuracy(z1, z2)`, pair cosine)
- Evaluator buffers split into x1/x2 and z1/z2
- Tests and examples constructing pair-only dummy datasets

The central migration is to standardize on a **single canonical batch schema**:

- `views_activations`: `(B, K, T, H)`
- `view_indices`: `(B, K)` (optional for non-layer-aware encoders)
- `halu`: `(B,)`
- `hashkey`: `List[str]` or equivalent sample IDs

Everything downstream (collate, trainers, evaluator, metrics, tests, docs) should consume this shape directly.

---

## Canonical Data Contract (New)

## Per-sample dataset item

```python
{
    "hashkey": str,
    "halu": torch.Tensor,          # scalar float/bool; collate -> (B,)
    "all_activations": List[torch.Tensor],
    "views_activations": torch.Tensor,  # (K, T, H)
    "view_indices": torch.Tensor,       # (K,) indices into relevant_layers
    "input_length": int,
}
```

## Collated batch

```python
{
    "hashkey": List[str],
    "halu": torch.Tensor,              # (B,)
    "views_activations": torch.Tensor, # (B, K, T, H)
    "view_indices": torch.Tensor,      # (B, K)
    "input_length": torch.Tensor,      # (B,)
}
```

## Derived tensors in trainer/evaluator

- Flatten views for one-pass encoding:
  - `x = views_activations.reshape(B*K, T, H)`
- Layer-aware encoder path:
  - `layer_idx = view_indices.reshape(B*K)`
- Embedding restore:
  - `z = model(...)` gives `(B*K, D)`
  - `z_views = z.reshape(B, K, D)` for SupConLoss

This is the key to keeping code simple and avoiding K loops over model forward unless memory constraints force chunking.

---

## Component-by-Component Audit and Required Changes

## 1) Data generation/parsing (`activation_logging/activation_parser.py`)

### Current state

- `ActivationDataset.__getitem__` returns pair fields:
  - `layer1_activations`, `layer2_activations`
  - `layer1_idx`, `layer2_idx`
- Layer selection logic samples exactly two layers.
- `min_target_layers` semantics currently pair-oriented.

### Required changes

1. Add `num_views: int` to `ActivationDataset` and `ActivationParser.get_dataset(...)`.
2. Replace 2-layer sampling with K-layer sampling strategy:
   - With fixed layer: include fixed index once and sample remaining `K-1` from others.
   - Without fixed layer: sample `K` indices from available layers.
3. Define behavior when available layers `< K`:
   - Decide policy globally (recommended: hard fail for strictness, or configurable `sampling_with_replacement`).
4. Return `views_activations` and `view_indices` only (remove pair keys).
5. Update docstrings and typing to no longer mention two views.

### Notes

- The existing `all_activations` can remain for classifier/OOD use-cases.
- This is the source of truth; once updated, pair fields should not be emitted anywhere.

---

## 2) Contrastive collate (`activation_research/training.py::_contrastive_collate_min`)

### Current state

- Explicitly stacks `layer1_activations` and `layer2_activations`.
- Emits `layer1_idx`, `layer2_idx`.

### Required changes

1. Rename to something K-view explicit, e.g., `_contrastive_collate_kview`.
2. Stack `views_activations` to `(B, K, T, H)`.
3. Stack `view_indices` to `(B, K)` when present.
4. Preserve `halu`, `hashkey`, optional metadata.
5. Remove all references to pair-specific fields.

---

## 3) Trainer configs (`activation_research/trainer.py`)

### Current state

- `ContrastiveTrainerConfig` has no `num_views`.
- Batch assumptions are pair-only in both trainers.

### Required changes

1. Add config fields:
   - `num_views: int`
   - Optional: `view_sampling_with_replacement: bool`
   - Optional: `train_forward_chunk_views: Optional[int]` for memory control.
2. Validate `num_views >= 2` at config/trainer init.
3. Ensure dataloader setup uses K-view collate function.

---

## 4) `ContrastiveTrainer.training_step` (`activation_research/trainer.py`)

### Current state

- Reads `layer1_activations`/`layer2_activations`.
- Computes `z1`, `z2`, then `torch.stack([z1, z2], dim=1)`.
- Metrics are pair-only.

### Required changes

1. Read `views_activations` `(B, K, T, H)`.
2. Flatten to `(B*K, T, H)`, encode once, reshape to `(B, K, D)`.
3. Feed `z_views` directly to `SupConLoss`.
4. Keep label/sample-id logic but ensure sample IDs remain per base sample (`B`), not per view.
5. Replace pair metrics with K-view metrics:
   - `intra_sample_cosine_mean`
   - `view_retrieval_top1` (optional)
   - `inter_sample_margin` (optional)

---

## 5) `LayerAwareContrastiveTrainer.training_step` (`activation_research/trainer.py`)

### Current state

- Uses pair activations and pair layer indices.

### Required changes

1. Read `views_activations` and `view_indices`.
2. Flatten both:
   - `x_flat: (B*K, T, H)`
   - `layer_idx_flat: (B*K,)`
3. Single forward pass for all views with `layer_idx=layer_idx_flat`.
4. Reshape embeddings to `(B, K, D)` before loss.
5. Adopt same K-view metrics as non-layer-aware trainer.

---

## 6) Class-based validation evaluator (`activation_research/contrastive_evaluator.py`)

### Current state

- Buffers `_buffer_x1`, `_buffer_x2`, `_buffer_l1`, `_buffer_l2`.
- `_process_full_batch` consumes pair tensors.
- Computes pair-only metrics.

### Required changes

1. Replace pair buffers with K-view buffers:
   - `_buffer_views`, `_buffer_view_indices`, `_buffer_labels`
2. In validation step, append `(B_sub, K, T, H)` slices.
3. In flush, concatenate to `(B_full, K, T, H)`, flatten -> encode -> reshape `(B_full, K, D)`.
4. Loss: `SupConLoss(z_views, labels=...)`.
5. Metrics: replace pairing accuracy/cosine with K-view counterparts.
6. Update `ContrastiveEvalState` fields (`total_acc` -> K-view metric naming).

---

## 7) Functional evaluator path (`activation_research/evaluation.py::evaluate`)

### Current state

- Legacy evaluate path also pair-only (`buffer_x1/x2`, `z1/z2`).
- Used by older training utilities and examples.

### Required changes

1. Rewrite evaluate to K-view tensor flow (same strategy as class-based evaluator).
2. Replace `evaluator_manager.accumulate_batch(z1, z2, ...)` with K-view accumulation interface.
3. Remove or migrate pair helper functions:
   - `pairing_accuracy`
   - `average_cosine_similarity`

---

## 8) Evaluator manager (`activation_research/evaluator_manager.py`)

### Current state

- Stores `accumulated_z1`, `accumulated_z2`.
- Records use `{z1, z2, ...}` schema.

### Required changes

1. Change storage to one tensor list, e.g., `accumulated_z_views` with shape `(B, K, D)`.
2. New API:
   - `accumulate_batch(z_views: torch.Tensor, hashkeys=None, labels=None)`
3. Record schema becomes:
   - `{"z_views": Tensor[K, D], "hashkey": ..., "halu": ...}`
4. Update example cosine evaluator to aggregate over all within-sample view pairs.

---

## 9) Legacy training entrypoint (`activation_research/training.py::train_contrastive`)

### Current state

- Contains explicit pair buffers and pair metrics.

### Required changes

1. Apply same K-view forward flow as class-based trainer.
2. Replace pair buffer vars with generic view buffer.
3. Ensure `batch_size/sub_batch_size` logic still works with K-view tensors.
4. Keep checkpoint keys stable where possible, but metric names should move to K-view semantics.

---

## 10) Model calibration path (`activation_research/model.py`)

### Current state

- `calibrate` reads `batch.get("layer1_activations", ...)`.

### Required changes

1. Switch calibration input extraction to `views_activations`.
2. Flatten `(B, K, T, H)` to token-level vectors for norm stats.
3. Remove fallback to `layer1_activations` once migration completes.

---

## 11) Tests (`tests/test_contrastive_trainer_smoke.py`)

### Current state

- Dummy datasets emit pair fields only.

### Required changes

1. Rewrite dummy datasets to emit:
   - `views_activations: (K, T, H)`
   - `view_indices: (K,)`
2. Add parametrized K tests (`K=2,4,8` at minimum).
3. Validate trainer/evaluator run + checkpoint/resume for K>2.
4. Add shape assertions in collate and training step.
5. Remove pair-only assertions and naming.

---

## 12) Examples/docs (`activation_research/evaluator_manager_example.py`, notebooks)

### Current state

- Examples and comments repeatedly reference z1/z2 and pair metrics.

### Required changes

1. Update all examples to `z_views` terminology.
2. Replace pair metric printouts with K-view metrics.
3. Update notebook cells that assume `layer1_*`/`layer2_*`.

---

## K-View Metrics Recommendations

Since `pairing_accuracy` is undefined for `K>2` without special handling, use metrics that generalize:

1. **Intra-sample cosine mean**
   - Mean cosine across all $
\binom{K}{2}
$ view pairs per sample.
2. **Intra-vs-inter margin**
   - Mean positive similarity minus mean negative similarity.
3. **View retrieval top-1 (sample ID retrieval)**
   - For each view embedding, nearest neighbor among embeddings of one selected alternate view set should share sample ID.
4. **SupCon loss itself**
   - Keep as primary training objective metric.

Recommended minimal set for logs/checkpoints:

- `train_loss`, `val_loss`
- `train_intra_cos`, `val_intra_cos`
- `train_intra_inter_margin`, `val_intra_inter_margin`

---

## Sampling Rules for K Views

Define one policy and keep it consistent everywhere:

### Strict policy (recommended for this migration)

- Require at least `num_views` valid target layers per sample.
- If not enough layers, raise and filter those samples upstream.

### Alternative policy

- Allow replacement when `available_layers < num_views`.
- Must be explicit via config; otherwise reproducibility and interpretation suffer.

Given your requirement for strict cleanup and no legacy behavior, strict policy is cleaner.

---

## Memory/Compute Considerations

For fixed encoder and sequence shape, compute roughly scales with `K`.

Rule of thumb for equal memory footprint:

$$
\text{new_batch_size} \approx \text{old_batch_size} \times \frac{2}{K}
$$

So if old `(K=2, batch=512)`:

- `K=4` -> batch ~256
- `K=8` -> batch ~128

Use gradient accumulation if throughput drops too much.

---

## Migration Sequence (Recommended)

1. **Data contract first**: dataset + collate emit only K-view fields.
2. **Trainer update**: both trainer classes consume K-view batches.
3. **Evaluator update**: class-based + functional evaluator use K-view.
4. **Manager + examples**: switch z1/z2 storage/docs to z_views.
5. **Tests**: convert smoke tests and add K parametrization.
6. **Cleanup**: remove pair helper functions and pair-named metrics.

Do this in one branch with failing tests up front, then green end-to-end.

---

## Acceptance Criteria

Migration is complete when all of the following hold:

1. No training/evaluation code references:
   - `layer1_activations`, `layer2_activations`
   - `layer1_idx`, `layer2_idx`
   - `z1`, `z2` pair APIs (except maybe historical comments/docs)
2. Contrastive trainers run for `K=4` and `K=8` without schema adapters.
3. Validation and metrics are K-view aware.
4. Smoke tests pass for at least `K in {2, 4, 8}`.
5. Docs/examples reflect only K-view contracts.

---

## Concrete Refactor Checklist

- [ ] `activation_logging/activation_parser.py`
  - [ ] Add `num_views`
  - [ ] Emit `views_activations`, `view_indices`
  - [ ] Remove pair fields
- [ ] `activation_research/training.py`
  - [ ] Replace `_contrastive_collate_min`
  - [ ] Rewrite `train_contrastive` K-view flow
- [ ] `activation_research/trainer.py`
  - [ ] Add `num_views` config
  - [ ] Rewrite both `training_step` methods
  - [ ] Update metric names
- [ ] `activation_research/contrastive_evaluator.py`
  - [ ] Replace pair buffers with K-view buffers
  - [ ] K-view metric aggregation
- [ ] `activation_research/evaluation.py`
  - [ ] Rewrite functional evaluate for K-view
- [ ] `activation_research/evaluator_manager.py`
  - [ ] Replace z1/z2 accumulation with z_views
- [ ] `activation_research/model.py`
  - [ ] Calibration consumes `views_activations`
- [ ] `tests/test_contrastive_trainer_smoke.py`
  - [ ] K-view dummy datasets
  - [ ] Parametrize by K
- [ ] `activation_research/evaluator_manager_example.py`
  - [ ] Update APIs and examples to z_views

---

## Non-Goals

- This document does not redesign SupCon loss itself.
- This document does not propose mixed legacy adapters.
- This document does not change OOD metric evaluator internals beyond required input schema migration.

---

## Final Recommendation

Adopt a **single strict K-view batch schema** now and migrate all components in one coherent pass. Avoid temporary pair adapters; they prolong complexity and increase bug surface area.
