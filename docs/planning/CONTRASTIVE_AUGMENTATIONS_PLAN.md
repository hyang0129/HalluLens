# Plan: Intra-View Augmentations for Contrastive Activation Learning

## Background

The current contrastive setup forms positive pairs by sampling **different LLM layers** of the same example as views (`activation_parser.py::_select_view_indices`). The shared `ProgressiveCompressor` is layer-agnostic, so pulling cross-layer views together implicitly trains a layer-invariant `z`. I.i.d. dropout inside the encoder is a small extra perturbation, not the dominant view-generator.

This plan adds **intra-view augmentations**: random transforms applied to each individual `(L_response_tokens, D=4096)` activation tensor *before* the encoder, independently for each of the K views. This is orthogonal to (and stacks with) the existing cross-layer view sampling.

The hypothesis: cross-layer sampling alone may make the encoder collapse to layer-invariant features that are *too* invariant to nuisances within a single layer (specific tokens, specific feature dims). Intra-view perturbations regularize each view independently, the same way SimCLR's color jitter / crop regularize each image view.

## Insertion point

A single transform fn `augment(view: Tensor[L, D]) -> Tensor[L, D]` plugged into the dataset between layer-selection and view-stacking. Concretely, edit `activation_parser.py` around [line 171](activation_logging/activation_parser.py#L171):

```python
selected_views = [filled_activations[i] for i in selected_view_indices]
selected_views = [act.squeeze(0) if act.ndim == 3 and act.shape[0] == 1 else act for act in selected_views]
# NEW: per-view augmentation
if self.augment_fn is not None:
    selected_views = [self.augment_fn(v) for v in selected_views]
views_activations = torch.stack(selected_views, dim=0)
```

Pass `augment_fn` and an `augmentations_config` through the dataset constructor; build the composition once at dataset init.

Apply augmentation **only on the train split, not eval**. The eval pipeline already passes `is_training=False` paths through the dataset; gate `augment_fn` on that.

## Per-augmentation specs

All operate on a single view of shape `(L, D)` where `L` is the response-token count (variable, padded), `D = 4096`. A response mask is available — augmentations that drop tokens must respect it (only drop within the valid response span; never pad tokens).

### 1. Whole-token dropout (`p_token = 0.15`)

```python
def whole_token_dropout(x, mask, p):
    keep = (torch.rand(x.shape[0], device=x.device) > p) | ~mask
    return x * keep.unsqueeze(-1)
```

Independently zero entire token rows with probability `p`. Each zeroed token contributes nothing to attention or to the mean-pool. Cheap, no rescaling needed (mean-pool naturally adjusts; the reduced number of contributing tokens lowers the mean magnitude proportionally — this *is* the augmentation).

Hyperparameter: `p_token ∈ {0.10, 0.15, 0.25}` to sweep.

### 2. Span masking (`p_span = 0.15`, `mean_span_len = 3`)

```python
def span_mask(x, mask, p, mean_len):
    L = mask.sum().item()
    n_spans = max(1, int(p * L / mean_len))
    for _ in range(n_spans):
        start = random.randint(0, L - 1)
        length = max(1, int(random.gauss(mean_len, 1.0)))
        x[start:start+length] = 0
    return x
```

Sample geometric-ish spans, zero them. Total fraction zeroed `≈ p_span`. Stronger than (1) because correlated zeros are harder to compensate. Use `[MASK]`-vector replacement instead of zero if zero turns out to be too distinguishable from real activations (a learned bias term is fine; no need for a true `[MASK]` token).

Hyperparameter: `p_span ∈ {0.10, 0.15}`, `mean_span_len ∈ {2, 3, 5}`.

### 4. Boundary crop (`crop_max = 0.20`)

```python
def boundary_crop(x, mask, crop_max):
    L = mask.sum().item()
    drop_left  = random.randint(0, int(crop_max * L))
    drop_right = random.randint(0, int(crop_max * L))
    new_mask = mask.clone()
    # zero out cropped boundary tokens
    valid_idx = torch.where(mask)[0]
    if drop_left > 0:
        x[valid_idx[:drop_left]] = 0
    if drop_right > 0:
        x[valid_idx[-drop_right:]] = 0
    return x
```

Drop a random prefix and/or suffix of the response (up to 20% each side). Preserves the middle of the answer. The cheapest length-augmentation; nice diversity signal for two views of the same example.

Combine cautiously with span masking — together they can erase too much of short responses (PopQA answers can be <5 tokens). Add a guard: never drop more than half the response.

### 6. Channel dropout (`p_channel = 0.10`)

```python
def channel_dropout(x, mask, p):
    keep = torch.rand(x.shape[-1], device=x.device) > p
    return x * keep.unsqueeze(0)
```

Zero out entire feature columns of the `(L, D)` tensor, same dropped-set across all tokens of this view. This is the standard `nn.Dropout2d`-style channel dropout, applied along the D axis.

Hyperparameter: `p_channel ∈ {0.05, 0.10, 0.20}`.

### 8. Calibrated Gaussian noise (`σ = 0.1`)

Requires per-feature std precomputed once over the training set per (model, dataset, layer). Cheap one-time pass:

```python
# precompute (D,) std per layer
std_per_feat = activations.std(dim=(0, 1))  # over batch, sequence
```

```python
def calibrated_gaussian(x, std_per_feat, sigma):
    return x + torch.randn_like(x) * std_per_feat * sigma
```

Without per-feature scaling, isotropic noise is dominated by the few high-variance dims and barely perturbs the rest. With calibration, every dim gets perturbed to the same SNR.

Hyperparameter: `σ ∈ {0.05, 0.10, 0.20}`. Cache the per-feature stds in the dataset as `feature_std_layer{ℓ}.npy` next to the zarr.

### 9. Multiplicative jitter (`σ_jitter = 0.05`)

```python
def jitter(x, mask, sigma):
    scale = 1.0 + torch.randn(x.shape[0], 1, device=x.device) * sigma
    return x * scale
```

Per-token multiplicative scale `(1 + ε)`, ε ~ N(0, σ). Mimics the natural per-token magnitude variation in the residual stream. Cheap; preserves direction.

Hyperparameter: `σ_jitter ∈ {0.03, 0.05, 0.10}`.

## Composition

Each view goes through a **random composition** of the six transforms. Two reasonable strategies:

**(a) All-in, fixed-rate.** Always apply all six in fixed order: `boundary_crop → span_mask → whole_token_dropout → channel_dropout → jitter → gaussian_noise`. Rates set so the *expected* fraction of input zeroed/perturbed is moderate (~30%). Simpler, more reproducible.

**(b) RandAugment-style.** Per view, sample `k=3` of the six uniformly; apply at full rate. More diversity between views.

Start with (a). It's deterministic given the rates and easier to ablate.

## Label-preservation sanity check

Before training, verify augmentations don't break positive-pair semantics:

1. Take a held-out 1000-sample subset.
2. For each, apply the augmentation pipeline 10×; encode each view with a current best checkpoint; compute pairwise cosine similarity within an example.
3. Compare to between-example similarity. The augmented-self distribution should sit cleanly above the between-example distribution.

Decision rule: if median(within-example) < 75th-percentile(between-example), augmentations are too aggressive — back off rates by half and retry.

A second check: verify hallucination AUROC of a frozen-encoder linear probe doesn't drop on augmented inputs. If it tanks, the augmentation is destroying the signal we want to preserve.

## Ablation matrix

Holding everything else fixed (encoder, layers, lr, seeds, contrastive + logprob aux as the current best), run:

| Run | aug |
|---|---|
| **A0** baseline (current best, no intra-view aug) | – |
| **A1** dropout family (1 + 2 + 4) | token + span + crop |
| **A2** noise family (6 + 8 + 9) | channel + gaussian + jitter |
| **A3** all six | full pipeline |

A1 vs A2 separates "remove information" augmentations from "perturb information" augmentations. A3 tests stacking. We do not run each of the six in isolation — that's 6 extra cells for marginal information; the family-level ablation captures the relevant structure.

Datasets: PopQA test + SciQ test on Qwen3-8B and Llama-3.1-8B. 4 (model, dataset) cells × 4 runs × 3 seeds = 48 runs. Cut to PopQA-only (24 runs) for the first pass.

## Decision criteria

Headline: **A3 vs A0** mean AUROC over seeds, both datasets.

- **A3 > A0 by ≥1 AUROC point** → land the augmentations as default. Promote winner of `{A1, A2, A3}` if one of the families is clearly better than the union.
- **A3 ≈ A0** → cross-layer sampling already saturates the augmentation budget. Document and shelve.
- **A3 < A0** → augmentations are net-harmful given an already-strong cross-layer scheme. Most likely cause: the cross-layer signal is the *whole* signal and intra-view perturbation just adds noise. Diagnose with the sanity check before discarding.

## Compute cost

All augmentations are O(L · D) elementwise ops, applied in the dataset (CPU). Profile first — if augmentation dominates dataloader time, move to GPU as a pre-encoder transform. For `L ≈ 50`, `D = 4096`, the per-view cost is ~200K ops, negligible.

Per-feature std caching (#8) requires one pass over the train zarr per (model, dataset, layer); ~minutes. Run once per dataset config.

## Out of scope (intentionally)

- Augmentation #5 (token shuffling): order may carry hallucination signal (early commitment vs late drift); skip until we know.
- Augmentation #7 (grouped feature dropout): requires structure we don't have on-hand; channel dropout #6 is the simpler version.
- Augmentation #14 (stochastic depth on the encoder): orthogonal class — encoder regularization, not data augmentation. Separate experiment.
- Augmentations #10, #11, #13: covered separately — see chat for explanations. #11 is largely subsumed by existing cross-layer view sampling.
