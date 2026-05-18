# Implementation Spec: Issue #81 — Augmentation + Label-Convention Grid

## Goal

Implement the 10-config experiment grid from issue #81. This requires:
1. A new `activation_research/augmentations.py` module
2. Wiring augmentations into the ContrastiveTrainer training step
3. A `data.augmentations` config block read in `scripts/run_experiment.py`
4. A `flip_auroc` flag in the evaluation pipeline (for `ignore_label=0` configs)
5. 10 method config files (`configs/methods/contrastive_logprob_recon_b{0..9}.json`)

Nothing else. Do not refactor unrelated code.

---

## 1. `activation_research/augmentations.py` (new file)

### Tensor shapes

All augmentations receive and return `views` of shape `(B, K, T, H)`:
- B = batch size
- K = num_views (always 2 in our setup)
- T = sequence length (pad_length + 1 = 64)
- H = hidden dim (varies by model layer)

They also receive `labels` of shape `(B,)` — needed by mixup.

### Functions to implement

#### `whole_token_dropout(views, labels, p=0.15, asymmetric=False)`

Zero out entire token positions (dim=2). For each sample, sample a binary mask of shape `(T,)` where each position is zeroed with probability `p`. Apply the same mask across all H features for that position (i.e., mask shape broadcasts as `(1, 1, T, 1)` after unsqueeze).

If `asymmetric=True`, apply the mask only to `views[:, 0, :, :]` (view index 0). View index 1 is left unchanged.

Do NOT use in-place ops on the input tensor — return a new tensor.

#### `channel_dropout(views, labels, p=0.10, asymmetric=False)`

Zero out feature channels (dim=3). For each sample, sample a binary mask of shape `(H,)` where each channel is zeroed with probability `p`. Apply the same mask across all T positions (mask broadcasts as `(1, 1, 1, H)`).

If `asymmetric=True`, apply only to `views[:, 0, :, :]`.

#### `mixup_intra_label(views, labels, alpha=0.4, asymmetric=False)`

Convex combination of same-label activation pairs. For each sample i, find another sample j in the batch with the same label. Mix: `lam * views[i] + (1-lam) * views[j]` where `lam ~ Beta(alpha, alpha)`.

Implementation details:
- Draw one `lam` per sample from `Beta(alpha, alpha)`. Shape: `(B, 1, 1, 1)` for broadcasting.
- For each label value, build an index permutation over samples with that label. Use `torch.randperm` on the index list for each label group. Samples in a singleton group (only one with that label) mix with themselves (lam=1.0 effectively, or just skip).
- Apply to both views by default. If `asymmetric=True`, apply only to `views[:, 0, :, :]` (compute the mix for view 0 only; view 1 stays clean).

**Important:** Under `ignore_label=1` (the default), hallu examples (label=1) never form positive pairs with other hallu examples in the SupCon loss. Mixing two hallu activations still produces a sample with label=1 that has no cross-sample positive pairs. This is a known limitation documented in the issue — do not try to work around it here.

### `AugmentationComposer`

```python
class AugmentationComposer:
    def __init__(self, augmentations: list[dict], asymmetric: bool = False):
        ...
    def __call__(self, views: Tensor, labels: Tensor) -> Tensor:
        ...
```

Each entry in `augmentations` is a dict with keys `type` and `params`. Supported types: `whole_token_dropout`, `channel_dropout`, `mixup_intra_label`. The `asymmetric` flag from the composer is passed to each augmentation function (overrides any per-aug asymmetric setting).

Apply augmentations in sequence. Return the modified views tensor.

If `augmentations` is empty or None, return views unchanged.

---

## 2. Wire augmentation into `activation_research/trainer.py`

In `ContrastiveTrainer.training_step` (around line 618), after loading `views = batch["views_activations"].to(self.device)` and before the model forward pass, add:

```python
if self.augment_fn is not None:
    labels_for_aug = batch["halu"].to(self.device)
    views = self.augment_fn(views, labels_for_aug)
```

Add `augment_fn=None` as a constructor parameter to `ContrastiveTrainer.__init__` and store it as `self.augment_fn`.

Also add the cosine-similarity diagnostic: after augmentation (if any), compute and log `mean cosine(view_0_raw, view_1_raw)` in the metrics dict for the first 100 steps only. Key: `"aug_view_cosine"`. This requires computing cosine on the raw (pre-model) views flattened to `(B, T*H)`. Skip this logging after step 100 to avoid overhead.

---

## 3. Wire augmentation into `scripts/run_experiment.py`

In `run_contrastive` (around line 106 where `ds_kwargs` is built), after reading `data_cfg`, also read augmentation config:

```python
aug_cfg = data_cfg.get("augmentations", None)
augment_fn = None
if aug_cfg:
    from activation_research.augmentations import AugmentationComposer
    augment_fn = AugmentationComposer(
        augmentations=aug_cfg.get("ops", []),
        asymmetric=aug_cfg.get("asymmetric", False),
    )
```

Then pass `augment_fn=augment_fn` when constructing `ContrastiveTrainer`.

Find where `ContrastiveTrainer` is instantiated in `run_experiment.py` and add the `augment_fn` kwarg.

---

## 4. `flip_auroc` in evaluation

In `activation_research/evaluation.py`, find where `roc_auc_score` is called. Add a `flip_auroc: bool = False` parameter to the relevant eval function. When `flip_auroc=True`, negate the prediction scores before passing to `roc_auc_score` (i.e., `scores = -scores`). This correctly handles configs where `ignore_label=0` (hallu=class): the model places correct examples far from the hallu cluster, so their distance score is high — negating makes high score = likely hallu = label 1.

Also pass `flip_auroc` through from `run_experiment.py`. In method configs, add `"flip_auroc": true` to the `evaluation` block for configs where `ignore_label=0`.

---

## 5. The 10 config files

Base config to copy from: `configs/methods/contrastive_logprob_recon.json`

All configs share the same base except for the fields noted below. Create `configs/methods/contrastive_logprob_recon_b{0..9}.json`.

### B0 — baseline (identical to existing base config, just renamed)
```json
{
  "name": "contrastive_logprob_recon_b0",
  "routine": "contrastive_logprob_recon",
  ...same as contrastive_logprob_recon.json...
}
```
No `data.augmentations` block. `training.ignore_label: 1`.

### B1 — token dropout, both views
Add to `data`:
```json
"augmentations": {
  "asymmetric": false,
  "ops": [{"type": "whole_token_dropout", "params": {"p": 0.15}}]
}
```
`training.ignore_label: 1`

### B2 — channel dropout, both views
```json
"augmentations": {
  "asymmetric": false,
  "ops": [{"type": "channel_dropout", "params": {"p": 0.10}}]
}
```
`training.ignore_label: 1`

### B3 — MixUp intra-label, both views
```json
"augmentations": {
  "asymmetric": false,
  "ops": [{"type": "mixup_intra_label", "params": {"alpha": 0.4}}]
}
```
`training.ignore_label: 1`

### B4 — asymmetric token dropout
```json
"augmentations": {
  "asymmetric": true,
  "ops": [{"type": "whole_token_dropout", "params": {"p": 0.15}}]
}
```
`training.ignore_label: 1`

### B5 — flip (no aug, hallu=class)
No `data.augmentations` block.
`training.ignore_label: 0`
`evaluation.flip_auroc: true`

### B6 — two-class SupCon (no aug)
No `data.augmentations` block.
`training.ignore_label: null`  ← JSON null; in Python this becomes None, which means SupConLoss falls back to SimCLR/unsupervised mask. **Wait** — re-read the SupConLoss code: when `labels` is passed but `ignore_label=None`, the `ignore_label` check `labels == self.ignore_label` will never match (None != any int). So setting `ignore_label=None` in the SupConLoss constructor gives standard two-class SupCon where all same-label pairs are positive. Verify this is the case before implementing; if `ignore_label` defaults to `-1` in the constructor, use `-1` (or another sentinel) for B6 — whichever value is never a real label.

Actually: look at the SupConLoss constructor — `ignore_label=-1` by default. Labels are 0 and 1. So passing `ignore_label=-1` (the default) gives standard two-class SupCon. Use `"ignore_label": -1` for B6.

### B7 — asymmetric token dropout + two-class
```json
"augmentations": {"asymmetric": true, "ops": [{"type": "whole_token_dropout", "params": {"p": 0.15}}]}
```
`training.ignore_label: -1`

### B8 — token dropout + flip
```json
"augmentations": {"asymmetric": false, "ops": [{"type": "whole_token_dropout", "params": {"p": 0.15}}]}
```
`training.ignore_label: 0`
`evaluation.flip_auroc: true`

### B9 — asymmetric token dropout + flip
```json
"augmentations": {"asymmetric": true, "ops": [{"type": "whole_token_dropout", "params": {"p": 0.15}}]}
```
`training.ignore_label: 0`
`evaluation.flip_auroc: true`

---

## 6. Verification

After implementing, verify with a dry-run (no GPU needed):

```python
import torch
from activation_research.augmentations import AugmentationComposer

views = torch.randn(4, 2, 64, 128)
labels = torch.tensor([0, 0, 1, 1])

# Token dropout
comp = AugmentationComposer([{"type": "whole_token_dropout", "params": {"p": 0.15}}])
out = comp(views, labels)
assert out.shape == views.shape
# Some positions should be zero
assert (out == 0).any()

# Asymmetric: view 1 should be unchanged
comp_asym = AugmentationComposer([{"type": "whole_token_dropout", "params": {"p": 0.15}}], asymmetric=True)
out_asym = comp_asym(views, labels)
assert torch.allclose(out_asym[:, 1], views[:, 1])  # view 1 unchanged

# MixUp: output should differ from input but same shape
comp_mix = AugmentationComposer([{"type": "mixup_intra_label", "params": {"alpha": 0.4}}])
out_mix = comp_mix(views, labels)
assert out_mix.shape == views.shape
```

Also verify configs load without error:
```python
import json, glob
for f in glob.glob("configs/methods/contrastive_logprob_recon_b*.json"):
    cfg = json.load(open(f))
    assert "name" in cfg
    assert "training" in cfg
    assert "ignore_label" in cfg["training"]
print("all configs OK")
```

---

## Files to create/modify

| File | Action |
|------|--------|
| `activation_research/augmentations.py` | Create |
| `activation_research/trainer.py` | Modify — add `augment_fn` param + call in training_step + diagnostic |
| `scripts/run_experiment.py` | Modify — read `data.augmentations`, build AugmentationComposer, pass to trainer |
| `activation_research/evaluation.py` | Modify — add `flip_auroc` param, negate scores when true |
| `configs/methods/contrastive_logprob_recon_b0.json` through `b9.json` | Create (10 files) |

Do **not** modify `activation_logging/activation_parser.py` — augmentation is cleanest at the training-step level where batch context (all labels) is available, not at the dataset level.

Do **not** modify `activation_research/training.py`'s `train_contrastive` function — the ContrastiveTrainer class in `trainer.py` is the active code path for all current experiments. Confirm this by grepping for where `ContrastiveTrainer` vs `train_contrastive` is actually called from `run_experiment.py`.
