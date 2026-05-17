# Implementation Spec: Combined Logprob + Attention Reconstruction Contrastive Model (Issue #75)

**Goal:** Implement a single new contrastive model class — `LogprobAttnReconProgressiveCompressor` — that adds an **attention-reconstruction auxiliary head** (Mechanism K) alongside the existing logprob-reconstruction head (Mechanism F), with the attention prediction direction selectable as `forward`, `backward`, or `both`. Paired with a new memmap-backed dataset class that consumes Issue #72's capture layout and emits the same per-item dict shape as the existing zarr-backed `PreloadedActivationDataset`, so the existing contrastive trainer consumes it without changes.

## 1. Scope

### In scope (this PR)
- New dataset class `MemmapContrastiveDataset` in `activation_research/memmap_contrastive_dataset.py`. Reads from Issue #72's capture layout (`shared/icr_capture/{dataset}_{model}/`); emits the dict shape that the existing contrastive trainer expects (matches `PreloadedActivationDataset.__getitem__`).
- New model class `LogprobAttnReconProgressiveCompressor` in `activation_research/model.py`. Carries both F and K auxiliary decoder heads in one nn.Module; `forward(x)` returns `z` only (inference path unchanged).
- New trainer `train_contrastive_logprob_attn_recon` in `activation_research/training.py`. Loss assembly extends `train_contrastive_logprob_recon` by one term per active attention direction.
- New collate fn `_contrastive_collate_with_logprob_attn` in `activation_research/training.py`.
- New config file `configs/methods/contrastive_logprob_attn_recon.json`.
- Unit tests: dataset round-trip on synthetic memmap data, model forward/recon shapes, trainer smoke test on toy data.

### Out of scope (follow-up PRs)
- Wiring into `scripts/run_experiment.py` method dispatch (separate PR once #72-format data is available on a real dataset).
- `attn_target='coarse'` and `attn_target='full'` decoder heads — this PR scaffolds the dispatch but only `'stats'` is implemented and tested (the only honestly feasible variant from `(B, 512)` pooled vector — see issue #75 risk #4).
- Real training runs on HotpotQA + PopQA — gated on the actual memmap captures existing (#72 produces the writer; capture runs are separate).
- Theoretical justification entry §5K in `THEORETICAL_JUSTIFICATION.md` (single-line cross-reference is added; full mechanism writeup is a paper-writing task, not a code change).

## 2. Dependencies

- Issue #72 merged to main (writer + memmap layout exist) ✓ — PR #73 merged 2026-05-17.
- Issue #70 merged (ICR probe scaffolding, `ICRDataset` reference for the memmap reader pattern) ✓ — PR #74 merged 2026-05-17.

No data dependency for this PR: unit tests synthesize memmap files of the right shape. **Real training is gated on actual capture data existing for HotpotQA + PopQA in the #72 layout**, which is a separate operational step.

## 3. Data layout (consumed)

The new dataset class reads from the Issue #72 capture layout at `shared/icr_capture/{dataset}_{model_slug}/`:

| File | Shape | dtype | Purpose |
|---|---|---|---|
| `config.json` | — | json | `num_layers`, `hidden_dim`, `r_max`, `max_response_len`, `max_prompt_len`, `n_samples`, `response_logprobs_top_k` |
| `meta.jsonl` | one line per sample | json | `sample_index`, `prompt_hash`, `hallucinated`, `prompt_len`, `response_len` |
| `response_activations.npy` | `(N, num_layers+1, max_response_len, hidden_dim)` | fp16 | per-token response activations at each layer |
| `response_attention.npy` | `(N, num_layers, r_max, r_max)` | fp16 | head-averaged response-to-response attention per layer |
| `response_token_logprobs.npy` | `(N, max_response_len)` | fp32 (NaN-padded) | per-token generated logprobs |
| `response_topk_token_ids.npy` | `(N, max_response_len, top_k)` | int32 (-1 padded) | top-K alternative tokens |
| `response_topk_logprobs.npy` | `(N, max_response_len, top_k)` | fp32 (NaN-padded) | top-K alternative logprobs |
| `response_token_ids.npy` | `(N, max_response_len)` | int32 (-1 padded) | generated token IDs |
| `response_len.npy` | `(N,)` | int32 | actual response length per sample |
| `prompt_len.npy` | `(N,)` | int32 | actual prompt length per sample |

This matches the documented layout in `activation_logging/inference_capture_writer.py` and the reader pattern in `activation_research/icr_dataset.py::_MemmapMode`.

## 4. New dataset class

### 4.1 File: `activation_research/memmap_contrastive_dataset.py`

```python
class MemmapContrastiveDataset(torch.utils.data.Dataset):
    """Memmap-backed contrastive dataset — same per-item dict shape as
    activation_logging.activation_parser.PreloadedActivationDataset.

    Reads Issue #72 capture layout from `capture_dir`; emits a dict per
    __getitem__ that the existing contrastive trainer consumes verbatim.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        # Split (same semantics as ICRDataset)
        split: Literal["train", "val", "test", "all"] = "train",
        val_fraction: float | None = None,
        random_seed: int = 42,
        # Contrastive view sampling (same semantics as PreloadedActivationDataset)
        num_views: int = 2,
        relevant_layers: list[int] | None = None,
        fixed_layer: int | None = None,
        view_sampling_with_replacement: bool = False,
        # Logprob recon (Mechanism F) — optional emit
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        pad_length: int | None = None,  # defaults to config.max_response_len
        # Attention recon (Mechanism K) — optional emit
        include_response_attention: bool = False,
        attention_summary: Literal["stats"] = "stats",
        attention_target_layer_offset_forward: int | None = None,
        attention_target_layer_offset_backward: int | None = None,
    ): ...
```

### 4.2 Per-item dict contract

Returned by `__getitem__(idx)`:

**Always:**
- `hashkey: str` — `prompt_hash` from meta.jsonl
- `halu: torch.Tensor` scalar float32 — 1.0 if hallucinated else 0.0
- `views_activations: torch.Tensor` (K, T, H) fp32 — K layer views sampled from `relevant_layers`
- `view_indices: torch.Tensor` (K,) long — model layer indices for each view
- `input_length: int` — `prompt_len` from meta.jsonl (parity with PreloadedActivationDataset, for masking)

**When `include_response_logprobs=True`:**
- `response_token_ids: torch.Tensor` (pad_length,) int32 (-1 padded)
- `response_token_logprobs: torch.Tensor` (pad_length,) fp32 (NaN-padded)
- `response_topk_ids: torch.Tensor` (pad_length, top_k) int32 (-1 padded)
- `response_topk_logprobs: torch.Tensor` (pad_length, top_k) fp32 (NaN-padded)
- `response_token_mask: torch.Tensor` (pad_length,) bool — `True` where token is real

**When `include_response_attention=True`:**
- `attention_forward: torch.Tensor` (K, num_stat_features) fp32 — emitted iff `attention_target_layer_offset_forward is not None`. One row per view; per-view target layer = `view_indices[k] + offset_forward` clamped to `[0, num_layers-1]`. Out-of-range views get NaN rows; the trainer's variance-suppression handles them.
- `attention_backward: torch.Tensor` (K, num_stat_features) fp32 — same semantics, target layer = `view_indices[k] - offset_backward`.

`num_stat_features = 3` for `attention_summary='stats'` (entropy, focal_frac, self_mass per layer; see §4.4). `coarse` and `full` summaries are reserved API but not implemented in this PR — passing them raises `NotImplementedError`.

### 4.3 View sampling

Identical contract to `PreloadedActivationDataset._select_view_indices`:
- If `fixed_layer` is set, the first view is `fixed_layer`; remaining views sampled (with or without replacement) from `relevant_layers \ {fixed_layer}`.
- Else, `num_views` views sampled (with or without replacement) from `relevant_layers`.

### 4.4 Attention summary stats (per layer ℓ, head-averaged input is the `(r_max, r_max)` block)

Three scalars per layer; computed per call (cheap — `r_max=64`, ~4K ops per scalar):

```
entropy_ℓ      = mean over query rows of -Σ_k a · log(a + ε)   over key positions
focal_frac_ℓ   = mean over query rows of max_k a              (peak mass per row)
self_mass_ℓ    = mean over query rows of a[k=q]               (self-attention diagonal)
```

Stats are masked by `response_len`: rows beyond `response_len` are excluded from the mean. If a sample's `response_len == 0` the stats row is NaN-filled (the variance threshold in the trainer will suppress those samples from the recon gradient — same pattern as logprob recon).

### 4.5 Memory model

`response_activations.npy` is `(N, L+1, T_max, H)` — for Qwen3-8B (L=36, T_max=256, H=4096, fp16) that's ~76 MB/sample. We **memmap, not preload** — the file may be many GB. `__getitem__` copies the `(K, T_max, H)` slice for the K sampled views into a contiguous fp32 tensor before returning. No persistent RAM growth across workers.

Comparison to `PreloadedActivationDataset`:
- That class preloads everything into RAM (`cache` numpy array) and slices.
- The memmap variant trades RAM for per-iter file I/O. On Linux with OS page cache this is comparable to preloaded for hot data; on Windows or first-pass training it's slower. The trainer is GPU-bound either way, so the I/O cost is masked.

### 4.6 Split logic

Reuse `_make_split_indices` from `activation_research/icr_dataset.py`:
- Same two-stage stratified split.
- `split='all'` exposes every sample (for separate-test-cell capture directories).
- Same `val_fraction` and `random_seed` semantics.

Import the helper directly rather than duplicating it.

## 5. New model class

### 5.1 File: `activation_research/model.py` (append)

```python
class LogprobAttnReconProgressiveCompressor(nn.Module):
    """Mechanism F (logprob recon) + Mechanism K (attention recon) combined.

    Loss assembled by train_contrastive_logprob_attn_recon:
        L = L_SupCon(z)
          + λ_lp · L_recon_logprob(g_lp(z), ℓ)
          + Σ_d λ_attn · L_recon_attn(g_attn^d(z), A_d)

    Inference: forward(x) returns z; both decoders discarded.
    """

    def __init__(
        self, *,
        input_dim: int = 4096,
        final_dim: int = 512,
        dropout: float = 0.1,
        input_dropout: float = 0.2,
        normalize_input: bool = False,
        # F head
        recon_seq_len: int = 64,
        recon_hidden_dim: int = 256,
        recon_lambda: float = 1.0,
        logprob_var_threshold: float = 1e-4,
        # K head
        attn_direction: Literal["forward", "backward", "both", "none"] = "backward",
        attn_offset_k: int = 4,
        attn_target: Literal["stats", "coarse", "full"] = "stats",
        attn_num_stat_features: int = 3,
        attn_recon_hidden_dim: int = 256,
        attn_recon_lambda: float = 1.0,
        attn_var_threshold: float = 1e-5,
    ): ...

    def forward(self, x):                            # → z (B, final_dim)
    def forward_with_recon(self, x):                 # → (z, lp_pred, {direction: attn_pred})
    def recon_loss_lp(self, lp_pred, lp_target):     # MSE + variance suppression
    def recon_loss_attn(self, attn_pred, attn_target):  # MSE + variance suppression
```

### 5.2 Internal structure

- `self.encoder = ProgressiveCompressor(...)` — same as existing.
- `self.lp_decoder = nn.Sequential(Linear(final_dim, recon_hidden_dim), GELU, Linear(recon_hidden_dim, recon_seq_len))` — mirrors `LogprobReconProgressiveCompressor`.
- `self.attn_decoders = nn.ModuleDict({direction: nn.Sequential(Linear(final_dim, attn_recon_hidden_dim), GELU, Linear(attn_recon_hidden_dim, num_stat_features))})`
  - Keys are subset of `{"forward", "backward"}` depending on `attn_direction`.
  - `attn_direction="none"` makes attn_decoders empty (degenerates to logprob-only, equivalent to `LogprobReconProgressiveCompressor`).

### 5.3 `forward_with_recon(x)` return shape

Returns `(z, lp_pred, attn_pred_by_direction)`:
- `z: (B, final_dim)`
- `lp_pred: (B, recon_seq_len)` — or `None` if `recon_lambda == 0`
- `attn_pred_by_direction: dict[str, Tensor]` — keys are `"forward"` and/or `"backward"`; each value is `(B, num_stat_features)`. Empty dict if `attn_direction == "none"` or `attn_recon_lambda == 0`.

### 5.4 Loss methods

`recon_loss_lp(lp_pred, lp_target)` — copied verbatim from existing `LogprobReconProgressiveCompressor.recon_loss` (NaN-mask + variance-threshold suppression + linear-interpolation resample to `recon_seq_len`, MSE).

`recon_loss_attn(attn_pred, attn_target)` — same pattern:
- NaN-mask `attn_target` (rows where `response_len == 0` are NaN).
- Compute batch variance; if below `attn_var_threshold`, return zero + `{"suppressed": True}`.
- Else: MSE between `attn_pred` and `attn_target`, ignoring NaN rows.
- No interpolation needed — shapes match by construction.

## 6. New trainer + collate

### 6.1 File: `activation_research/training.py` (append)

```python
def _contrastive_collate_with_logprob_attn(batch):
    """Collate that pads/stacks logprob + attention summary fields alongside
    the standard contrastive fields.

    Same behaviour as _contrastive_collate_with_logprob for the logprob
    branch.  Adds:
        attention_forward    (B, K, num_stat_features)  — when present in batch
        attention_backward   (B, K, num_stat_features)  — when present in batch

    Either field is omitted from the output if not present in all items
    (trainer falls back to no-K-recon for that direction).
    """


def train_contrastive_logprob_attn_recon(
    model, train_dataset, test_dataset=None,
    *,
    epochs=10, batch_size=512, lr=1e-6, temperature=0.07,
    device="cuda", num_workers=16, sub_batch_size=64,
    checkpoint_dir="checkpoints", save_every=1, resume_from=None,
    persistent_workers=True, cleanup_legacy_checkpoints=True,
    snapshot_every=0, snapshot_keep_last=5,
    use_labels=False, ignore_label=-1,
    same_sample_weight=1.0, same_class_weight=1.0,
    balanced_sampling=False,
    recon_lambda=None,           # override model.recon_lambda
    attn_recon_lambda=None,      # override model.attn_recon_lambda
    use_infinite_index_stream=False,
    grad_clip_norm=None,
):
    """Train LogprobAttnReconProgressiveCompressor with both auxes.

    Loss per step:
        L = L_SupCon(z) + λ_lp · L_lp(g_lp(z), ℓ) + Σ_d λ_attn · L_attn(g_d(z), A_d)

    Mirrors train_contrastive_logprob_recon's skeleton (microbatch
    buffering, atomic checkpoint save, snapshot pruning, optional infinite
    index stream).  Extends only the per-batch loss-assembly block by
    adding one MSE term per active attention direction.
    """
```

### 6.2 Loss-assembly logic (inside trainer)

```
1. Standard buffered microbatch accumulation (unchanged from F trainer).
2. When buffer full:
     z_flat, lp_pred_flat, attn_pred_by_dir = model.forward_with_recon(x_flat)
     supcon = SupConLoss(z_views, ...)
     lp_loss = model.recon_loss_lp(lp_pred_flat, lp_target_expanded)
              if lp_pred_flat is not None and λ_lp > 0 else 0
     attn_loss = sum(model.recon_loss_attn(attn_pred_by_dir[d], target_d)
                     for d in active_directions
                     if λ_attn > 0)
     total = supcon + λ_lp · lp_loss + λ_attn · attn_loss
     ... backward, step ...
3. Per-direction diagnostics logged (suppressed flag, variance) alongside
   the existing logprob diagnostics.
```

### 6.3 Target reshape for attention

The dataset emits `attention_{direction}: (B, K, num_stat_features)`. The trainer flattens views to `(B*K, num_stat_features)` to match the flattened `z_flat` shape, identically to how `views_activations` and `logprob` are handled today.

## 7. Config file

`configs/methods/contrastive_logprob_attn_recon.json`:

```json
{
  "method_name": "contrastive_logprob_attn_recon",
  "model_class": "LogprobAttnReconProgressiveCompressor",
  "trainer": "train_contrastive_logprob_attn_recon",
  "model_params": {
    "input_dim": 4096,
    "final_dim": 512,
    "dropout": 0.1,
    "input_dropout": 0.3,
    "normalize_input": true,
    "recon_seq_len": 64,
    "recon_hidden_dim": 256,
    "recon_lambda": 1.0,
    "logprob_var_threshold": 1e-4,
    "attn_direction": "backward",
    "attn_offset_k": 4,
    "attn_target": "stats",
    "attn_num_stat_features": 3,
    "attn_recon_hidden_dim": 256,
    "attn_recon_lambda": 1.0,
    "attn_var_threshold": 1e-5
  },
  "training_params": {
    "epochs": 150,
    "batch_size": 512,
    "sub_batch_size": 64,
    "lr": 1e-6,
    "temperature": 0.07,
    "use_labels": true,
    "balanced_sampling": true
  },
  "dataset_params": {
    "num_views": 2,
    "include_response_logprobs": true,
    "include_response_attention": true,
    "attention_target_layer_offset_backward": 4,
    "attention_target_layer_offset_forward": null
  }
}
```

(The dispatch wiring that reads this config and constructs model/trainer/dataset lives in `scripts/run_experiment.py` — out of scope for this PR.)

## 8. Unit tests

### 8.1 `tests/test_memmap_contrastive_dataset.py`

- **Round-trip**: synthesize a fake memmap capture (5 samples, num_layers=4, hidden_dim=8, r_max=4, max_response_len=6) with `_make_fake_capture()` helper. Open dataset with `include_response_logprobs=True, include_response_attention=True, attention_target_layer_offset_backward=2`. Assert:
  - `__len__` returns 5 (after split='all').
  - `__getitem__(0)['views_activations'].shape == (num_views, max_response_len, hidden_dim)`.
  - `__getitem__(0)['view_indices'].shape == (num_views,)`.
  - `__getitem__(0)['response_token_logprobs'].shape == (max_response_len,)`.
  - `__getitem__(0)['attention_backward'].shape == (num_views, 3)`.
  - NaN locations in `response_token_logprobs` correspond to padding positions (past `response_len`).
  - Out-of-range attention target layers (e.g. for layer 0 backward) yield NaN stat rows.
- **Splits**: assert `split='train'`, `split='val'`, `split='test'` partition the sample set with no overlap and stratify on `hallucinated`.
- **View determinism**: same `random_seed` produces the same `view_indices` across two dataset constructions.
- **Field omission**: with `include_response_logprobs=False`, the dict does not contain `response_token_logprobs`. With `include_response_attention=False`, no `attention_forward` / `attention_backward` keys.

### 8.2 `tests/test_logprob_attn_recon_model.py`

- **`forward(x)`** with `x.shape == (2, 6, 8)` returns `(2, final_dim)` — inference path unchanged.
- **`forward_with_recon(x)`** with `attn_direction='backward'` returns 3-tuple `(z, lp_pred, {'backward': attn_pred})`.
- **`forward_with_recon(x)`** with `attn_direction='both'` returns `{'forward': ..., 'backward': ...}` (two decoders).
- **`forward_with_recon(x)`** with `attn_direction='none'` returns empty attn dict.
- **`recon_loss_lp`** matches `LogprobReconProgressiveCompressor.recon_loss` numerically on synthetic input (refactored from same code).
- **`recon_loss_attn`** returns finite MSE on non-degenerate target; returns zero + `suppressed=True` when batch variance is below threshold.
- **Inference equivalence**: same `final_proj` weights → same `z` between `LogprobAttnReconProgressiveCompressor.forward(x)` and a plain `ProgressiveCompressor.forward(x)`.

### 8.3 `tests/test_logprob_attn_recon_trainer.py`

- Smoke test: 50 synthetic samples, 2 epochs, batch_size=8, sub_batch_size=4, `attn_direction='backward'`, K1 stats. Loss decreases. Checkpoint written. No NaN gradients. No errors when both auxes active simultaneously.
- Loss decomposition: verify `total_loss == supcon + λ_lp · lp_loss + Σ λ_attn · attn_loss` exactly on one step (read from training diagnostics dict).
- Falls back to SupCon + F-only when batch lacks attention fields (degrade path).

## 9. Files to create / modify

| File | Action | Purpose |
|---|---|---|
| `activation_research/memmap_contrastive_dataset.py` | Create | `MemmapContrastiveDataset` class |
| `activation_research/model.py` | Modify | Add `LogprobAttnReconProgressiveCompressor` |
| `activation_research/training.py` | Modify | Add `train_contrastive_logprob_attn_recon` + `_contrastive_collate_with_logprob_attn` |
| `configs/methods/contrastive_logprob_attn_recon.json` | Create | Default method config |
| `tests/test_memmap_contrastive_dataset.py` | Create | Dataset unit tests + synthetic capture helper |
| `tests/test_logprob_attn_recon_model.py` | Create | Model unit tests |
| `tests/test_logprob_attn_recon_trainer.py` | Create | Trainer smoke test |
| `specs/issue_75_combined_logprob_attn_recon.md` | Create (this file) | Implementation spec |

## 10. Out of scope follow-ups

After this PR lands and #72-format captures exist for HotpotQA + PopQA:

1. **Wire into `scripts/run_experiment.py`** — add `run_contrastive_logprob_attn_recon(...)` similar to `run_contrastive_logprob_recon` (or reuse via dispatch table — depends on existing dispatch shape).
2. **Add `"contrastive_logprob_attn_recon"` to `configs/experiments/baseline_comparison_*.json`** for HotpotQA + PopQA on both models, 5 seeds.
3. **Implement K2 (coarse 8×8 binned) and K3 (full r_max × r_max) variants** if K1 stats shows meaningful F+K > F separation in the primary ablation.
4. **`THEORETICAL_JUSTIFICATION.md` §5K writeup** — narrative justification for the new mechanism in paper-ready prose; references this spec for code.

## 11. Risk register (issue-local)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Memmap-backed dataset slower than preloaded zarr on small datasets (full-dataset I/O per epoch) | medium | Trainer is GPU-bound at the recommended batch sizes; OS page cache covers hot data after first epoch. If measurable, add a `preload=True` flag in a follow-up PR. |
| Attention summary stats degenerate (variance below threshold on most samples) | low-medium | Variance suppression returns zero loss without poisoning gradients; the diagnostic logs flag suppression rate. If suppression is too aggressive, drop threshold in config. |
| `attn_target='coarse'` / `'full'` raise `NotImplementedError` if user accidentally configures them | low | Explicit error message points at this spec § for the rationale. |
| Out-of-range layer offsets (e.g. `view_indices[k]=0` with `backward_offset=4`) | high | Dataset emits NaN stat rows; variance suppression handles them. Trainer logs suppression rate per direction. |
| F+K-fwd shows no improvement over F-only (the "overlap with SupCon" hypothesis materializes) | medium | Pre-registered decision rule in #75 — record finding and write up in §3 of the paper. Not a code change. |
