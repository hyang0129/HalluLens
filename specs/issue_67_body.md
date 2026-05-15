## Background

The current headline method (`contrastive_logprob_recon`) outperforms SAPLMA. But the gap is attributable to **three** simultaneous design choices, not one:

1. Asymmetric supervised contrastive loss (vs. SAPLMA's BCE)
2. Layer pairs as views (vs. SAPLMA's single layer)
3. Logprob reconstruction auxiliary loss (vs. SAPLMA's no auxiliary)

For the paper's framing — *contrastive learning method for hallucination detection* — we need to isolate the contribution of (1) the **contrastive objective itself** as distinct from the **recon trick**. The cleanest way to do that is to add the recon auxiliary to SAPLMA and see whether SAPLMA's gap with the full method shrinks.

A reviewer will ask: *"What if you just gave SAPLMA the same recon auxiliary? Would it match your contrastive method?"* We currently can't answer.

## Hypothesis

The full method's gain over SAPLMA decomposes into:

- **Contribution of contrastive structure (layer-pair InfoNCE supervision)**: full method AUROC − SAPLMA+recon AUROC
- **Contribution of recon auxiliary alone**: SAPLMA+recon AUROC − SAPLMA AUROC

If `SAPLMA+recon ≈ full method`, the contrastive structure adds little and the paper's framing must pivot toward "logprob-recon as an auxiliary regression target for hallucination probing" — a real but narrower contribution. If `SAPLMA+recon < full method` by a meaningful margin, the contrastive structure carries genuine independent contribution.

Either outcome usefully constrains §3 of the paper draft.

## Architecture

Add a `SaplmaWithReconHead` model class in `activation_research/model.py`:

- **Body**: rebuild the existing `SimpleHaluClassifier` `nn.Sequential` ([model.py:284-301](activation_research/model.py#L284-L301)) split into two pieces so the penultimate (dim 512) representation is addressable as `z`:
  - `self.body = nn.Sequential(Linear(4096, 2048), ReLU, Dropout, Linear(2048, 1024), ReLU, Dropout, Linear(1024, 512), ReLU, Dropout)` → outputs `(B, 512)`
  - `self.head = nn.Linear(512, 1)`
- **Recon decoder**: same shape as `LogprobReconProgressiveCompressor` ([model.py:172-176](activation_research/model.py#L172-L176)) — `Linear(512, recon_hidden_dim) → GELU → Linear(recon_hidden_dim, recon_seq_len)`.
- **forward(x)**: returns `sigmoid(head(body(x[:, -1, :])))` — identical inference path to plain SAPLMA. The decoder is discarded at inference.
- **forward_with_recon(x)**: returns `(sigmoid_logit, z, logprob_pred)`.
- **recon_loss(logprob_pred, logprob_target)**: copied verbatim from `LogprobReconProgressiveCompressor.recon_loss` ([model.py:201-242](activation_research/model.py#L201-L242)) — NaN-mask + variance-threshold suppression + linear-interpolation resample to `recon_seq_len`, MSE.

Training objective: `L = L_BCE(sigmoid_logit, y_label) + λ · L_recon(g(z), per-token logprobs)`.

## Data path

**No new dataset class needed.** `run_saplma` currently calls `train_ds.get_single_layer_dataset(probe_layer)` ([run_experiment.py:553-554](scripts/run_experiment.py#L553-L554)), which strips logprob fields. For SAPLMA+recon we instead build the train/test datasets directly with:

```python
ap.get_dataset(
    "train",
    relevant_layers=relevant_layers,
    num_views=1,
    fixed_layer=<probe_layer_pos>,   # positional index into relevant_layers
    include_response_logprobs=True,
    response_logprobs_top_k=20,
    preload=True,
)
```

`PreloadedActivationDataset._select_view_indices` with `num_views=1, fixed_layer=pos` deterministically returns `[pos]`, so `views_activations` is shape `(1, T, H)` — identical to what `SingleLayerDataset` returns to `LinearProbeTrainer`. The `_get_logprobs` branch ([activation_parser.py:641-642](activation_logging/activation_parser.py#L641-L642)) attaches `response_token_logprobs` (NaN-padded `(pad_length,)`) to each item.

## Trainer

New `SaplmaReconTrainer(LinearProbeTrainer)` in `activation_research/trainer.py`. Only override `training_step`:

- Get `x = batch["views_activations"]` shape `(B, 1, T, H)`, squeeze view dim → `(B, T, H)`.
- Get `labels = batch["halu"].float().view(-1, 1)`.
- Call `sigmoid_logit, z, logprob_pred = model.forward_with_recon(x)`.
- `bce = BCELoss(sigmoid_logit, labels)`.
- Get `logprob = batch["response_token_logprobs"]` shape `(B, pad_length)`. Replicate the NaN→row-mean fill from [training.py:956-960](activation_research/training.py#L956-L960) (per-row `nanmean`, mask-fill).
- `recon, _diag = model.recon_loss(logprob_pred, logprob)`.
- Return `bce + λ · recon`, with `acc` metric on the sigmoid_logit threshold.

`validate()` is inherited unchanged — inference uses only the sigmoid logit, so AUROC reporting matches plain SAPLMA exactly.

New `SaplmaReconTrainerConfig(LinearProbeTrainerConfig)` adds one field: `recon_lambda: float = 1.0`.

## Ablation matrix

| Run | Method | Datasets | Models | Seeds |
|-----|--------|----------|--------|-------|
| C (this issue) | `saplma_logprob_recon` (new) | HotpotQA + PopQA | Llama-3.1-8B + Qwen3-8B | 0, 1, 2, 3, 4 |
| Reference (complete) | `saplma` | — | — | — |
| Reference (complete) | `contrastive_logprob_recon` | — | — | — |

Seeds match the existing `training_seeds` in the four `baseline_comparison_*` configs so the new ablation is directly comparable to the cached `saplma` and `contrastive_logprob_recon` results.

Total new runs: 2 datasets × 2 models × 5 seeds = **20 runs** on cached activations. No new GPU inference.

## Code changes required

- [ ] Implement `SaplmaWithReconHead` in `activation_research/model.py` (body / head split, recon head, `forward_with_recon`, `recon_loss`)
- [ ] Implement `SaplmaReconTrainerConfig` + `SaplmaReconTrainer` in `activation_research/trainer.py`
- [ ] Add `run_saplma_logprob_recon` to `scripts/run_experiment.py` and wire into the routine dispatch (around [run_experiment.py:2143](scripts/run_experiment.py#L2143))
- [ ] Create `configs/methods/saplma_logprob_recon.json` (mirrors `saplma.json` plus `recon_seq_len: 64`, `recon_hidden_dim: 256`, `recon_lambda: 1.0`, `logprob_var_threshold: 1e-4`)
- [ ] Unit test (`tests/test_saplma_recon.py`): forward returns `(B, 1)` sigmoid in `[0, 1]`; `forward_with_recon` returns 3-tuple with finite `recon_loss`; classification inference path is identical to a plain `SimpleHaluClassifier` with the same body+head weights

## Config additions

- [ ] Add `"saplma_logprob_recon"` to method list in `configs/experiments/baseline_comparison_hotpotqa.json`
- [ ] Same for `baseline_comparison_hotpotqa_qwen3.json`
- [ ] Same for `baseline_comparison_popqa.json`
- [ ] Same for `baseline_comparison_popqa_qwen3.json`

## Dispatch + verify

- [ ] Dispatch via `scripts/gpu_dispatch.py` with `resume=True`
- [ ] `python scripts/results_table.py` shows 20/20 cells complete for `saplma_logprob_recon`
- [ ] AUROC numbers reported in a comment alongside `saplma` and `contrastive_logprob_recon` for direct three-way comparison

## Decision rules

- **SAPLMA+recon ≈ full method (within seed CI)** → contrastive structure is decorative; paper's contribution narrows to "logprob-recon as auxiliary target for hallucination probing." Pivot §3 framing accordingly. SEP comparison becomes more central; theory section emphasizes the recon target choice over the contrastive structure.
- **SAPLMA+recon noticeably > SAPLMA but < full method** → recon helps any probe, AND contrastive structure adds independent value on top. This is the *best* outcome for the paper — we get a clean attribution. §3 theory frames the contribution as the *interaction* of supervised contrastive + recon.
- **SAPLMA+recon ≈ SAPLMA (recon doesn't help)** → recon's value is contingent on the contrastive setup. The two losses interact non-trivially; the §3 theory needs to address this interaction explicitly rather than treating recon as a general-purpose auxiliary.

## Out of scope for this issue

- The complementary "supervised contrastive WITHOUT recon" ablation — tracked in #66.
- Linear probe + recon variant — less informative than SAPLMA+recon (SAPLMA is the stronger reference baseline).
- SmolLM3 — out of scope per [PAPER_ROADMAP.md](PAPER_ROADMAP.md) §5.

## Related

- Companion ablation: #66 (supervised contrastive WITHOUT logprob recon).
- Discussed in conversation 2026-05-15 around the §3 theory section of [paper/theory_outline.md](paper/theory_outline.md).
- Connects to [PAPER_ROADMAP.md](PAPER_ROADMAP.md) §4 item 2 (SAPLMA baseline rationale) and item 7 (loss decomposition).
