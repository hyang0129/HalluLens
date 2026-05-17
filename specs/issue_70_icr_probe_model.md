# Implementation Spec: ICR Probe — Score Computation, Probe Model, Training & Evaluation (Issue #70)

**Goal:** Reproduce the ICR Probe baseline (Zhang et al., ACL 2025; [arXiv:2507.16488](https://arxiv.org/abs/2507.16488); code at [github.com/XavierZhang2002/ICR_Probe](https://github.com/XavierZhang2002/ICR_Probe)) in our experiment framework, evaluated head-to-head against our contrastive method on HotpotQA first (Phase 1), with optional rollout to the remaining 5 datasets (Phase 2).

This issue assumes Issue #69 has produced `attention.zarr` next to each `activations.zarr`, with the reader API and dataset wrapper described there.

---

## 1. Scope

### In scope
- Computation of the ICR Score (`ICR_ℓ`) from paired `(Δh_ℓ, attn_ℓ)` per sample.
- Probe MLP matching the paper's architecture.
- Trainer that fits into the existing `activation_research/trainer.py` framework.
- Evaluation producing the same artifacts as our other baselines (`eval_metrics.json`, `predictions.csv`).
- Experiment configs at `configs/experiments/baseline_comparison_hotpotqa.json` (and 5 more for Phase 2) updated to include `icr_probe` as a method.

### Out of scope
- Attention map generation / storage (Issue #69).
- Any change to the contrastive method, linear probe, or SAPLMA.
- Cross-dataset transfer of the ICR probe (Issue #62 may pick this up later; not in this issue).

### Phases
- **Phase 1 (this issue, mandatory):** HotpotQA, both models, 5 seeds.
- **Phase 2 (this issue, conditional):** the remaining 5 datasets, gated on Phase 1 numbers being within ~0.03 AUROC of the paper's reported ~0.84 on HotpotQA (Gemma-2). If Phase 1 reproduces the trend, roll out to Phase 2. If Phase 1 is dramatically off, **stop and diagnose** before scaling.

---

## 2. Dependencies

This issue **cannot start** until Issue #69 has:
1. Resolved every §2 open question and written `notes/icr_probe_paper_notes.md`.
2. Passed the §4.5 numerical-equivalence validation on at least HotpotQA for both models.
3. Generated `attention.zarr` for HotpotQA test + train, both models.

If the implementer of this issue is the same person, they have the paper notes in hand. If different, this issue's first step is reading those notes.

---

## 3. ICR Score: computation specification

This section's defaults are placeholders pending the answers Issue #69 produced for §2 of that spec. **Before implementing, replace the placeholders with the verified-from-paper values** and update this spec in the PR.

### 3.1 Per-layer score (single-module variant)

For each sample `s`, layer `ℓ`, query position `q`:

```
Δh = h_ℓ[q] - h_{ℓ-1}[q]                                 # (H,)
W_U = model.lm_head.weight                                # (V, H)  unembedding

# §2 answer 5 will pin down EXACTLY how the projection is computed.
# Placeholder (variant A): project Δh onto vocab logits, restrict to
# tokens at the top-k attention-weighted KEY positions of attn_ℓ[q].

attn_q = attn_ℓ[q]                                        # (T,) head-averaged
topk_keys = argsort(attn_q)[-K:]                          # K key positions
topk_token_ids = sequence_token_ids[topk_keys]            # the actual token IDs at those positions

# Δh's projection onto those token directions:
proj_logits = W_U[topk_token_ids] @ Δh                    # (K,)
Proj_ℓ = softmax(proj_logits)                             # distribution over K tokens
Attn_ℓ_topk = softmax(attn_q[topk_keys])                  # distribution over the same K positions

ICR_ℓ = JSD(Proj_ℓ, Attn_ℓ_topk)                          # scalar
```

`K` is a paper hyperparameter (§2 answer 3 of Issue #69).

Per-sample feature vector for the probe:
```
icr_vec = [ICR_0, ICR_1, ..., ICR_{L-1}]                  # (L,)
```

### 3.2 Per-layer + per-module variant (if §2 answer 2 says modules are split)

If the paper produces separate scores for the attention sublayer's update and the MLP sublayer's update:
```
Δh_attn_ℓ = h^post-attn_ℓ - h_{ℓ-1}      # not directly cached!
Δh_mlp_ℓ  = h_ℓ - h^post-attn_ℓ
```

Note that we cache `h_{ℓ-1}` and `h_ℓ` (the residual stream *between* blocks), but NOT the intermediate `h^post-attn_ℓ`. To get this, Issue #69 must additionally store `Δh_attn` per layer per query position (added to the `mlp_updates` array under a different name — clarify with Issue #69 author).

If the paper uses a single update per layer (the residual-stream delta), Issue #69 does not need to store anything extra; we compute everything from `h_{ℓ-1}` and `h_ℓ` already in `activations.zarr`.

**The implementer must clarify which variant applies based on `notes/icr_probe_paper_notes.md` §2 answer 2 before writing any code.**

### 3.3 JSD definition

Standard symmetric KL:
```
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)   where M = 0.5*(P+Q)
```
Use `scipy.spatial.distance.jensenshannon` (squared if the paper uses JSD; un-squared if the paper uses JS distance — verify).

### 3.4 Numerical-stability gotchas

- Pad keys in `attn_q` beyond `seq_len` are zero. Mask them before `argsort` (otherwise top-k can include padding positions, yielding garbage token IDs).
- If `seq_len < K`, fall back to using all available keys.
- `Proj_ℓ` and `Attn_ℓ_topk` must use the same key ordering. Sort `topk_keys` before indexing both to avoid silent misalignment.

---

## 4. Probe model

### 4.1 Architecture (matches paper)

```
class ICRProbe(nn.Module):
    """ICR Probe MLP from Zhang et al. ACL 2025.

    Input:  (B, L)              # per-layer ICR scores
    Output: (B,)                # logit; sigmoid → P(hallucinated)

    Architecture: L → 128 → 64 → 32 → 1   (ReLU between hidden layers).
    Paper reports <16K params total when L ≤ 42; for Llama-3.1-8B (L=32 blocks
    or 33 including embeddings) this is ~12K params.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_layers, 128), nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):  # x: (B, L)
        return self.net(x).squeeze(-1)
```

If §2 answer 2 indicates per-module scores (2L-dim input), parameterize `num_layers` as the effective input dim.

### 4.2 Place in code

New file: `activation_research/icr_probe.py`. Importable from `activation_research/__init__.py`.

---

## 5. Trainer

### 5.1 Pattern to follow

Mirror the structure of [`activation_research/trainer.py`](../activation_research/trainer.py) — specifically the `LinearProbeTrainer` path, since the probe is also a small classifier on a fixed-dim input.

Key differences from `LinearProbeTrainer`:
- Input is the L-dim ICR score vector, not raw activations at one layer.
- Score computation lives in `icr_score.py` (new) and is called by the dataset wrapper.

### 5.2 New file: `activation_research/icr_score.py`

```python
def compute_icr_scores_batch(
    delta_h: torch.Tensor,      # (B, L, Q_max, H)
    attn_probs: torch.Tensor,   # (B, L, Q_max, T_max)
    token_ids: torch.Tensor,    # (B, T_max)         # sequence token IDs
    seq_len: torch.Tensor,      # (B,)
    W_U: torch.Tensor,          # (V, H)             # unembedding matrix
    top_k: int,
    head_aggregation: str = "mean",
) -> torch.Tensor:
    """Compute per-layer ICR scores for a batch.

    Returns: (B, L) tensor of ICR scores (currently assumes Q_max==1).
    If Q_max > 1, returns (B, L, Q_max) and the caller decides aggregation.
    """
```

This is pure tensor algebra; no model loading. Lives on GPU during training but can run on CPU for eval.

### 5.3 Caching ICR scores

Computing ICR is cheap (one matmul against `W_U[topk_ids]` per layer, one JSD per layer), but doing it in every training batch is wasteful when the input data is fixed. Option:
- **Precompute once** per (model, dataset, split) into `icr_scores.zarr` shape `(N, L)` and load directly during training.
- Place: `shared/<dataset>_<model_slug>/icr_scores.zarr` next to `attention.zarr`.

Add a `scripts/precompute_icr_scores.py` driver. ~minutes per (model, dataset).

### 5.4 New file: `activation_research/icr_trainer.py`

```python
class ICRProbeTrainer:
    """Trainer for ICRProbe over precomputed icr_scores.zarr.

    Mirrors LinearProbeTrainer's CSV / checkpoint outputs:
      - linear_probe_last.pt   (state_dict + cfg)
      - eval_metrics.json
      - predictions.csv
    """
```

Train: BCE-with-logits loss, Adam, early stopping on validation AUROC. Hyperparameters from paper §exp; record what was used in PR.

---

## 6. Dataset wrapper

New file: `activation_research/icr_dataset.py` (already created in Issue #69 if it loads raw `(Δh, attn)`; this issue extends it to load precomputed ICR scores).

```python
class ICRScoreDataset(torch.utils.data.Dataset):
    """Reads icr_scores.zarr; yields {'hashkey', 'halu', 'icr_scores' (L,)}."""
```

---

## 7. Integration with experiment configs

### 7.1 Add `icr_probe` as a method

Update `configs/experiments/baseline_comparison_hotpotqa.json` (and `_qwen3` variant) to include:
```json
{
  "name": "icr_probe",
  "trainer": "icr_probe",
  "model_params": {
    "num_layers": null    // auto-resolve from icr_scores.zarr metadata
  },
  "training_params": {
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "patience": 10
  },
  "score_params": {
    "top_k": null,                       // from paper notes
    "head_aggregation": null,            // from paper notes
    "query_position_rule": null          // from paper notes
  },
  "attention_zarr": "shared/<dataset>_<model_slug>/attention.zarr",
  "icr_scores_zarr": "shared/<dataset>_<model_slug>/icr_scores.zarr"
}
```

### 7.2 Wire the trainer into the dispatch table

In `scripts/run_experiment.py` (or whichever entry point dispatches trainers — implementer to locate), add:
```python
if method == "icr_probe":
    trainer = ICRProbeTrainer(...)
```

### 7.3 Resolve `num_layers` automatically

The probe's input dim depends on whether ICR Probe uses single-module or per-module scores. Read from `icr_scores.zarr/meta/config.json` rather than the experiment config.

---

## 8. Evaluation

### 8.1 Artifacts (must match other baselines so existing aggregation code works)

For each (model, dataset, seed) run:
```
runs/baseline_comparison_<dataset>/<dataset>/icr_probe/seed_<s>/
  config.json
  linear_probe_last.pt        # state_dict + cfg (filename match — see §10)
  eval_metrics.json           # {"test_auroc": ..., "selected_layer": null, ...}
  eval_metrics_extended.json  # AUPRC, ECE, FPR@95, CIs (issue #59 schema)
  predictions.csv             # hashkey, halu, score
```

`selected_layer` is `null` for ICR Probe since the probe consumes all layers simultaneously.

### 8.2 Metrics

Same as every other baseline:
- AUROC (primary)
- AUPRC, ECE, FPR@95, bootstrap 95% CIs (computed by the existing `scripts/recompute_metrics.py` from `predictions.csv` — no new code per issue #59)

---

## 9. Sanity checks (gating, before launching full Phase 1)

**Gate philosophy.** The paper's ~0.84 AUROC on HotpotQA is on **Gemma-2**.
We run on Llama-3.1-8B-Instruct and Qwen3-8B — different model family,
tokenizer, chat template, residual stream geometry. There is no published
number for our models to match. The earlier "within ~0.03 of paper's 0.84"
language was cross-model and conceptually wobbly; it has been redefined to
check the actually-meaningful properties.

The full sanity-check template lives at [`notes/icr_probe_sanity.md`](../notes/icr_probe_sanity.md)
(empty in the PR, filled after the captures complete on Empire AI).
Summary of the gates:

1. **Score distribution sane.** `icr_scores.npy` has no NaN/Inf, values fit
   in `[0, ~1.0]` (JSD bound is ln(2) ≈ 0.693), per-layer mean varies, and
   most layers have non-trivial std.
2. **Train-set discriminability.** At least one layer's raw single-feature
   AUROC on train labels > 0.55, AND ≥25% of layers clear 0.55. Confirms the
   score has signal *before* training the probe.
3. **End-to-end pipeline runs.** Tiny-budget probe (5 epochs) completes
   without errors, writes the expected artifacts (`linear_probe_last.pt`,
   `eval_metrics.json`, `predictions.csv`), and produces test AUROC > 0.55.
4. **Competitive vs other baselines on same data.** After Phase 1's 5 seeds
   land, mean AUROC > 0.55 (robust above chance) AND ICR Probe is positioned
   defensibly against `contrastive` / `saplma` / `linear_probe` / `llmsknow_probe`
   on the same `(model, dataset)`. Mid-pack is acceptable.

**The top_p deviation (PR plan §A.2) is investigated only if gate 3 collapses
to chance.** A below-paper number that's still well above chance is a fine,
publishable result — *not* a trigger for remediation.

Paste sanity-check results into the PR before launching the full Phase 1 sweep.

---

## 10. Filename collision warning

`LinearProbeTrainer` writes `linear_probe_last.pt`. SAPLMA's `SimpleHaluClassifier` uses the same filename via shared trainer. The ICR Probe will collide if the same filename is used in the same directory — but since each method is in its own `<dataset>/<method>/<seed>` subdirectory, there is no collision in practice. Mirror the convention: write `linear_probe_last.pt` (it is a linear-probe-flavored MLP). This keeps `transfer_eval.py` and similar generic loaders working without special-casing ICR Probe.

If the implementer prefers a distinct filename (`icr_probe_last.pt`), they must also update the generic loaders that key on filename (see `specs/issue_62_transfer_matrix.md` §6). Default: reuse `linear_probe_last.pt`, no loader change needed.

---

## 11. Files to create / modify

| File | Action | Purpose |
|---|---|---|
| `activation_research/icr_probe.py` | Create | `ICRProbe` MLP class |
| `activation_research/icr_score.py` | Create | Batched ICR Score computation |
| `activation_research/icr_dataset.py` | Modify (from Issue #69) | Add `ICRScoreDataset` reading precomputed scores |
| `activation_research/icr_trainer.py` | Create | Trainer that fits ICRProbe; emits standard artifacts |
| `scripts/precompute_icr_scores.py` | Create | CLI driver: input attention.zarr + activations.zarr, output icr_scores.zarr |
| `configs/experiments/baseline_comparison_hotpotqa.json` | Modify | Add `icr_probe` method block |
| `configs/experiments/baseline_comparison_hotpotqa_qwen3.json` | Modify | Same |
| `scripts/run_experiment.py` (or trainer dispatch) | Modify | Wire `icr_probe` into dispatch |
| `tests/test_icr_score.py` | Create | Unit tests on synthetic data + one real sample sanity check |
| `tests/test_icr_probe.py` | Create | Probe forward/backward + tiny training loop convergence |
| `notes/icr_probe_sanity.md` | Create | §9 sanity check results |

---

## 12. CLI for `scripts/precompute_icr_scores.py`

```
--attention-zarr    str, required
--activations-zarr  str, required
--model             str, required             # HF model ID, for W_U lookup
--output-zarr       str, required             # icr_scores.zarr path
--top-k             int, default <from paper>
--head-aggregation  str, default <from paper>
--device            str, default "cuda"
--batch-size        int, default 64
--resume            flag
```

Output layout:
```
icr_scores.zarr/
  arrays/
    icr_scores      shape (N, L)  dtype float32
    sample_key      shape (N,)    dtype |S64
  meta/
    index.jsonl
    config.json     # mirror of attention.zarr/meta/config.json + top_k, head_aggregation
```

---

## 13. Phase 1 launch plan

After §9 sanity checks pass:
```bash
# Llama-3.1-8B HotpotQA, 5 seeds
for s in 0 1 2 3 4; do
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa.json \
    --methods icr_probe \
    --seeds $s
done

# Qwen3-8B HotpotQA, 5 seeds
for s in 0 1 2 3 4; do
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa_qwen3.json \
    --methods icr_probe \
    --seeds $s
done
```

(Concrete invocation depends on the actual `run_experiment.py` API — the implementer should mirror how SAPLMA is launched today; see `scripts/run_saplma_hotpotqa_nq.sh`.)

Expected runtime per seed: minutes (the probe is tiny; data is precomputed). 10 seeds total in under an hour of GPU.

---

## 14. Decision gate before Phase 2 rollout

The paper's ~0.84 number was on **Gemma-2**, not on Llama or Qwen3, so it is
not a target for our Phase 1. The decision rests on the redefined §9 gates
applied to our `(model, dataset)` results.

Three branches:
- **Phase 1 passes §9 gates 1–4 on both models** → roll out to Phase 2
  (popqa, mmlu, natural_questions, sciq, searchqa). The reimplementation is
  sound on our regime; scaling out is mechanical.
- **Phase 1 passes gates 1–3 but ICR Probe sits in the lower half vs other
  baselines on both models** → document the finding and proceed with Phase 2
  anyway. That's a publishable result about the method as applied to
  Llama/Qwen3 outputs; no need to delay.
- **Phase 1 fails gate 3 (probe at chance on at least one model)** →
  stop and diagnose. Top suspects: (a) the top_p deviation noted in the PR
  (remediate via a follow-up `scripts/recompute_icr_scores.py`); (b) score
  formula bug not caught by unit tests; (c) data-pipeline alignment between
  `meta.jsonl` indices and `icr_scores.npy` rows.

Document the branch taken in the PR.

---

## 15. Risk register (issue-local)

| Risk | Likelihood | Mitigation |
|---|---|---|
| §3 placeholders chosen wrong despite reading paper | low | Resolved via `notes/icr_probe_paper_notes.md`; §9 sanity checks 1–2 would catch a gross score-formula bug |
| ICR Probe beats our contrastive method on our regime | low-medium | This is the per-issue manifestation of the §11.4 risk in `PAPER_ROADMAP.md`. Decide on data — a competitive baseline is a feature, not a failure. |
| Phase 1 probe collapses to chance AUROC | medium | §9 gate 3 catches this. Diagnose: top_p deviation (plan §A.2), score-formula bug, or data alignment issue. Up to 3 days of debugging budget; beyond that, fall back to "we tried, here is the diagnosis" cite-only paragraph |
| `W_U` lookup is wrong (e.g. tied weights handled differently across HF model families) | low | Sanity check: `W_U.shape == (vocab_size, hidden_size)`; verify by decoding `argmax(W_U @ h_last)` matches the model's predicted next token on one sample |
| Top-k indices include padding positions (yields garbage token IDs) | medium | §3.4 explicitly handles this — implementer must add masking unit test |
| Filename collision with linear probe causes generic loaders to load the wrong checkpoint | low | Each method has its own subdirectory; documented in §10 |
