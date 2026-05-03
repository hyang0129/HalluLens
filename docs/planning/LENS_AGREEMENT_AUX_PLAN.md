# Plan: Logit-Lens Agreement as an Additional Aux Objective

## Goal

Test whether adding a **logit-lens top-1 agreement** auxiliary head on top of the existing **contrastive + log-prob reconstruction** training improves hallucination detection. One new head, one new label channel, otherwise identical to the current best recipe.

## Hypothesis

The log-prob aux head teaches the encoder *how confident the model was at each emitted token*. The lens-agreement head teaches *how early the model committed to its final answer* — a different facet of the same forward pass that tracks convergence dynamics rather than peak-confidence.

Concrete prediction: lens-agreement head adds 0.5–2 AUROC points on PopQA test (long-tail factoid, where late/oscillating convergence should be common in hallucinations) over the contrastive + log-prob baseline.

---

## Current scaffolding we extend

- `activation_research/model.py` → `LogprobReconProgressiveCompressor`: encoder + auxiliary decoder `g: R^d → R^recon_seq_len` predicting per-token logprob sequence.
- `activation_research/training.py` → `train_contrastive_logprob_recon`: training loop that adds `λ_recon · MSE(g(z), logprob_seq)` to the contrastive loss. Collation via `_contrastive_collate_with_logprob`.
- `activation_logging/activation_parser.py` → `include_response_logprobs=True` already plumbs per-token logprob sequences through the dataset.

The lens head should mirror this structure exactly: another `R^d → R^seq_len` decoder, BCE loss against a per-token binary target.

---

## Step 1 — Pre-compute lens-agreement labels

**One-shot offline pass per (model, dataset). Done once, stored alongside activations.**

Script: `scripts/compute_lens_agreement.py` (new).

Inputs:
- A zarr activation store (already has hidden states for `--train-layers`, e.g. 14–29).
- The same model (Llama-3.1-8B-Instruct or Qwen3-8B) — needed only for `model.model.norm` and `model.lm_head`.

For each sample, for each stored layer ℓ, for each response-span token position t:

```
h_ℓ,t                  ← stored activation  (already in zarr)
logits_ℓ,t  = lm_head(final_norm(h_ℓ,t))
logits_F,t  = lm_head(final_norm(h_F,t))             # F = final layer
agree_ℓ,t   = 1[ argmax(logits_ℓ,t) == argmax(logits_F,t) ]
```

Output: a new zarr array `lens_agreement` of shape `(N, num_train_layers, max_response_len)` (uint8, NaN-equivalent = 255 for padding) under the same store. Co-located so dataset loading is one pull.

**Notes / gotchas:**
- Apply `model.model.norm` *before* `lm_head` — Llama / Qwen3 final LN is non-trivial. Skipping it gives noisier labels.
- The final layer's hidden state must be among the stored layers, or stored separately. Audit with `python scripts/audit_datasets.py --model … --zarr` and confirm layer 31 (Llama) / layer 35 (Qwen3) is present. If not, dump it once.
- For sampled (non-greedy) generations, use `argmax(logits_F,t)` as the "final-layer top-1" — the *emitted* token is a sample, not the model's best guess at that position.
- Top-1 argmax can be computed in fp16 without numerical issues; keep this on GPU.
- Throughput: for each stored layer this is one matmul `(T × d_model) @ (d_model × |V|)` followed by argmax. ~|V| ≈ 128k for Llama 3, ~150k for Qwen3. Should run at >1000 samples/s on H200.

**Sanity-check artifact:** the script also writes `lens_agreement_summary.json` containing per-layer mean agreement (over response tokens), bucketed by hallucinated vs non-hallucinated examples. We inspect this *before* running training (see Step 2).

## Step 2 — Pre-training sanity checks

Before any training run, verify the labels carry signal. Cheap and saves a wasted training run.

1. **Layer-monotonicity check.** Plot `mean(agree_ℓ)` by layer ℓ on the train split. Expect a roughly monotonic rise from ~0.1 (low layers) toward 1.0 (final). If flat, lens labels are uninformative and we abort.
2. **Class-separation check.** For each layer, plot `mean(agree_ℓ | hallucinated)` vs `mean(agree_ℓ | correct)`. The gap is the upper bound on what a noiseless head could learn from this signal alone. Aim for ≥2 percentage points at one or more layers; if the curves are within noise everywhere, abort.
3. **Correlation with logprob.** For each layer, compute Pearson `r` between `agree_ℓ,t` and `logprob_t`. If `r > 0.9` everywhere, the lens head is mostly redundant with the existing logprob head and we should not expect lift. We'd still run, but with tempered expectations.

Decision rule: proceed to Step 3 only if checks 1 and 2 pass on at least one of {PopQA train, SciQ train}.

## Step 3 — Implement the dual-aux model and trainer

**New module: `LogprobLensReconProgressiveCompressor`** in `activation_research/model.py`.

- Same encoder as `LogprobReconProgressiveCompressor`.
- Two decoders sharing the bottleneck `z`:
  - `g_logprob: R^d → R^L_logprob` (existing, regression target)
  - `g_lens: R^d → R^L_lens` (new, binary target → BCE-with-logits)
- `forward(x, layer_idx)` returns `(z, logprob_pred, lens_logits)`.
- `recon_loss(...)` extended to return both component losses for logging.

**New trainer: `train_contrastive_logprob_lens_recon`** in `activation_research/training.py`.

- Copy `train_contrastive_logprob_recon` and extend.
- Collate: extend `_contrastive_collate_with_logprob` → `_contrastive_collate_with_logprob_lens` to also stack `lens_agreement` (uint8 → float, with a separate validity mask).
- Loss: `total = L_contrastive + λ_logprob · L_logprob + λ_lens · L_lens`.
  - `L_lens = BCEWithLogits(lens_logits, lens_target)` averaged over valid response tokens, then averaged over the layer dimension if predicting all train-layers' agreement labels (see below).
- NaN/pad handling: mirror the existing `nan_mask` pattern for logprobs. Pad value 255 → invalid → masked out.

**Head shape choice — single-layer-target vs multi-layer-target:**

- *Option A (simpler, recommended first):* `g_lens` predicts only one specific layer's agreement (e.g., layer 22's agreement with the final layer). Output dim = `L_lens = response_len`. The encoder reads its own layer; lens label may be from a *different* layer, which mitigates the leakage concern (see "Subtle point" in `AUXILIARY_OBJECTIVES.md`).
- *Option B:* predict agreement across all train layers simultaneously. Output dim = `num_train_layers × response_len`. More signal but more capacity needed and harder to interpret.

Start with Option A: encoder reads layers in `--train-layers 14-29` (existing convention), aux head predicts agreement at **layer 22 with final**. This decouples encoder layer from prediction layer.

**Loss weighting starting point:**
- `λ_logprob`: keep current best value (whatever the existing best run uses — read it off `configs/experiments/baseline_comparison_*.json`).
- `λ_lens = 1.0` initially. BCE is naturally well-scaled, but sweep `{0.25, 1.0, 4.0}` if the first run shows the head is either ignored (gradient norm tiny) or dominating (logprob/contrastive losses fail to converge).

## Step 4 — Experiment matrix

Minimal, three-cell ablation. All on the same encoder architecture, layers, optimizer, seed set.

| Run | contrastive | logprob aux | lens aux |
|---|---|---|---|
| **B0** baseline (current best) | ✓ | ✓ | – |
| **B1** lens-only aux | ✓ | – | ✓ |
| **B2** both aux (the proposal) | ✓ | ✓ | ✓ |

B1 isolates lens contribution independent of logprob; B2 is the headline. We do **not** run a contrastive-only cell — we already have that number.

**Datasets:** PopQA test + SciQ test, both Qwen3-8B and Llama-3.1-8B-Instruct → 4 (model, dataset) cells × 3 runs = 12 training runs.

**Seeds:** 3 seeds per cell (use the existing seed-split infrastructure). 36 total runs.

If compute is tight, drop to PopQA-only first (the dataset where the hypothesis is strongest), 6 runs, decide from there.

**Configs:**
- Add `configs/experiments/lens_aux_popqa_qwen3.json`, `lens_aux_popqa_llama3.json`, `lens_aux_sciq_qwen3.json`, `lens_aux_sciq_llama3.json`. Each enumerates B0/B1/B2 × seeds via the existing run-spec mechanism in `scripts/experiment_utils.py`.
- Reuse `--train-layers 14-29` from current best.

**Dispatch:** `scripts/gpu_dispatch.py run --min-vram 30 bash scripts/run_lens_aux_<dataset>_<model>.sh` per the project convention. **Do not start jobs without explicit user approval** — flag this and wait.

## Step 5 — Evaluation

Reuse existing evaluation pipeline. Primary metric: **AUROC of hallucination detection on the held-out test split**, mean ± std across seeds.

Decision criteria:
- **B2 > B0** by ≥1 AUROC point (mean over seeds, both datasets) → lens aux is a keeper. Land it as a config option in the trainer; add to default best-run.
- **B2 ≈ B0** (within seed noise) → lens aux is informationally redundant given logprob. Document and shelve.
- **B1 ≫ B0** but **B2 ≈ B0** → lens carries signal but not stackable with logprob. Worth investigating; possibly a loss-weighting issue.
- **B2 < B0** → aux objective is hurting. Most likely cause: leakage-style overfitting or wrong loss weight. Diagnose before discarding.

Secondary diagnostics (not gates):
- Per-layer probe AUROC on `z` after training — does the lens aux change *which* layer encodes hallucination signal best?
- Calibration (ECE) on the test split.

## Step 6 — Threats to validity / failure modes to watch

- **Label leakage via shared geometry.** The encoder sees `h_ℓ`; the label is a deterministic function of `h_ℓ` (and `h_F`). If the head can re-implement `lm_head ∘ final_norm` it perfectly fits the label without learning anything useful. Mitigations: shallow head (already), and Option-A's layer-decoupling.
- **Logprob already encodes most of the signal.** If `agree_ℓ,t` and `logprob_t` are highly correlated (Step 2 check 3), lens adds little. Pre-empted by the sanity check — we'd know before training.
- **Final-layer-top-1 ≠ gold token under sampling.** For non-greedy generations the agreement target measures "did the model commit to its final greedy candidate early," which may not be exactly what we want. A useful follow-up — not part of this plan — is `agree-with-gold` for short-answer datasets where the gold token is unambiguous.
- **Tuned-lens labels would be cleaner.** Vanilla lens has known basis-mismatch issues. If results are close-to-significant but not over the bar, training a tuned lens (Belrose et al.) and re-running is the next move. Not in scope here — keep the first experiment cheap.

## Step 7 — Write-up artifacts

- `results/lens_aux_<dataset>_<model>.md` — table of seeds × runs × AUROC, plus the Step 2 sanity-check plots.
- One-line update in `docs/planning/SOTA_TRACKER.md` if B2 beats B0.
- If lens aux lands: a short note in `CLAUDE.md` under the training section listing the new flag(s).

---

## Concrete deliverables checklist

1. `scripts/compute_lens_agreement.py` — offline label generator.
2. `lens_agreement` array + `lens_agreement_summary.json` written into the existing zarr stores for the four (model, dataset) pairs.
3. Sanity-check report (Step 2) — gate before any training.
4. `LogprobLensReconProgressiveCompressor` in `activation_research/model.py`.
5. `train_contrastive_logprob_lens_recon` in `activation_research/training.py`, plus updated collate.
6. Four experiment configs under `configs/experiments/lens_aux_*.json`.
7. Four dispatch scripts under `scripts/run_lens_aux_*.sh` (do not auto-launch).
8. Results table + decision per Step 5.

## Out of scope (intentionally)

- Tuned lens — vanilla first.
- Predicting agreement at multiple layers simultaneously — Option A only.
- Other aux heads (entropy, P(True), semantic entropy) — separate plan; we test one variable at a time.
- New datasets beyond PopQA / SciQ — add only after the two-dataset signal is clear.
