# Methods — Outline

Structural skeleton only. Each subsection lists what the prose will cover, not the prose itself. Placeholders in `[brackets]` mark spots where the author will decide what to import from planning/theory docs.

---

## 3. Method

### 3.1 Problem Setup

- **Task framing.** Intrinsic hallucination detection: given a model `M`, a prompt `x`, and `M`'s greedy generation `y`, produce a scalar score `s(x, y) ∈ ℝ` such that higher scores correspond to hallucinated generations.
- **Access regime.** White-box, single forward pass. We have read access to intermediate residual-stream activations but do not modify `M`.
- **Notation.** Define `h_ℓ(x, y) ∈ ℝ^d` = last-token hidden state at layer `ℓ` of `M`'s residual stream when reading `(x, y)`. `L` = total layers. `y_label ∈ {0, 1}` = ground-truth hallucination label from the benchmark's evaluator.
- **What we are *not* doing.** No retrieval, no external verifier, no multi-sample generation in the headline method — these define the comparison classes in §4.

### 3.2 Activation Extraction

- **Layer selection.** We extract a contiguous band of mid-to-late layers `[ℓ_lo, ℓ_hi]`. State the chosen band and the rationale (cross-layer-coherence theory motivates mid-to-late residual stream — defer the full argument to §[theory section]).
- **Token position.** Last token of the generation, after the final answer token. State why: [rationale — likely the position where the answer is fully integrated].
- **Storage.** Activations cached once per (model, dataset, split) in zarr. No re-extraction across training seeds or hyperparameter sweeps.

### 3.3 Contrastive Compression Architecture

- **High-level diagram.** Two views per example, each view being a single layer's activation `h_ℓ`. Each view passes through a shared compression network `f_θ`. Loss couples the views in representation space.
- **`ProgressiveCompressor` ƒ_θ.** Describe the architecture: input dim, hidden dims, output (compressed) dim, activation function, normalization. [Confirm exact dims against `activation_research/model.py`.]
- **View construction.** Each training example yields two views by sampling a pair of layers `(ℓ_a, ℓ_b)` from the cached band. State whether layers are sampled uniformly, with a fixed pair, or scheduled — [defer to §[layer-pair ablation] for sensitivity].
- **Why layer pairs as views (1 sentence).** Different layers encode the same generation under different inductive biases; consistent representations across layer pairs should correlate with epistemic stability. Full justification: §[theory section].

### 3.4 Training Objective

Total loss is a sum of two terms:

```
L = L_contrastive + λ · L_logprob_recon
```

- **`L_contrastive` — SimCLR-style InfoNCE.** Positive pair: two views of the same `(x, y)`. Negatives: all other examples in the batch. Temperature `τ`. State the exact form and `τ` value.
- **`L_logprob_recon` — auxiliary reconstruction loss.** From the compressed representation, predict the per-token logprobs of `y` under `M`. State the head architecture (linear? MLP?) and the regression target (mean logprob? sequence-level?). [Confirm against `activation_research/training.py`.]
- **Why both terms (1 sentence + cite §[ablation]).** SimCLR alone produces representations orthogonal to the hallucination axis (the unsupervised-only ablation gives AUROC ≈ 0.5); the logprob-recon term grounds the embedding in a quantity correlated with model uncertainty. We report this empirically in the ablation; the theoretical argument is in §[theory section].

### 3.5 Inference Scoring

Given a trained `f_θ` and a held-out example `(x, y)`, compute `z = f_θ(h_ℓ(x, y))` and produce `s(x, y)` via one of three scorers:

- **KNN (headline scorer).** Distance to the `k`-th nearest neighbor in the training-set embedding bank, computed over [labeled-positive set / full training set]. State `k` and the distance metric.
- **Cosine (ablation).** Mean cosine similarity to the training-set positive class (or appropriate variant — state precisely).
- **Mahalanobis (ablation).** Distance to the training-set class-conditional Gaussian under the pooled covariance.

Choice of KNN as headline is empirical (seed-0 evidence, §[results]); cosine and Mahalanobis are reported as ablation rows.

### 3.6 Implementation Details

- **Hyperparameters.** Optimizer, learning rate, batch size, training epochs, temperature `τ`, λ for the recon loss, weight decay. State once; defer ablations to the ablation section.
- **Training data.** Per-(model, dataset) training: we train on the dataset's train split only (no cross-dataset training); cross-dataset transfer is reported separately in §[transfer section]. Split seed fixed at 42; 5 training seeds {0, 5, 26, 42, 63} for variance estimates.
- **Reproducibility.** All cached activations stored at known paths; configs in `configs/experiments/baseline_comparison_*.json`; training entry point `scripts/train_activation_model.py`. Random seeds control model init, batch order, and KNN bank sampling.
- **Compute.** Single 80GB GPU sufficient for one full (dataset, model, method, seed) cell once activations are cached. Cite total GPU-hours in §[experiments].

---

## Cross-references this outline expects to call

- §[theory section]: cross-layer-coherence justification for layer pairs as views, and for the necessity of the recon term
- §[ablation section]: SimCLR-only vs full loss; layer-pair sensitivity; scorer choice
- §[experiments / results]: the headline numbers backing "KNN beats cosine/Mahalanobis"
- §[transfer section]: source→target evaluation procedure
- §[appendix]: full hyperparameter table, full `ProgressiveCompressor` spec, exact training-time logs

Resolve these forward references before submission — every `§[...]` placeholder above must point to a real section anchor.
