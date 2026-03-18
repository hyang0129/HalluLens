# Novel Hallucination Detection Methods
> Generated via researcher–critic debate (4 rounds each). Methods 4 and 5 in progress.

---

## Method 1: Cross-Layer Attention on Aligned Projections

### Overview

Detect hallucinations by learning cross-layer relationships from aligned, dimension-reduced LLM activations. A fixed PCA projection (zero learned params) compresses each layer's activations while preserving cross-layer dimensional correspondence, then a lightweight bidirectional transformer learns which cross-layer interaction patterns distinguish truthful from hallucinated generations. Contrastive training with aggressive layer-subset augmentation (50% retention) forces the model to learn representations invariant to which specific layers are observed.

### Architecture

**Stage 1 — PCA Projection (zero learnable parameters):**

Compute PCA on training-set activations *pooled across all layers*, producing a projection matrix P. Apply the same P to every layer:

```python
# One-time computation on training data:
# X_train ∈ R^{(N * K) × 4096}  where N=samples, K=16 layers
# Fit PCA on X_train, keep top d=128 components:
# P ∈ R^{4096 × 128}, mu ∈ R^{4096}  (both frozen)

for k in layers:  # {14, 15, ..., 29}
    a_k = pool(activations_k)           # (B, seq_len, 4096) → (B, 4096)
    h_k = (a_k - mu) @ P                # (B, 128)
```

**Token pooling:** Learned scalar gate g ∈ [0,1] (1 parameter, sigmoid-parameterized):
```
pool(x) = g * x.mean(dim=1) + (1 - g) * x[:, -1, :]    # (B, 4096)
```

**Why PCA over random projection:** PCA computed across all layers captures 85–95% of variance vs. ~3% per dimension for random projection. Hallucination signal likely occupies a low-dimensional subspace; PCA concentrates it. Shared projection matrix P guarantees cross-layer dimensional alignment.

**Stage 2 — Lightweight Bidirectional Cross-Layer Transformer:**

```python
H = stack([h_14, ..., h_29])               # (B, K, 128), K=16
H = LayerNorm(H)
H = H + LearnedLayerPositionalEncoding(K)  # indexed by absolute layer position
Z = BidirectionalTransformer(H)            # (B, K, 128), 1 layer, 4 heads
z = MeanPool(Z, dim=1)                     # (B, 128)
```

Transformer spec: 1 layer, 4 heads, dim=128, FFN dim=256, dropout=0.1. ~130K learnable params total (+1 gate param).

### Training

**View generation — two random 50% layer subsets:**

```python
for v in {1, 2}:
    subset_v = random_sample(K=16, keep=8, without_replacement=True)
    H_v = stack([h_k for k in sorted(subset_v)])    # (B, 8, 128)
    H_v = LayerNorm(H_v)
    H_v = H_v + LearnedLayerPositionalEncoding(subset_v)  # absolute position PE
    Z_v = BidirectionalTransformer(H_v)
    z_v = MeanPool(Z_v, dim=1)
```

Expected overlap: ~4 of 8 layers. Forces the model to prove cross-layer signal is redundant across subsets. No MASK tokens.

**Loss:** Pure supervised contrastive:
```
L = SupConLoss(z_views, labels=halu_labels, temperature=0.07)
```

### Scoring

Mahalanobis distance or k-NN on z ∈ R^128. Same pipeline as current system. All K=16 layers at test time.

### Key Insight

Current SupCon treats inter-layer differences as nuisance to be discarded (same-sample layers as positive pairs). This method captures inter-layer relationships as signal. The 50% layer-subset augmentation forces the model to learn representations invariant to *which* layers are observed, while the bidirectional transformer learns *how* the layers relate.

### Ablation Plan

1. **Pairwise cosine similarity MLP (critical):** 120 pairwise cosines → 2-layer MLP. If this matches the transformer, attention is unjustified — use the MLP.
2. **Concatenation baseline:** Concat K vectors (16×128=2048) → linear to 128 with SupCon.
3. **Single-layer oracle:** Best single layer, PCA-projected to 128, Mahalanobis.
4. **Unordered baseline:** Shuffle layer positional encodings at train time.
5. **Learning curve analysis:** N = {100, 250, 500, 1000}.

---

## Method 2: Cross-Layer Feature Profiles for Hallucination Detection

### Overview

Detect hallucinations using **hand-crafted cross-layer features** extracted from PCA-aligned layer activations, scored with L1-regularized logistic regression. Honestly framed as feature engineering + simple classifier. The contribution: pairwise geometric statistics between all layer pairs capture cross-layer dynamics that single-layer methods miss.

### Input Representation

Same PCA projection as Method 1 (4096→128, shared P, frozen). Mean-pool over tokens before projection.

### Feature Extraction

**Tier 1 — Raw Cross-Layer Statistics (zero learned params, primary):**

For each of the 120 undirected pairs (i, j) where i < j:
```python
dist(i, j)       = || h_i - h_j ||²
cos(i, j)        = (h_i · h_j) / (||h_i|| · ||h_j||)
norm_ratio(i, j) = ||h_j|| / max(||h_i||, eps)
```

Plus 16 per-layer norms: `norm(k) = ||h_k||`

**Total Tier 1: 376 dimensions.**

**Tier 2 — Per-Pair Linear Prediction Residuals (optional, requires N_truthful > 256):**

For each directed pair (s, t) where s < t, fit ridge regression on truthful training data:
```
W_{s,t} = (H_s^T H_s + λI)^{-1} H_s^T H_t   [closed-form, λ via LOOCV]
pred_residual(s, t) = || W_{s,t} @ h_s - h_t ||²
```

**Total Tier 2: 120 dimensions. Combined: 496 dimensions.**

*Tier 2 activation criterion:* Only computed when N_truthful > 256. Below this, ridge is dominated by regularization and residuals are artifacts.

### Scoring: L1 Logistic Regression with Nested Cross-Validation

```python
score = LogisticRegression(features, penalty='l1', solver='saga')
```

**Protocol — nested 5×3 CV to prevent leakage:**
- Inner loop (3-fold): tune C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}; feature selection (R² > 0.1 threshold for Tier 2 pairs; correlation filter |r| > 0.95) — all computed on inner training data only
- Outer loop (5-fold): evaluate on held-out test fold
- Report: mean ± std AUROC across 5 outer folds

L1 naturally handles feature selection; non-zero coefficients reveal which cross-layer statistics matter.

### Ablation Plan

1. **Tier 1 L1-logreg vs. single-layer linear probe:** Core experiment. Does cross-layer signal exist?
2. **Tier 1 + Tier 2 vs. Tier 1 only (when N > 256):** Does learned residual beat raw geometry?
3. **Feature ablation within Tier 1:** L2 only (120-dim) vs. cosine only (120-dim) vs. all (376-dim).
4. **L1 feature importance:** Inspect top-10 features by |coefficient| — mechanistic insight.

---

## Method 3: Targeted Logprob Shape Features for Hallucination Detection

### Overview

Detect hallucinations using **per-token logprob temporal features** that capture confidence dynamics within a generation. A compact 19-feature representation extending the existing token entropy baseline (mean/min logprob, mean entropy) with distributional, trend, localized-anomaly, and half-contrast features. All computed on the original variable-length token sequence — no resampling, no neural network.

**Key scientific question:** Does temporal confidence structure (how confidence evolves within a generation) add discriminative value beyond global confidence statistics (mean, min)?

### Signal Definitions

```python
logprob(t) = lp_t                                          # per-token logprob (stored in Zarr)
entropy(t) = -sum_k p_k * log(p_k) from top-k logprobs     # per-token entropy
```

No surprise signal — dropped due to collinearity with logprob and undefined semantics after resampling.

### Feature Groups (19 total)

**Group 1 — Global Statistics (7 features):**
```python
mean_logprob      # existing baseline
std_logprob
min_logprob       # existing baseline
mean_entropy      # existing baseline
std_entropy
skewness_logprob  # left-skew = scattered low-confidence tokens
response_length = log(T)
```

**Group 2 — Trend Features (4 features, length-invariant):**
```python
logprob_slope  = OLS slope of logprob(t) vs. t/T
entropy_slope  = OLS slope of entropy(t) vs. t/T
logprob_r2     = R² of logprob linear fit
entropy_r2     = R² of entropy linear fit
```
`logprob_slope < 0` captures "confidence declines through response" (confident prefix → hallucinated suffix).

**Group 3 — Localized Anomaly Features (4 features):**
```python
max_logprob_drop_normalized = max(logprob(t) - logprob(t+1)) / std_logprob
    # Normalized by std → removes length-dependent scaling (E[max] ~ O(sqrt(log T)))

drop_position = argmax(logprob(t) - logprob(t+1)) / T
    # Normalized position of steepest drop

longest_low_run = (longest consecutive run where logprob(t) < Q25(logprob)) / T
    # Data-adaptive threshold (Q25), no hyperparameter. Genuinely temporal — shuffling
    # tokens changes this value.

run_position = start_of_longest_low_run / T
    # WHERE the sustained low-confidence span occurs (late = hallucinated suffix)
```

**Group 4 — First-Half vs. Second-Half Contrast (4 features):**
```python
logprob_half_delta = mean(logprob[T/2:]) - mean(logprob[:T/2])
entropy_half_delta = mean(entropy[T/2:]) - mean(entropy[:T/2])
logprob_half_ratio = std(logprob[T/2:]) / max(std(logprob[:T/2]), eps)
entropy_half_ratio = std(entropy[T/2:]) / max(std(entropy[:T/2]), eps)
```
Minimum T=6 tokens required; shorter responses get these set to 0.

### Scoring

L1 logistic regression, nested 5×3 CV (same protocol as Method 2):
```python
score = LogisticRegression(features_19dim, penalty='l1', solver='saga')
C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}
```

### Ablation Plan

1. **Core question — Group 1 only (7-dim) vs. Full 19-dim:** The only high-power comparison. If full model wins by >3 AUROC, temporal structure is real.
2. **Subsumption check:** Existing 3-feature baseline vs. Group 1 (7-dim). Validates enhanced globals.
3. **Learning curve:** N = {50, 100, 200, 500} × 10 random seeds. Determines if temporal signal is data-limited.
4. **Properly structured ensemble with Method 2:** Method 2 alone vs. Method 2 + Groups 2-4 only (376+12=388-dim). Groups 2-4 only — excludes Group 1 globals to avoid redundancy with Method 2's per-layer norms.

### Honest Expected Outcomes

- **Most likely:** Group 1 improves 2–4 AUROC over 3-feature baseline; Groups 2–4 add 0–2 more (within noise at N=500).
- **Best case:** Sharp-drop and trend features capture "confident hallucination" pattern, adding 3–5 AUROC over global stats alone.
- **Worst case:** All temporal features add <1 AUROC over Group 1 — reduces to "use 7 global stats instead of 3."

All three outcomes are publishable.

---

## Method 4: Factored Additive Ensemble over Layer Activations

### Overview

Factorize hallucination detection into 16 independent per-layer subproblems, each solved by a small MLP on 64-dim PCA projections, combined via L1-regularized stacking. This imposes a rank-16 structural constraint (one discriminative direction per layer) as an alternative inductive bias to L1 sparsity on concatenated features. The experiment tests which constraint better fits the data at small N.

**Honest framing**: Method 4 and L1-on-concat represent two competing structural constraints (rank-16 factored vs. sparse flat), and the experiment determines which better fits this data. No optimality claims.

### Architecture

**Input**: 64-dim PCA per layer (fitted on truthful training data only), all 16 layers {14,...,29}. Each sample produces 16 vectors in R^64.

**Stage 1 — 16 Independent Per-Layer MLPs with Out-of-Fold Prediction**:

For each layer l in {14,...,29}:
- Architecture: R^64 → 32 hidden (ReLU) → 1 raw logit
- Loss: binary cross-entropy with logits
- Optimizer: Adam, lr=1e-3, L2 weight decay=1e-3
- Training: fixed 50 epochs, cosine LR decay (1e-3 → 1e-5), no early stopping, no validation split

Out-of-fold (OOF) procedure on inner training data (~400 samples across 5 inner folds):
```python
for k in 1..5:
    train MLP_l on folds {1..5}\k  (~320 samples)
    predict raw logits on fold k   (~80 samples)
# result: every inner sample has leak-free OOF logit for each layer
# after OOF: retrain each MLP_l on all ~400 inner samples for test inference
```

Output: [~400 × 16] matrix of OOF logits for stacker training.

**Stage 2 — L1 Stacking on OOF Logits**:
```python
# Input: 16 OOF logits per sample, all ~400 inner samples
# Model: L1-regularized logistic regression (17 params: 16 weights + 1 bias)
# L1 penalty tuned via 5-fold CV within stacking data
# Functions: rescales per-layer logits, selects informative layers, combines additively
```

**Parameter count**: 16 MLPs × 2,113 params + 17 stacker = ~33,825 total.

### Evaluation

Nested 5-fold outer CV. Per outer fold: run full inner OOF pipeline, apply to test fold. Report mean ± std AUROC across 5 folds.

### Diagnostics

1. **Per-layer R² linearity test**: After training, compare each MLP's predictions to a linear logistic regression on the same data. If R² > 0.9, nonlinearity contributed nothing — use the linear variant (Ablation 2) instead.
2. **Stacker weight profile**: Which layers receive nonzero weights, and do they align with independent per-layer probing AUROC rankings?
3. **Per-fold AUROC variance**: With ~100 test samples per fold, variance across folds indicates estimate reliability.

### Success Criteria

Method 4 is justified over L1-on-concat only if BOTH:
- (a) Stacker-selected layers align with independent per-layer probing AUROC rankings (confirming factored structure finds real layer-level signal, not stacking artifacts)
- (b) Mean R² < 0.9 across layers (confirming real nonlinearity contributed)

If (a) but not (b): use the linear factored variant (Ablation 2) instead of the MLP version.
If neither: abandon Method 4, use L1-on-concat.

### Expected Outcome

- **Baselines**: Linear probe best layer (~0.68), Mahalanobis best layer (~0.68), L1 logistic on 1024-dim concat (~0.70-0.72)
- **Target**: 0.72-0.75 AUROC
- **Most likely**: Method 4 and L1-on-concat within 1-2 AUROC of each other. Diagnostic value is understanding WHICH structural constraint fits better, not a dramatic performance gap.

### Ablation Plan

1. **L1 logistic regression on 1024-dim concatenated PCA features** — The critical head-to-head. Same 5-fold outer CV. Tests sparsity constraint vs. rank constraint. If L1 matches or wins, Method 4's factored structure adds no value.
2. **Linear per-layer classifiers + OOF + stacking** — Replace MLPs with logistic regression per layer. Tests whether nonlinearity contributes. Prediction: matches MLPs (R² > 0.95), confirming the factored structure is doing the work. If this matches full Method 4, it becomes the preferred variant.
3. **R² linearity diagnostic** — Computed post-hoc on Method 4's trained MLPs. Not a separate training run.
4. **Full MLP on 1024-dim concat (~33K params)** — Tests cross-layer interactions. If it beats Method 4, the additive assumption is wrong. If Method 4 beats it, factored structure provides beneficial regularization at this data scale.

---

## Method 5: Token-Level Intermediate Layer Anomaly Detection

### Overview

Method 5 tests whether decomposing sample-level activation anomaly detection to token-level resolution adds discriminative information. Methods 1/2/4 detect hallucination via sample-level (mean-pooled) activation features. Method 5 asks: does the within-response **distribution** of per-token anomaly scores — particularly the tail — carry signal that mean-pooling destroys?

**Core hypothesis**: Hallucinated short-form QA responses contain a small number of fabricated factual tokens whose intermediate-layer activations are distributional outliers relative to truthful token activations. Mean-pooling dilutes this sparse signal; tail-sensitive aggregation statistics (outlier fraction, top-k mean) preserve it.

This is explicitly framed as an incremental extension of Method 2's sample-level anomaly detection to token-level resolution — not a standalone novelty claim.

### Input Representation

Per-token activations at all 16 layers {14,...,29}, each h_l^t in R^4096, for the **middle 80% of generated tokens** (first and last 10% excluded to remove position-dependent edge effects without requiring position correction).

### Pipeline

**Step 1 — Token-Level PCA**:
```python
# For each layer l:
# From each truthful training response, sample 1 random token from middle 80%
# → ~250 independent token-level activations (one per generation)
# Fit PCA_l on these 250 observations, d=32 components
# Apply PCA_l to all middle-80% tokens: z_l^t = PCA_l(h_l^t) ∈ R^32
```
One random token per generation ensures independence (~250 independent samples, 7.8:1 ratio).

**Step 2 — Token-Level Anomaly Scoring**:
```python
# For each layer l:
# Compute mu_l, Sigma_l (Ledoit-Wolf shrinkage) from same 250 PCA-projected token samples
# For each token t: anomaly_l^t = Mahalanobis(z_l^t, mu_l, Sigma_l)
# Q95_l = 95th percentile of anomaly scores across all truthful training tokens
#   (all tokens, not just 250 — percentile estimation is robust to within-gen correlation)
```

**Limitation**: PCA and Mahalanobis statistics are fitted on the same 250 tokens (circularity). Ledoit-Wolf shrinkage partially mitigates eigenvalue instability; condition number diagnostics (see below) flag cases where this is problematic.

**Step 3 — Aggregate to Sequence-Level Features (64 total)**:
```python
for l in layers:  # 16 layers
    scores = [anomaly_l^t for t in middle_80%_tokens]
    k = max(3, len(scores) // 20)

    mean_anomaly_l    = mean(scores)
    std_anomaly_l     = std(scores)
    outlier_frac_l    = fraction(scores > Q95_l)
    topk_mean_l       = mean(top_k(scores, k))
```

`mean_anomaly` approximates mean-pooled Mahalanobis (baseline-matching). `outlier_frac` and `topk_mean` are robust to dilution by surrounding truthful tokens — the key novel features.

**Step 4 — Classification**:
L1-regularized logistic regression on 64 features. Nested 5-fold CV (same protocol as Methods 2-3). All fitting (PCA, Mahalanobis, Q95, classifier) within each outer training fold.

**Parameter count**: 65 learned parameters (64 weights + 1 bias). Everything else is fitted statistics.

### Central Falsification Test

Report individual feature AUROC for each aggregation type (averaged across 16 layers):

**Method 5 is falsified** (should be abandoned in favor of standard per-layer Mahalanobis) only if BOTH:
- (a) Mean features (`mean_anomaly`) beat tail features (`outlier_frac`, `topk_mean`) by **>2 AUROC points** on individual feature AUROCs
- (b) Tail features have nonzero L1 coefficients in **fewer than 3 of 5 outer folds**

If either criterion is not met, tail features are considered contributing.

### Diagnostics

1. **Individual feature AUROCs** (with bootstrap CIs) — falsification test
2. **L1 coefficient profile** — which layers and feature types have nonzero coefficients across outer folds
3. **Covariance condition number** per layer at each PCA dimension in the dim sweep — distinguishes signal effects from numerical breakdown (values >100 flagged as unreliable)

### Expected Outcome

- **Baselines**: Mahalanobis on mean-pooled activations (~0.68), Method 3 logprob features (~0.69-0.71)
- **Target**: 0.70-0.74 AUROC
- **Most likely**: Tail features provide +1-3 AUROC over mean-only features at late layers. L1 selects mix of tail and mean features from layers 23-29. Overall 0.71-0.73 AUROC.
- **Downside**: If hallucination is a whole-response phenomenon (all tokens shift uniformly), falsification test triggers and Method 5 should be abandoned.

### Ablation Plan

1. **Mahalanobis on mean-pooled activations (existing baseline)** — Tests whether any token-level decomposition adds value. If mean-pooled matches, token-level is unnecessary.
2. **Tail-only features vs. mean-only features** — L1 on {outlier_frac, topk_mean} (32 features) vs. {mean_anomaly, std_anomaly} (32 features). THE key diagnostic for the sparse-fabrication hypothesis.
3. **Include edge tokens (full response) vs. middle 80% only** — Tests whether edge exclusion removes confounds or discards informative tokens.
4. **PCA dimension sweep: d = 16, 32, 48** — Report covariance condition number at each d. Expect d=32 as sweet spot (d=48 may show numerical instability at 5.2:1 ratio).
