# Should We Resample Our Activations?

A structured debate on whether generating multiple responses per question improves contrastive representation learning for hallucination detection.

---

## The Hypothesis

> The contrastive representation learning across layers suffers from a lack of same-question-different-answer examples. Because we only get one answer per question, we may be learning more about the question than the answer — effectively identifying which questions lead to hallucinations rather than which answers are hallucinations. By resampling answers, we can improve our contrastive learning process and overall performance.

**Framing the choice:** This is fundamentally a question of where to invest effort — resampling existing data (new inference runs, more responses per question) vs. better use of existing data (algorithmic and architectural improvements to the current pipeline).

---

## Current System Summary

- **One response per question**, labeled hallucinated (1.0) or non-hallucinated (0.0)
- **K "views"** = the same response sampled from K different layers (14–29), NOT different responses
- **Positive pairs:** same-sample layer views (weight=1.0), same-label different-samples (weight=0.5)
- **Loss:** SupConLoss at temperature=0.07, batch_size=512
- **Encoder:** ProgressiveCompressor, sequence-level pooling

---

## The Case For Resampling

### 1. The Question-Identity Confound Is Structurally Inevitable

With one response per question, the mapping `question → label` is deterministic in training data. The encoder cannot distinguish between:
- **(Desired):** "These activations share hallucination-specific patterns"
- **(Confound):** "These activations came from structurally similar questions (hard, obscure, multi-hop)"

No amount of tuning the existing loss can break this collinearity — the confound is in the data structure itself.

### 2. Contrastive Learning Theory Requires the Right Invariance

SupCon learns invariance to whatever varies *within* positive pairs. Current positive pairs vary only in:
- Which layer (same response, different depth) — weight 1.0
- Which question (same label, different prompt) — weight 0.5

Neither pair type includes the critical missing invariance: **same question, different label**. The encoder has no training signal that forces it to separate "model is confabulating" from "this is a hard question." The missing invariance is the NLP equivalent of a data augmentation gap in vision contrastive learning.

### 3. The Specific Failure Mode

At temperature=0.07 with batch_size=512, the sharp loss landscape is dominated by whatever explains the most label variance — and question difficulty is correlated with hallucination rate. The encoder will use question identity as a shortcut because it provides the most stable, high-magnitude gradient signal.

**Expected symptom:** High in-distribution AUROC (recognizing familiar question clusters) but degraded out-of-distribution AUROC on held-out question types. This is testable.

### 4. What Resampling Provides

Same-question, different-label pairs create the strongest possible gradient signal: two activations from identical prompts that must be placed on opposite sides of the decision boundary. The only way to satisfy this is to encode **response-level** features (generated tokens, uncertainty patterns, factual coherence) rather than **prompt-level** features (entity obscurity, question structure).

### 5. Expected Gains

| Metric | Expected Impact | Reason |
|--------|----------------|--------|
| AUROC on held-out questions | Large improvement | Breaks question-identity generalization ceiling |
| Calibration (ECE, Brier) | Moderate improvement | Confidence becomes conditional on response, not question |
| Precision at high recall | Improvement on "easy question, hallucinated answer" cases | Currently under-represented in training |
| Question-identity probe accuracy | Should decrease | Diagnostic of reduced confound |

---

## The Case Against Resampling

### 1. The Diagnosis Is Unverified

The question-identity confound is a hypothesis, not an observed failure mode. Before investing in resampling, the confound should be confirmed via:
- Linear probing: can a classifier predict question identity from the learned embeddings?
- Within-question variance analysis in the latent space
- Performance stratified by question hallucination rate

Simpler explanations for poor performance exist: suboptimal temperature, weak architecture, noisy labels, or simply undertrained models. These are cheaper to investigate.

### 2. The "Same Hallucination" Problem

LLMs under standard sampling temperatures are nearly deterministic on factual questions. Resampling a question the model reliably gets wrong produces:

```
Sample 1: "The Eiffel Tower was built in 1887."  [hallucination, label=1]
Sample 2: "The Eiffel Tower was built in 1887."  [hallucination, label=1]
Sample 3: "The Eiffel Tower was built in 1887."  [hallucination, label=1]
```

Three correlated copies of the same activation pattern, all the same label. Zero new contrastive signal. The confound is not broken; it is replicated.

**The epistemic trap:** Resampling is most useful when the model is uncertain (stochastic outputs, mixed labels per question). But when the model is uncertain, labels are least reliable. Resampling is least useful (all same answer) precisely when labels are most reliable.

### 3. Class Imbalance Amplification

If a "hard" question reliably produces hallucinated responses, resampling amplifies the class imbalance: more hallucinated examples per hard question, fewer per easy question. The within-class pairs added have weight=0.5 and contribute little novel signal. The encoder now has more correlated hallucinated examples clustered around hard question types — reinforcing, not breaking, the confound.

### 4. Alternative Explanations for Performance Gaps

- **Layer selection:** Layers 14-29 may not be the optimal range; factual recall mechanisms are layer-specific in Llama-3.1-8B. This is an empirical question.
- **Sequence-level pooling:** Mean pooling destroys token-level information about *where* the hallucination occurs.
- **Weak positive pairs:** Treating layer 14 and layer 28 as equivalent views is architecturally unjustified — they encode qualitatively different information.
- **Temperature miscalibration:** τ=0.07 is borrowed from vision; language activation spaces have different geometry.
- **Binary label noise:** Partial correctness, hedged hallucinations, and coarse annotation collapse important distinctions.

### 5. Better Alternatives With Lower Cost

| Intervention | Compute Cost | Expected Gain |
|---|---|---|
| Diagnose confound with probing classifiers | Trivial | Confirms/denies the hypothesis |
| Temperature sweep (τ ∈ [0.05, 0.5]) | Trivial | High leverage, zero data cost |
| Loss weight tuning (same_sample_weight, same_class_weight) | Trivial | Directly controls pair weighting |
| Hard negative mining | O(batch²), zero new data | Informative pairs from existing data |
| Token-level contrastive objective | Medium | Richer signal from claim-bearing tokens |
| Learned layer-attention fusion | Medium | Adaptive layer importance per input |
| Resampling (3x) | 3x inference + storage | Dependent on stochasticity of model |

---

## Key Points of Agreement

Both positions agree on:

1. **The core goal:** The encoder should learn response-level hallucination features, not question-level difficulty features
2. **The contrastive signal problem is real:** Whether caused by question-identity confound or other factors, the positive pair diversity is insufficient
3. **Resampling is not wrong in principle** — it is a valid intervention if designed carefully (high temperature, varied prompting, chain-of-thought variations)
4. **Verification before investment:** The confound hypothesis should be confirmed before committing to the 3x compute overhead

---

## Resolution: When to Resample, and How

### First: Diagnose

Before resampling, run the diagnostic:
```python
# Train a linear probe on learned embeddings to predict question_id
# If accuracy >> chance, the confound is real
```
If the probe shows strong question-identity encoding, resampling is justified. If not, algorithmic improvements should come first.

### Second: Resample Strategically, Not Naively

If resampling is warranted, naive temperature sampling is insufficient. Effective resampling requires:
- **High sampling temperature** (T ≥ 0.8) to generate semantically distinct responses
- **Varied prompting strategies** (paraphrased questions, chain-of-thought, few-shot variants) to maximize activation diversity
- **Label verification across samples** to confirm within-question label variation exists before using the sample
- **Discarding within-question same-label resamples** that provide no contrastive signal

### Third: Combine With Algorithmic Improvements

Resampling and algorithmic improvements are not mutually exclusive. The highest-value path is:
1. Fix temperature and pair weights (cost: trivial)
2. Implement hard negative mining (cost: low)
3. If confound is confirmed by probing, add targeted resampling for questions with confirmed within-question label variation

---

## Verdict

**Do not resample as a first intervention.** The question-identity confound is plausible but unverified; the algorithmic alternatives are cheaper and potentially higher-value; and naive resampling is likely to reproduce the same hallucinated answers rather than generate the diverse cross-label pairs that would fix the problem.

**Resampling becomes the right investment** when:
- Probing classifiers confirm strong question-identity encoding in learned representations
- Low-cost alternatives (temperature, loss weights, hard negative mining) have been explored and their gains plateaued
- A sampling strategy is in place that actually produces within-question label variation (not just correlated copies)

The core insight from the supporter is correct: the data structure has a confound. The core insight from the critic is also correct: you need to confirm the confound exists and exhaust cheaper fixes before paying 3x inference cost for potentially correlated duplicates.

---

## Diagnosing the Confound: Experimental Plan

The following experiments were designed through a researcher/critic debate to confirm or deny whether the encoder learns question-level features rather than response-level hallucination features. One naive diagnostic — predicting question_id from embeddings — is meaningless here: with 1 activation per question, every question is a unique class and the problem is trivially perfect.

### Structural Caveat (applies to all experiments)

The dataset has multiple questions per Wikipedia entity, and entities with mixed hallucination labels exist. The confound likely operates at the **entity level** (obscure entities → more hallucinations), not the question level per se. All diagnostics below test for entity-level confound, which is the most plausible form of the question-identity problem.

---

### Priority 1 — Prompt vs. Response Token Activation Split

**What it tests:** Whether the encoder's discriminative signal comes from prompt-token positions (confound) or response-token positions (genuine hallucination detection).

Each stored activation tensor covers all token positions. Split into:
- `A_prompt` — activations at prompt token positions (mean-pooled)
- `A_response` — activations at response token positions (mean-pooled)

Train separate hallucination probes on each. If `A_prompt` achieves similar or higher AUROC than `A_response`, the encoder is exploiting the question representation. If `A_response` dominates, the signal is response-driven.

**Cost:** Zero new inference. Requires per-token activation storage — verify whether Zarr logs store per-token activations or only pooled. If only pooled, this requires one re-inference pass on REMOTE_GPU.

**Critic's note:** This is the most mechanistically precise test. It directly separates prompt-driven from response-driven signal without relying on entity groupings.

---

### Priority 2 — PCA Structure: Entity vs. Label Variance

**What it tests:** Whether the dominant axes of variation in the embedding space align with entity identity or hallucination label.

1. Compute PCA on the full embedding matrix (N_samples × embedding_dim)
2. For each of the top K principal components (K = 20–50), compute:
   - `R²_entity`: variance explained by entity_id (via ANOVA or entity-dummy regression)
   - `R²_label`: variance explained by hallucination label
3. Plot R²_entity vs. R²_label per PC

**Confound pattern:** Top PCs are entity-aligned; label-aligned PCs appear only in PC 10+.
**Healthy pattern:** Label-aligned PCs appear in top 3–5 PCs.

**Cost:** Zero new inference. Runs on LOCAL_CPU in minutes using existing embeddings from a trained checkpoint.

---

### Priority 3 — Layer-Wise Confound Localization

**What it tests:** Whether entity-level and label-level information are encoded at different layers — and whether the contrastive training window (layers 14-29) happens to include the most confounded layers.

For each layer 14–29, train two lightweight linear probes:
- Probe A: predict hallucination label from that layer's mean-pooled activations
- Probe B: predict entity_id from that layer's mean-pooled activations (multi-class)

Plot both accuracy curves against layer index.

**Actionable outcome:** If entity-decodability peaks in early layers (14-18) and hallucination-decodability peaks in later layers (24-29), the training window should be narrowed to avoid the confounded layers. If they covary throughout, the problem is structural, not a layer-selection issue.

**Cost:** Zero new inference. 32 lightweight logistic regression probes on stored per-layer activations. LOCAL_CPU feasible.

---

### Priority 4 — Entity-Stratified Cross-Validation (With Normalization)

**What it tests:** Whether AUROC drops when entities are fully held out of training (all questions from an entity go to train OR test, never both).

Run evaluation under:
- **Random split:** Questions split randomly (entity leakage allowed)
- **Entity-stratified split:** All questions from each entity assigned to one split only

**Critical normalization (critic's requirement):** Run the same two splits on a text-only baseline (logistic regression on question length, entity description length, `h_score_cat`). The diagnostic signal is the *excess* drop in the activation model beyond what the text baseline drops:

```
confound_signal = delta_AUROC(activations) - delta_AUROC(text_baseline)
```

If `confound_signal ≈ 0`, the activation encoder's drop is explained by distribution shift, not entity-specific confounding. If `confound_signal > 0`, entity-level features are being exploited by the activation encoder specifically.

**Cost:** Zero new inference.

---

### Priority 5 — Within-Entity AUROC on Mixed-Label Entities

**What it tests:** Whether the encoder can separate hallucinated from non-hallucinated responses *within* a single entity — controlling for entity identity completely.

For entities with both hallucinated and non-hallucinated questions (N ≥ 4 per entity), compute AUROC using only within-entity comparisons (score = cosine distance from within-entity non-halu centroid).

**Confound pattern:** Mean within-entity AUROC ≈ 0.5 — the encoder cannot separate within-entity.
**Healthy pattern:** Mean within-entity AUROC substantially above 0.5 (e.g., 0.65+).

**Critic's note:** Report the full distribution (per-entity AUROC histogram), not just the mean. Also report within-entity AUROC as a function of entity sample size — if AUROC rises with more samples per entity, the signal is real but weak, not absent.

**Cost:** Zero new inference.

---

### Priority 6 — Label-Shuffle Permutation Test (Corrected Design)

The originally proposed design (shuffle labels within entity, check if AUROC stays above 0.5) is ambiguous for identity confounds. The corrected design:

Train a **linear probe** on frozen encoder embeddings. Shuffle labels 500–1000 times and record probe AUROC under each shuffle. If the true-label AUROC falls above the 95th percentile of the shuffle distribution, the encoder contains genuine label-predictive geometry. Report `p-value = rank(true AUROC) / n_permutations`.

**Cost:** Zero new inference. Trivially fast.

---

### Priority 7 — Multi-Response Inference (Ground Truth Test)

The only fully definitive test. Generate 4–5 responses per question at temperature ≥ 0.8 for a held-out subset of ~200 questions (100 entities). Label each response via the existing evaluation pipeline. For each question, compute within-question AUROC: can the encoder rank hallucinated responses above non-hallucinated ones for the *same question*?

**Confound pattern:** Within-question AUROC ≈ 0.5 (same question → same embedding regardless of response content).
**Healthy pattern:** Within-question AUROC substantially above 0.5.

**Critical metric refinement (critic's note):** Don't just cluster same-question embeddings — measure the **cross-label within-question silhouette score** specifically. Clustering is expected even without the confound because all 5 responses share prompt tokens.

**Cost:** ~2 GPU hours on REMOTE_GPU. Only run if Priorities 1–6 are inconclusive.

---

### Summary Table

| Priority | Experiment | New Inference? | Answers |
|---|---|---|---|
| 1 | Prompt vs. response token split | Only if pooled storage | Are prompt or response tokens the signal source? |
| 2 | PCA entity vs. label variance | No | Does embedding space geometry favor entity or label? |
| 3 | Layer-wise probe accuracy | No | Which layers are confounded? |
| 4 | Entity-stratified CV (normalized) | No | Does entity hold-out drop exceed text-baseline drop? |
| 5 | Within-entity AUROC | No | Can encoder discriminate within one entity? |
| 6 | Linear probe permutation test | No | Does encoder geometry contain genuine label signal? |
| 7 | Multi-response inference | Yes (~2 GPU hours) | Definitive within-question discriminability |

Run Priorities 1–3 first — they are mechanistic (locate the confound structurally) and have zero data cost. Priorities 4–6 are statistical validation. Priority 7 is the ground truth confirmation if earlier results are ambiguous.
