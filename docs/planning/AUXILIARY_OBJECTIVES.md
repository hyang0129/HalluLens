# Auxiliary Objectives for Hallucination Detection

## Motivation

When we added a **log-probability prediction head** as an auxiliary objective alongside our (non-contrastive) classifier, hallucination-detection performance improved. The intuition:

> An auxiliary task that predicts something *correlated with hallucination* forces the encoder to retain features that the main classifier might otherwise discard. The main hallucinated-vs-not label is binary and noisy; auxiliary signals are denser and inject useful inductive bias.

This is the standard multi-task / auxiliary-head argument: shared representations regularize each other, and gradient signal from the aux head reaches features that a single binary head would compress away.

The natural follow-up: **what else could we predict?** Below are candidates grouped by what kind of signal they expose.

---

## 1. Token-distribution signals (uncertainty proxies)

Cheap to compute, already correlated with hallucination, and densely available per token.

| Target | Why it might help |
|---|---|
| **Next-token entropy** | Captures distributional spread, not just argmax. Complements log-prob (peakedness vs mass). |
| **Top-1 / top-2 margin** | Decision-boundary tightness; flat tops indicate the model is hedging. |
| **Min-token-prob across answer span** | The weakest link in the answer is often where confabulation happens. |
| **Sequence perplexity of the generated answer** | Aggregates uncertainty across the whole answer rather than a single token. |
| **Rank of the gold token** (when available) | Sharper than log-prob at distinguishing "in the top few" vs "completely off." |

These are essentially "different views of the output distribution." Cheap because they're already in the logits.

## 2. Self-consistency / sample-variance signals

Multiple-sample agreement is one of the strongest known hallucination signals (SelfCheckGPT, semantic entropy). If the encoder can predict it from a *single* forward pass, we get the signal at inference without resampling.

| Target | Why it might help |
|---|---|
| **Semantic entropy over k samples** | Direct proxy for the canonical "would this answer be stable?" question. |
| **Greedy-vs-sampled agreement** (binary) | Cheap label, captures whether temperature unlocks a different answer. |
| **# distinct answer clusters** across k samples | Coarse but informative; classification rather than regression. |
| **Pairwise embedding similarity of samples** | Continuous, well-defined gradient. |

Cost: requires offline sampling to *generate* labels, but only at training time.

## 3. Knowledge / popularity signals

Hallucinations concentrate on long-tail entities. If activations encode "how familiar is this entity," the classifier can use that prior.

| Target | Why it might help |
|---|---|
| **Entity popularity** (Wikipedia pageviews, PopQA's `s_pop`) | Direct long-tail indicator; PopQA already exposes it. |
| **Pretraining frequency tier** (bucketed) | Same idea, classification version. |
| **"Was this entity in the prompt?"** | Forces locality awareness: copying vs generating. |

PopQA gives this label for free. SciQ / SearchQA / MMLU don't, but Wikipedia-page-view lookups are cheap to add.

## 4. Self-evaluation signals (P(True)-style)

The model itself, prompted differently, often knows. If we can predict *its own* self-evaluation from intermediate activations, we recover that signal without a second pass.

| Target | Why it might help |
|---|---|
| **P(True)** (Kadavath): probability the model assigns to "the answer is correct" when asked | Strong empirical baseline; making it predictable distills it into the encoder. |
| **Verbalized confidence** ("I'm 80% sure") parsed from a confidence-elicitation prompt | Same idea, different elicitation format. |
| **Refusal/hedging probability** | Predict whether the model *would* hedge if asked to. |

## 5. Internal-dynamics signals (no extra labels needed)

These come for free from the forward pass — no annotation, no resampling.

| Target | Why it might help |
|---|---|
| **Logit-lens prediction at earlier layers** | Forces the layer to encode "where the answer is converging." Disagreement between early and late lens predictions correlates with hallucination. |
| **Final hidden state** (predict from middle layer) | Forward distillation; encourages the middle layer to carry late-layer semantics. |
| **Attention entropy / head specialization at later layers** | Captures "is the model still searching or has it committed?" |
| **Layer-to-layer activation delta norm** | Proxy for how much processing remains; flat trajectories often indicate confident (correct or confidently-wrong) answers. |

These are attractive because the labels are *deterministic functions of the same forward pass*, so dataset construction is trivial.

## 6. Question / context features

Predicting properties of the *input* rather than the output. Useful when input difficulty drives hallucination.

| Target | Why it might help |
|---|---|
| **Question difficulty** (model's empirical accuracy on similar questions) | Directly conditions on "is this the kind of question I get wrong?" |
| **Question type** (factoid / reasoning / ambiguous) | Different failure modes; lets the head route. |
| **Answer cardinality** (unique answer vs many acceptable) | Ambiguity is often misclassified as hallucination. |

Labels for these typically need a separate annotation or heuristic pipeline.

---

## How to choose among these

Roughly ranked by **expected ROI** for our setup (Llama-3.1-8B / Qwen3-8B on PopQA / SciQ / SearchQA / MMLU):

1. **Next-token entropy + top-k margin** — almost free, complementary to log-prob, no new pipeline.
2. **Logit-lens at intermediate layers** — free labels, directly tied to the "intermediate representation" thesis of the paper.
3. **Semantic entropy / sample agreement** — strongest empirical signal in the literature; expensive to label but only once.
4. **P(True)** — well-validated; needs a second forward pass per training example, but offline only.
5. **Entity popularity** — free for PopQA, easy to add elsewhere; helps long-tail explicitly.
6. **Final hidden state distillation** — cheapest of all; worth trying as an ablation.

Lower priority for now:
- Question-feature prediction (label cost is high relative to expected lift).
- Verbalized confidence (noisy, format-sensitive).

## Compatibility with the contrastive setup

The contrastive objective pulls together activations from correct generations and pushes apart hallucinated ones. Auxiliary regression/classification heads attach to the *same encoder* and are co-trained. Two design choices to settle before running:

- **Head placement**: per-layer aux head vs single head on the pooled contrastive embedding. Per-layer is more aligned with the existing `train-layers` interface and probes which layer encodes which signal.
- **Loss weighting**: start with equal weighting then sweep; multi-task training is sensitive here. A single dominant aux loss can hijack the encoder.

## Suggested first experiment

Smallest change that tests the hypothesis beyond log-prob:

- Same encoder + contrastive loss as current best run.
- Add **two** aux heads: (a) next-token entropy regression, (b) logit-lens top-1 agreement with final layer (binary).
- Compare against: contrastive-only, contrastive + log-prob (current best).
- Dataset: PopQA test (long-tail signal) + SciQ test (clean factoid baseline).

If either head yields lift comparable to the log-prob head, the auxiliary-objective hypothesis generalizes and the next step is the more expensive labels (semantic entropy, P(True)).

---

## Detailed spec: the two proposed aux heads

### (a) Next-token entropy regression

**What it is.** For every token position `t` in the generated answer, the model produced a distribution `p_t = softmax(logits_t)` over the vocab. The label is the Shannon entropy of that distribution:

```
H_t = -Σ_v p_t(v) · log p_t(v)
```

The aux head, attached to the intermediate-layer activation at position `t`, regresses to `H_t`.

**Why it complements log-prob.** The current log-prob head predicts `log p_t(y_t)` — "how confident was the model in the token it actually emitted." Entropy predicts "how peaked was the *whole* distribution."

These differ when:
- The model is confident in the *wrong* token (low entropy, low log-prob of gold) → entropy disagrees with log-prob.
- The distribution is bimodal between two reasonable answers (moderate entropy, moderate log-prob) → entropy captures the spread that log-prob misses.
- A single token has 0.4 mass and the rest is diffuse vs. two tokens at 0.4/0.4 → both have similar top-1 prob but very different entropy.

So entropy is a strict-second-moment signal that log-prob does not encode.

**Label computation.**
- Already free if the inference run captured logits. Check `activation_logging/` — if logits aren't stored per-token, this requires either (i) re-running with logit capture, or (ii) computing entropy on the fly during the activation-dump pass.
- Truncate-to-top-K entropy is a reasonable approximation if vocab-wide softmax is too memory-heavy: `H ≈ -Σ_{v ∈ topK} p_v log p_v + tail_correction`. K=100 is usually within 1% of full entropy for LM distributions.
- Numerical: clamp `p_t(v)` to `≥ 1e-12` before log; or use `log_softmax` and compute `-Σ exp(log_p) · log_p`.

**Head shape.** Single linear layer per train layer → scalar, MSE loss against `H_t`. Normalize the target: entropies for an 8B model on factoid answers typically range 0–6 nats; standardize to zero mean / unit variance over the training set so the loss scale matches the contrastive loss.

**Token aggregation.** Either (a) per-token loss, averaged over the answer span (matches the contrastive setup if it's also per-token), or (b) predict only the entropy of the *last* answer token (cleaner, weaker signal). Start with (a).

**Loss weight.** Begin with `λ_entropy = 0.1 · λ_logprob` since entropy has higher variance than log-prob and can dominate gradients if unweighted. Sweep `{0.05, 0.1, 0.5, 1.0}`.

**Sanity checks before training.**
- Pearson correlation between `H_t` and ground-truth hallucination label on a held-out set. If `|r| < 0.05`, the signal is too weak to help — pick a different aux objective.
- Check label distribution: if entropy is bimodal at 0 and ~max, regression may be wasteful and a binary "high/low entropy" head is simpler.

### (b) Logit-lens top-1 agreement (binary)

**What it is.** The "logit lens" projects an intermediate-layer hidden state through the final unembedding matrix `W_U`, producing a pseudo-prediction at that layer:

```
logits_ℓ,t = LayerNorm_final(h_ℓ,t) · W_U
ŷ_ℓ,t     = argmax(logits_ℓ,t)
```

The label is whether layer `ℓ`'s top-1 matches the *final* layer's top-1:

```
agree_ℓ,t = 1[ŷ_ℓ,t == ŷ_final,t]
```

The aux head — a binary classifier on the layer-`ℓ` activation — predicts this label.

**Why this is interesting for hallucination detection.** Empirically (logit-lens / tuned-lens literature), in factual recall the answer "snaps in" at a specific layer band. Hallucinations tend to:
- Snap later (the model is still computing when emission happens), or
- Oscillate (early agreement, then disagreement, then re-agreement), or
- Show early commitment to a *wrong* token that later gets corrected — or doesn't.

A head that predicts "does this layer already agree with the final answer" forces the encoder to represent **convergence dynamics**, which a binary correctness label doesn't directly supervise.

**Label computation.**
- Free from the forward pass — no resampling, no external annotation.
- Requires applying the model's final `LayerNorm` + `lm_head` to intermediate hidden states. For Llama / Qwen3 this is `model.model.norm` followed by `model.lm_head`. Apply both; skipping the final LN gives a noisier lens.
- `ŷ_final` is just the emitted token (greedy) or the argmax of the final-layer logits (for sampled generations, use the argmax — emitted token is a sample, not the model's "best guess").
- Pre-compute once when dumping activations; store as a `(num_layers, seq_len)` boolean array per example.

**Head shape.** Linear → 2-class logits, BCE/cross-entropy loss. Per-layer instance — each train layer gets its own agreement label (the label *changes* by layer, which is the whole point).

**Subtle point: label leakage.** The label `agree_ℓ,t` is computed from `h_ℓ,t` itself via `W_U`. The aux head is a *different* function of `h_ℓ,t` predicting that same label. This is fine — it's not leakage in the train/test sense — but it does mean a sufficiently expressive aux head can perfectly fit the label by re-implementing the unembedding. Mitigations:
- Keep the head shallow (single linear layer, low rank if needed).
- Optionally predict agreement at a *different* layer than the encoder layer (e.g., encoder reads layer 14, predicts whether layer 20's lens agrees with final). This forces forward prediction rather than reconstruction.
- The shallow-head approach is fine for the first experiment; revisit if results look suspicious.

**Token aggregation.** Per-token, averaged over the answer span — same as entropy.

**Loss weight.** Binary cross-entropy is naturally well-scaled; start with `λ_lens = λ_logprob` and sweep `{0.5, 1.0, 2.0}`.

**Sanity checks before training.**
- Plot `mean(agree_ℓ,t)` by layer on the train set. Expect a monotonic-ish rise from low layers to ~1.0 at the final layer. If it's flat, the lens is uninformative for these models (would be surprising) and the head won't help.
- Compare mean-agreement curves between hallucinated and non-hallucinated examples. If they overlap completely, the *signal* is weak even though the *labels* are well-defined; consider switching to "agreement with the *gold* token" instead of with the final-layer prediction. (Requires gold token alignment, which we have for short-answer datasets.)

### Joint training recipe

```
total_loss = L_contrastive
           + λ_logprob · L_logprob          (existing)
           + λ_entropy · L_entropy          (new, head a)
           + λ_lens    · L_lens             (new, head b)
```

- Same encoder, three aux heads, one contrastive objective.
- Gradients from all heads flow into the encoder; heads themselves are independent.
- Per-layer heads if `--train-layers` spans multiple layers (matches existing convention).

### Ablation matrix (minimal)

| Run | contrastive | logprob | entropy | lens |
|---|---|---|---|---|
| baseline | ✓ | – | – | – |
| current best | ✓ | ✓ | – | – |
| +entropy | ✓ | ✓ | ✓ | – |
| +lens | ✓ | ✓ | – | ✓ |
| +both | ✓ | ✓ | ✓ | ✓ |
| entropy-only | ✓ | – | ✓ | – |
| lens-only | ✓ | – | – | ✓ |

The last two rows isolate each new head's contribution independent of log-prob, which matters if entropy/lens substantially overlap with log-prob in what they teach the encoder.

### Datasets

PopQA test + SciQ test for the first sweep — long-tail and clean-factoid respectively. Add SearchQA / MMLU only if both heads show consistent lift on the first two; otherwise the marginal information per dataset is not worth the compute.
