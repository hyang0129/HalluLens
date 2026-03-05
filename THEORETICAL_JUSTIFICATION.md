# Theoretical Justification: Detecting Intrinsic Hallucinations via Intermediate-Layer Compression

## Core Hypothesis

Intrinsic hallucinations — cases where the model generates claims that contradict its own parametric knowledge — leave a detectable signature in the intermediate-layer activations of an LLM. A learned compression of those activations can extract and amplify this signal, enabling reliable hallucination detection without external retrieval or multiple generations.

---

## Justification 1: Compression Exposes Incoherence in Factual Grounding (Information Bottleneck)

**The argument.** The information bottleneck framework (Tishby & Schwartz-Ziv, 2017) characterizes deep networks as performing successive lossy compressions: each layer discards information irrelevant to the prediction task while preserving task-critical structure. For an LLM generating a *factually grounded* answer, the intermediate layers enact a coherent compression pathway: raw token statistics are progressively refined into semantically structured, factually consistent representations. Every layer "agrees" on the same underlying fact.

When the model *hallucinates*, this coherent pathway breaks down. The model is generating plausible surface text without a stable factual referent in its parametric memory. The resulting activations encode a compositional fiction — internally inconsistent across the representational hierarchy. A learned compressor (e.g., the `ProgressiveCompressor` in `activation_research/model.py`) trained to map these activations into a low-dimensional embedding space will map coherent, factually grounded responses to a tight, well-structured region of that space, and incoherent hallucinated responses to diffuse or outlying regions.

**Why compression specifically helps.** Raw high-dimensional activations (4096 dims) are high-rank and entangled with task-irrelevant variation (input formatting, stylistic register, entity surface form). The compression forces the model to discard this surface variation and retain only the information that is consistent across training examples — which, under contrastive training with hallucination labels, is precisely the factual grounding signal.

**Prediction.** Hallucinated responses should lie far from the distribution of correct responses in the compressed embedding space — measurable via Mahalanobis distance or K-NN distance. The separation should be larger in the compressed space than in the raw activation space, because compression removes uninformative dimensions.

**Critical challenge.** The information bottleneck argument assumes the compressor is trained to discard the *right* information. If the training set is small or unbalanced, the compressor may discard factual grounding signal along with noise. This motivates the contrastive objective: it explicitly supervises which dimensions to preserve by requiring that the same example's hallucination label is reflected in the embedding geometry.

---

## Justification 2: Cross-Layer Consistency as a Proxy for Internal Certainty (Multi-View Coherence)

**The argument.** A transformer's residual stream accumulates information through successive layers. For a factual claim the model has stored reliably in its weights, the representation of that claim should be *stable* across layers: early-middle layers build compositional semantic structure, middle layers perform factual retrieval and association, and late layers encode output-format and syntactic constraints. When all stages operate on the same well-grounded fact, the activations at layer $l_1$ and layer $l_2$ are "views" of the same underlying knowledge state — semantically similar up to a learned rotation.

When the model hallucinates, this cross-layer stability is disrupted. The late layers decode fluent output from an internal state that the middle layers never fully grounded in a consistent factual association. The representation at a late layer is "ahead" of where the middle layers left off — the two views become inconsistent.

**Why contrastive training captures this.** The contrastive objective in `training.py` treats activations from two different layers of the same generation as two "views" and trains the compressor to align them. For correct answers, the compressor learns to produce similar embeddings from both views (high inter-layer agreement). For hallucinated answers, the views are structurally inconsistent, so the compressor cannot achieve this alignment — producing embeddings with high inter-layer distance. At inference time, this divergence between the two compressed views becomes the detection signal.

This is empirically grounded in the "layer sweep" finding recorded in `PAPER_ROADMAP.md` (Section IDEAS_TO_TEST §C8): the strongest detection signal comes from *specific layer pairs*, not all pairs uniformly — consistent with the prediction that cross-layer inconsistency is layer-specific and concentrated in particular transitions in the residual stream.

**Prediction.** The detection signal should peak at layer pairs that span a factual retrieval transition (typically mid-to-late layers, e.g. layers 14–29 in a 32-layer LLaMA model). Pairs confined to early or very late layers should show weaker separation.

**Critical challenge.** Cross-layer consistency is also disrupted by input complexity and ambiguity — difficult questions may exhibit high layer divergence even when the model answers correctly. This motivates the relative Mahalanobis baseline (IDEAS_TO_TEST §A2): normalizing the detection signal by a background distribution can remove the "hardness" confound.

---

## Justification 3: Activations Encode Epistemic State Separably from Output (Intrinsic Knowledge Representation)

**The argument.** An LLM's output is a deterministic function of its final hidden state through a single linear projection (the unembedding matrix). This projection is optimized for fluency and task format — not for calibrated uncertainty. It is a lossy encoding: much of the information encoded in the high-dimensional final hidden state is not recoverable from the output distribution alone. The implication is that the model's *epistemic state* — its internal representation of whether it knows a fact reliably — is encoded in the residual stream in a form that is not well-reflected in output logprob or token probability.

This is the central insight formalized in "LLMs Know More Than They Show" (Orgad et al.) and "Detecting LLM Hallucination Through Layer-wise Information Deficiency" — both referenced in `PAPER_ROADMAP.md` §3C. The model's internal representations carry factual uncertainty signals that are discarded before output generation.

**Why intermediate layers, not the last layer.** The last layer's activations are shaped heavily by the output projection objective — they are optimized to produce the next token, not to represent epistemic uncertainty. Intermediate layers, operating before this final collapse, preserve more of the model's internal uncertainty structure. This predicts the core empirical finding in the roadmap: *contrastive training on intermediate-layer activations outperforms a last-layer classifier baseline*.

**Why compression helps extract this signal.** The intrinsic uncertainty signal is not a single dimension of the activation space — it is a distributed, entangled pattern across the 4096-dimensional residual stream. A learned compression trained with contrastive supervision disentangles this signal from unrelated variation (entity salience, positional effects, surface form) and projects it into a low-dimensional space where distance metrics like Mahalanobis become effective. The `ProgressiveCompressor` architecture — cascaded dimensionality reduction via transformer blocks — is well-suited to this because it can learn nonlinear rotations of the activation space that expose the hallucination-correlated subspace.

**Prediction.** Logprob-based baselines (from `external/LLMsKnow/logprob_detection.py`) should be strictly weaker than intermediate-layer activation methods, because logprob is derived from the same lossy final projection that discards the epistemic signal. The activation-based method should show the largest advantage on cases where the model is *confidently wrong* — high output logprob but incorrect answer — which are precisely the intrinsic hallucinations this method targets.

**Critical challenge.** The signal is only accessible if activations are logged from an open-weights model with internal access. This limits the method to white-box settings. Furthermore, the signal is expected to be model-architecture-specific: the layer at which factual retrieval occurs varies across model families, requiring layer-sweep calibration per model. The paper should clearly bound the method's applicability to this white-box setting and compare only to other white-box baselines (probing, logprob) on equal footing.

---

---

## Justification 4: Output Logit Uncertainty as a Compression-Guiding Auxiliary Signal

**The general argument.** Justification 3 establishes that logprob is a *weaker* signal than intermediate-layer activations because the output projection discards epistemic structure. This does not preclude logprob from being useful as a *conditioning* or *routing* signal over the compression pathway. The key insight is that logprob encodes *where in the response sequence* the model expressed uncertainty — positional uncertainty — whereas activations encode *what epistemic state the model was in* — representational uncertainty. These two uncertainty axes are not redundant; a model can be internally incoherent (bad activations) while producing high-confidence tokens (low-entropy logprob), or internally coherent while generating semantically uncertain hedges. Combining both axes into the compression should yield richer separation than either alone.

Five distinct mechanisms follow. They vary in invasiveness (from lightweight pooling changes to new contrastive view types) and in their data requirements (some need per-token logprobs at training time; others only need them at inference).

---

### Mechanism A: Uncertainty-Weighted Sequence Pooling

**The argument.** `ProgressiveCompressor` currently performs uniform mean pooling over the sequence dimension after the final transformer block ([`model.py:102`](activation_research/model.py#L102)). This treats every generated token position equally — a position where the model was highly confident and a position where it was nearly random contribute identically to the compressed summary vector. However, uncertain token positions (low log probability of the generated token) are precisely where the model's epistemic state was weakest during generation. These positions are more likely to be the *sites of hallucination events* — where the model committed to a token without strong parametric backing. Upweighting these positions during pooling biases the compressed representation toward the most diagnostically informative parts of the sequence.

Formally, given per-token log probabilities $\ell_i = \log p(t_i \mid t_{<i}, \text{prompt})$ for generated tokens $t_1, \ldots, t_L$, define uncertainty weights:
$$w_i = \frac{\exp(-\lambda \cdot \ell_i)}{\sum_j \exp(-\lambda \cdot \ell_j)}$$
where $\lambda > 0$ is a temperature parameter controlling sharpness. The pooling step becomes $z_{\text{pool}} = \sum_i w_i h_i$ where $h_i$ is the token representation after the final transformer block. At $\lambda = 0$ this recovers uniform mean pooling. At large $\lambda$ it approaches max-uncertainty pooling — selecting the single most uncertain token.

This is a **no-parameter change** to the compressor's forward pass (given logprobs as an auxiliary input) and is compatible with all existing training infrastructure. The weights can also be computed from token entropy $H_i = -\sum_v p_{iv} \log p_{iv}$ (over the full vocabulary distribution) rather than from the scalar log probability of the chosen token, which captures full distributional uncertainty rather than the realized token's probability.

**Prediction.** The pooling-reweighted compressor should show better hallucination detection on *short factual answers* where hallucination is concentrated in one or two content tokens (e.g., "The capital of X is [wrong city]") — here the uncertain token is the factual claim, and its activation should dominate the pooled summary. Benefit should be smaller on long-form text where uncertainty is distributed broadly.

**Critical challenge.** If the model is *confidently wrong* — the core intrinsic hallucination case — the logprob of the hallucinated token will be *high* (the model was certain of the wrong answer). In this regime, uncertainty-weighted pooling actively deprioritizes the hallucinated token position, making it *worse* than mean pooling. This is the fundamental limitation of logprob as a signal: it cannot distinguish *confident-correct* from *confident-wrong* at the token level. Uncertainty-weighted pooling helps primarily for cases where the model expresses overt uncertainty, not for confident intrinsic hallucinations. This argues for treating it as a *complementary* rather than *replacement* mechanism.

---

### Mechanism B: Entropy-Profile Conditioning via Extended FiLM

**The argument.** The existing `FiLMConditioner` and `LayerAwareProgressiveCompressor` condition the encoder on a *discrete* layer index — a one-hot style signal distinguishing which layer's activations are being compressed. This captures inter-layer distribution shift but carries no information about the *generation event's* uncertainty profile. A natural extension is to condition the encoder on a *continuous* summary of the output entropy profile, treating it as a "difficulty" or "epistemic regime" descriptor.

The entropy profile of a response can be summarized into a fixed-size descriptor in several ways:

- **Scalar summary:** Mean token entropy $\bar{H} = \frac{1}{L}\sum_i H_i$, capturing overall response difficulty.
- **Percentile summary:** $(H_{\text{min}}, H_{\text{p25}}, H_{\text{median}}, H_{\text{p75}}, H_{\text{max}})$, capturing the *shape* of the entropy distribution rather than just its level.
- **Temporal summary:** Entropy at the first content token, peak entropy position, and entropy at the final token — capturing the *trajectory* of uncertainty across the response.
- **Spectral summary:** DFT coefficients of the entropy sequence, capturing oscillatory patterns (e.g., regular uncertainty spikes at semantic boundaries).

The extended FiLM applies a learned affine transformation to the encoder representations conditioned on this descriptor:
$$h' = \gamma(e_H) \odot h + \beta(e_H)$$
where $e_H$ is the entropy summary vector and $\gamma, \beta$ are learned MLP projections (rather than embedding lookup tables, since $e_H$ is now continuous). This is a direct extension of the existing `FiLMConditioner` from discrete to continuous conditioning.

**Why this helps the compression.** The compressor must currently handle activations from generations with very different difficulty profiles — a trivial factual question and a contested historical question both produce 4096-dimensional activations at layer 22, but the "meaning" of those activations differs by epistemic regime. Entropy conditioning allows the compressor to recalibrate its representation based on how uncertain the generation event was overall, potentially separating the "hard but correct" regime from the "hard and hallucinated" regime.

**Prediction.** Entropy conditioning should show the largest benefit on *calibrated* models where output entropy correlates with factual uncertainty. On poorly calibrated models (overconfident across the board), the entropy profile carries little signal and conditioning will degrade to near-identity transforms. Empirically, LLaMA-3.1 Instruct shows reasonable calibration on factual QA, suggesting this mechanism is viable for the primary target model.

**Critical challenge.** Computing full vocabulary entropy $H_i$ requires materializing the full softmax distribution at every token position — $O(L \times V)$ where $V \approx 128K$ for LLaMA-3 vocabulary. This is expensive to store alongside activations. A practical approximation uses only the top-$k$ probability mass (e.g., top-100 tokens) to estimate entropy, which is available from vLLM's `logprobs=k` output parameter. This approximation is accurate when the distribution is concentrated but degrades when probability mass is spread beyond the top-$k$.

---

### Mechanism C: Logprob Trajectory as a Third Contrastive View

**The argument.** The current contrastive training treats activations from two different LLM layers as two "views" of the same generation event. The multi-view hypothesis says that for correct answers, both views should be geometrically aligned (the same fact is represented at both layers), while for hallucinations the views are inconsistent. This view-alignment signal is the core detection mechanism.

The framework generalizes immediately to heterogeneous views: the "view" need not be an activation tensor — it can be any representation of the generation event. The per-token logprob sequence $(\ell_1, \ldots, \ell_L)$ is a natural third view, encoding the model's *output-space expression* of uncertainty during generation. It is structurally different from activation views:

- Activation views are high-dimensional ($4096$-dim per token), encode internal representations, and come from a specific layer.
- The logprob view is 1-dimensional per token, encodes output distributions, and is layer-agnostic.

A learned encoder $f_{\text{lp}}: \mathbb{R}^L \to \mathbb{R}^d$ (e.g., a 1D convolution over the sequence followed by a linear projection) maps the logprob sequence into the same embedding space as the activation views. The contrastive loss then has three views per sample instead of two:

$$\mathcal{L} = \mathcal{L}_{\text{SupCon}}(z_{\text{layer}_1}, z_{\text{layer}_2}, z_{\text{logprob}})$$

The cross-view alignment between activation views and the logprob view is the diagnostic signal. For *confident hallucinations*, the activation views are internally inconsistent (large $\|z_{\text{layer}_1} - z_{\text{layer}_2}\|$) while the logprob view is confidently structured (the model "knew" its wrong answer). This cross-modal misalignment — $z_{\text{logprob}}$ aligns poorly with either activation view — is a richer signal than within-activation inconsistency alone.

For *overtly uncertain hallucinations*, the logprob view is diffuse (uncertain), activation views are also inconsistent, and the three-way misalignment is larger than two-view misalignment. Both regimes benefit.

**Architectural note.** The logprob encoder $f_{\text{lp}}$ should be lightweight relative to the activation encoder (which processes 4096-dimensional tensors) because the input is only 1-dimensional per token. A 3-layer 1D conv with channel sizes $[32, 64, d]$ and global average pooling is sufficient to produce a $d$-dimensional embedding. The full architecture is a multi-branch compressor with a shared contrastive head.

**Prediction.** Three-view contrastive training should outperform two-view training specifically on *confident hallucinations* — cases where the model was sure of the wrong answer. These are the hardest cases for the existing two-view approach (both activation views are consistently "certain" even though wrong). The logprob view adds an orthogonal channel that may catch these cases through subtle distributional signatures in the output entropy profile.

**Critical challenge.** Per-token logprobs must be stored alongside activations in the zarr/LMDB dataset. The current `ZarrActivationsLogger` does not store logprobs, and the `Choice` schema in `server.py` has `logprobs` as `Optional[Any]`. Enabling this requires: (1) passing `logprobs=True` (or `logprobs=k`) to the vLLM API at inference time, (2) extracting and aligning the per-token logprob sequence with the activation sequence in `zarr_activations_logger.py`, and (3) updating the contrastive dataset loader to expose logprob tensors alongside activation views. This is a non-trivial pipeline addition but is technically clean.

---

### Mechanism D: Uncertainty Residualization to Isolate Epistemic Signal

**The argument.** Raw per-token log probabilities conflate two distinct uncertainty sources:

1. **Aleatoric (linguistic) uncertainty**: Uncertainty due to genuine lexical ambiguity, stylistic variation, or discourse structure — independent of factual knowledge. Articles, prepositions, discourse connectives, and punctuation tokens all carry high uncertainty for this reason.
2. **Epistemic (factual) uncertainty**: Uncertainty due to the model's lack of reliable parametric knowledge about the specific claim being generated. This is the signal we care about for hallucination detection.

Simply using raw logprobs (as in LLMsKnow baselines) mixes these two components. A residualization approach separates them. Define a *baseline logprob* $\hat{\ell}_i$ as the expected log probability of token $t_i$ under a context-agnostic reference — for example, the logprob of the same token drawn from the model's unconditional distribution, or the mean logprob of that token type (function word vs. content word) computed over a large reference corpus. The *factual uncertainty residual* is:

$$r_i = \ell_i - \hat{\ell}_i$$

A large negative residual ($r_i \ll 0$) indicates the model found this token surprisingly unlikely given its context *above and beyond* baseline linguistic difficulty — strong evidence of factual uncertainty. A near-zero residual on a content token indicates the model was confident in the factual claim.

In practice, a lightweight approximation: tokenize the response, classify each token as a content token (entity, number, named place, date) vs. a function token using part-of-speech tagging, and compute the residual as $r_i = \ell_i - \text{mean}(\ell_{\text{function tokens}})$. This normalizes out the "level" of the logprob distribution and leaves the factual uncertainty *shape*.

The residualized signal $\mathbf{r} = (r_1, \ldots, r_L)$ is a higher-SNR input to Mechanisms A, B, and C above. Uncertainty-weighted pooling using $\mathbf{r}$ instead of $\ell$ focuses specifically on factual tokens where the model expressed anomalous uncertainty; entropy FiLM conditioning on $\mathbf{r}$ captures factual difficulty rather than overall linguistic complexity; the logprob contrastive view trained on $\mathbf{r}$ should generalize better across domains.

**Prediction.** Residualized logprob should outperform raw logprob in all three dependent mechanisms (A, B, C) on domain-mixed test sets where baseline linguistic difficulty varies substantially across examples. The benefit will be smaller on domain-homogeneous benchmarks like PreciseWikiQA where the linguistic style is uniform.

**Critical challenge.** Accurate residualization requires identifying content vs. function tokens, which is non-trivial for free-form generation (POS tagging can be applied offline but adds latency). A cruder approximation — subtracting the mean logprob over the response — is parameter-free and always applicable, though it only removes the "level" effect without content/function discrimination.

---

### Mechanism E: Uncertainty Trajectory as a Temporal Pattern Classifier

**The argument.** The previous mechanisms treat the logprob sequence either as a set of independent per-token scalars (A, B, D) or as a sequence to be encoded holistically (C). A distinct perspective focuses on the *temporal dynamics* of the uncertainty trajectory — how the model's confidence evolves over the course of generating a response.

Hallucination events have a characteristic temporal signature. When a model generates a factual claim it doesn't reliably know, it often exhibits a brief *uncertainty spike* at the key content token (low logprob at the claim itself), followed by a *confidence recovery* as it generates fluent continuation text from the now-committed-to (but wrong) premise. This spike-then-recovery pattern in the logprob sequence is distinct from both: (a) uniformly low confidence across the response (a hard question answered with genuine uncertainty hedging), and (b) uniformly high confidence (a correctly recalled fact).

The spike-then-recovery pattern can be detected by computing:
- **Peak-to-mean ratio:** $\max_i (-\ell_i) / \text{mean}_i(-\ell_i)$ — high for spiky, low for uniform uncertainty.
- **Position of entropy peak:** Whether the maximum uncertainty occurs early (first content token) vs. late (closing of a long argument) vs. in the middle.
- **Variance of the uncertainty trajectory:** Low variance = uniformly uncertain or uniformly confident; high variance = mixed confidence (potentially the spike-then-recovery pattern).
- **Autocorrelation at lag 1:** Measures whether uncertainty is persistent (slow drift) vs. spiky (rapid alternation).

These scalar features compose a *temporal fingerprint* of the generation's confidence dynamics, distinct from any single-position statistic. They can be concatenated with the compressed activation embedding before the final contrastive head, providing a lightweight augmentation that requires no architectural changes to the `ProgressiveCompressor` itself.

**Prediction.** The temporal fingerprint should show the strongest benefit on *short factual answers* (1–5 content tokens) where the spike-then-recovery pattern is visible within a short sequence. On long-form generation (paragraphs), the signal is washed out by many confidence transitions and the fingerprint becomes less discriminative. This predicts a task-dependent interaction: the mechanism is most valuable on PreciseWikiQA (short answers) and least valuable on LongWiki (paragraph-length responses).

**Critical challenge.** For very short responses (1–3 tokens), the trajectory statistics are degenerate (e.g., variance of a length-1 sequence is undefined). Special-casing or minimum-length filtering is required.

---

## Summary Table

| Justification | Core mechanism | Key prediction | Primary failure mode |
|---|---|---|---|
| 1. Information Bottleneck | Incoherent activations compress poorly / to outlier regions | Mahalanobis distance > in compressed space than raw | Compressor trained on insufficient / imbalanced data |
| 2. Cross-Layer Consistency | Hallucinations disrupt inter-layer agreement in residual stream | Best signal at mid-to-late layer pairs; peaks at factual retrieval layers | Hard inputs confound layer divergence signal |
| 3. Intrinsic Epistemic Encoding | Epistemic uncertainty is encoded in intermediate layers but lost in output projection | Activation method > logprob baseline; strongest on confidently-wrong cases | White-box only; layer positions are model-specific |
| 4A. Uncertainty-Weighted Pooling | Low-logprob positions upweighted in sequence pooling | Better detection when hallucination is concentrated at specific content tokens | Ineffective on confident-wrong hallucinations (core intrinsic case) |
| 4B. Entropy FiLM Conditioning | Continuous entropy profile conditions activation encoder | Better calibration across difficulty regimes | Requires accurate entropy estimation; degrades on overconfident models |
| 4C. Logprob Contrastive View | Logprob trajectory encoded as a third view in multi-view contrastive training | Catches confident hallucinations missed by within-activation inconsistency | Requires logprob storage in data pipeline |
| 4D. Uncertainty Residualization | Subtract baseline linguistic difficulty to isolate factual uncertainty signal | Reduces domain-shift degradation of logprob-based mechanisms | Content/function token classification adds preprocessing complexity |
| 4E. Uncertainty Trajectory Fingerprint | Temporal dynamics of confidence (spike-then-recovery) encoded as auxiliary features | Best signal on short-answer tasks; spiky trajectories distinguish intrinsic hallucinations | Degenerate for very short responses; washed out in long-form generation |

**Relationship between justifications.** Justification 4 does not contradict Justification 3. The core claim of Justification 3 remains: *logprob alone is weaker than activations for detecting intrinsic hallucinations.* Justification 4 adds: *but logprob carries positional and temporal uncertainty structure that activations lack, and using it to guide the compression pathway — not replace it — improves the richness of the learned representations.* The two justifications are complementary channels, not competing accounts. The ideal system uses intermediate-layer activations as the primary representational basis (Justifications 1–3), with output logprob uncertainty as a compression-routing and pooling-guidance signal (Justification 4).

---

## Justification 5: Logprob as a Training-Time Shaping Signal for the Learned Compression

**The framing shift.** Mechanisms 4A–4E use logprob as a *runtime feature* — they modify the compressor's forward pass or add auxiliary inputs available at inference. The mechanisms below are categorically different: they use logprob exclusively as a *training-time supervision signal* that shapes which dimensions of activation space the compressor learns to project onto. The logprob signal is consumed during training and then discarded. At inference the compressor operates identically to the standard `ProgressiveCompressor` — no logprob input is required. The benefit is a learned projection that is intrinsically more uncertainty-aware because training steered it toward the uncertainty-correlated subspace of the residual stream.

This matters for deployment: it decouples the requirement for logprob access (training infrastructure) from the deployment requirement (inference infrastructure), allowing the richer representations to be used in any setting where logprob is unavailable at inference time.

The unifying theoretical principle: the compressor's SupConLoss training signal tells it *which examples to separate* but not *which directions of the activation space are uncertainty-relevant*. Logprob-informed training signals provide this directional guidance, biasing the learned projection toward the activation dimensions that co-vary with output uncertainty — precisely the dimensions Justification 3 argues carry epistemic information that the output projection discards.

---

### Mechanism F: Auxiliary Logprob Reconstruction (Predictive Coding Regularizer)

**The argument.** Add a lightweight auxiliary decoder $g: \mathbb{R}^d \to \mathbb{R}^L$ that, during training, reconstructs the per-token logprob sequence from the compressed embedding $z$. The training objective becomes:

$$\mathcal{L} = \mathcal{L}_{\text{SupCon}}(z) + \lambda \cdot \mathcal{L}_{\text{recon}}\!\left(g(z),\, \boldsymbol{\ell}\right)$$

where $\boldsymbol{\ell} = (\ell_1, \ldots, \ell_L)$ is the per-token logprob sequence and $\mathcal{L}_{\text{recon}}$ is mean-squared error. The auxiliary decoder is discarded at inference; only $z$ is retained.

**Why this shapes the learned projection.** The SupConLoss trains the compressor to produce embeddings that cluster by hallucination label. But the contrastive objective is agnostic to *which directions* of the activation space are used to achieve this clustering — it will exploit any available dimensions, including those irrelevant to uncertainty. The reconstruction auxiliary task adds a second objective: $z$ must encode information predictive of the logprob trajectory. Since logprob trajectory is correlated with factual uncertainty (even imperfectly), the compressor is steered toward the uncertainty-encoding subspace of the 4096-dimensional residual stream. The SupConLoss then operates in a subspace already aligned with epistemic structure.

This is an instance of *predictive coding as representation regularization*: the encoder is forced to preserve information about the "lower-level" signal (logprob) in its high-level compressed representation. In neuroscientific predictive coding models, higher-level representations are shaped by their ability to predict lower-level observations. Here the "lower-level observation" is the output logprob sequence, and the representation is the compressed activation embedding.

**Practical implementation.** The auxiliary decoder can be a 2-layer MLP: $\mathbb{R}^d \to \mathbb{R}^{2d} \to \mathbb{R}^L$ with ReLU, where $L$ is the maximum response length (padded and masked). For variable-length responses, mask the reconstruction loss over padding positions. The $\lambda$ coefficient should be annealed: start at $\lambda_0 = 0.1$ and decay geometrically, so the reconstruction objective regularizes early training (when the compressor's projection is most malleable) and contributes less as the contrastive objective dominates in later epochs.

**Prediction.** Embeddings trained with logprob reconstruction should show larger Mahalanobis distance between correct and hallucinated responses in the compressed space (Justification 1's core prediction) because the projection is aligned with uncertainty-correlated dimensions. The reconstruction head's test loss is a diagnostic: if it remains high, the compressor failed to encode logprob-relevant structure; if it falls, the projection is uncertainty-aligned. The hallucination detection benefit should be largest when the auxiliary decoder achieves a test reconstruction loss below a threshold (model-dependent calibration criterion).

**Critical challenge.** If the model is overconfident (logprob near-uniform across responses regardless of correctness), the reconstruction objective provides no useful steering — the compressor will learn to output a constant logprob prediction, and the auxiliary gradient carries no directional information. Adding a diagnostic check on train-set logprob variance (and suppressing the auxiliary term when it is below a threshold) prevents wasted capacity.

---

### Mechanism G: Confident-Hallucination Curriculum via Hard Negative Mining

**The argument.** In the `SupConLoss` training loop, the contrastive gradient signal is dominated by "easy negatives" — hallucinated examples where logprob correctly signals uncertainty, making the separation obvious even from raw logprob statistics. Hard negatives — hallucinated examples with *high* logprob (confidently wrong) — are the diagnostically important cases, but they contribute proportionally to the batch composition, which is typically balanced by correctness label rather than by logprob-correctness alignment.

Define a *deceptiveness score* for each hallucinated training example $i$:
$$d_i = \text{MeanLogprob}(i) \quad \text{where } y_i = 1 \text{ (hallucinated)}$$
High $d_i$ indicates a confident hallucination — the model generated the wrong answer fluently and with high probability. These are the cases where logprob-only detectors fail completely and where activation-based methods must carry the full detection burden.

A curriculum mechanism progressively increases the proportion of high-$d_i$ negatives within each training batch. In early epochs ($e < e_{\text{warmup}}$), negatives are sampled uniformly — the compressor learns the general contrastive structure. After warmup, the sampling distribution shifts toward high-$d_i$ negatives: specifically, sample negatives with probability proportional to $\text{softmax}(\beta \cdot d_i)$ where $\beta$ increases linearly with epoch. In final epochs, the compressor sees predominantly confident-hallucination negatives, forcing it to develop discriminators based purely on activation structure.

**Effect on the learned projection.** To separate confident-wrong from confident-correct samples — which share indistinguishable logprob statistics by construction — the compressor must learn to project onto the activation dimensions encoding *factual grounding coherence*, not merely output-uncertainty. This is precisely the cross-layer consistency signal from Justification 2: coherent factual retrieval produces stable representations across layers (low cross-layer distance), while confident hallucinations produce representations that are layer-inconsistent despite confident output. The curriculum forces the compressor to discover and encode this cross-layer consistency structure.

**Prediction.** The curriculum-trained compressor should show the largest improvement on the "confidently wrong" test subset — examples where logprob-only baselines assign high correctness probability but the actual label is hallucinated. The improvement on uncertain hallucinations (where logprob is informative) should be neutral or slightly negative (due to reduced training signal from easy negatives). This is an acceptable tradeoff: the confidently-wrong regime is precisely where existing methods fail and where the HalluLens approach has the strongest theoretical advantage.

**Critical challenge.** Confident hallucinations may be rare in the training set. The model must have a firmly held false parametric belief to produce a confident wrong answer — this is less common than "uncertain hallucinations" where the model confabulates because it has no relevant knowledge. If $d_i$ is high for only a small fraction of examples, the curriculum sampling will over-replicate a few examples, risking overfitting. Combining with data augmentation (paraphrasing the question to elicit the same confident hallucination from a different surface form) addresses this.

---

### Mechanism H: Logprob-Calibration Residual Weighting

**The argument.** Each training example has a *logprob prediction error* — the discrepancy between what logprob predicts about correctness and the actual label. Define:

$$e_i = \left|\sigma\!\left(s_{\text{lp}}(i)\right) - y_i\right|$$

where $s_{\text{lp}}(i)$ is a scalar logprob-derived score (e.g., mean log probability of generated tokens, mapped through a learned monotone function $\sigma$), and $y_i \in \{0, 1\}$ is the correctness label. High $e_i$ means logprob was maximally wrong for this sample: confident-wrong ($s_{\text{lp}}$ high, $y_i = 1$, so $e_i \approx 1$) or uncertain-correct ($s_{\text{lp}}$ low, $y_i = 0$, so $e_i \approx 1$). Low $e_i$ means logprob was sufficient.

Weight the contrastive loss contribution of sample $i$ by $e_i$:
$$\mathcal{L} = \frac{\sum_i e_i \cdot \mathcal{L}_{\text{SupCon}}^{(i)}}{\sum_i e_i}$$

This is a principled form of *instance-level boosting*: the compressor allocates its representational capacity to the examples where logprob fails, developing activation-space discriminators specifically calibrated to the logprob-failure regime.

**Relationship to Mechanism G.** Mechanism G changes *which samples enter each batch* (sampling curriculum). Mechanism H changes *how much each sample contributes to the gradient* (loss weighting). Both focus the compressor on logprob-failure cases, but H operates over all examples in each batch rather than filtering the sampling distribution. H is cheaper to implement (no modified sampler required) and applies to iterable/streaming datasets where sampling probabilities cannot be precomputed.

**The gradient geometry interpretation.** Standard SupConLoss pushes all same-class pairs together and all cross-class pairs apart uniformly. Weighting by $e_i$ means the compressor's embedding space is primarily shaped by examples where logprob gives misleading signal. In the embedding space learned under this weighting, the "decision boundary" between correct and hallucinated responses is orthogonal to the logprob direction by construction — because samples on the logprob-predicted-correct side of the boundary are heavily upweighted for separation. This produces the maximal complementarity between activation-based and logprob-based detectors.

**Prediction.** A linear probe trained on $z$ should show near-zero correlation with mean logprob (since the embedding space was shaped around logprob-orthogonal discriminators). The combined detector ($z$ + logprob score) should achieve AUROC strictly better than either alone, and the improvement should be approximately additive (since the two channels carry disjoint information by design). This is testable by measuring the correlation between logprob scores and the compressor's embedded distance to the correct-class centroid.

**Critical challenge.** Computing $e_i$ requires running a logprob-to-correctness calibration pass before training begins. If the training set is collected incrementally (streaming), $e_i$ must be updated online, requiring a running calibration of $\sigma$. A simpler approximation: use $\hat{e}_i = \mathbb{1}[\text{logprob\_quartile}(i) \neq \text{correctness}(i)]$ — a binary flag for examples where logprob and correctness are on opposite ends. This is computationally cheap and captures the main effect.

---

### Mechanism I: Logprob-Gated Attention Temperature During Training

**The argument.** Inside each `TransformerBlock` within `ProgressiveCompressor`, self-attention is computed with uniform temperature $1/\sqrt{d_k}$ at every token position. During training, the temperature at each position can be modulated by the token's logprob:

$$\tau_i = \tau_{\text{base}} \cdot \exp(-\alpha \cdot \tilde{\ell}_i)$$

where $\tilde{\ell}_i = (\ell_{\text{max}} - \ell_i) / \sigma_\ell$ is the normalized *negative* log probability (high value = uncertain) and $\alpha > 0$ controls modulation strength. High uncertainty at position $i$ → lower $\tau_i$ → sharper, more concentrated self-attention at that position. Confident positions → higher $\tau_i$ → softer, more diffuse attention.

**Effect on learned representations.** Sharper attention at uncertain positions means those positions receive stronger, more focal gradient signal through the attention mechanism. The transformer blocks develop representations where uncertain-position tokens attend strongly to their context and context attends strongly back to them — encoding the inter-token relationships that characterize uncertain generation events. Confident positions contribute broadly but without focal structure.

The key insight: this modulation is applied only during training. At inference, attention reverts to uniform temperature (no logprob available). However, the *weights* that were learned under uncertainty-gated training encode inter-token relationships characteristic of uncertain positions. The compressor "learned how to look at" uncertain regions of the sequence and retains this knowledge in its weights even when the gating signal is absent. This is directly analogous to how dropout-trained networks retain robustness to missing inputs even when dropout is disabled at inference: the training distribution shaped the learned features, not just the current forward pass.

**Connection to cross-layer consistency.** The uncertain positions in the logprob sequence correspond to positions where the model's residual stream had the least stable factual grounding (by Justification 3). Training the compressor to attend focally to these positions causes it to develop representations of exactly those positions — the ones carrying the most hallucination-relevant signal. The attention-gating is therefore a way of implementing "attend to the signal" as a training-time prior, without hard-coding which positions are signal-bearing (which logprob tells us, probabilistically, at training time).

**Prediction.** The compressor trained with uncertainty-gated attention should produce embeddings where the activation-space direction most predictive of hallucination labels aligns with the activation-space direction most predictive of per-position uncertainty. This alignment can be measured by computing the cosine similarity between (a) the principal component of $z_{\text{hallucinated}} - z_{\text{correct}}$ and (b) the principal component of the gradient of logprob reconstruction w.r.t. $z$. In the standard compressor, these directions are uncorrelated; in the gated compressor, they should align.

**Critical challenge.** Logprob-gated attention requires per-token logprob to be aligned with the activation sequence, which must be ensured by the data pipeline. If the activation sequence and logprob sequence cover different token ranges (e.g., activations over the full prompt+response, logprob only over the response), careful index alignment is required. The `zarr_activations_logger.py` currently supports `sequence_mode="response"` which provides this alignment, but it must be explicitly selected.

---

### Mechanism J: Adversarial Logprob Decoupling via Gradient Reversal

**The argument.** Mechanisms F–I steer the compressor *toward* uncertainty-correlated dimensions of activation space. Mechanism J is the structural complement: it steers the compressor *away* from logprob-predictable dimensions, forcing $z$ to encode only the activation-derived epistemic signal that is genuinely orthogonal to output logprob.

Architecture: attach a logprob prediction head $g: \mathbb{R}^d \to \mathbb{R}$ to $z$ through a gradient reversal layer (GRL; Ganin et al., 2016). In the forward pass, the GRL is the identity. In the backward pass, it negates the gradient with scale $\mu$:

$$\frac{\partial \mathcal{L}_g}{\partial \theta_{\text{enc}}} \leftarrow -\mu \cdot \frac{\partial \mathcal{L}_g}{\partial z} \cdot \frac{\partial z}{\partial \theta_{\text{enc}}}$$

The head $g$ minimizes $\mathcal{L}_g = \|g(z) - \bar{\ell}\|^2$ (predicting mean logprob). The encoder receives *reversed* gradients: it is simultaneously trained to produce $z$ useful for contrastive discrimination (via SupConLoss) and *uninformative* about logprob (via gradient reversal). The stable equilibrium of this adversarial game is a $z$ that maximally separates hallucination classes while being a worst-case input for logprob prediction.

**Why this produces maximal inference-time complementarity.** If $z$ is forced to be independent of logprob during training, then at inference the combined detector ($z$ + logprob) carries zero redundancy between its two components. Every bit of logprob's detection signal is genuinely additive to $z$'s signal, rather than re-encoding information already present in $z$. Information-theoretically: $\text{MI}(z, \text{logprob}) \to 0$ under adversarial decoupling, so the joint mutual information $\text{MI}(z + \text{logprob}, y) \geq \text{MI}(z, y) + \text{MI}(\text{logprob}, y)$ (since the two channels are nearly independent). This is the maximum possible additive benefit from combining the two channels.

**Relationship to domain adversarial training.** This is formally identical to the domain adversarial neural network (DANN) of Ganin et al., where "domain" is replaced by "logprob regime." In DANN, the encoder is made invariant to domain labels; here it is made invariant to logprob statistics. The same theoretical guarantees apply: under infinite data and model capacity, the encoder learns a representation that is sufficient for the main task (hallucination classification) while being maximally uninformative about the nuisance variable (logprob).

**Prediction.** Post-training mutual information $\text{MI}(z, \bar{\ell})$ should be near zero (measurable via a held-out linear probe: $R^2$ of predicting mean logprob from $z$ should approach chance). The combined detector AUROC should exceed the sum-of-independent-contributions baseline, because the enforced independence guarantees zero information overlap. The improvement in AUROC over activation-only baseline should correlate with the standalone logprob AUROC — in datasets where logprob is more informative, decoupling produces larger gains because there is more independent information to add.

**Critical challenge.** Gradient reversal training is notoriously unstable. The adversarial game between the encoder and the logprob head can oscillate rather than converge: the head adapts faster than the encoder can avoid, leading to a moving target. Standard stabilization techniques apply: (1) lower learning rate for the encoder relative to the head, (2) $\mu$ scheduling (start near zero, anneal up), (3) spectral normalization on $g$ to prevent the head from dominating the encoder's gradient signal. The instability risk is lower here than in image-domain GANs because $g$ is a linear head predicting a scalar, not a full discriminator — the adversarial game is well-conditioned.

---

## Summary Table

| Justification | Core mechanism | Key prediction | Primary failure mode |
|---|---|---|---|
| 1. Information Bottleneck | Incoherent activations compress poorly / to outlier regions | Mahalanobis distance > in compressed space than raw | Compressor trained on insufficient / imbalanced data |
| 2. Cross-Layer Consistency | Hallucinations disrupt inter-layer agreement in residual stream | Best signal at mid-to-late layer pairs; peaks at factual retrieval layers | Hard inputs confound layer divergence signal |
| 3. Intrinsic Epistemic Encoding | Epistemic uncertainty is encoded in intermediate layers but lost in output projection | Activation method > logprob baseline; strongest on confidently-wrong cases | White-box only; layer positions are model-specific |
| 4A. Uncertainty-Weighted Pooling | Low-logprob positions upweighted in sequence pooling | Better detection when hallucination is concentrated at specific content tokens | Ineffective on confident-wrong hallucinations (core intrinsic case) |
| 4B. Entropy FiLM Conditioning | Continuous entropy profile conditions activation encoder | Better calibration across difficulty regimes | Requires accurate entropy estimation; degrades on overconfident models |
| 4C. Logprob Contrastive View | Logprob trajectory encoded as a third view in multi-view contrastive training | Catches confident hallucinations missed by within-activation inconsistency | Requires logprob storage in data pipeline |
| 4D. Uncertainty Residualization | Subtract baseline linguistic difficulty to isolate factual uncertainty signal | Reduces domain-shift degradation of logprob-based mechanisms | Content/function token classification adds preprocessing complexity |
| 4E. Uncertainty Trajectory Fingerprint | Temporal dynamics of confidence (spike-then-recovery) encoded as auxiliary features | Best signal on short-answer tasks; spiky trajectories distinguish intrinsic hallucinations | Degenerate for very short responses; washed out in long-form generation |
| 5F. Predictive Coding Regularizer | Auxiliary logprob reconstruction forces z to encode uncertainty-relevant activation dims | Larger Mahalanobis separation; reconstruction loss as diagnostic of z quality | Useless when logprob is uncorrelated with correctness (overconfident models) |
| 5G. Confident-Hallucination Curriculum | Hard negative mining weights training batch toward high-logprob hallucinations | Largest improvement on confidently-wrong test subset | Rare confident hallucinations → overfitting risk without augmentation |
| 5H. Calibration-Residual Loss Weighting | Per-sample loss weight = magnitude of logprob prediction error | z direction near-orthogonal to logprob; additive gain at inference | Requires calibration pass before training; streaming datasets complicate this |
| 5I. Logprob-Gated Attention Temperature | Uncertain positions receive sharper self-attention during training only | z-hallucination direction aligns with z-uncertainty direction | Requires token-level logprob aligned to activation sequence in dataset |
| 5J. Adversarial Logprob Decoupling | Gradient reversal enforces MI(z, logprob) ≈ 0 | AUROC gain from combining z + logprob approaches theoretical maximum | Adversarial instability; requires careful learning rate and α scheduling |

**Relationship between Justifications 4 and 5.** Justification 4 mechanisms require logprob at inference time (they are forward-pass modifications or auxiliary inputs). Justification 5 mechanisms require logprob only at training time — the inference-time compressor is architecturally identical to the baseline `ProgressiveCompressor`. This makes J5 mechanisms deployable in any white-box setting where activations are accessible, even if the inference server does not return logprobs. The two families are compatible and composable: a compressor trained with J5 mechanisms (uncertainty-aware projection) can additionally be combined at inference with J4 mechanisms (uncertainty-weighted pooling, entropy conditioning) for maximum performance.
