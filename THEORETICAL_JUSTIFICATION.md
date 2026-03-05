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

## Summary Table

| Justification | Core mechanism | Key prediction | Primary failure mode |
|---|---|---|---|
| 1. Information Bottleneck | Incoherent activations compress poorly / to outlier regions | Mahalanobis distance > in compressed space than raw | Compressor trained on insufficient / imbalanced data |
| 2. Cross-Layer Consistency | Hallucinations disrupt inter-layer agreement in residual stream | Best signal at mid-to-late layer pairs; peaks at factual retrieval layers | Hard inputs confound layer divergence signal |
| 3. Intrinsic Epistemic Encoding | Epistemic uncertainty is encoded in intermediate layers but lost in output projection | Activation method > logprob baseline; strongest on confidently-wrong cases | White-box only; layer positions are model-specific |

The three justifications are complementary: (1) describes *what* the compression is doing geometrically, (2) describes *where in the network* the signal originates, and (3) describes *why* intermediate activations carry information that output-space methods cannot access. Together they motivate the design choices in HalluLens — contrastive training, multi-layer views, and progressive compression — as theoretically coherent rather than empirically opportunistic.
