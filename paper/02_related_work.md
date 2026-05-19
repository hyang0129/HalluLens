# Related Work — Outline

Structural skeleton **plus** fully materialized reference list. By the time a writing agent picks this up, every citation that will appear in §2 is named here with bibkey, summary, and the precise point our prose draws from it. The writing agent should not need to research further; if a claim cannot be supported from the bullets below, surface it to the human rather than inventing a source.

**Bib-policy reminder.** Bibkeys in the "References — materialized for §2" section below are already in `paper/references.bib`. Any inline cite marked `[PENDING-APPROVAL]` in the bullets below, plus everything in the "Pending-approval candidates" section at the very bottom, is **not** yet in `.bib` and must be human-approved before insertion (per `paper/references.bib` policy header and `CLAUDE.md`). As of 2026-05-19: `zhang2025icr` (ICR Probe), `suresh2025clap` (CLAP), and `barshalom2025actvit` (ACT-ViT) are pending-approval; the first and third are live baselines in the §4 grid. All three were verified as real papers on 2026-05-19 (see fix notes in the candidate entries below) and need human sign-off before insertion.

Target length: ~0.75 page. Three subsections, 3–4 references each. End with one positioning sentence.

---

## 2. Related Work

### 2.1 Activation probing for LLM behavior

**Framing sentence.** Probing intermediate activations to predict downstream behavior is a well-trodden axis; our method extends it from *read-out* to *learned, multi-layer compression*.

**Bullets — write each as 1–2 sentences in prose:**

- **Linear classifier probes as the canonical read-out.** Cite `alain2017understanding` for the original "freeze the network, train a linear classifier on each layer's activations" formulation. State the takeaway the prose will use: linear probes reveal that intermediate layers contain task-relevant linear structure, which is the empirical basis for the single-layer linear-probe baseline in §4. Frame it as a *floor* — the simplest read-out we should beat.

- **Probing as an analysis methodology — survey-level framing.** Cite `belinkov2019analysis` for the framing that probing answers "what does layer ℓ know?" rather than "how should we extract a behavior signal?". This is the lever we use to argue that a *learned compression* is a different (and we claim better) question than what probing typically asks. Use this in one sentence to position §2.1 as adjacent to but distinct from classical probing.

- **SAPLMA — the closest *single-layer* MLP probe.** Cite `azaria2023internal`. Technical content the prose should land:
  - Trains a feed-forward MLP (~11M params per their setup) on a *single layer's* hidden state of the model's own internal representation of a generated statement, with a binary truth/false target.
  - Establishes that the signal is recoverable without retrieval and without sampling, but assumes (a) the right layer is known a priori and (b) a fixed-shape MLP on one layer suffices.
  - Position our method against SAPLMA as: same access regime, learned *contrastive* objective, *multi-layer* (layer-pair) views — none of which SAPLMA does. SAPLMA is the supervised MLP baseline §4 reports alongside the single-layer linear probe; it is the natural strawman-killer for "linear probe is a strawman," but it is *not* the closest method architecturally — see ICR Probe below.

- **ICR Probe — multi-layer scalar-summary probe.** [PENDING-APPROVAL: full candidate entry below — Zhang et al. ACL 2025, arXiv:2507.16488. Add to `.bib` only after human approval.] Technical content the prose should land:
  - Computes a per-layer Jensen-Shannon-divergence-style score from the residual stream at each layer (an L-dimensional vector per generation) and feeds it through an `L → 128 → 64 → 32 → 1` MLP with BatchNorm + LeakyReLU + Dropout, trained on a binary hallucination label.
  - Same access regime (frozen LM activations), same target (binary hallucination), and — crucially — *multi-layer*. The structural gap is therefore narrower than "no one uses multi-layer activations": ICR uses *per-layer scalar features* concatenated as an MLP input, not contrastive coupling between layer representations.
  - The prose must also flag the methodology issue that §4 picks up: the ICR paper's labeling pipeline (`output_judge.jsonl`) is not documented or released and the training code is non-functional, so our reproduction lands at ~0.675 AUROC on Llama-3.1-8B with substring labels rather than the paper's 0.7982 figure on Llama-3. We compare against our faithful re-implementation of the published algorithm and disclose the discrepancy in a methodology footnote (see §4 item 10 of the roadmap).
  - Position as: same access regime, multi-layer extraction, supervised-from-scratch — but per-layer *scalar* features rather than learned views over the full residual stream.

- **CLAP — contemporaneous *cross-layer-attention* probe.** [PENDING-APPROVAL: full candidate entry below — Suresh et al., arXiv:2509.09700, Sep 2025; TRUST-AI workshop at ECAI 2025. Surfaced by the 2026-05-19 literature pass; not currently in our baseline grid. Add to `.bib` only after human approval.] Technical content the prose should land:
  - Treats all residual-stream layers' representations of a generation as a *token sequence over the layer axis*, runs a transformer encoder with a learnable CLS token over that sequence, trains end-to-end with **binary cross-entropy** on hallucination labels.
  - Same problem, same input regime as ours; not contrastive (supervised BCE, no InfoNCE, no layer-pair view definition). Architecturally distinct from ACT-ViT (which patches the `(L × N × D)` tensor as an image) and from ICR Probe (which reduces each layer to a JSD scalar) — CLAP applies cross-layer *attention* with no spatial-token folding.
  - Position as: contemporaneous direct competitor in hallucination detection that uses cross-layer structure but **does not** use the contrastive view definition our method introduces. **Resolved 2026-05-19:** cite in §2.1, **do not** run as a §4 baseline. Scope justification: ACT-ViT subsumes the supervised-multi-layer-classifier architectural class. Flag in §9 limitations as a recent method we cite but do not reproduce.
  - Why this matters for §2.3 novelty: CLAP is exactly the kind of paper the broad novelty claim must survive. It surfaced in the literature pass and it does **not** collapse the claim — verified its method uses BCE, not InfoNCE, and supervised single-objective training, not contrastive view-pair extraction. The §2.3 framing-level verdict accounts for it.

- **ACT-ViT — the architecturally closest *multi-layer* prior method.** [PENDING-APPROVAL: full candidate entry below — Bar-Shalom et al., arXiv:2510.00296. Add to `.bib` only after human approval.] Technical content the prose should land:
  - Treats the full activation tensor of a generation `(L × N × D)` as an "image," max-pools the `(L, N)` plane to a fixed `(L_p, N_p)` grid, projects the hidden dimension through a per-LLM `LinearAdapter`, tiles into `patch_h × patch_w` patches, and runs a standard ViT encoder on the patch sequence to a binary hallucination output.
  - This is the architecturally closest prior method to ours in the sense that matters: it consumes the *full residual-stream tensor across multiple layers and tokens* — not per-layer scalars (ICR), not a single layer (SAPLMA / linear probe). The remaining gap between ACT-ViT and us is **how** the multi-layer structure is exploited: ACT-ViT runs a supervised ViT classifier over a pooled image; we use *contrastive coupling between layer-pair representations* of the same generation as the extraction mechanism, with an InfoNCE objective and a logprob-recon auxiliary loss.
  - The prose should be careful here: with ACT-ViT in the comparison, the §2.3 novelty claim cannot be "we are the first to use multi-layer residual-stream activations." It must be the narrower claim "we are the first to extract the hallucination signal via *contrastive coupling between layer-pair representations* under a mutual-information lower-bound objective."
  - Position as: same access regime, multi-layer + full residual-stream extraction, *supervised* end-to-end ViT classifier — versus our *self-supervised contrastive* objective with layer-pair views.

- **Linear truth/factuality directions in activation space.** Cite `li2023iti` (ITI — Inference-Time Intervention) and `marks2024geometry` (geometry of truth, COLM 2024). Technical content:
  - ITI identifies attention-head-level directions correlated with truthfulness and shows shifting activations along those directions changes truthfulness of outputs — *causal* evidence that the residual stream carries the signal we want to read out.
  - Marks & Tegmark show that true/false statements live on a near-linear manifold in late-layer activations, and that the direction generalizes across datasets.
  - Use both jointly in one sentence: "Prior work shows hallucination-relevant structure exists in residual-stream activations, primarily in mid-to-late layers, and is at least partially linear — which motivates why a small learned compression is plausible and why we extract from a band of mid-to-late layers (§3.2)."

**What §2.1 establishes for the rest of the paper.** Activations carry the signal; existing probes (a) read a single layer linearly, (b) read a single layer with a fixed MLP head (SAPLMA), (c) summarize each layer with a scalar score and run an MLP over the concatenated vector (ICR Probe), or (d) treat the full `(layers × tokens × hidden)` tensor as an image and run a supervised ViT (ACT-ViT). Nobody has used *contrastive coupling between layer-pair representations* of a frozen LM's residual stream as the extraction mechanism — that is the precise gap §2.3 stakes out. This is also where the prose foreshadows that *naive multi-layer concatenation* of raw activations is the wrong baseline — reported as an ablation in §7.2, not as a main competitor — because the inductive bias tying the layers together must be chosen (contrastive views, per-layer JSD scores, ViT patches), not skipped.

---

### 2.2 Hallucination detection

**Framing sentence.** Detection methods partition by access regime (white/black-box) and by forward-pass count (single vs. multi-sample). Our method sits in the *single-pass, white-box* cell; §2.2 names the occupants of the other cells and the closest white-box competitors.

**Bullets — write each as 1–2 sentences:**

- **Output-space scalars (single-pass, black-box-compatible).** Cite `kadavath2022language`. Technical content:
  - Three signals: per-token logprob, predictive entropy over the next-token distribution, and `P(true)` (ask the model "is the above answer correct?" and read the token probability).
  - Establishes the empirical claim "models partly know what they don't know," which justifies treating logprob/entropy/P(true) as legitimate baselines rather than strawmen.
  - These are the cheapest baselines in §4 and define the lower-compute end of the §5.3 compute-matched axis. Prose should say: "these signals discard the bulk of the model's internal state and are our floor."

- **Multi-sample consistency (multi-pass).** Cite `farquhar2024semantic` and `manakul2023selfcheckgpt`. Technical content:
  - `farquhar2024semantic` (semantic entropy, Nature 2024): sample K generations, cluster by NLI-defined semantic equivalence, take entropy over cluster probabilities. Pays K× the forward-pass cost; SOTA-class detection signal on free-form QA.
  - `manakul2023selfcheckgpt` (SelfCheckGPT, EMNLP 2023): sample K generations, score consistency of the original answer against the K samples via NLI, BERTScore, or n-gram overlap.
  - These define the *upper* end of the §5.3 compute axis. The prose should set up the §5.3 framing explicitly: any single-pass method must justify itself against K=10 sampling methods, and our compute-matched plot does this.

- **Probe-on-uncertainty hybrids — the genuine threat.** Cite `kossen2024semantic` (SEP, "Semantic Entropy Probes"). Technical content:
  - Trains a linear probe on a single forward pass's activations to *predict* the length-normalized semantic-entropy score. Recovers a multi-sample method's signal at single-pass cost.
  - In §4 we implement this exactly: SEP-SE = ridge-regression probe on last-token activations at the linear-probe layer, target = length-normalized SE, evaluated as a hallucination detector via AUROC against the binary halu label on the test split. See [`tasks/sampling_baselines/sep.py`](tasks/sampling_baselines/sep.py).
  - **Do NOT mention "SEP-binary."** It is not in `kossen2024semantic` (Kossen's probe target is the SE score, not the halu label) and it is not implemented in our codebase — [`tasks/sampling_baselines/sep.py:6`](tasks/sampling_baselines/sep.py#L6) explicitly states "SEP-binary (a logistic probe on halu labels directly) is not implemented." A previous outline-writer agent (a model) confabulated "SEP-binary" as if it were part of Kossen's paper. It is not. SEP-binary mentions in `PAPER_ROADMAP.md`, `paper/outline.md`, and `results/README.md` were purged in the same 2026-05-19 pass.
  - The prose must describe SEP fairly and accurately — *do not* preemptively diminish it. SEP-SE is the closest single-pass white-box competitor to our method. The actual comparison happens in §5; §2 only sets up the framing: "SEP-SE fixes the probe target to an uncertainty proxy; we learn the representation under a contrastive objective and recover the signal from layer-pair coherence."

- **Retrieval-augmented fact-checking (scoped out).** Cite `min2023factscore` once. Technical content:
  - FActScore decomposes a long-form generation into atomic facts and checks each against a retrieved knowledge corpus.
  - Use a single sentence to scope this *out*: we assume no external knowledge source and target short-form QA, so retrieval-based methods are not in our comparison class. This is also why long-form factuality is in §9 Limitations.

**What §2.2 establishes for the rest of the paper.** The detection landscape has three cells we care about (output scalars, multi-sample, probe-on-uncertainty); §5.3 evaluates against representatives of each at matched compute; SEP-SE (Kossen et al. 2024) is the closest white-box single-pass competitor.

---

### 2.3 Contrastive representation learning

**Framing sentence.** Self-supervised contrastive learning is well-established for visual and textual representations; its application to *intermediate activations of a frozen LLM*, with *layer pairs as views*, is the non-obvious move.

**Bullets — write each as 1–2 sentences:**

- **InfoNCE — the loss family.** Cite `oord2018cpc`. Technical content:
  - Introduces the InfoNCE loss as a lower bound on mutual information between two views: a positive pair drawn from the joint distribution and `N-1` negatives drawn from the marginal.
  - This is the loss we use in §3.4. The prose should briefly state the form (positive pair, in-batch negatives, temperature) so §3.4 can refer back without re-introducing it.

- **SimCLR — the canonical "two-augmented-views" setup.** Cite `chen2020simclr`. Technical content:
  - Establishes the contrastive-learning recipe for visual representations: two random augmentations of the same image form the positive pair, all other images in the batch form negatives, InfoNCE is the loss.
  - This is the *reference point* the prose contrasts against: in SimCLR the views are augmentations of the input; in our method the views are *different layers' activations* of the same generation. One sentence here is enough — the detailed structural argument lives in §3.3.

- **Contrastive learning of sentence representations.** Cite `gao2021simcse`. Technical content:
  - Shows the contrastive framework transfers from vision to text, with dropout-as-augmentation serving as the view-generation mechanism for sentence embeddings.
  - The prose uses this to establish "contrastive learning is not vision-specific" without spending more than one sentence on it. The point is to make the *layer-pair-as-view* move read as a continuation of an established direction rather than a wholly new technique.

- **The structural departure (no citation — we are claiming the gap).** State explicitly that multi-layer activation probing *itself* is not the gap: ICR Probe concatenates per-layer JSD scalars through an MLP, and ACT-ViT runs a supervised ViT over the full `(L × N × D)` activation tensor (both in §2.1). The load-bearing novelty is the **view definition** (layer-pair representations of the same generation as positive views) and the **objective** (self-supervised contrastive + reconstruction), not "we use more than one layer." The exact claim depends on the pre-submission literature pass — see "Framing-level decision" immediately below. The prose should hedge ("to our knowledge") regardless of which level lands.

- **Framing-level decision — literature pass verdict 2026-05-19: BROAD is defensible (with hedging).** The novelty claim has three levels (narrow / medium / broad — definitions retained below for the human's reference). A six-axis literature pass (cross-layer contrastive in NLP/LLMs, InfoNCE between layers / depth-axis CPC, contrastive knowledge distillation, vision intra-network contrastive, mech-interp / SAE cross-layer, "layer as view" phrasings) returned **no precedent** for the specific move: positive InfoNCE pairs as two different layers of the same un-augmented forward pass through a frozen network, used as a frozen-backbone feature extractor. The two closest patterns are (a) contrastive distillation (CRD / CoDIR — Tian et al. 2020, Sun et al. 2020) which pairs the *same layer across teacher/student networks*; (b) Deep InfoMax / Contrastive Deep Supervision which pair *augmented views within a layer*. Neither matches. CLAP (the contemporaneous hallucination-detection paper surfaced by the search) uses cross-layer attention with BCE, not InfoNCE with layer-pair views — also does not collapse the claim.
  - **Recommended verdict:** ship the **broad** claim, hedged as "to our knowledge" + a footnote crediting CRD/CoDIR (cross-network contrast) and CDS (within-layer-augmentation contrast) as the closest related machinery in different problem settings. This recasts the paper from "new hallucination detector" to "new probing methodology demonstrated on hallucination detection" — a stronger pitch for EMNLP Main, and the abstract / §1 should be drafted in this register.
  - **Level definitions (retained for the writing agent / human reference).**
    - **Narrow.** "*To our knowledge, contrastive coupling between layer-pair representations under an InfoNCE objective has not been used as a hallucination-detection extractor.*"
    - **Medium.** "*…has not been used to extract any behavioral property of a frozen LLM.*"
    - **Broad (recommended).** "*…has not been used as an extraction mechanism at all — across modalities or domains.*"
  - **Sequencing.** Verdict resolved → §2.3 fourth bullet, the positioning sentence, §1 contribution claims, and the abstract should be drafted to the **broad** level with the "to our knowledge" hedge and the CRD/CoDIR/CDS footnote. The §2.3 fourth bullet "load-bearing novelty is the view definition + objective" framing stands. Do **not** rewrite §2.3 / §1 / abstract to broad until the human signs off on the level — the level decision propagates outside §2.
  - **Reviewer-rebuttal fallback.** If a reviewer surfaces a precedent at the broad level, graceful-degrade to medium. If at medium, degrade to narrow. If at narrow (i.e. a prior hallucination-detection paper uses layer-pair InfoNCE — none found 2026-05-19), the contribution claim must pivot off "view definition" onto whatever else still holds (empirical wins, transfer, layer-pair interpretability, compute-matching vs. sampling) — see §9 risk register.

**What §2.3 establishes for the rest of the paper.** The contrastive-learning machinery is borrowed; the *view definition* (layer pairs of a frozen LM's residual stream) is the new ingredient and is paid off in §3.3.

---

### Positioning sentence (closes §2)

A single complex sentence naming all three threads and stating the gap our method fills. Draft:

> "Our method sits at the intersection of these threads: it is a *learned probe* on residual-stream activations (§2.1) trained with a *contrastive objective* (§2.3) and benchmarked against the *single-pass and multi-sample detection baselines* (§2.2), with the new ingredient being the use of layer pairs — rather than data augmentations — as contrastive views."

Finalize after §3.3 (contrastive architecture) and §5 (results framing) freeze, so the positioning sentence matches the contribution claims verbatim.

---

## Open questions for §2

1. **How much space for retrieval/RAG-style fact-checking?** ~~Open.~~ **Resolved 2026-05-19:** keep at one sentence in §2.2 citing `min2023factscore`. RAG (Lewis 2020) not added — retrieval is out-of-scope (white-box + short-form QA premise) and §9 limitations carries the longer treatment of long-form factuality. Saves ~0.15 page for §5.
2. **Do we need a cross-layer-representation precedent in §2.3, or do we claim the gap?** ~~Currently claiming the gap.~~ **Resolved 2026-05-19 (full):** Six-axis literature pass returned **broad is defensible** with hedging. No precedent found for layer-pair InfoNCE positive views over a frozen network as an extractor — across NLP behavioral probing, depth-axis CPC, contrastive KD, vision intra-network, mech-interp/SAE, or "layer as view" phrasings. Closest adjacent machinery: CRD / CoDIR (cross-network same-layer contrast) and Contrastive Deep Supervision (within-layer augmented-view contrast). Contemporaneous hallucination-detection competitor CLAP (arXiv:2509.09700) uses cross-layer attention with BCE, not contrastive — added to §2.1 as a citation. Recommended verdict and the reviewer-rebuttal fallback live in the "Framing-level decision" bullet in §2.3.
3. **SEP attribution — `SEP-binary` is a confabulation; purge it.** ~~Previously: "describe SEP fairly; SEP-binary is a genuine threat."~~ **Resolved 2026-05-19:** Verified against [`tasks/sampling_baselines/sep.py:6`](tasks/sampling_baselines/sep.py#L6): SEP-binary is **not** in Kossen et al. 2024 and is **not** implemented in our codebase. A previous outline-writer agent (a model) hallucinated "SEP-binary" as if it were part of Kossen's paper; we then propagated that into the outline. What we actually compare against is **SEP-SE** — a ridge-regression probe on activations trained to predict the length-normalized semantic-entropy score, evaluated as a hallucination detector via AUROC against binary halu labels on test (the standard Kossen recipe). §2.2 SEP bullet rewritten accordingly. Cross-file purge in the same pass: `PAPER_ROADMAP.md` (§2 status table, §4 must-add list, §6 matrix, §7 GPU cuts, §8 figure list, §9 risk register), `paper/outline.md` (§4 baselines, §5.3 compute-matched, §8 risk-register mention, Open Question 2), `results/README.md` (family 2 description + scope + paths + role + framing).
4. **Order of subsections.** ~~Open.~~ **Resolved 2026-05-19:** Keep **probing → detection → contrastive**. Rationale: the broad novelty claim (verdict in §2.3 "Framing-level decision") frames the paper as a methods contribution — "new probing methodology demonstrated on hallucination detection" — so §2 should read like a methods paper, leading with the technique family (probing), then the problem domain (detection), then the specific machinery (contrastive). Reverse order would mis-set the §1 framing as application-led.

---

## Cross-references this outline expects to call

- §3.3 (contrastive architecture) — §2.3 sets up the "views are layer pairs" framing that §3.3 pays off.
- §4 (experimental setup, baselines list) — §2.2 names the methods; §4 names the implementations. Avoid duplication: §2 cites the *paper*, §4 cites the *config / our reproduction*.
- §5.3 (compute-matched comparison) — §2.2 must mention the compute axis so §5.3 doesn't introduce it cold.
- §7.2 (multi-layer probe ablation) — §2.1 should foreshadow that naive multi-layer concatenation is *not* the right baseline, so §7.2 reads as confirmation rather than surprise.

---

## References — materialized for §2

Every entry below is already in `paper/references.bib`. Bibkey, full citation context, and the *role each reference plays in our prose* are stated so the writing agent does not need to re-research.

### §2.1 — Activation probing

- **`alain2017understanding`** — Alain & Bengio, "Understanding intermediate layers using linear classifier probes," ICLR Workshop 2017 (arXiv:1610.01644).
  - **What they did.** Trained a linear classifier on the frozen activations of each layer of an image classifier, plotted per-layer probe accuracy.
  - **Why we cite.** Canonical origin of "linear probe on a single layer" methodology. Anchors the framing that probing is well-established for read-out, which lets us position contrastive compression as a *next step*, not a replacement.
  - **One-sentence role in our prose.** "Linear probing of frozen intermediate activations has been the workhorse since `alain2017understanding`."

- **`belinkov2019analysis`** — Belinkov & Glass, "Analysis Methods in Neural Language Processing: A Survey," TACL 2019.
  - **What they did.** Surveyed probing and analysis methods for NLP, including diagnostic classifiers.
  - **Why we cite.** Survey-level anchor so the prose can claim "an established analysis methodology" in one sentence without piling on individual probing papers. Useful for one citation, not the main argument.
  - **One-sentence role in our prose.** "Probing as a diagnostic methodology is reviewed in `belinkov2019analysis`."

- **`azaria2023internal`** — Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying," Findings of EMNLP 2023.
  - **What they did.** SAPLMA: feed-forward MLP probe (~11M params) on a single layer's hidden state of an LLM's representation of its own generated statement, trained to predict truth/false on a curated TRUE/FALSE statement dataset.
  - **Why we cite.** Closest prior method in spirit. The architecturally-distinct comparison point in §4 (activation-space probe with a non-trivial head, but single-layer, supervised-from-scratch, no contrastive objective). Our learned compression is the natural extension.
  - **One-sentence role in our prose.** "SAPLMA (`azaria2023internal`) shows a non-linear head on a single layer's hidden state recovers a strong truth signal; we extend this from a fixed MLP on one layer to a learned contrastive compression over layer pairs."

- **`li2023iti`** — Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model," NeurIPS 2023 (arXiv:2306.03341).
  - **What they did.** Identified attention-head-level directions correlated with truthfulness via linear probes on TruthfulQA; showed shifting activations along those directions at inference time improves truthful-output rates — *causal* evidence the residual stream carries the signal.
  - **Why we cite.** Provides causal (not just correlational) grounding for the claim that hallucination-relevant information lives in the residual stream. Justifies our extraction site (§3.2).
  - **One-sentence role in our prose.** "ITI (`li2023iti`) provides causal evidence that residual-stream activations carry truth-relevant structure."

- **`marks2024geometry`** — Marks & Tegmark, "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets," COLM 2024 (arXiv:2310.06824).
  - **What they did.** Showed true/false statements lie on a near-linear manifold in late-layer activations and that the truth direction generalizes across datasets.
  - **Why we cite.** Empirical grounding for "the signal is mid-to-late residual stream" and partial linearity — which is why a *small* learned compression is plausible (we are not learning a massive nonlinearity, just one that combines layer pairs).
  - **One-sentence role in our prose.** "`marks2024geometry` show the relevant structure is near-linear in late layers and transfers across datasets, consistent with the mid-to-late band we extract from."

### §2.2 — Hallucination detection

- **`kadavath2022language`** — Kadavath et al., "Language Models (Mostly) Know What They Know," arXiv:2207.05221, 2022.
  - **What they did.** Showed LLMs can self-evaluate via token logprobs, predictive entropy, and an explicit `P(true)` follow-up question — partial calibration without external signals.
  - **Why we cite.** Defines the output-space scalar baselines in §4. Also justifies that these are not strawmen but legitimately informative signals, which strengthens our headline result by raising the floor.
  - **One-sentence role in our prose.** "Output-space scalars — token logprob, predictive entropy, and `P(true)` — were established as partial calibration signals by `kadavath2022language`."

- **`farquhar2024semantic`** — Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in large language models using semantic entropy," Nature 2024.
  - **What they did.** Semantic entropy: sample K generations, cluster by bidirectional NLI entailment, compute entropy over clusters. Strong free-form QA hallucination signal at the cost of K× forward passes.
  - **Why we cite.** The premier multi-sample baseline. Defines the upper end of the §5.3 compute axis and motivates why a single-pass method needs a compute-matched evaluation.
  - **One-sentence role in our prose.** "Semantic entropy (`farquhar2024semantic`) is the strongest multi-sample baseline and defines the K=10 cluster in our compute-matched comparison."

- **`manakul2023selfcheckgpt`** — Manakul, Liusie, Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," EMNLP 2023.
  - **What they did.** Sample K generations from a black-box model; score the original answer's consistency against the samples via NLI, BERTScore, or n-gram overlap.
  - **Why we cite.** Black-box multi-sample baseline; the NLI variant is the headline cell in our §5.3 K=10 comparison. Establishes the consistency-across-samples idea independently of semantic entropy's clustering machinery.
  - **One-sentence role in our prose.** "SelfCheckGPT (`manakul2023selfcheckgpt`) provides a black-box multi-sample baseline via consistency scoring."

- **`kossen2024semantic`** — Kossen et al., "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs," arXiv:2406.15927, 2024.
  - **What they did.** Linear probe trained on a single forward pass's activations to predict the *length-normalized semantic-entropy score*. Single-pass cost, multi-sample signal.
  - **What they did NOT do.** Kossen's probe target is the SE *score*, not a binary hallucination label. An earlier draft of this outline confabulated a "SEP-binary" variant and attributed it to Kossen — that variant is not in the paper and is not implemented in our codebase (see [`tasks/sampling_baselines/sep.py:6`](tasks/sampling_baselines/sep.py#L6)). Do not cite it.
  - **Why we cite.** SEP-SE is the closest single-pass white-box competitor to our method. §2.2 must set up the comparison fairly; the actual head-to-head happens in §5.
  - **One-sentence role in our prose.** "SEP (`kossen2024semantic`) closes the cost gap: a probe trained on a single forward pass to predict the semantic-entropy score is the closest direct competitor to our method."

- **`min2023factscore`** — Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023.
  - **What they did.** Decompose long-form generations into atomic facts; check each against a retrieved corpus to score factual precision.
  - **Why we cite.** Scoping device. Used in one sentence to mark retrieval-based and long-form-targeted methods as out-of-scope for our problem class.
  - **One-sentence role in our prose.** "Retrieval-based long-form fact verification (e.g. FActScore, `min2023factscore`) targets a different problem — long-form, open-world, knowledge-grounded — and is out of scope here."

### §2.3 — Contrastive representation learning

- **`oord2018cpc`** — van den Oord, Li, Vinyals, "Representation Learning with Contrastive Predictive Coding," arXiv:1807.03748, 2018.
  - **What they did.** Introduced InfoNCE as a contrastive lower bound on mutual information between two views.
  - **Why we cite.** This is the loss we use in §3.4. One citation establishes provenance; the actual loss form is restated in §3.4.
  - **One-sentence role in our prose.** "The contrastive objective we use is InfoNCE (`oord2018cpc`)."

- **`chen2020simclr`** — Chen, Kornblith, Norouzi, Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.
  - **What they did.** Canonical "two-augmented-views-of-the-same-image" contrastive recipe for visual representations.
  - **Why we cite.** The reference point our view-construction departs from. The prose contrasts SimCLR's *data-augmentation views* against our *layer-pair views*.
  - **One-sentence role in our prose.** "In contrast to SimCLR's data-augmentation views (`chen2020simclr`), our positive pairs are different layers' representations of the same generation."

- **`gao2021simcse`** — Gao, Yao, Chen, "SimCSE: Simple Contrastive Learning of Sentence Embeddings," EMNLP 2021.
  - **What they did.** Showed the contrastive framework transfers from vision to text; uses dropout as the view-generation mechanism for sentence embeddings.
  - **Why we cite.** One-sentence reference to establish that contrastive learning is not vision-specific, so layer-pair views read as a continuation of an existing direction rather than a wholly new technique.
  - **One-sentence role in our prose.** "Contrastive learning has been ported to text (e.g. SimCSE, `gao2021simcse`), though always with views constructed from inputs rather than from internal model state."

---

## Pending-approval candidates — NOT in references.bib

Live candidates a writing agent may want to cite. **Do not add to `.bib` without human verification** (per `paper/references.bib` policy header and `CLAUDE.md`). Each entry below states what the candidate would buy us and what the human should verify before approval.

- **(REQUIRED — live baseline in the grid)** Zhang et al., "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs," ACL 2025, arXiv:2507.16488. Proposed bibkey: `zhang2025icr`.
  - *What they did.* Per-layer Jensen-Shannon-divergence-style scores from the residual stream (L-dim vector per generation), fed through an `L → 128 → 64 → 32 → 1` MLP (BatchNorm + LeakyReLU + Dropout) trained to predict a binary hallucination label. Paper reports ~0.84 AUROC on HotpotQA with Gemma-2 and 0.7982 on Llama-3.
  - *Would buy.* The multi-layer-scalar-probe slot in §2.1 — the second-closest activation-space prior method. Without it, §2.1 leaves a citable competitor unmentioned and the §4 baseline grid (`icr_probe` in `configs/methods/`, PR #74) is unanchored in §2.
  - *Verify before adding to `.bib`.* (a) arXiv ID 2507.16488 + title confirmed 2026-05-19; ACL Anthology entry should be cross-checked before submission. (b) Venue resolved 2026-05-19: **ACL 2025 Main Conference** (not Findings). (c) The methodology footnote we plan in §4 item 10 (irreproducibility — `output_judge.jsonl` not released, training code references undefined `_load_data()`, GitHub Issues #3 and #5 unanswered) is **not yet re-checked against the upstream repo**; verify before submission.

- **(REQUIRED for §2.1 — surfaced by 2026-05-19 literature pass; NOT in our baseline grid)** Suresh, Aljundi, Nkisi-Orji, Wiratunga, "Cross-Layer Attention Probing for Fine-Grained Hallucination Detection" (CLAP), arXiv:2509.09700, Sep 2025; to appear at the TRUST-AI workshop, ECAI 2025. Proposed bibkey: `suresh2025clap`. **Correction 2026-05-19:** an earlier draft of this candidate entry attributed CLAP to "Liu et al." — the agent's first search returned that string and the outline propagated it without verification. The actual first author is Malavika Suresh; the bibkey was changed from `liu2025clap` accordingly. Any prose, code comment, or config that says "Liu et al." or `liu2025clap` for CLAP is wrong and should be swept.
  - *What they did.* Treats all residual-stream layers' representations of a generation as a token sequence over the layer axis; runs a transformer encoder with a learnable CLS token over that sequence; trains end-to-end with binary cross-entropy on hallucination labels.
  - *Would buy.* The "contemporaneous direct competitor" slot in §2.1 — a recent (Sep 2025) hallucination-detection paper using cross-layer structure. Failing to cite it would leave §2.1 looking like it missed the closest contemporaneous work. **Critically: this is the most recent precedent that the broad novelty claim in §2.3 had to survive, and it does — CLAP uses BCE, not InfoNCE, and no layer-pair view definition. Citing CLAP makes the novelty claim *stronger* in print, not weaker, because it shows we are aware of the closest contemporaneous work and our move is distinct.**
  - *Verify before adding to `.bib`.* All items below were resolved 2026-05-19. (a) Full author list: Malavika Suresh, Rahaf Aljundi, Ikechukwu Nkisi-Orji, Nirmalie Wiratunga. (b) arXiv ID 2509.09700 resolves to the title above. (c) Venue: workshop publication (TRUST-AI @ ECAI 2025); cite as workshop paper, not main-conference.
  - *Scope decision — resolved 2026-05-19.* CLAP is cited in §2.1, **not** added to the §4 baseline grid. Justification: ACT-ViT subsumes the supervised-multi-layer-classifier architectural class. The §9 limitations section should note CLAP as a contemporaneous method we cite but do not reproduce.

- **(REQUIRED — live baseline in the grid)** Bar-Shalom, Frasca, Galron, Ziser, Maron, "Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT," NeurIPS 2025, arXiv:2510.00296. Proposed bibkey: `barshalom2025actvit`. **Correction 2026-05-19:** an earlier draft of this entry rendered the title as "ACT-ViT: A Vision Transformer Over Full LLM Activation Tensors for Hallucination Detection" — that was a paraphrase, not the actual title. The real title leads with "Beyond Token Probes:". Any prose, code comment, or config docstring that uses the paraphrased title should be swept.
  - *What they did.* Treats the full activation tensor `(L × N × D)` of a generation as an image; adaptive max-pool to `(L_p, N_p)`, project hidden dim via a per-LLM `LinearAdapter`, tile into patches, run a standard ViT encoder on the patch sequence to a binary hallucination output.
  - *Would buy.* The architecturally-closest-prior-method slot in §2.1 — full-tensor multi-layer extraction. Crucially, this is in our baseline grid (`act_vit` in `configs/methods/act_vit.json`, wired into all 12 `baseline_comparison_*_memmap.json` configs), so §2 *must* name it or §4 reads as introducing a baseline cold. It also forces the §2.3 novelty claim to its narrow, defensible form ("contrastive coupling between layer-pair views," not "multi-layer probing").
  - *Verify before adding to `.bib`.* All items below were resolved 2026-05-19. (a) Full author list: Guy Bar-Shalom, Fabrizio Frasca, Yaniv Galron, Yftah Ziser, Haggai Maron — surname spelling `Bar-Shalom` (hyphen, capital S) confirmed. (b) Venue: accepted at NeurIPS 2025; cite as NeurIPS paper, not arXiv-only. (c) arXiv ID 2510.00296 resolves to the title above.

- **(probing depth/structure)** Hewitt & Manning, "A Structural Probe for Finding Syntax in Word Representations," NAACL 2019. *Would buy:* additional anchor in §2.1 for "probing has identified structured information in layers." *Verify:* is it actually needed given the survey citation? Recommendation: omit unless §2.1 feels under-cited after first draft.

- **(multi-view / cross-layer in NLP)** Tenney et al., "BERT Rediscovers the Classical NLP Pipeline," ACL 2019. *Would buy:* precedent for "different layers encode different aspects of the same input," which weakens our novelty claim slightly but strengthens the *motivation* for layer-pair views in §3.3. *Verify:* trade-off between motivation strength and novelty framing.

- **(hallucination — broader survey)** Ji et al., "Survey of Hallucination in Natural Language Generation," ACM Computing Surveys 2022 (arXiv:2202.03629; 13 authors confirmed 2026-05-19). *Would buy:* one-citation anchor for "hallucination is a recognized problem class." *Verify:* may already be covered by §1 (Introduction) citations — avoid double-citing.

- **(retrieval baseline)** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020 (RAG). *Would buy:* canonical retrieval citation if §2.2 needs to scope retrieval out more rigorously. *Verify:* `min2023factscore` may suffice; only add if a reviewer flags the omission.

- **(literature pass for the novelty claim in §2.3 — complete 2026-05-19)** Six axes run (NLP/LLM cross-layer contrastive, depth-axis CPC, contrastive KD, vision intra-network, mech-interp/SAE, "layer as view"). **Verdict: broad is defensible.** No precedent found; closest adjacent machinery is CRD/CoDIR (cross-network same-layer contrast) and CDS (within-layer augmented-view contrast). Contemporaneous hallucination-detection competitor CLAP added to §2.1 (separate pending-approval entry above). See §2.3 "Framing-level decision" for the verdict and the reviewer-rebuttal fallback policy. No further search is owed before submission.

  **Adjacent-machinery citations the broad-claim footnote will need.** If we ship the broad claim, the prose owes a short footnote crediting the closest related machinery in different problem settings — these are *adjacent precedents*, not collapses. All three are pending-approval (verify and route through human approval before `.bib` insert):
  - Tian et al., "Contrastive Representation Distillation" (CRD), ICLR 2020. Proposed bibkey: `tian2020crd`. URL: https://openreview.net/pdf?id=SkgpBJrtvS. *Role:* "Contrastive coupling between matching layers of two different networks has been used for distillation."
  - Sun et al., "Contrastive Distillation on Intermediate Representations" (CoDIR), EMNLP 2020. Proposed bibkey: `sun2020codir`. URL: https://aclanthology.org/2020.emnlp-main.36/. *Role:* same as CRD but specifically on BERT intermediate layers — the closest distillation precedent in LM-space.
  - Zhang et al., "Contrastive Deep Supervision" (CDS), ECCV 2022, arXiv:2207.05306. Proposed bibkey: `zhang2022cds`. URL: https://arxiv.org/abs/2207.05306. *Role:* "InfoNCE on augmented views *within* an intermediate layer has been used as auxiliary supervision," distinct from our InfoNCE *between* layers of an un-augmented forward pass.
