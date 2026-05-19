# Related Work — Outline

Structural skeleton **plus** fully materialized reference list. By the time a writing agent picks this up, every citation that will appear in §2 is named here with bibkey, summary, and the precise point our prose draws from it. The writing agent should not need to research further; if a claim cannot be supported from the bullets below, surface it to the human rather than inventing a source.

**Bib-policy reminder.** All bibkeys referenced below are already in `paper/references.bib`. Any additional citations a writing agent feels are needed must go to the human for approval *before* being added to the `.bib`. See the "Pending-approval candidates" section at the very bottom for refs currently flagged but not yet in `.bib`.

Target length: ~0.75 page. Three subsections, 3–4 references each. End with one positioning sentence.

---

## 2. Related Work

### 2.1 Activation probing for LLM behavior

**Framing sentence.** Probing intermediate activations to predict downstream behavior is a well-trodden axis; our method extends it from *read-out* to *learned, multi-layer compression*.

**Bullets — write each as 1–2 sentences in prose:**

- **Linear classifier probes as the canonical read-out.** Cite `alain2017understanding` for the original "freeze the network, train a linear classifier on each layer's activations" formulation. State the takeaway the prose will use: linear probes reveal that intermediate layers contain task-relevant linear structure, which is the empirical basis for the single-layer linear-probe baseline in §4. Frame it as a *floor* — the simplest read-out we should beat.

- **Probing as an analysis methodology — survey-level framing.** Cite `belinkov2019analysis` for the framing that probing answers "what does layer ℓ know?" rather than "how should we extract a behavior signal?". This is the lever we use to argue that a *learned compression* is a different (and we claim better) question than what probing typically asks. Use this in one sentence to position §2.1 as adjacent to but distinct from classical probing.

- **SAPLMA — the closest prior method.** Cite `azaria2023internal`. Technical content the prose should land:
  - Trains a feed-forward MLP (~11M params per their setup) on a *single layer's* hidden state of the model's own internal representation of a generated statement, with a binary truth/false target.
  - Establishes that the signal is recoverable without retrieval and without sampling, but assumes (a) the right layer is known a priori and (b) a fixed-shape MLP suffices.
  - This is the architecturally closest baseline and the one §4 reports as an activation-space probe alongside the single-layer linear probe. Position our method as: same access regime, learned *contrastive* objective, and *multi-layer* (layer-pair) views — none of which SAPLMA does.

- **Linear truth/factuality directions in activation space.** Cite `li2023iti` (ITI — Inference-Time Intervention) and `marks2024geometry` (geometry of truth, COLM 2024). Technical content:
  - ITI identifies attention-head-level directions correlated with truthfulness and shows shifting activations along those directions changes truthfulness of outputs — *causal* evidence that the residual stream carries the signal we want to read out.
  - Marks & Tegmark show that true/false statements live on a near-linear manifold in late-layer activations, and that the direction generalizes across datasets.
  - Use both jointly in one sentence: "Prior work shows hallucination-relevant structure exists in residual-stream activations, primarily in mid-to-late layers, and is at least partially linear — which motivates why a small learned compression is plausible and why we extract from a band of mid-to-late layers (§3.2)."

**What §2.1 establishes for the rest of the paper.** Activations carry the signal; existing probes either read a single layer linearly or with a fixed MLP head; nobody has used contrastive coupling *across layers* as the extraction mechanism.

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

- **Probe-on-uncertainty hybrids — the genuine threat.** Cite `kossen2024semantic` (SEP). Technical content:
  - Trains a probe on activations to *predict* the semantic-entropy score from a *single* forward pass — i.e. recovers a multi-sample method's signal at single-pass cost.
  - As a "free byproduct," the same probe head can be retargeted to predict the binary hallucination label directly (the "SEP-binary" variant our risk register flags as the live competitor).
  - The prose must describe SEP fairly and accurately — *do not* preemptively diminish it. The actual comparison happens in §5; §2 only sets up the framing: "SEP fixes the probe target to an uncertainty proxy; we learn the representation under a contrastive objective and recover the signal from layer-pair coherence."

- **Retrieval-augmented fact-checking (scoped out).** Cite `min2023factscore` once. Technical content:
  - FActScore decomposes a long-form generation into atomic facts and checks each against a retrieved knowledge corpus.
  - Use a single sentence to scope this *out*: we assume no external knowledge source and target short-form QA, so retrieval-based methods are not in our comparison class. This is also why long-form factuality is in §9 Limitations.

**What §2.2 establishes for the rest of the paper.** The detection landscape has three cells we care about (output scalars, multi-sample, probe-on-uncertainty); §5.3 evaluates against representatives of each at matched compute; SEP-binary is the closest white-box single-pass competitor.

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

- **The structural departure (no citation — we are claiming the gap).** State explicitly that, to our knowledge, contrastive coupling *between different layers' representations of the same generation in a frozen language model* has not been used as a hallucination-detection extractor. This is the load-bearing novelty claim. The prose should hedge ("to our knowledge") and the author should do a final pre-submission literature pass; see "Pending-approval candidates" for the live search axes.

**What §2.3 establishes for the rest of the paper.** The contrastive-learning machinery is borrowed; the *view definition* (layer pairs of a frozen LM's residual stream) is the new ingredient and is paid off in §3.3.

---

### Positioning sentence (closes §2)

A single complex sentence naming all three threads and stating the gap our method fills. Draft:

> "Our method sits at the intersection of these threads: it is a *learned probe* on residual-stream activations (§2.1) trained with a *contrastive objective* (§2.3) and benchmarked against the *single-pass and multi-sample detection baselines* (§2.2), with the new ingredient being the use of layer pairs — rather than data augmentations — as contrastive views."

Finalize after §3.3 (contrastive architecture) and §5 (results framing) freeze, so the positioning sentence matches the contribution claims verbatim.

---

## Open questions for §2

1. **How much space for retrieval/RAG-style fact-checking?** Currently one sentence under §2.2 citing `min2023factscore`. Recommendation: keep it that way — saves ~0.15 page for §5.
2. **Do we need a cross-layer-representation precedent in §2.3, or do we claim the gap?** Currently claiming the gap. Action item: literature pass before week-2 drafting; if a clean precedent surfaces, route through human approval and add to `.bib`. Search axes listed in "Pending-approval candidates."
3. **SEP framing tone.** SEP-binary is a genuine threat. §2.2 must describe SEP fairly; do *not* diminish it in related work — the comparison happens in §5.
4. **Order of subsections.** Current order: probing → detection → contrastive. Alternative: detection → probing → contrastive (lead with the problem, not the technique family). Decision needed before week-2 drafting.

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
  - **What they did.** Probe trained on a single forward pass's activations to predict the *semantic entropy score*. Single-pass cost, multi-sample signal. The same probe head can be retargeted to a binary hallucination label as a "free byproduct" (the SEP-binary variant).
  - **Why we cite.** The closest single-pass white-box competitor to our method. The risk register lists SEP-binary as the live threat; §2.2 must set up the comparison fairly.
  - **One-sentence role in our prose.** "SEP (`kossen2024semantic`) closes the cost gap: a probe trained to predict semantic entropy from a single forward pass — and, as a byproduct, the binary hallucination label — is the closest direct competitor to our method."

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

- **(probing depth/structure)** Hewitt & Manning, "A Structural Probe for Finding Syntax in Word Representations," NAACL 2019. *Would buy:* additional anchor in §2.1 for "probing has identified structured information in layers." *Verify:* is it actually needed given the survey citation? Recommendation: omit unless §2.1 feels under-cited after first draft.

- **(multi-view / cross-layer in NLP)** Tenney et al., "BERT Rediscovers the Classical NLP Pipeline," ACL 2019. *Would buy:* precedent for "different layers encode different aspects of the same input," which weakens our novelty claim slightly but strengthens the *motivation* for layer-pair views in §3.3. *Verify:* trade-off between motivation strength and novelty framing.

- **(hallucination — broader survey)** Ji et al., "Survey of Hallucination in Natural Language Generation," ACM Computing Surveys 2023. *Would buy:* one-citation anchor for "hallucination is a recognized problem class." *Verify:* may already be covered by §1 (Introduction) citations — avoid double-citing.

- **(retrieval baseline)** Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020 (RAG). *Would buy:* canonical retrieval citation if §2.2 needs to scope retrieval out more rigorously. *Verify:* `min2023factscore` may suffice; only add if a reviewer flags the omission.

- **(literature pass for the novelty claim in §2.3)** Search axis: "cross-layer contrastive learning of internal representations of frozen language models for hallucination/factuality detection." Action: human should run this search before submission; if a precedent surfaces, route the citation through approval. Currently we hedge with "to our knowledge."
