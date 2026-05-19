# Related Work — Outline

Structural skeleton only. Each subsection lists what the prose will cover and which references will anchor it — not the prose itself. `[bracketed]` items mark spots where the author will decide what to import from the SOTA tracker (with explicit permission per the paper-writing rule).

Target length: ~0.75 page. Three subsections, 3–4 references each. End with one positioning sentence.

---

## 2. Related Work

### 2.1 Activation probing for LLM behavior

- **Framing sentence.** Probing intermediate activations to predict downstream behavior is a well-trodden axis; we extend it from *read-out* to *learned compression*.
- **Threads to cover.**
  - Classical linear probes on hidden states — frame as the "single layer, single linear map" baseline that defines what our compression must beat. [Anchor citation: representation-probing literature, e.g. Alain & Bengio; Belinkov & Glass survey.]
  - SAPLMA (Azaria & Mitchell, 2023) — the established MLP probe on a single layer's activations. Position as the closest prior method in spirit (probe on activations) but architecturally distinct (no contrastive objective, no cross-layer view).
  - Truthfulness / factuality directions in activation space — ITI (Li et al.), Marks & Tegmark, and the "linear truth direction" line. Use these to motivate *why* mid-to-late residual stream carries the signal, not just that it does.
  - [Optional 4th: Gurnee & Tegmark or similar "features as linear directions" work — defer decision until §3.3 framing freezes.]
- **What we take from this thread.** The empirical fact that hallucination-relevant structure lives in the residual stream.
- **What we depart from.** Single-layer assumption. Static (non-learned) compression. No coupling between layers.

### 2.2 Hallucination detection

- **Framing sentence.** Detection methods partition by access regime (white/black-box) and by forward-pass count (single vs. multi-sample). We situate ours and the baselines along both axes.
- **Threads to cover.**
  - **Output-space scalars (single-pass, black-box-compatible).** Token logprob, predictive entropy, and P(true) — Kadavath et al. (2022) "Language Models (Mostly) Know What They Know". These are the cheapest baselines and define the lower-compute end of §5.3.
  - **Multi-sample consistency (multi-pass, black-box).** Semantic entropy (Farquhar et al., 2024, Nature); SelfCheckGPT (Manakul et al., 2023). These define the higher-compute end of the compute-matched comparison — important to cite for the compute-axis framing.
  - **Probe-on-uncertainty hybrids.** SEP (Kossen et al., 2024) — train a probe to predict semantic entropy from activations. Position carefully: SEP-binary is the *genuine threat* per the risk register, so the §2 framing should not dismiss it. Acknowledge SEP introduced the "free byproduct" of predicting binary labels from the same probe head.
  - [Optional: retrieval-augmented checking — RAG-style fact-verification (e.g. Min et al. FActScore). Cite briefly to scope it *out* of our problem class, since we assume no external knowledge source.]
- **What we take.** The benchmark set (PreciseWikiQA / LLMsKnow lineage), the AUROC/AUPRC convention, the compute-matched evaluation idea.
- **What we depart from.** Output-space methods discard the bulk of the model's internal state. Sampling methods pay K× the forward-pass cost. SEP fixes the probe target to uncertainty rather than learning a representation; we learn the representation.

### 2.3 Contrastive representation learning

- **Framing sentence.** Self-supervised contrastive learning is well-established for visual and textual representations; its application to *intermediate activations of a frozen LLM*, with *layer pairs as views*, is the non-obvious move.
- **Threads to cover.**
  - SimCLR (Chen et al., 2020) and InfoNCE (Oord et al., 2018) — the loss family we use. One sentence on the canonical "two augmentations of the same image" setup, to set up the contrast with our setup.
  - Contrastive learning of sentence/text representations — SimCSE (Gao et al., 2021); brief mention to establish that the contrastive framework already migrated from vision to language.
  - [Optional: cross-layer or multi-view representation work in vision/NLP that uses *internal* views rather than data augmentations. Decision pending: include if a clean precedent exists; otherwise note the gap explicitly.]
- **What we take.** The InfoNCE objective and the temperature/negatives machinery.
- **What we depart from.** Views are not data augmentations of the input; they are different layers' representations of the same generation. This is a structural reinterpretation of "view," and §3.3 should foreground it.

---

### Positioning sentence (closes §2)

One sentence — a single complex sentence is fine — that names all three threads and states the gap our method fills. Draft skeleton:

> "Our method sits at the intersection of these threads: it is a *learned probe* (§2.1) trained with a *contrastive objective* (§2.3) and evaluated against the *single-pass and multi-sample detection baselines* (§2.2), with the new ingredient being the use of layer pairs as contrastive views."

Finalize the exact phrasing after §3.3 (contrastive architecture) and §5 (results framing) freeze, so the positioning sentence matches the contribution claims verbatim.

---

## Open questions for §2

1. **How much space for retrieval/RAG-style fact-checking?** Currently scoped out in one sentence under §2.2. Decision: keep it that way, or break out a fourth subsection? Recommendation: keep it scoped out — saves ~0.15 page for §5.
2. **Do we cite a cross-layer-representation precedent in §2.3, or claim the gap?** Depends on whether a clean prior exists. Action: literature pass before week 2.
3. **SEP framing tone.** SEP-binary is a genuine threat. §2.2 must describe SEP fairly and accurately; do *not* preemptively diminish it in related work — the comparison happens in §5.
4. **Order of subsections.** Current order: probing → detection → contrastive. Alternative: detection → probing → contrastive (lead with the problem, not the technique family). Decision needed before week 2 drafting.

---

## Cross-references this outline expects to call

- §3.3 (contrastive architecture) — §2.3 should set up the "views are layer pairs" framing that §3.3 pays off.
- §4 (experimental setup, baselines list) — §2.2 names the methods; §4 names the implementations. Avoid duplication: §2 cites the *paper*, §4 cites the *config / our reproduction*.
- §5.3 (compute-matched comparison) — §2.2 must mention the compute axis so §5.3 doesn't introduce it cold.
- §7.2 (multi-layer probe ablation) — §2.1 should foreshadow that naive multi-layer concatenation is *not* the right baseline, so §7.2 reads as confirmation rather than surprise.
