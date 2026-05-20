# Taxonomy Decision — Plan & Findings Log

**Status:** OPEN — awaiting findings sweep before edits begin
**Created:** 2026-05-20
**Owner:** human + writing agent

This document records (a) the decision to **sidestep the intrinsic-vs-extrinsic hallucination taxonomy** in this paper, (b) the rationale and sources, (c) the preliminary edit list, and (d) a findings section where sonnet agents accumulate every inconsistency across `paper/*` that the decision creates.

Edits do **not** start until the findings sweep completes and the human approves the consolidated edit list.

---

## 1. The question

§3 of the current draft uses the word "intrinsic" twice in load-bearing positions:

- **§3.1 line 23** of [`03_method.md`](03_method.md): "Intrinsic hallucination detection in a single-pass white-box regime."
- **§3.2 Claim 1 caveat** of [`03_method.md`](03_method.md): "this argument is architectural and tells us only that the joint distribution carries *some* shared information — it says nothing about whether that shared information is *relevant* to hallucination."

These statements presuppose a particular reading of the intrinsic-vs-extrinsic taxonomy, but the taxonomy itself is **contested** in the literature:

| Framework | Closed-book QA against gold references (HotpotQA, NQ, PopQA, SciQ, SearchQA, MMLU) |
|---|---|
| **Ji et al. 2022 (NLG survey, extends Maynez 2020)** | Treated as *intrinsic* with source = world knowledge ([survey](https://arxiv.org/html/2202.03629v6), §2.5 on GQA) |
| **HalluLens (Bang et al., ACL 2025)** — the paper this codebase descends from | Reclassified as *extrinsic*: the input prompt does not carry the answer-supporting evidence; the "oracle truth" sits in training data ([HalluLens](https://aclanthology.org/2025.acl-long.1176/)) |
| **ACT-ViT (Bar-Shalom et al., NeurIPS 2025)** — our architecturally-closest baseline | Sidesteps the taxonomy entirely. Defines hallucination broadly as "outputs unfaithful to the input *or* external facts"; uses fully reference-based labels (`y_i = 1 iff matches gold answer`); evaluates on both grounded (HQA-Wc) and closed-book (HQA) tasks under one definition without distinguishing ([arXiv:2510.00296](https://arxiv.org/abs/2510.00296)) |

The current §3 prose implicitly straddles the Maynez/Ji reading (when calling the target "intrinsic detection") and the HalluLens reading (which would call the same target extrinsic) — and the discrepancy compounds in the §3.2 information-theoretic caveat, where introducing an explicit `world` random variable matters only if the framework treats world facts as not-encoded-in-(x,y).

## 2. The decision — Option A: Sidestep

We adopt the **ACT-ViT-style operational definition** and decline to commit to the intrinsic-vs-extrinsic taxonomy in this paper. Concretely:

> Hallucination is defined operationally as the per-benchmark gold-reference evaluator's binary judgment of the model's greedy generation: `y_label = 1` when the evaluator marks the generation **incorrect**, `y_label = 0` otherwise. We acknowledge the intrinsic-vs-extrinsic taxonomy of Ji et al. (2022) and the HalluLens reframing (Bang et al., 2025) but treat the taxonomic placement of our targeted regime as orthogonal to the architectural argument in §3.2.

**Note on label convention.** ACT-ViT uses `y_i = 1 for correct`, which inverts our `y_label = 1 for hallucinated`. The decision adopts our codebase's convention (`y_label = 1 hallucinated`) for §3 prose; this is purely notational and does not affect the architectural argument.

## 3. Why Option A — rationale

1. **Closest baseline precedent.** ACT-ViT is the §2.3-anchor baseline (architecturally closest, multi-layer activation-tensor probe). It does not engage the intrinsic/extrinsic taxonomy and uses the same gold-reference labeling we use. Following its lead is the conservative move — and `02_related_work.md` already positions ACT-ViT as the architectural anchor.

2. **HalluLens itself is uncomfortable with the listed benchmarks.** HalluLens separates "hallucination" (vs training/input) from "factuality" (vs external truth) and introduces PreciseWikiQA as a new *extrinsic* task precisely because closed-book QA against world facts was not cleanly served by the older taxonomy. Importing HalluLens's letter would force a "this is extrinsic in the HalluLens sense" paragraph that adds complexity without changing any empirical claim.

3. **DPI chain simplifies.** Under sidestep, `y_label = g_evaluator(x, y)` is a deterministic function of (x, y) for a fixed evaluator. The `world` random variable that the original draft of Corollary A.2.1's caveat introduced becomes unnecessary — the surviving caveat is the cleaner statement that `h` is a lossy compression of (x, y) at any single layer.

4. **Reviewer surface.** Picking either Maynez/Ji or HalluLens invites a reviewer to challenge the choice. Sidestepping with explicit acknowledgment ("we follow ACT-ViT's operational definition") puts the choice on solid prior-art ground.

5. **§9 limitations becomes honest.** The taxonomic question is genuinely open in 2026; we declare it unresolved in §9 rather than pretending we resolved it.

## 4. Preliminary edit list (pre-findings-sweep)

These are the edits we **expect** to make based on the current understanding of §3. The findings sweep in §5 will expand this list.

### Edit set A — `03_method.md`

- **A1.** §3.1 line 23: rephrase "Intrinsic hallucination detection in a single-pass white-box regime" → "**Single-pass white-box hallucination detection**" + one new sentence stating the operational definition and naming the ACT-ViT precedent.
- **A2.** §3.2 Claim 1 caveat (currently amended to reference Corollary A.2.1, lines 43): keep the corollary reference; drop the `y_label = g(x, y, world)` apparatus and replace with the cleaner "`h` is a lossy compression of (x, y); architectural argument gives only an upper bound on `I(h; y_label)`" framing.
- **A3.** Corollary A.2.1 (line 179 area): drop the `world`-variable DPI chain from the caveat. State the surviving caveat in 2 sentences instead of 5.

### Edit set B — `outline.md` / abstract draft / `figures_outline.md`

- *Findings-sweep agent must surface any place these documents use the word "intrinsic" in a taxonomic (not methodological) sense and flag for rewording. (No `01_intro.md` exists yet — intro framing currently lives in `outline.md`.)*

### Edit set C — `02_related_work.md`

- *Findings-sweep agent must surface any §2 statement that commits to a taxonomic position on the benchmarks or on the detection target.*

### Edit set D — `04_experimental_setup.md`

- *Findings-sweep agent must surface any §4 statement about how labels are assigned, whether the dataset positions itself as intrinsic/extrinsic, and whether the labeling protocol is described compatibly with the operational definition above.*

### Edit set E — §9 limitations slot (does not yet exist as its own file)

- **E1.** When §9 lands, include one short paragraph stating that the taxonomic question (intrinsic vs extrinsic) is unresolved by this paper, citing Ji et al. 2022 and HalluLens, and forwarding to follow-up work. For now, record the obligation in `outline.md` §9 if a §9 placeholder exists there.

## 5. Findings — to be filled by sonnet agents (NEXT STEP)

The agents below will sweep their assigned files for any statement that conflicts with the Option A decision. Each finding should be recorded as a bullet under the corresponding subsection:

```
- [file:line] **Quote or paraphrase.** What the statement says, what conflicts with Option A, recommended action (reword / cut / leave + add caveat).
```

### 5.1 `03_method.md` — additional findings beyond A1–A3

*Sweep complete 2026-05-20 (Explore agent — findings transcribed from agent report).*

Beyond the three pre-identified edits (A1, A2, A3), no additional substantive conflicts found. The §3.2 Claims 1–3 prose, §3.3 architecture description, and §3.4 attribution-table framing are all operationally compatible with Option A. Specific hits:

- [03_method.md:23, §3.1 Task bullet] **"Intrinsic hallucination detection in a single-pass white-box regime."** Taxonomic use of "intrinsic" as a label for the detection target. **Conflict:** Already captured by Edit A1.

- [03_method.md:43, Claim 1 caveat] **"`y` is the response tokens, not the hallucination label `y_label`, and the architecture alone does not pin `I(h_{ℓ_a}; h_{ℓ_b}; y_label) > 0`."** Methodologically compatible with Option A — correctly distinguishes response from label. The conflict is downstream in Corollary A.2.1's formal apparatus, not in the caveat prose itself. **Conflict:** Already captured by Edit A2 (Claim 1 caveat needs the `world` reference dropped; the response-vs-label distinction is kept).

- [03_method.md:179, Corollary A.2.1 caveat block] **"by DPI through `y_label = g(x, y, world)`, `I(h_{ℓ_a}; h_{ℓ_b}; y_label) ≤ I(h_{ℓ_b}; y, world)`, which is not pinned positive by architecture alone."** The `world` random variable as a constituent of `y_label`'s generation process. **Conflict:** Direct contradiction with Option A's operational definition `y_label = g_evaluator(x, y)`. Already captured by Edit A3. **Note:** the surrounding 5 sentences of caveat (including the "what this strengthens / does not strengthen" paragraph structure) should collapse to 2 sentences: (1) label-relevance caveat survives because `h` is a lossy compression of (x, y), and (2) under the operational definition, label-relevance is empirical (deferred to Claim 2 + trainability punchline), not architectural.

- [03_method.md:25, §3.1 Access regime] **"No retrieval, no external verifier, no resampling."** Compatible — refers to *inference-time* access; the training-time gold-reference evaluator is not an "external verifier" in this sense. **Conflict:** None. **Action:** Leave as-is.

- [03_method.md:27, §3.1 Notation] **"`y_label` may have evaluator noise (carried forward to §9 limitations)."** Correctly identifies the evaluator as the label source. **Conflict:** None. **Action:** Leave as-is; this is in fact the seed of the operational definition we're now formalizing in §3.1.

- [03_method.md, §3.3 throughout] No instance of "intrinsic," "extrinsic," "world," "external truth," or "factuality" found. Architecture description is fully operational.

- [03_method.md, §3.4 throughout] No instance of taxonomic language. Attribution table framings are purely empirical.

**Summary.** Findings sweep confirms the conflict surface in §3 is exactly the three pre-identified edits (A1, A2, A3). No new edits owed.

### 5.2 `02_related_work.md` — findings

*Agent assignment: scan §2 for any taxonomic commitment, especially in §2.1 (activation probing), §2.2 (single-pass baselines), and §2.3 (novelty claim). Flag any prose that presupposes a particular reading of intrinsic/extrinsic. Also check the materialized references — do any cited works' framings get adopted implicitly?*

**CLEAN — no conflicts found.** The file is a detailed outline and reference guide for a writing agent, not prose. It contains no actual §2 text, no instances of "intrinsic" or "extrinsic," and no taxonomic commitments. The outline's guidance to the writing agent is consistent with Option A throughout:

- [02_related_work.md:42-45, §2.1 ACT-ViT framing] **"ACT-ViT does not engage the intrinsic/extrinsic taxonomy" and uses gold-reference labeling.** The outline correctly positions ACT-ViT as the procedural precedent for Option A. **Conflict:** none — already aligned. **Recommended action:** none.

- [02_related_work.md:47-50, §2.1 ITI/marks guidance] **Prose will cite truth/factuality directions in activations.** The outline carefully avoids framing this as taxonomic evidence; uses neutral language ("hallucination-relevant structure"). **Conflict:** none. **Recommended action:** none.

- [02_related_work.md:80, 204, §2.2 FActScore framing] **"We assume no external knowledge source and target short-form QA."** This is methodologically neutral, compatible with operational definition, makes no taxonomic claim. **Conflict:** none. **Recommended action:** none.

- [02_related_work.md:104-114, §2.3 novelty claim framing] **Outline stakes broad novelty on methodology (layer-pair InfoNCE) not taxonomy.** Narrowest formulation (line 109) frames contribution as architectural, not taxonomic. **Conflict:** none — presupposes no intrinsic-vs-extrinsic stance. **Recommended action:** When prose materializes §2.3, writing agent must ensure cited adjacent machinery (CRD/CoDIR, Contrastive Deep Supervision) is not inadvertently framed as solving "intrinsic" or "extrinsic" problems.

- [02_related_work.md:52, §2.1 summary] **"Hallucination-relevant structure."** Neutral language, no taxonomy. **Conflict:** none. **Recommended action:** none.

- [02_related_work.md:82, §2.2 summary] **Detection landscape partitioned operationally by access regime and compute cost, not by intrinsic-vs-extrinsic.** **Conflict:** none. **Recommended action:** none.

### 5.3 `04_experimental_setup.md` — findings

*Sweep complete 2026-05-20 (Explore agent — findings transcribed from agent report).*

**CLEAN — no taxonomic conflicts found.** §4 describes the labeling protocol operationally throughout, in a form fully compatible with Option A. Specific hits:

- [04_experimental_setup.md:44, §4.2 Labeling convention] **"Substring-match against the gold answer is the binary hallucination label across all six datasets."** Pure operational definition. **Conflict:** None — this *is* the operational definition Option A formalizes. **Action:** Leave as-is. When §4.2 prose materializes, surface this sentence explicitly as the canonical statement of `y_label = 1 iff substring-match fails`.

- [04_experimental_setup.md:41, §4.2 PopQA description] **"Retrieval-free."** Methodological framing (dataset composition), not taxonomic. **Conflict:** None. **Action:** Leave as-is.

- [04_experimental_setup.md:90, §4.x truth-direction reference] **"Truth-direction evidence"** in technical context. Refers to activation-geometry literature (Marks et al.), not to a taxonomic claim about hallucination classification. **Conflict:** None. **Action:** Leave as-is.

- [04_experimental_setup.md:93, §4.x] **"Truthful-class embedding."** Implementation-detail terminology for the loss function (one-class SupCon anchor side). Not a taxonomic claim. **Conflict:** None. **Action:** Leave as-is.

- [04_experimental_setup.md, all six datasets (HotpotQA, NQ, PopQA, SciQ, SearchQA, MMLU)] No per-benchmark prose adopts an intrinsic-or-extrinsic stance. Each dataset is framed as a test bed for the operational label.

- [04_experimental_setup.md, references-materialized section] No cited work's framing is implicitly adopted in a taxonomic sense.

**Summary.** §4 is ready for Option A as-is. No edits owed beyond ensuring the materialized §4.2 prose explicitly states the operational definition (which is already what the outline prescribes).

### 5.4 `01_intro.md` / `outline.md` / abstract — findings

*Agent assignment: scan the top-level outline, intro, and any abstract draft for taxonomic claims, the words "intrinsic"/"extrinsic," and the framing of the detection target. Also flag any contribution-bullet that presupposes a taxonomic position.*

**ONE CONFLICT FOUND.**

- [outline.md:17] **"intrinsic LLM hallucination detection, white-box single-pass"** in abstract framing bullet. This phrase presupposes the intrinsic-vs-extrinsic taxonomy and explicitly adopts the "intrinsic" position. **Context:** The abstract outline lists this as one of the framing sentences for the abstract section. **Sense of "intrinsic":** Taxonomic — frames the problem as intrinsic hallucination (hallucinated information not present in the input) rather than extrinsic. **Conflict with Option A:** Option A sidesteps the taxonomy entirely; explicitly using "intrinsic" in the abstract framing commits to the taxonomy and violates the operational-definition approach. **Recommended action:** Delete "intrinsic" and replace with neutral operational language. Suggested rewording: "LLM hallucination detection in a white-box single-pass regime" or "white-box single-pass hallucination detection against gold-reference labels." Remove the taxonomic modifier before prose draft materializes. **§9 limitations slot:** §9 exists as an outline placeholder (lines 125–130 of outline.md) with four bullets on scope; no separate `09_limitations.md` file yet exists. When §9 materializes, Edit E1 (per §4 of plan) should land there — one paragraph stating the taxonomic question is unresolved, citing Ji et al. 2022 and HalluLens, forwarding to follow-up work.

The three contribution bullets (lines 31–34) contain no explicit taxonomic language and make no presuppositions about the intrinsic-vs-extrinsic framing; they are framed architecturally and operationally.


### 5.5 `05_results.md` and downstream — findings

*Agent assignment: scan §5–§9 outlines for taxonomic commitments, especially any place the prose interprets a per-benchmark result in terms of intrinsic-or-extrinsic. Flag the §9 slot where Edit E1 lands.*

**CLEAN — no conflicts found.** `05_results.md` is a fully materialized outline of the Results section structure and narrative positioning, not prose. It contains zero instances of "intrinsic," "extrinsic," "world," "external," "ground truth," or "factuality." 

All narrative moves in the outline are operationally framed:
- §5.1 cluster-gap framing compares methods by AUROC across dataset benchmarks, not by taxonomy.
- §5.2 within-cluster comparison ("ours vs. strongest learned baselines") is purely empirical, matches-or-outperforms language.
- §5.3 compute-matched framing pitches K=1 single-pass against K=10 sampling by forward-pass count, not by detection regime.
- §5.4 calibration section treats score-to-probability mapping as methodological, not taxonomic.

The outline correctly:
- Forwards methodological "why" questions to §3, not to taxonomy.
- Forwards NQ shortfall interpretation to §8 (discussion), leaving §5 as result-dense and argument-light.
- Preserves the SAPLMA attribution story for §7.1 (loss decomposition), not settling it in §5.
- Hedges verbs ("matches-or-outperforms") until ACT-ViT 5/5 seeds close, ensuring no over-commitment.

**§9 slot:** No separate `09_limitations.md` file yet exists. The outline lists §9 as a pending file (line 169 of outline.md). When §9 materializes, Edit E1 belongs there as per §4 of this plan.

**Recommended action:** When writing agents materialize §5 prose from this outline, ensure:
1. No per-benchmark narrative leans on "intrinsic" or "extrinsic" framing (e.g., avoid "on intrinsic QA benchmarks like PopQA").
2. The MMLU footnote (§5.1 line 38) notes ACT-ViT limitation neutrally — as an architecture-token interaction, not as evidence for or against taxonomy.
3. Forward all "why is NQ lower" interpretation to §8; §5.1 states the fact and forwards only.


### 5.6 References — findings

*Agent assignment: scan `paper/references.bib` (read-only — do NOT edit) for any cited works whose taxonomic framing we implicitly adopt by citing them. Also check whether Ji et al. 2022 and the HalluLens paper itself are in `.bib` already; if not, flag for human approval (per `CLAUDE.md`, do not add to `.bib` without explicit approval).*

**CRITICAL REFERENCES MISSING — TWO FINDINGS.**

1. **Ji et al. 2022 NLG survey absent.** Per plan §1 (lines 24, 48), `ji2022survey` or `ji2023survey` (the "survey" variant, which covers the intrinsic-vs-extrinsic taxonomy extension from Maynez 2020) is cited in the plan rationale but **not present in `references.bib`**. Plan §5 (plan line 188–190, the "Bibliography state" section) confirms this: "The only candidates that are *not* in `references.bib` are the four §2 optionals (Hewitt & Manning, Tenney et al., **Ji et al.**, Lewis et al. RAG)." **Conflict with Option A:** The plan strategy (§2 of plan, line 48) explicitly states "We declare it [the taxonomic question] unresolved in §9 rather than pretending we resolved it" and cites Ji et al. 2022 as a required forward reference in §9's taxonomy-honesty paragraph (Edit E1). **Recommended action:** Flag for human approval (per `CLAUDE.md`) to add Ji et al. 2022 NLG survey to `references.bib` before §9 prose locks. Suggested bibkey: `ji2022survey` (lowercase-author-year convention per line 30 of references.bib). Do not add without explicit human OK.

2. **HalluLens paper absent.** Per plan §1 (line 25), Bang et al. ACL 2025 "HalluLens" (`bang2025hallulens` or similar) is **not present in `references.bib`**. Plan cite notes the paper reclassifies closed-book QA as extrinsic and introduces PreciseWikiQA as an extrinsic task (plan rationale §2 line 42). **Conflict with Option A:** The decision to sidestep the taxonomy (plan §2 line 34) explicitly "acknowledge[s] the intrinsic-vs-extrinsic taxonomy of Ji et al. (2022) and the HalluLens reframing (Bang et al., 2025)" and names this acknowledgment as part of the operational-definition paragraph. If HalluLens is not cited, the acknowledgment cannot land in prose. **Recommended action:** Flag for human approval to add HalluLens (Bang et al., 2025, ACL) to `references.bib` before §3.1 and §9 prose materialize. Suggested bibkey: `bang2025hallulens`. Do not add without explicit human OK. **Note:** Both papers are foundational to the decision rationale and must appear in the final paper for the Option A choice to be transparent to reviewers.

**Cited works — no taxonomy-framing conflicts found.** All currently cited works in `references.bib` (lines 1–248) are cited operationally:
- Activation-probing papers (§2.1) are cited for methodology (linear probes, ViT tensors), not for taxonomic positioning.
- Hallucination-detection papers (§2.2) are cited for method class (output-space, sampling), not for taxonomy.
- Contrastive learning papers (§2.3) are cited for loss machinery (SimCLR, SupCon), not for regime positioning.
- Information-theoretic papers (§3 method) are cited for bounds and mutual-information machinery, not for taxonomy.

No entry's framing is adopted implicitly in a way that commits to intrinsic-vs-extrinsic.

**macros.tex — no findings.** Brief skim (lines 1–153) confirms the file is purely technical: LaTeX result-lookup macros (`\result`, `\resdelta`, etc.) for binding numbers from CSV at build time. Zero taxonomy-related content.


## 6. Consolidated edit list — ready for human approval

Post-sweep, the full edit set is small and well-localized. Findings confirm no surprises outside §3 and `outline.md`:

| ID | File | What | Status |
|---|---|---|---|
| **A1** | [03_method.md:23](03_method.md) | Reword §3.1 Task bullet: drop "Intrinsic" qualifier; add one sentence with operational definition + ACT-ViT precedent | Pre-planned |
| **A2** | [03_method.md:43](03_method.md) | §3.2 Claim 1 caveat: keep Corollary A.2.1 reference + response-vs-label distinction; drop `world` reference | Pre-planned |
| **A3** | [03_method.md:179](03_method.md) | Collapse Corollary A.2.1's 5-sentence caveat to 2 sentences; drop `world`-variable DPI chain | Pre-planned |
| **B1** | [outline.md:17](outline.md) | Abstract framing bullet: replace "intrinsic LLM hallucination detection, white-box single-pass" with neutral operational phrasing | New from sweep |
| **E1-placeholder** | `outline.md` §9 block (lines 125–130) | Add a bullet reserving one §9 paragraph for the taxonomic-honesty acknowledgment (citing Ji 2022 + HalluLens). Defer prose until `09_limitations.md` materializes | New from sweep |
| **F1** | `references.bib` | **Human approval required** to add (a) Ji et al. 2022 NLG survey (suggested key `ji2022survey`, arXiv:2202.03629) and (b) HalluLens / Bang et al. ACL 2025 (suggested key `bang2025hallulens`). Both are foundational to the Option A acknowledgment; without them, §3.1 cannot cite the taxonomy it sidesteps. **Do not add to `.bib` without explicit human OK** (per `paper/CLAUDE.md`). | New from sweep — gated on human |

**No edits owed in:** `02_related_work.md`, `04_experimental_setup.md`, `05_results.md`, `figures_outline.md`, `macros.tex`. All four are already operationally framed.

**Risk note.** Edit B1 lands in `outline.md`'s abstract framing, which is read by the writing agent when materializing the abstract. Catching it pre-materialization (i.e., now) is cheap; catching it post-materialization would require re-editing prose. Recommend B1 lands in the same pass as A1–A3.

## 7. Decision log

- **2026-05-20 (initial).** Option A selected after literature sweep (Ji 2022, HalluLens 2025, ACT-ViT 2025). See §1–§3.
- **2026-05-20 (sweep).** Findings sweep complete across all paper files. Net edit list: 5 prose edits (A1, A2, A3, B1, E1-placeholder) + 2 bib additions gated on human approval (F1). No additional conflicts found in §2, §4, §5, figures outline, or macros.
- *(further entries as edits are made)*

## 7. Out of scope for this plan

- Any change to the labeling code, dataset configs, or evaluator scripts in the Python codebase. This is a **paper-prose** decision; the operational definition matches what the code already does.
- Re-running any experiment. No empirical claim shifts under Option A.
- The §7.3 ablation framings, §5 narrative framings, or §1 contribution bullets — those depend on data, not on taxonomy. Findings agents may flag taxonomic *language* in those files but must not propose framing changes here.
