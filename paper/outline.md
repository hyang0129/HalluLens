# Paper Outline — EMNLP 2026 Submission

**Working title (provisional, 2026-05-20).** *Detecting Hallucinations with Contrastive Large Language Model Representations (CLLMR)*. Task-first framing puts "Detecting Hallucinations" in the first three words (mean-margin headline lead — Open Question 6 below; not the compute-efficiency lead, which is shared with ACT-ViT at K=1 and therefore non-differentiating). Method-name backronym `CLLMR` = *Contrastive Large Language Model Representations*, structurally echoing SimCLR (`chen2020simclr`).

**⚠ Naming-collision flag for CLLMR.** Web search 2026-05-20 surfaced [arXiv:2409.20052](https://arxiv.org/html/2409.20052) (Liu et al., updated April 2025) — "CLLMR" already in use as *Counterfactual Large Language Model Recommendation* in the recsys/causal-inference literature. Different subfield (recsys vs. hallucination detection), so reviewer-flag risk is low, but the cite-graph collision is permanent. Near-collision: [CLLMRec](https://arxiv.org/abs/2511.17041) (recsys, also 2025). Open Question 7 below tracks the *keep-CLLMR-anyway vs. switch* decision. Title and method-name are both **provisional until that resolves**; do not propagate `CLLMR` into prose, bib entries, or figure captions until the open question closes. Candidate alternatives if a switch is preferred: **LPCR** (Layer-Pair Contrastive Representations) and **CL²R** (Cross-Layer Contrastive Representations) — both encode the layer-pair structure that `CLLMR` does not; neither yet checked for collisions.

Target: 8-page long paper (Main or Findings). References + appendix unlimited.

Structural skeleton only. Each section lists what the prose will cover, not the prose itself. `[bracketed]` items mark spots where the author decides whether to import from planning docs.

**Reference coverage note.** The per-section outlines (`02_related_work.md`, `03_method.md`, `04_experimental_setup.md`) are the authoritative reference lists for their sections and may contain additional citations not enumerated here. This file summarizes structure and key anchors only — absence of a citation from this outline is not an inconsistency.

---

## Abstract (~180 words)

- One-sentence framing of the problem (intrinsic LLM hallucination detection, white-box single-pass).
- One-sentence statement of the method (learned contrastive compression of intermediate-layer activations with a logprob-recon auxiliary loss).
- One-sentence statement of the comparison classes (output-space scalar baselines, single-layer linear probe, MLP probe, sampling-based methods).
- One-sentence headline result naming the AUROC delta and the scope (two model families, six datasets, five seeds).
- One-sentence claim about transfer + layer-pair concentration (the cross-layer-coherence prediction).
- **Do not finalize until §6 numbers freeze** (week 3 of writing plan).

---

## 1. Introduction (~1 page)

- **The phenomenon.** LLMs hallucinate; the cost is asymmetric — wrong answers presented with high fluency are downstream-damaging.
- **The detection axis we work in.** White-box, single forward pass. Distinguish from: retrieval-augmented checking, multi-sample consistency methods, prompt-based self-evaluation.
- **What's already known about activations and hallucination.** A single layer carries some signal (linear-probe literature), but it's not clear which layer or what structure to extract.
- **Our contribution (3 bullets, paper-claim-aligned).**
  1. A learned contrastive compression of intermediate activations is competitive with or outperforms the strongest learned-activation baselines (single-layer linear probe, SAPLMA, LLMsKnow, ACT-ViT) and clearly outperforms sampling-based methods at matched compute. Exact verb (*beats* vs. *matches-or-outperforms*) and headline-cell selection pending ACT-ViT 5/5-seed completion — see Open Question 6.
  2. The signal is consistent across two model families, six datasets, and five training seeds (for our method; ACT-ViT seeds still completing on a subset of cells), and transfers across datasets without retraining.
  3. The effective signal concentrates in mid-to-late residual-stream layer pairs, matching a cross-layer-coherence prediction.
- **Roadmap of the paper.** One paragraph.

---

## 2. Related Work (~0.75 page)

Detailed outline in [02_related_work.md](02_related_work.md). Summary structure (three subsections, ~3–4 references each):

- **2.1 Activation probing for LLM behavior.** Linear probes (Alain & Bengio), SAPLMA single-layer MLP (Azaria & Mitchell), ICR Probe per-layer-scalar MLP (Zhang et al. 2025), CLAP cross-layer attention (Suresh et al. 2025), ACT-ViT full-tensor ViT (Bar-Shalom et al. 2025), plus residual-stream truth-direction evidence (ITI, Marks & Tegmark). Sets our method up as a *learned* extension of probing rather than a replacement.
- **2.2 Hallucination detection.** Output-space scalars (Kadavath), multi-sample consistency (semantic entropy — Farquhar; SelfCheckGPT — Manakul), retrieval-augmented checking scoped out (FActScore — Min). Add one-sentence footnote: SEP (Kossen et al. 2024) trains a linear probe to predict SE; since we report both direct SE and linear probes on the same activations, SEP is covered by their union — not run as a standalone experiment.
- **2.3 Contrastive representation learning.** SimCLR (Chen et al.), InfoNCE (van den Oord et al.), SimCSE (Gao et al.); broad-novelty claim ships hedged with a footnote crediting CRD/CoDIR/CDS as adjacent machinery in different problem settings. Views here are layer pairs, not data augmentations or different networks.

End the section with one sentence positioning our work relative to the three threads.

---

## 3. Method (~2 pages)

Detailed outline in [03_method.md](03_method.md), which supersedes `methods_outline.md` and `theory_outline.md`. Summary structure:

- 3.1 Problem setup + notation
- 3.2 Why a learned cross-layer compression can carry the signal — the information-theoretic argument (folded in, not a standalone theory section)
- 3.3 What we built — supervised contrastive over layer pairs with logprob reconstruction (`ProgressiveCompressor`, layer-pair views, asymmetric `ignore_label` SupCon, logprob-recon auxiliary, KNN/cosine/Mahalanobis scorers, implementation summary)
- 3.4 What the method predicts — the 2×2 attribution table (SupCon-asymm × recon) with four pre-committed outcome-conditional framings

**Theoretical framing** is §3.2 (~half the section), with the full information-bound derivation in Appendix A. Budget is slightly above the original 1.5–2 page target because the theory is absorbed in-section; if the page budget binds, §3.4 collapses first into a single paragraph forwarding to §7.1.

---

## 4. Experimental Setup (~0.5–0.75 page)

- **Models.** Llama-3.1-8B-Instruct, Qwen3-8B. State why these two (open weights, two distinct training pipelines, 8B-scale).
- **Datasets.** HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA. State per-dataset: task type, train/test sizes, evaluator used to label hallucinations, class imbalance ratio. (Movies excluded — no train split.)
- **Baselines.** Three classes:
  1. Output-space scalar: logprob, token entropy, P(true).
  2. Activation-space probes: single-layer linear probe (the "obvious" baseline), SAPLMA (~11M-param MLP, established literature baseline), LLMsKnow (Slobodkin et al. 2023, layer-wise probing baseline), ACT-ViT (Bar-Shalom et al. 2025, full-tensor ViT on activations).
  3. Sampling-based: SE (length-normalized headline + discrete supplementary), SelfCheckGPT (NLI headline + BERTScore + n-gram supplementary).
  - **[TODO: one-sentence parry needed here]** Naive multi-layer concat is excluded from the baseline list; ACT-ViT is the principled learned multi-layer comparison and LLMsKnow covers layer-wise probing — add one sentence justifying the omission on compute/design grounds so reviewers don't flag it as an oversight.
- **Training procedure.** One paragraph: five seeds, cached activations reused across all methods, single 80 GB GPU per cell. Do not reproduce hyperparameters in prose — forward readers to the codebase on GitHub for full implementation details.
- **Metrics.** AUROC ± std across 5 seeds (main paper only). AUPRC, ECE, FPR@95 in supplementary tables.
- **Compute budget summary** (1–2 sentences; full breakdown in appendix).

---

## 5. Main Results (~1.5 pages)

- **5.1 Headline table.** AUROC ± std (across 5 seeds) for every (model, dataset, method) cell. State which cells the contrastive method wins, by how much, and where it does not.
- **5.2 Headline figure.** Per-dataset AUROC bars, both models, baseline cluster vs. ours.
- **5.3 Compute-matched comparison.** AUROC vs. forward-pass count. K=1 cluster: ours, linear probe, SAPLMA, P(true). K=10 cluster: SE (length-normalized), SelfCheckGPT-NLI. One panel per dataset for the 5 free-form datasets; MMLU shows K=1 cluster only.
- **5.4 Calibration.** Reliability diagrams for one dataset per model. ECE in supplementary.

Numbers freeze before this section is written. Until then, table/figure slots are reserved but blank.

---

## 6. Cross-Dataset Transfer (~0.5 page)

- **Procedure.** Train on dataset A, evaluate on dataset B's test split. No retraining, no fine-tuning. Both models. All ordered pairs.
- **Headline table.** 6×6 source→target AUROC heatmap per model. Off-diagonal mean + worst case in the body.
- **One paragraph of interpretation.** Where transfer breaks down and why.

---

## 7. Ablations (~1 page)

- **7.1 Loss decomposition.** SimCLR-only (AUROC ≈ 0.5, our negative result — this is itself a contribution) vs. logprob-recon-only vs. full loss. On 2 representative datasets, 5 seeds. The SimCLR-only cell coincides with **Variant 1** of §7.3 — one run, two ablations.
- **7.2 Layer-pair sensitivity.** Sweep early/mid/late layer pairs on 1–2 datasets. Connects directly to the theoretical framing in §3.3.
- **7.3 Contrastive variant — which side gets labeled.** Holds the recon auxiliary, architecture, layer pair, and scorer fixed; varies only how labels enter the contrastive loss. Four variants, on 2 representative datasets, 5 seeds:
  1. **Unsupervised contrastive (SimCLR over layer pairs).** `use_labels: false`. No label information — layer-pair views are the only positive-pair signal. **Early result: ~0.6 AUROC** — above chance but well below headline, confirming label information is necessary but not sufficient.
  2. **Hallucinations as outliers — truthful labeled, hallucinated unlabeled** *(headline; `use_labels: true, ignore_label: 1`)*. Truthful class is the coherent inlier cluster; hallucinated instances are required only to be view-consistent with themselves. This is the §3.3 / §5 configuration. **Early result: ~0.8 AUROC.**
  3. **Inverse — hallucinated labeled, truthful unlabeled** *(`use_labels: true, ignore_label: 0`)*. Hallucinated class treated as the coherent cluster; truthful instances ignored. **Early result: ≈ Variant 2 ±1 AUROC point** — varies by dataset/model but tracks the headline within noise. Implication: the *direction* of the one-class anchor does not strongly matter; what matters is that one class is unlabeled as the outlier role. The §3.3 "hallucinations are not a coherent class" framing needs revision — see §3.3 update below.
  4. **Symmetric SupCon — both classes labeled** *(`use_labels: true`, no `ignore_label`)*. Both classes pulled into their own coherent clusters; standard SupCon (`khosla2020supcon`). **Early result: drops ~10 AUROC points vs. Variant 2** (i.e., ≈ 0.7 AUROC) — worse than the one-class variants but still above unlabeled. This is a confirmed prediction of Corollary A.1.1: forcing `I(Z; C=1)` to be positive when class 1 lacks coherent latent structure adds gradient noise that actively fights the class-0 term. See [`03_method.md`](03_method.md) Theorems section.
  - **Preliminary ordering (early results, 2026-05-20): {2 ≈ 3} >> {4} > {1}.** Confirmed framing: **Variant 2 ≈ Variant 3 > Variant 4 > Variant 1.** Maps to the third pre-committed framing below — now effectively resolved pending final 5-seed runs.
  - **What this resolves.** Removes the §3.3 hedge ("we have not run and do not commit to running [symmetric SupCon]") and the symmetric-equivalent question raised in [`03_method.md`](03_method.md) Open Question 4. Pre-committed framings (for reference; early data points to the third):
    - **Variant 2 > {3, 4} ≈ Variant 1:** asymmetric `ignore_label=1` is doing the work; both the label-asymmetry *and* the choice of which side to anchor matter. Cleanest support for the current §3.3 prose. *Not supported by early results.*
    - **Variant 2 ≈ Variant 4 > {3, 1}:** any supervised contrastive over layer pairs works; the asymmetry is decorative. *Not supported — Variant 4 drops 10 points.*
    - **Variant 2 ≈ Variant 3 > Variant 4 > Variant 1:** *(early results point here)* the *asymmetry* helps but the *direction* doesn't; either class can serve as the one-class anchor. What is contraindicated is labeling *both* classes. §3.3 "hallucinations are not a coherent class" → revise to "labeling both classes simultaneously forces an incoherent second cluster that actively hurts." Variant 4's drop is a confirmed prediction of Corollary A.1.1, not just an empirical surprise.
    - **All four tied:** label structure of the contrastive loss is irrelevant; recon does all the work. *Not supported — large gaps between variants.*
- **7.4 Scorer choice.** Cosine vs. Mahalanobis vs. KNN. Justifies the KNN headline.
- **7.5 Alternative Augmetnations.** see https://github.com/hyang0129/HalluLens/issues/109
---

## 8. Discussion (~0.5 page)

- **Where the method works.** Mid-to-late layers, both model families, free-form QA + multi-choice MMLU.
- **Where it doesn't.** Whatever the data shows — call it out honestly. Live candidates from the 2026-05-20 draft headline pass: NQ on both models (ACT-ViT modestly above ours, ~0.013–0.018 AUROC), several near-ties on HotpotQA / SciQ / SearchQA. Original risk-register candidates (Qwen weaker than Llama, SAPLMA parity) appear *not* to fire on current numbers but stay in the watch list until 5/5 seeds complete. Update this section after numbers freeze.
- **What the layer-pair concentration result means.** Brief — full theoretical argument in the appendix.

---

## 9. Limitations (~0.25 page)

- Two model families, 8B scale only. No 70B, no closed models.
- Short-form QA only. Long-form (FactScore-style) is out of scope.
- White-box. The method does not apply to API-only models.
- Single language (English). Multilingual is future work.

---

## 10. Conclusion (~0.25 page)

Three sentences, mirroring the abstract's claim. Do not finalize until the abstract is finalized.

---

## Appendix

- **A. Full theoretical derivation.** Cross-layer-coherence argument in full. Imported from `THEORETICAL_JUSTIFICATION.md` *only with explicit author sign-off per the paper-writing rule*.
- **B. Full hyperparameter table.** Every config in `configs/experiments/baseline_comparison_*.json`.
- **C. Full results tables.** Every (model, dataset, method, seed) cell, AUROC + AUPRC + ECE + FPR@95.
- **D. Dataset details.** Per-dataset evaluator prompts, class imbalance, split sizes, exact preprocessing.
- **E. Compute breakdown.** GPU-hours per phase per model.
- **F. Reproducibility checklist.** EMNLP-mandated artifact.

---

## File map for `paper/`

Each section below should eventually have its own file. Naming convention: `NN_section_name.md`.

| File | Status | Owner |
|---|---|---|
| `outline.md` | ✅ this file | — |
| `methods_outline.md` | ⚠️ superseded by `03_method.md` (delete after final review) | — |
| `theory_outline.md` | ⚠️ superseded by `03_method.md` §3.2 (delete after final review) | — |
| `00_abstract.md` | pending | freeze last |
| `01_introduction.md` | pending | week 1 |
| `02_related_work.md` | ✅ outline materialized (prose pending) | week 2 |
| `03_method.md` | ✅ outline materialized (prose pending; depends on §3.2 pending bib + Figure 1 source) | week 2 |
| `04_experimental_setup.md` | pending | week 2 |
| `05_results.md` | pending | week 3 (after numbers freeze) |
| `06_transfer.md` | pending | week 3 |
| `07_ablations.md` | pending | week 3 |
| `08_discussion.md` | pending | week 3 |
| `09_limitations.md` | pending | week 4 |
| `10_conclusion.md` | pending | week 4 |
| `appendix/` | pending | rolling |

---

## Open structural questions

1. ~~**Theory as §3 subsection vs. its own section.**~~ **Resolved 2026-05-19** — folded into §3 as §3.2 (information-theoretic argument), not promoted to a standalone Theory section. Rationale: the load-bearing theoretical move is short, structurally inseparable from the architecture, and the 8-page EMNLP main paper does not have room for a dedicated theory section. Full information-bound derivation lives in Appendix A. See [`03_method.md`](03_method.md) structural-decision header. Soft commit revisable post-internal-review.
2. ~~**Where does SEP-binary / SEP-SE land in §5?**~~ **Removed** — SEP-binary was a confabulation; SEP-SE cut 2026-05-20. Both upper-bounded by max(SE, linear probe), already reported. Defused with a one-sentence footnote in §2.2.
3. **Headline framing across model families.** Preliminary numbers (2026-05-20 draft headline table) suggest Qwen tracks slightly *above* Llama on most cells (mean ours: Qwen 0.852, Llama 0.812), inverting the original "what if Qwen is materially weaker" worry. The opposite framing question is now live — whether the abstract leads with Qwen, leads with Llama, or treats both symmetrically. Don't pre-commit; revisit once 5/5 seeds complete across baselines.
4. **§2.3 novelty-claim level — broad / medium / narrow.** Resolved 2026-05-19 to **broad with hedging** (see `02_related_work.md` §2.3 framing-level decision). Reviewer-rebuttal fallback to medium then narrow is documented there. Abstract and §1 contribution claims must be drafted to the broad level once §3 / §5 freeze.
5. **§3.4 attribution headline — locked when #66 (SupCon-asymm only) and #67 (SAPLMA + recon) land.** Until then, §3.4 prose is the *menu* of four outcome-conditional framings. The headline framing of §3 and the abstract cannot finalize until these resolve. See [`03_method.md`](03_method.md) §3.4. **Related:** the §7.3 contrastive-variant ablation (added 2026-05-20) now also bears on §3.3's load-bearing "asymmetric `ignore_label`" claim — Variant 4 (symmetric SupCon) is the run §3.3 previously declined to commit to. §3.3 prose stays hedged until Variants 2/3/4 land.
6. **ACT-ViT as the primary learned competitor.** Per the 2026-05-20 draft headline pass ([`../results/draft_headline_table.md`](../results/draft_headline_table.md)), ACT-ViT is materially closer than the original outline implied. Mean across both models: ours ≈ 0.83 vs. ACT-ViT ≈ 0.79 (≈ +0.04 on the cells ACT-ViT has reported). Win pattern concentrates in PopQA (~+0.09 both models, 5/5 seeds) and MMLU (~+0.15, ACT-ViT only 2/5 seeds — gap may move). On HotpotQA / SciQ / SearchQA the two methods sit within ±0.03; on NQ ACT-ViT is modestly ahead on both models. **Lead-framing decision (2026-05-20 working commit):** of the three live framings — mean-margin / PopQA-and-MMLU-headline-cells / parameter-or-compute-efficiency-at-matched-AUROC — the working choice is **mean-margin**, because (a) theory is the slim half of §3 and cannot anchor the headline, and (b) compute-efficiency is shared with ACT-ViT (also K=1 activation-space) and therefore non-differentiating against the strongest learned competitor. This commit is provisional and revisits once ACT-ViT 5/5 seeds close on the five outstanding cells — see [`05_results.md`](05_results.md) Open Question 6. Implications: (a) the §1 contribution verb against learned baselines should not lock until ACT-ViT seeds finish; (b) the §1 / §8 / abstract framing all assume the mean-margin lead and need re-drafting if it moves; (c) an interpretation question remains open: ACT-ViT-vs-ours wins concentrate on already-high-AUROC datasets (PopQA is the highest-AUROC cell on the table), not on the hardest-to-detect ones (SciQ, NQ) — does our method *push the ceiling* or *recover the floor*? Decide in §8 after numbers freeze.

7. **CLLMR naming collision — keep or switch.** Working title (top of outline) uses `CLLMR` = *Contrastive Large Language Model Representations* as the method-name backronym, echoing SimCLR (`chen2020simclr`). 2026-05-20 web search surfaced a permanent cite-graph collision with Liu et al., [arXiv:2409.20052](https://arxiv.org/html/2409.20052) (recsys / causal inference), and a near-collision with [arXiv:2511.17041 CLLMRec](https://arxiv.org/abs/2511.17041). Subfield distance makes reviewer-flag risk low but Google Scholar / arXiv search confusion permanent. Options: (a) **keep CLLMR** (status quo; brand-first, ties cleanly to SimCLR, accepts cross-subfield search collision); (b) **switch to LPCR** = *Layer-Pair Contrastive Representations* (additionally encodes the layer-pair-view structure that §3.3 makes load-bearing; collision-check pending); (c) **switch to CL²R** = *Cross-Layer Contrastive Representations* (most distinctive visually, but the superscript reads awkwardly in plain-text contexts including BibTeX keys and grep). Decision blocker: human needs to weigh cite-graph search-uniqueness vs. SimCLR-shape brand recognition. Until resolved, do **not** propagate `CLLMR` into prose, bib entries, or figure captions; references in this outline and in [`05_results.md`](05_results.md) carry the provisional flag.

---

## Bibliography state (as of 2026-05-19)

All §2 anchor citations and §3 load-bearing citations are in [`references.bib`](references.bib). Most recent additions: `khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, `hjelm2019deepinfomax` (all five for §3.2/§3.3, human-approved 2026-05-19 from the verification pass in [`03_method.md`](03_method.md) pending-approval section). The only candidates that are *not* in `references.bib` are the four §2 optionals (Hewitt & Manning, Tenney et al., Ji et al., Lewis et al. RAG) — verified 2026-05-19, defer insertion until a draft surfaces a gap.
