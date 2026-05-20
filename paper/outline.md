# Paper Outline — EMNLP 2026 Submission

Target: 8-page long paper (Main or Findings). References + appendix unlimited.

Structural skeleton only. Each section lists what the prose will cover, not the prose itself. `[bracketed]` items mark spots where the author decides whether to import from planning docs.

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
  - Note the deliberate omission: multi-layer linear probe is in §[ablations] only — it underperforms the single-layer probe and is reported to motivate the learned compression.
- **Metrics.** AUROC (headline), AUPRC (paired in main table), ECE + FPR@95 + bootstrap 95% CIs in supplementary tables.
- **Compute budget summary** (1–2 sentences; full breakdown in appendix).

---

## 5. Main Results (~1.5 pages)

- **5.1 Headline table.** AUROC ± 95% CI for every (model, dataset, method) cell. State which cells the contrastive method wins, by how much, and where it does not.
- **5.2 Headline figure.** Per-dataset AUROC bars, both models, baseline cluster vs. ours.
- **5.3 Compute-matched comparison.** AUROC vs. forward-pass count. K=1 cluster: ours, linear probe, SAPLMA, P(true). K=10 cluster: SE (length-normalized), SelfCheckGPT-NLI. One panel per dataset for the 5 free-form datasets; MMLU shows K=1 cluster only.
- **5.4 Calibration.** Reliability diagrams for one dataset per model. ECE in the main table.

Numbers freeze before this section is written. Until then, table/figure slots are reserved but blank.

---

## 6. Cross-Dataset Transfer (~0.5 page)

- **Procedure.** Train on dataset A, evaluate on dataset B's test split. No retraining, no fine-tuning. Both models. All ordered pairs.
- **Headline table.** 6×6 source→target AUROC heatmap per model. Off-diagonal mean + worst case in the body.
- **One paragraph of interpretation.** Where transfer breaks down and why.

---

## 7. Ablations (~1 page)

- **7.1 Loss decomposition.** SimCLR-only (AUROC ≈ 0.5, our negative result — this is itself a contribution) vs. logprob-recon-only vs. full loss. On 2 representative datasets, 5 seeds.
- **7.2 Multi-layer probe (motivation row).** Concatenating layers 14–29 underperforms the single-layer probe at layer 22. Use this to argue that naive multi-layer use is the wrong baseline; learned compression is the right one.
- **7.3 Layer-pair sensitivity.** Sweep early/mid/late layer pairs on 1–2 datasets. Connects directly to the theoretical framing in §3.3.
- **7.4 Scorer choice.** Cosine vs. Mahalanobis vs. KNN. Justifies the KNN headline.

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
5. **§3.4 attribution headline — locked when #66 (SupCon-asymm only) and #67 (SAPLMA + recon) land.** Until then, §3.4 prose is the *menu* of four outcome-conditional framings. The headline framing of §3 and the abstract cannot finalize until these resolve. See [`03_method.md`](03_method.md) §3.4.
6. **ACT-ViT as the primary learned competitor.** Per the 2026-05-20 draft headline pass ([`../results/draft_headline_table.md`](../results/draft_headline_table.md)), ACT-ViT is materially closer than the original outline implied. Mean across both models: ours ≈ 0.83 vs. ACT-ViT ≈ 0.79 (≈ +0.04 on the cells ACT-ViT has reported). Win pattern concentrates in PopQA (~+0.09 both models, 5/5 seeds) and MMLU (~+0.15, ACT-ViT only 2/5 seeds — gap may move). On HotpotQA / SciQ / SearchQA the two methods sit within ±0.03; on NQ ACT-ViT is modestly ahead on both models. Implications: (a) the §1 contribution verb against learned baselines should not lock until ACT-ViT seeds finish; (b) the §1 / §8 framing needs to choose between *mean-margin* story, *PopQA-and-MMLU-headline-cells* story, or *parameter-/compute-efficiency-at-matched-AUROC* story — pick after numbers freeze; (c) an interpretation question is also open: ACT-ViT-vs-ours wins concentrate on already-high-AUROC datasets (PopQA is the highest-AUROC cell on the table), not on the hardest-to-detect ones (SciQ, NQ) — does our method *push the ceiling* or *recover the floor*? Decide in §8 after numbers freeze.

---

## Bibliography state (as of 2026-05-19)

All §2 anchor citations and §3 load-bearing citations are in [`references.bib`](references.bib). Most recent additions: `khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, `hjelm2019deepinfomax` (all five for §3.2/§3.3, human-approved 2026-05-19 from the verification pass in [`03_method.md`](03_method.md) pending-approval section). The only candidates that are *not* in `references.bib` are the four §2 optionals (Hewitt & Manning, Tenney et al., Ji et al., Lewis et al. RAG) — verified 2026-05-19, defer insertion until a draft surfaces a gap.
