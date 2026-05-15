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
  1. A learned contrastive compression of intermediate activations beats linear probes, an 11M-param MLP probe (SAPLMA), and sampling-based methods at matched compute.
  2. The signal is consistent across two model families, six datasets, and five training seeds, and transfers across datasets without retraining.
  3. The effective signal concentrates in mid-to-late residual-stream layer pairs, matching a cross-layer-coherence prediction.
- **Roadmap of the paper.** One paragraph.

---

## 2. Related Work (~0.75 page)

Three subsections, each ~3–4 references:

- **2.1 Activation probing for LLM behavior.** Linear probes, SAPLMA, probing literature broadly. Set up our method as a *learned* extension of probing rather than a replacement.
- **2.2 Hallucination detection.** Output-space methods: logprob / entropy / P(true) (Kadavath); sampling-based: semantic entropy (Farquhar), SelfCheckGPT (Manakul); probe-on-uncertainty hybrids: SEP (Kossen).
- **2.3 Contrastive representation learning.** SimCLR, InfoNCE, and why self-supervised contrastive losses on intermediate activations is a non-obvious use of the framework — the views here are layer pairs, not data augmentations.

End the section with one sentence positioning our work relative to the three threads.

---

## 3. Method (~1.5–2 pages)

See [methods_outline.md](methods_outline.md) for the full subsection breakdown. Summary structure:

- 3.1 Problem setup + notation
- 3.2 Activation extraction (layer band, token position, caching)
- 3.3 Contrastive compression architecture (`ProgressiveCompressor`, layer-pair views)
- 3.4 Training objective (SimCLR + logprob-recon)
- 3.5 Inference scoring (KNN headline; cosine, Mahalanobis as ablations)
- 3.6 Implementation details

**Brief theoretical framing** (~3–4 sentences) lives inside §3.3. Full derivation of the cross-layer-coherence argument is in the appendix.

---

## 4. Experimental Setup (~0.5–0.75 page)

- **Models.** Llama-3.1-8B-Instruct, Qwen3-8B. State why these two (open weights, two distinct training pipelines, 8B-scale).
- **Datasets.** HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA. State per-dataset: task type, train/test sizes, evaluator used to label hallucinations, class imbalance ratio. (Movies excluded — no train split.)
- **Baselines.** Three classes:
  1. Output-space scalar: logprob, token entropy, P(true).
  2. Activation-space probes: single-layer linear probe (the "obvious" baseline), SAPLMA (~11M-param MLP, established literature baseline).
  3. Sampling-based: SE (length-normalized headline + discrete supplementary), SelfCheckGPT (NLI headline + BERTScore + n-gram supplementary), SEP-SE (Kossen probe-on-SE), SEP-binary (probe on binary label, free byproduct).
  - Note the deliberate omission: multi-layer linear probe is in §[ablations] only — it underperforms the single-layer probe and is reported to motivate the learned compression.
- **Metrics.** AUROC (headline), AUPRC (paired in main table), ECE + FPR@95 + bootstrap 95% CIs in supplementary tables.
- **Compute budget summary** (1–2 sentences; full breakdown in appendix).

---

## 5. Main Results (~1.5 pages)

- **5.1 Headline table.** AUROC ± 95% CI for every (model, dataset, method) cell. State which cells the contrastive method wins, by how much, and where it does not.
- **5.2 Headline figure.** Per-dataset AUROC bars, both models, baseline cluster vs. ours.
- **5.3 Compute-matched comparison.** AUROC vs. forward-pass count. K=1 cluster: ours, linear probe, SAPLMA, SEP-binary, SEP-SE, P(true). K=10 cluster: SE (length-normalized), SelfCheckGPT-NLI. One panel per dataset for the 5 free-form datasets; MMLU shows K=1 cluster only.
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
- **Where it doesn't.** Whatever the data shows — call it out honestly. (See roadmap §9 risk register for the live candidates: Qwen weaker than Llama, SEP-binary parity, etc. Update this section after numbers freeze.)
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
| `methods_outline.md` | ✅ done | — |
| `00_abstract.md` | pending | freeze last |
| `01_introduction.md` | pending | week 1 |
| `02_related_work.md` | pending | week 2 (uses SOTA tracker *with permission*) |
| `03_method.md` | pending | week 2 (expand from methods_outline.md) |
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

These need an explicit decision before §3 prose starts:

1. **Theory as §3 subsection vs. its own section.** Currently §3.3 holds 3–4 sentences with full derivation in appendix. Alternative: dedicated §3 Theory section before Method (would shift everything down). Decision needed.
2. **Where does SEP-binary land in §5?** It's compute-matched with our linear probe and is the genuine threat per the risk register. Options: (a) in the main table alongside linear probe, (b) as a dedicated subsection 5.5. Decision needed.
3. **Headline framing if Qwen is materially weaker than Llama.** Per risk register, this changes the abstract. Don't pre-commit; check before week 3.
