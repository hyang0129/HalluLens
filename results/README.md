# Results — catalog of result types

Three distinct result families feed the paper. They differ in *what's being compared* and *what compute regime they live in*, which is why they are tabled and discussed separately rather than concatenated into a single mega-table.

The single source of truth for cell-level completion is `python scripts/results_table.py` (writes `output/results_table/results_table.{json,csv}`). The descriptions below explain *what* each family is and *where* it sits in the paper; status numbers should be re-checked against the script before citing.

---

## 1. Baseline main paper headline results — activation-space, K=1 forward pass

**What it is.** The headline grid: our method vs. every activation-based and prompt-based competitor at **one forward pass per example**, on cached intermediate-layer activations.

**Methods in this family:**
- `contrastive_logprob_recon` — headline method (full)
- `linear_probe` — single-layer linear probe at layer 22
- `saplma` — `SimpleHaluClassifier` MLP (Azaria & Mitchell 2023)
- `llmsknow_probe` — LLMsKnow-style probe
- `logprob_baseline`, `token_entropy` — deterministic output-space scalars from the same forward pass
- `icr_probe` — Zhang et al. ACL 2025 (in flight, see roadmap §2)
- `p_true` — Kadavath et al. 2022 (output-space, one templated follow-up prompt; not started)
- `multi_layer_linear_probe` — ablation row only, dropped from main baseline grid (commit `fabf4de`)

**Scope.** 2 models (Llama-3.1-8B-Instruct, Qwen3-8B) × 6 datasets (HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA) × 5 seeds for trained methods; 1 run per cell for deterministic methods.

**Where the data lives.** `runs/baseline_comparison_{dataset}[_qwen3]/{dataset}/{method}/seed_*/eval_metrics.json`.

**Paper role.** Main table (AUROC + AUPRC ± 95% CI per model × dataset × method) and headline figure (per-dataset AUROC bars, baseline cluster vs. ours).

**Reviewer concern this family answers.** "Did you beat the obvious activation-based competitors at matched compute?"

---

## 2. Sampling-based main paper results — output-space, K=10 sampling

**What it is.** The strongest output-space competitors. These methods are *not directly comparable* on compute to family (1) because they require K=10 forward passes per example (sampling + entailment clustering or self-consistency). Reported as a **compute-matched** comparison panel rather than mixed into the headline table.

**Methods in this family:**
- **Semantic entropy** — Farquhar et al. *Nature* 2024. Length-normalized variant is headline; discrete variant supplementary.
- **SelfCheckGPT** — Manakul et al. EMNLP 2023. NLI variant is headline; BERTScore and n-gram variants supplementary.
- **SEP-SE** — Kossen et al. 2024. Linear probe on activations trained to predict length-normalized SE. Evaluated as a hallucination detector via AUROC against binary halu labels on test. Bridges K=10 → K=1 at single-forward-pass inference cost.

**Note (purged 2026-05-19).** Earlier drafts of this README listed a "SEP-binary" method (logistic probe on halu labels directly). That method was confabulated by an earlier writing-agent and was never implemented; see [`tasks/sampling_baselines/sep.py:6`](../tasks/sampling_baselines/sep.py#L6) ("SEP-binary … is not [implemented]"). It is also not present in Kossen et al. 2024 — their probe target is the SE score, not the halu label. Only SEP-SE is real and run.

**Scope.** 5 free-form datasets (HotpotQA, NQ, PopQA, SciQ, SearchQA) for SE / SelfCheckGPT / SEP-SE — MMLU excluded because NLI clustering degenerates on single-letter answer tokens. 2 models (Llama, Qwen3). SearchQA capped at 5k train / 10k test items to bound GPU. SEP-SE trained on 5k stratified train subset per (dataset, model).

**Where the data lives.** Sampling pass outputs in `output/sampling/` (see issue #49); SEP-SE training runs live alongside family (1) under `runs/baseline_comparison_*/.../sep_se/`.

**Paper role.** Compute-matched comparison panel — one panel per free-form dataset: K=1 cluster (ours, linear probe, SAPLMA, SEP-SE, P(true)) vs. K=10 cluster (SE-length-norm, SelfCheckGPT-NLI). Not in the headline AUROC table.

**Reviewer concern this family answers.** "Didn't compare to the strongest sampling-based baselines / didn't compare per-FLOP."

**Why separated from family (1).** Mixing K=1 and K=10 numbers in the same AUROC column invites the reviewer to read it as "ours loses to SE on dataset X" when the right comparison is per-compute. SEP-SE is the single-forward-pass bridge: it recovers the SE signal from one forward pass, so the K=1 cluster includes a method designed to approximate K=10 SE.

---

## 3. Loss-decomposition ablation — 2×2 attribution (issues #66 + #67, PR #68)

**What it is.** A small, surgical ablation that isolates which component of the full method carries the gain over SAPLMA. Two axes:
- **Loss objective:** BCE (SAPLMA) vs. supervised contrastive InfoNCE on layer pairs.
- **Recon auxiliary:** off vs. on (logprob reconstruction).

That gives a 2×2:

| | no recon | + recon |
|---|---|---|
| **BCE** | `saplma` | `saplma_logprob_recon` *(new)* |
| **Supervised InfoNCE** | `contrastive` *(new)* | `contrastive_logprob_recon` (= full method) |

**Methods that are new in this family.** `saplma_logprob_recon` (#67 — new `SaplmaWithReconHead` + `SaplmaReconTrainer`); `contrastive` (#66 — config-only, no new code, uses existing `ProgressiveCompressor` without the recon head).

**Scope.** 2 datasets (HotpotQA, PopQA) × 2 models (Llama-3.1-8B, Qwen3-8B) × 5 seeds × 2 new methods = 40 cells. All complete (2026-05-18).

**Where the data lives.** `runs/baseline_comparison_{hotpotqa,popqa}[_qwen3]/{hotpotqa,popqa}/{contrastive,saplma_logprob_recon}/seed_*/eval_metrics.json`.

**Paper role.** Ablation table in §3 (Methods / theory) supporting the framing of the contribution. **Not** in the headline AUROC table — the headline already includes `saplma` and `contrastive_logprob_recon` as competitors and the full method respectively; this ablation just adds the two off-diagonal corners to attribute the gap.

**Reviewer concern this family answers.** "How do you know the contrastive structure does the work, and not the recon auxiliary?"

**Verdict (recorded in PAPER_ROADMAP §4 item 7).** Supervised contrastive layer-pair InfoNCE carries essentially all the gain over SAPLMA (+0.02 to +0.14 AUROC across the four cells). Logprob-recon as a free-floating auxiliary is decorative — SAPLMA+recon ≈ SAPLMA in every cell (|Δ| ≤ 0.006). Recon helps on top of contrastive in exactly 1 of 4 cells (PopQA-Qwen3, +0.035); elsewhere flat or slightly negative. The earlier "recon is load-bearing" reading from the SimCLR-only ablation was a confound — that ablation correctly showed self-supervised contrastive fails, but the load-bearing component is the *supervised label structure*, not recon.

**Why separated from families (1) and (2).** It is not a baseline competition — it answers an attribution question internal to our own method. Its rows are not paper-table peers of `linear_probe` or `semantic_entropy`; they only make sense in pairs that span one axis of the 2×2.

---

## Cross-family notes

- **`saplma` and `contrastive_logprob_recon` appear in both family (1) and family (3).** In (1) they are competitors / our method against the field. In (3) they are corners of the 2×2 and only meaningful when juxtaposed with their `+recon` / `−recon` counterparts. The numbers are the same; the framing differs.
- **Cross-dataset transfer matrix** (PAPER_ROADMAP §4 item 5) is a separate analysis, not a result family — it reuses checkpoints from family (1) and reports a single off-diagonal AUROC summary.
- **Extended metrics (AUPRC, ECE, FPR@95, bootstrap CIs)** are CPU-side recomputes on `predictions.csv` from family (1) and (3); they are not a new family but a richer rendering of the same cells.
