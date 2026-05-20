# Paper Roadmap — EMNLP Submission

**Last updated:** 2026-05-20 (SEP-SE and SEP-binary explicitly cut — see decision below. LLM-judge ablation for ICR Probe explicitly cut — see #76. Verified the ICR paper's labeling protocol is undocumented (zero hits for "judge"/"evaluator"/"GPT" in paper text, no supplementary materials on ACL Anthology) and the released training code is broken (undefined `_load_data()`/`_create_data_loaders()` in `src/icr_probe.py`, public GitHub Issue #3 since Aug 2025 unresolved; third-party request for full pipeline GitHub Issue #5 Nov 2025 unanswered). We will not chase a number from an unreproducible paper; substring-match labeling stays primary, justified by literature standard (LLMsKnow, P(True), Farquhar et al.). Methodology footnote added to §4 item 10.)
**Previously:** 2026-05-17 (issue #70 ICR Probe implementation complete — PR #74 draft open; `activation_research/icr_probe.py + icr_trainer.py + icr_dataset.py` 24/24 tests green; capture workers draining on Empire AI (8 workers, hotpotqa × 2 models); sanity gate redefined — target is model-specific discriminability (>0.55 AUROC), NOT paper's Gemma-2 0.84 AUROC (different model family, gate was meaningless). Issue #72 memmap capture infra merged to main (`97d1353`). SAPLMA and Qwen3 seed sweeps ongoing per 2026-05-15.)
**Previously:** 2026-05-15 (reconciled §2 against `scripts/results_table.py` disk audit: Llama all 5 seeds complete across every trained method; sampling bundle #49 is **done** (60/60 cells); SAPLMA #51 is **in flight, not unstarted** (55/90 — Llama-complete, Qwen partial); Qwen3 has 21/132 cells outstanding from the #58 re-runs; SmolLM3 configs exist but are explicitly out of scope per §5.)
**Previously:** 2026-05-14 (extended-metrics row scoped to AUPRC + ECE + FPR@95 + bootstrap 95% CIs as a pure-CPU recompute on existing `predictions.csv` — see #59; llmsknow_probe seed bug fixed in #58, archived seed=null runs and dispatched 5-seed re-runs across 3 GPU nodes)
**Previously:** 2026-05-12 (SAPLMA baseline added to grid; semantic-entropy item rescoped to a sampling-based baselines bundle — SE + SelfCheckGPT + SEP-SE + SEP-binary across 5–6 datasets, see #49)
**Target venue:** EMNLP 2026 (Main / Findings)
**Submission window:** ~4 weeks from today
**Status:** seed-0 sweep complete; multi-seed expansion + second-model + missing baselines in flight

This roadmap is intentionally narrow. The broader ideas pool — extra contrastive variants, asymmetric encoders, K-view fusion, spectral analysis, retrieval baselines — lives in [`docs/planning/PAPER_ROADMAP_LEGACY.md`](docs/planning/PAPER_ROADMAP_LEGACY.md). Anything not listed below is **out of scope** for this submission. New ideas go through the legacy doc first; promote to here only if they unblock a reviewer concern, not because they're interesting.

Companion docs:
- Current numbers: [`results/seed0_results.md`](results/seed0_results.md)
- Theory: [`THEORETICAL_JUSTIFICATION.md`](THEORETICAL_JUSTIFICATION.md)
- SOTA tracker: [`docs/planning/SOTA_TRACKER.md`](docs/planning/SOTA_TRACKER.md)
- Schema/contract for runs: see legacy §5D

---

## 1. Paper claim (one paragraph)

A learned contrastive compression of intermediate-layer activations detects intrinsic LLM hallucinations more reliably than (a) logprob/entropy from the same generation, (b) a single-layer linear probe on the same activations, and (c) sampling-based output-uncertainty methods at matched compute. A naive multi-layer linear probe (concatenated lower-half layers) underperforms the single-layer probe — reported as an ablation, not a main baseline — which motivates the learned compression. The signal is consistent across two model families (Llama-3.1-8B-Instruct and Qwen3-8B), six QA/reasoning benchmarks (HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA), and five training seeds; it transfers across datasets without retraining; and it concentrates in mid-to-late residual-stream layer pairs as predicted by the cross-layer-coherence theory.

If any of those clauses fails to hold up empirically, the framing changes — that's fine, but **the framing must match what the data actually shows**. Don't write the abstract until §2 is closed.

---

## 2. What's done vs. what's missing

> **Source of truth:** run `python scripts/results_table.py` for the live cell-by-cell disk audit (training + sampling + SEP + P(true)). The table below is a hand-curated summary that can drift; the script is authoritative for any "is X done" question.

| Component | Status | Notes |
|---|---|---|
| Llama-3.1-8B inference + activation logging, 6 datasets | ✅ done | HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA (Movies dropped — no train split) |
| Llama-3.1-8B seed-0 training × 6 datasets × {linear, multi-layer, contrastive×3 scorers} | ✅ done | See [`results/seed0_results.md`](results/seed0_results.md) |
| Qwen3-8B inference + activations, 6 datasets | ✅ done | Substitutes the originally-planned Qwen2.5-7B (better availability, same scale) |
| Qwen3-8B seed-0 training × 6 datasets | ✅ done | See [`results/seed0_qwen3_results.md`](results/seed0_qwen3_results.md) |
| SearchQA Multi-Layer probe | ↪️ obsolete | Multi-layer probe dropped from main grid (see below) |
| Multi-layer probe dropped from main baseline grid | ✅ done | Commit `fabf4de`. Concatenating all 16 lower-half layers overfit — consistently underperformed the single-layer linear probe at layer 22. Keep as an **ablation row**, not a main baseline. Paper must explain the drop and report numbers from seed-0. |
| Movies anomaly (contrastive < linear probe) | ↪️ obsolete | Movies excluded from experiment matrix (no train split) |
| Llama seeds {1, 2, 3, 4} × 6 datasets × learned methods | ✅ done | All 132/132 Llama cells complete across linear / contrastive / saplma / llmsknow / logprob / token-entropy (verified via `results_table.py` 2026-05-15). |
| Qwen3-8B seed sweep, seeds {1, 2, 3, 4} × 6 datasets | 🟡 in flight | 111/132 cells complete; 21 remaining from the #58 re-runs dispatched 2026-05-14. Runner shells: `scripts/run_baseline_qwen3_seed{1..4}.sh`. |
| SmolLM3 third-model expansion | ⛔ out of scope | Configs exist (`configs/experiments/baseline_comparison_*.json` reference SmolLM3 → 162 pending cells), but per §5 we are not adding a third model for this submission. Configs left in place for the journal extension; do not dispatch. |
| SAPLMA baseline (`SimpleHaluClassifier`) on full grid | 🟡 in flight | 55/90 cells complete across {Llama-3.1-8B, Qwen3-8B} (Llama-complete, Qwen partial in the #58 sweep). Code at [`activation_research/model.py:276`](activation_research/model.py#L276); wired into the baseline grid. Forecloses "linear probe is a strawman." Tracked in [#51](https://github.com/hyang0129/HalluLens/issues/51). |
| P(true) baseline | ❌ not started | 0/12 cells. Cheap; one extra prompt per example. Tracked in [#50](https://github.com/hyang0129/HalluLens/issues/50). Output-space only — does not need activation logging; can run concurrently with the Qwen seed sweep. |
| Sampling-based baselines bundle (SE + SelfCheckGPT) | ✅ done | 60/60 sampling cells complete: SE × 3 variants (length-normalized, discrete, semantic-entropy) + SelfCheckGPT × 3 variants (NLI, BERTScore, n-gram) across 5 free-form datasets × 2 models. Tracked in [#49](https://github.com/hyang0129/HalluLens/issues/49). |
| SEP-SE + SEP-binary probes | ⛔ explicitly cut | Cut 2026-05-20. SEP is upper-bounded by max(SE, linear probe), both of which are already reported — no independent reviewer concern it would foreclose. "SEP-binary" is not a paper concept: it is Kossen et al.'s "accuracy probe" baseline, which is architecturally identical to our linear probe. Two independent Opus reviewer simulations agreed: absence is a minor weakness fully defused by a one-sentence footnote ("SEP (Kossen et al. 2024) trains a linear probe to predict SE; since we report both direct SE and linear probes on the same activations, SEP is covered by their union"). With ICR Probe (ACL 2025) and act_vit present, SEP is not the recency frontier. Add the footnote to §Related Work; do not run the experiments. |
| ICR Probe capture (HotpotQA × {Llama-3.1-8B, Qwen3-8B}) | 🟡 in flight | 8 capture workers draining on Empire AI (alphagpu04/07/17/19/23). Writes `shared/icr_capture/<cell>/` per sample: `icr_scores.npy` (N, L), `response_activations.npy`, `meta.jsonl`. Capture infra (issue #72) merged to main (`97d1353`). |
| ICR Probe baseline code (Zhang et al. ACL 2025) | 🟡 in review | PR #74 draft: `activation_research/icr_probe.py` (L → 128 → 64 → 32 → 1 MLP), `icr_trainer.py` (Adam + ReduceLROnPlateau + early-stop on val AUROC), `icr_dataset.py` (mode="memmap"), `run_experiment.py` dispatch, `configs/methods/icr_probe.json`. 24/24 tests green. Gated on sanity checks (Gates 0–4 per `notes/icr_probe_sanity.md`) before Phase 1 launch. Tracked in [#70](https://github.com/hyang0129/HalluLens/issues/70). |
| ICR Probe Phase 1 (HotpotQA × {Llama, Qwen3} × 5 seeds) | ❌ not started | Gated on: capture complete + Gate 0–3 passing (score sane, >0.55 single-layer AUROC on train, smoke test >0.55 on test). Gate target is model-specific — paper's Gemma-2 0.84 is not the bar. |
| ICR Probe Phase 2 (5 remaining datasets × 2 models × 5 seeds) | ❌ not started | Gated on Phase 1 Gate 4: competitive vs other baselines on same (model, dataset). Roll out if passes; document-and-proceed if mid-pack; stop only if at-chance. |
| ICR Probe LLM-judge labeling ablation | ❌ explicitly cut | Verified the ICR paper does not document the judge protocol (no model, prompt, or parsing rule in paper text; no supplementary materials) and the released code is non-functional (undefined data-loading methods, [Issue #3](https://github.com/XavierZhang2002/ICR_Probe/issues/3); third-party reproduction request [Issue #5](https://github.com/XavierZhang2002/ICR_Probe/issues/5) unanswered). We will not invent a surrogate judge and chase the paper's 0.798 number. Substring labels stay primary (literature standard: LLMsKnow, P(True), Farquhar et al.). Decision recorded in [#76](https://github.com/hyang0129/HalluLens/issues/76). If a reviewer demands the ablation at rebuttal, reopen #76. |
| Cross-dataset transfer table | ❌ not started | Train→test pairs across datasets, no new training |
| AUPRC, ECE, FPR@95, bootstrap 95% CIs | ❌ not started | Pure CPU recompute on existing `predictions.csv` (zero GPU). FPR@95 = standard operational detection metric (lower is better). Tracked in [#59](https://github.com/hyang0129/HalluLens/issues/59). |
| Ablation: SimCLR-only vs +logprob-recon | ✅ tried | Unsupervised SimCLR features + linear probe on top → ~random guessing. Logprob-recon aux loss is load-bearing. Need to log the run + write it up; no further compute. |
| Ablation: layer-pair sensitivity | ❌ not started | Reuses cached activations |
| Paper draft | ❌ not started | Target ~16h student time; LaTeX from week 3 |

---

## 3. Submission strategy

**Target: EMNLP only.** No NeurIPS detour. Rationale:
- NeurIPS / EMNLP both prohibit concurrent submission; withdrawal-and-resubmit is possible but NeurIPS reviews land in August, well *after* EMNLP submission closes — so the "free reviews" rationale doesn't apply.
- The week of GPU + writing time required for a NeurIPS smoke test is precisely the budget needed to add the second model + missing baselines that elevate the EMNLP version.
- Venue fit: this is an NLP-method paper, not an ML-theory paper.

**Reconsider only if** Qwen seed-0 results land cleanly in the same week and clearly support the claim — at that point a NeurIPS submission is no longer a smoke test. Decide on data, not calendar.

---

## 4. Must-add list (in priority order)

Each item names the reviewer concern it forecloses. Don't reorder without a reason.

1. **Second model: Qwen2.5-7B-Instruct.** Forecloses "results may be Llama-specific." This is the single highest-leverage missing piece. The Qwen 72B qgen plumbing is partially built (legacy §8.6); the 7B inference path needs `model_map` + `--quantization` wiring or the equivalent for non-quantized 7B. Prefer non-quantized 7B unless VRAM forces otherwise. *(Status note: substituted with Qwen3-8B-Instruct — see §2 status table.)*
2. **SAPLMA baseline** (Azaria & Mitchell 2023, "The Internal State of an LLM Knows When It's Lying"). Forecloses "linear probe is a strawman; you didn't compare to the established activation-based hallucination detector." The model is already implemented as `SimpleHaluClassifier` in [`activation_research/model.py:276`](activation_research/model.py#L276) — a 3-hidden-layer MLP (4096 → 2048 → 1024 → 512 → 1) on the last-token activations of a single layer, with ReLU + Dropout. It's wired into the trainer at [`scripts/train_activation_model.py:235`](scripts/train_activation_model.py#L235) as `simple_halu_classifier`, but **not in any `configs/experiments/baseline_comparison_*.json`** — so it's never been run in the baseline grid. The discriminating question this addresses: does our contrastive method beat (a) a 4K-param single linear hyperplane and (b) an 11M-param nonlinear MLP on the same activations, or only (a)? Without this, the paper's headline could collapse to "more parameters helps" rather than "contrastive structure helps." Activation-space; reuses cached activations; no new inference. Run on the full grid: 5 seeds × 6 datasets × {Llama-3.1-8B, Qwen3-8B} = 60 training runs. Probe layer should match `linear_probe` (layer 22) for an apples-to-apples comparison. Tracked in [#51](https://github.com/hyang0129/HalluLens/issues/51).
3. **P(true) baseline** (Kadavath et al. 2022). Forecloses "didn't compare to prompt-based self-evaluation." Implementation cost: one templated follow-up prompt per generated answer. Run on all 6 datasets, both models. **Output-space only — does not need activation logging**, so this can be dispatched independently of the seed sweeps. Tracked in [#50](https://github.com/hyang0129/HalluLens/issues/50).
4. **Sampling-based baselines bundle** — one K=10 sampling pass produces 4 baselines, each foreclosing a distinct reviewer concern:
   - **Semantic entropy** (Farquhar et al. *Nature* 2024) — forecloses "didn't compare to the strongest sampling-based method."
   - **SelfCheckGPT** (Manakul et al. EMNLP 2023; variants NLI / BERTScore / n-gram) — forecloses "didn't compare to the canonical black-box consistency baseline."
   ~~**SEP-SE** (Kossen et al. 2024) — cut 2026-05-20. SEP-binary is not a real paper concept. Both are upper-bounded by existing baselines. See §2 table entry for full rationale.~~

   **Scope:** 5 free-form datasets (HotpotQA, NQ, PopQA, SciQ, SearchQA) for SE / SelfCheckGPT. SearchQA capped at 10k test / 5k train items. Frame as **compute-matched comparison** (1 forward pass vs. 10 samples). Both models if budget allows; Llama-only is the firm fallback. **Sampling pass is output-space only.** Tracked in [#49](https://github.com/hyang0129/HalluLens/issues/49).
5. **Cross-dataset transfer table.** Forecloses "trained classifier may overfit per-dataset." For each (source, target) dataset pair, evaluate the source-trained checkpoint on the target test split. No new training. One supplementary table, one summary number in the body.
6. **AUPRC + ECE + bootstrap 95% CIs across seeds.** Forecloses "AUROC alone hides class imbalance / calibration issues / seed noise." Pure analysis; merge into the main table where space allows, push extras to appendix.
7. **Ablation: contrastive loss decomposition — writeup only.** Already tried: unsupervised SimCLR contrastive pretraining followed by a linear probe on the learned features → ~random (AUROC ≈ 0.5). Interpretation: the logprob-recon auxiliary loss carries the hallucination signal; SimCLR alone learns representations that are invariant to the right things for self-supervision but orthogonal to the hallucination axis. The logprob-recon-only variant is still worth running (1 retrain × seeds × 2 datasets, cached activations) to complete the decomposition. Either way, this ablation closes the "which part of the loss matters" question, and the negative SimCLR-only result is itself a useful contribution.
8. **Ablation: layer-pair sensitivity.** Sweep layer pairs on 1–2 datasets using cached layers 14–29 activations. Reuses cached activations. Connects directly to Justification 2.
9. **Multi-layer probe ablation + writeup.**
10. **ICR Probe baseline (Zhang et al. ACL 2025, arXiv:2507.16488).** Forecloses "didn't compare to activation-based probe methods from recent literature." Architecture: per-layer JSD scores (L-dim vector) fed through an L → 128 → 64 → 32 → 1 MLP with BatchNorm + LeakyReLU + Dropout. Paper reports ~0.84 AUROC on HotpotQA with **Gemma-2** and 0.7982 on Llama-3 — a different model family / model version from ours; our gate is model-specific discriminability (best single-layer AUROC > 0.55), not paper's number. Code complete in PR #74; Phase 1 launch (HotpotQA × {Llama, Qwen3} × 5 seeds) gated on sanity checks. Phase 2 rollout (5 remaining datasets) gated on Phase 1 competitive rank. Tracked in [#70](https://github.com/hyang0129/HalluLens/issues/70).

   **Methodology footnote for the paper.** Our reproduction lands at AUROC ≈ 0.675 under substring-match labeling, below the paper's 0.798 Llama-3 figure. We investigated and confirmed the ICR paper is not reproducible from released artifacts:
   - The training labels are loaded from an `output_judge.jsonl` artifact whose generation pipeline is not in the released repository. The paper text contains no description of the judge model, prompt, decoding settings, or parsing rule (verified: zero hits for "judge", "evaluator", "GPT", or "annotator" in the full arXiv v1 HTML or ACL Anthology PDF; no supplementary materials).
   - The released training script (`src/icr_probe.py`) calls `_load_data()` and `_create_data_loaders()`, neither of which is defined anywhere in the repository — flagged publicly in [GitHub Issue #3](https://github.com/XavierZhang2002/ICR_Probe/issues/3) since August 2025, unresolved.
   - A third-party request for the full reproduction code in [GitHub Issue #5](https://github.com/XavierZhang2002/ICR_Probe/issues/5) (November 2025) has gone unanswered.

   We decline to invent a surrogate labeling pipeline and chase the paper's headline number against an unreproducible target. Our reproduction follows the paper's published algorithm specification (§3, Algorithm 1) faithfully, uses substring-match labels (the dominant standard in the short-form QA probing literature — LLMsKnow, P(True), Farquhar et al.), and lands ICR Probe alongside SAPLMA (0.69), LN-Entropy (0.65), PPL (0.61), and our own contrastive method (0.85) on the same labels. Five other reproduced baselines land within reasonable range of the paper's reported values, providing positive evidence that our evaluation pipeline is sound. Decision recorded in [#76](https://github.com/hyang0129/HalluLens/issues/76). The multi-layer probe (concatenation of layers 14–29) was dropped from the main baseline grid because it consistently underperformed the single-layer linear probe at layer 22 — likely overfitting from the ~16× parameter inflation with no inductive bias tying the layers together. We owe the reader: (a) one ablation row showing seed-0 multi-layer numbers next to single-layer, (b) a sentence in §Methods explaining why naive concatenation is the wrong baseline and why our learned contrastive compression is the right comparison, (c) connect this to the layer-pair-sensitivity ablation (item 8) — the contrastive method picks specific layer pairs rather than dumping all of them in. This is now part of the **motivation** for the method, not a hole.

## 5. Won't-add list (explicit cuts — read this before scoping creep)

- ❌ More models beyond {Llama-3.1-8B, Qwen2.5-7B}. No 70B, no Mistral, no Gemma.
- ❌ Long-form benchmarks (LongWiki, FactScore-style). Save for the journal extension.
- ❌ Asymmetric / BYOL-style encoders, K>2 contrastive views, FiLM layer conditioning, spectral analysis. All in the legacy idea pool; none are required for the claim.
- ❌ SEP-SE / SEP-binary (Kossen et al. 2024). Upper-bounded by max(SE, linear probe), both reported. "SEP-binary" is not a paper concept. Defuse with a one-sentence footnote in §Related Work. See §2 status table for full rationale.
- ❌ Refactoring the storage layer (Zarr-only migration). Use what's already cached.
- ❌ A new model architecture. The current `ProgressiveCompressor` is the method; ablate it, don't replace it.
- ❌ Retrieval-augmented or external-verifier baselines. We're explicitly white-box; cite and bound the claim.
- ❌ TruthfulQA as a *training* benchmark. Eval-only transfer use, per the dataset-size constraint.

If a reviewer concern surfaces during writing that one of these would address, fix it in the limitations section instead of adding the experiment.

---

## 6. Experimental matrix (concrete runs)

Per (model, dataset, method) cell: 5 training seeds {0, 5, 26, 42, 63}, split seed 42, evaluation on the held-out test split. Logprob and entropy baselines are deterministic given fixed inference — one run per (model, dataset).

**Models:** {Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct}
**Datasets:** {HotpotQA, NQ, MMLU, Movies, PopQA, SciQ, SearchQA}
**Trained methods (per cell × 5 seeds):** {Linear probe, SAPLMA (`SimpleHaluClassifier`), Contrastive+Logprob-recon, ICR Probe (`icr_probe` — Phase 1 HotpotQA only; Phase 2 remaining datasets gated on Phase 1 competitive rank)}
**Ablation-only (seed-0, not in main grid):** Multi-layer probe (layers 14–29 concatenated) — reported once for motivation; see §4 item 9.
**Non-trained methods (per cell × 1):** {Logprob, Token entropy, P(true)}
**Sampling baselines** (5 free-form datasets only — MMLU excluded due to NLI-clustering degeneracy on letter tokens; SearchQA test capped at 10k items, seed=42):
- Semantic entropy — length-normalized (headline) + discrete (supplementary).
- SelfCheckGPT — NLI (headline) + BERTScore + n-gram (supplementary).
~~**Probe-style baselines from the same sampling pass (SEP, Kossen et al. 2024):** Cut 2026-05-20 — see §2 and §5 won't-add entry.~~

**Reported scorers for the contrastive method:** Cosine, Mahalanobis, KNN — KNN as the headline number based on seed-0 evidence; the other two as ablation rows.

**Ablations** (Llama only, on 2 representative datasets — pick HotpotQA + PopQA):
- SimCLR-only loss
- Logprob-recon-only loss
- Layer-pair sensitivity sweep (a coarse grid, not exhaustive — early/mid/late representatives)

**Transfer:** 7×7 source→target matrix per model; report off-diagonal mean and worst case.

---

## 7. GPU budget & dispatch order

GPU time is the binding constraint, not student time. Order matters.

| Order | Job | Why first | Approx. GPU-hours |
|---|---|---|---|
| 1 | Qwen2.5-7B inference + activation logging × 7 datasets × {train, test} splits | Longest-pole — everything downstream waits for these activations | ~1 week wall time at single-node throughput |
| 2 | Llama remaining seeds (×4) × 7 datasets × 3 trained methods | Cheap once activations are cached; CPU-feasible-ish but GPU faster | ~1–2 days |
| 3 | Qwen seeds × 7 × 3 methods | Starts as soon as job 1 produces each dataset's activations | ~2–3 days |
| 4 | P(true) inference on both models × 6 datasets — see [#50](https://github.com/hyang0129/HalluLens/issues/50) | Independent of training jobs; can interleave. Output-space only, so it can run on any free GPU regardless of activation-cache state. | ~1 day |
| 5 | Sampling-based baselines bundle (SE + SelfCheckGPT + SEP-SE) on 5 free-form datasets × both models — see [#49](https://github.com/hyang0129/HalluLens/issues/49). SEP-binary then trained for free on 6 datasets from cached activations. | One K=10 sampling pass yields inputs for 4 baselines (SearchQA capped 5k train / 10k test). Output-space only, so it can run on any free GPU regardless of activation-cache state. | ~25–30 GPU-hr full / ~12–15 GPU-hr Llama-only |
| 6 | Ablations (loss decomposition, layer-pair) | All on cached activations | <1 day |

Use `scripts/gpu_dispatch.py` for batch dispatch, `resume=True` everywhere. **Never submit or kill SLURM jobs without explicit approval** (per CLAUDE.md).

If GPU access slips by >3 days at any stage, cut #49's sampling-pass scope in this order: (1) drop SearchQA from #49 entirely (largest single dataset), (2) drop Qwen3-8B for SE + SelfCheckGPT + SEP-SE (Llama-only), (3) drop SciQ from the sampling-based methods, (4) drop SelfCheckGPT-BERTScore + n-gram (keep SelfCheckGPT-NLI). Then cut Qwen ablations. **Never cut SEP-binary** — it has no sampling cost and runs from cached activations. **Never cut the Qwen full seed sweep** — losing the second-model story is worse than losing sampling-based baselines.

---

## 8. Writing plan

Total student writing time: ~16 hours, parallelizable with GPU jobs.

| Week | Writing milestone |
|---|---|
| Week 1 (now) | Outline + abstract draft based on seed-0 + theoretical justification. Reserve table/figure slots. **Do not finalize the abstract** — it must match week-3 numbers. |
| Week 2 | Methods + theory sections (largely done — port from `THEORETICAL_JUSTIFICATION.md`). Related work pass with [`docs/planning/SOTA_TRACKER.md`](docs/planning/SOTA_TRACKER.md). |
| Week 3 | Results + ablations + transfer. Numbers freeze. Rewrite abstract + intro to match. |
| Week 4 | Limitations, discussion, polish, format check, supplementary appendix, anonymization audit, internal review pass. |

Tables/figures owners (each is a single deliverable):
- **Main table:** AUROC + AUPRC ± 95% CI per (model, dataset, method).
- **Headline figure:** Per-dataset AUROC bars, both models, with baseline cluster vs. ours.
- **Transfer matrix:** Heatmap of off-diagonal AUROC.
- **Ablation table:** Loss decomposition + layer-pair sweep.
- **Calibration figure:** Reliability diagrams for one dataset per model.
- **Compute-matched comparison:** AUROC per forward-pass count across covered datasets. K=1 cluster: ours, linear probe, SAPLMA, SEP-binary, SEP-SE, P(true). K=10 cluster: semantic entropy (length-normalized), SelfCheckGPT-NLI. One panel per dataset (5 free-form datasets; MMLU gets the K=1 cluster only).

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen results materially weaker than Llama | medium | Reframe as "model-family-dependent" honestly; Movies-style root-cause; this is a finding, not a failure — but it changes the abstract |
| SE or SelfCheckGPT-NLI beats us at 10× compute on its 5 datasets | medium | Pivot framing to compute-matched: ours wins per-FLOP. Theory still holds. |
| **SEP-binary** (genuinely compute-matched linear probe) beats us on some datasets | medium | This would be a real threat, not a per-FLOP escape hatch — SEP-binary uses the same forward-pass count and the same training data as our linear probe baseline. Mitigation depends on outcome: if SEP-binary < contrastive across the board, this is a strong positive. If SEP-binary ≈ contrastive, the contribution must rest on transfer (§4 item 5) + ablations + the SAPLMA gap. Decide on data, not in advance. |
| GPU node reclamation interrupts long Qwen inference | medium | `resume=True` everywhere; Zarr checkpointing; dispatch with `gpu_dispatch.py` not raw SSH |
| Movies anomaly is genuine ("contrastive prior hurts on small entity-domain datasets") | medium | Own it in §6 limitations + a one-paragraph diagnosis. This actually strengthens the paper if framed as bounded scope. |
| Multi-seed CIs reveal that contrastive vs. multi-layer probe gap is not significant on some datasets | medium-high | Headline becomes "consistent improvement at matched parameter count + transfer + Movies-style insight," not "always wins." Pre-commit to AUROC + AUPRC + win-rate-across-seeds reporting. |
| SAPLMA (~11M-param MLP) matches the contrastive method, suggesting the win is "more capacity" rather than "contrastive structure" | medium | The whole point of running it is to find out *before* a reviewer does. Mitigation depends on outcome: if SAPLMA < contrastive, this becomes a strong positive ("contrastive structure beats raw capacity at matched access"). If SAPLMA ≈ contrastive, the framing must pivot to compute/parameter efficiency, transferability, or layer-pair interpretability — pick whichever still holds on the data. Decide on data, not in advance. |
| ICR Probe collapses to chance AUROC (gate 3 fails) | low-medium | Trigger top_p remediation in `notes/icr_probe_sanity.md` §Check 5: recompute ICR scores using upstream-equivalent effective k = int(0.1 × (P+R)) rather than int(0.1 × R). If that restores signal, ship remediated scores; if not, downgrade ICR probe to "implementable but underperforms on our regime" and report it honestly. Do NOT chase paper's Gemma-2 0.84 — the target is model-specific. |
| Reviewer demands the LLM-judge labeling ablation for ICR Probe | medium | Pre-empt with the methodology footnote (§4 item 10) citing irreproducibility evidence (GitHub Issues #3 + #5, paper's silent labeling, ACL Anthology with no supplementary materials). Argue the burden of reproducibility lies with the original authors. If reviewer is unmoved at rebuttal, reopen [#76](https://github.com/hyang0129/HalluLens/issues/76) and run the HaluEval QA judge surrogate (~3-4 days). Do not pre-emptively run the ablation; the irreproducibility footnote is the stronger defense. |
| Class imbalance makes AUROC look strong but AUPRC weak on some datasets | medium | Already mitigated by reporting both. Disclose imbalance ratios in §3 dataset table. |
| Writing slips past deadline | low (8h budget × 2 contingency) | Hard cut: drop ablation §, push to appendix only |

---

## 10. Daily decision rules

- **Don't add experiments not on the must-add list** without first removing one. Scope creep is the failure mode here, not under-ambition.
- **Don't rerun completed seed-0 cells** to chase tiny improvements. Lock the seed-0 numbers; new methods get evaluated against the locked numbers.
- **Don't start writing before §6 ablations are at least sketched** — rewriting around new ablation results is more expensive than writing them up after.
- **Do flag any deviation** from this roadmap with a one-line entry in the file's "Last updated" section + a note in commits.
- **Do prefer producing the supplementary appendix early.** It's where reviewer concerns get parked cheaply.
