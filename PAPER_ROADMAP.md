# Paper Roadmap — EMNLP Submission

**Last updated:** 2026-05-12 (SAPLMA baseline added to grid; semantic-entropy item rescoped to a sampling-based baselines bundle — SE + SelfCheckGPT + SEP-SE + SEP-binary across 5–6 datasets, see #49)
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

| Component | Status | Notes |
|---|---|---|
| Llama-3.1-8B inference + activation logging, 6 datasets | ✅ done | HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA (Movies dropped — no train split) |
| Llama-3.1-8B seed-0 training × 6 datasets × {linear, multi-layer, contrastive×3 scorers} | ✅ done | See [`results/seed0_results.md`](results/seed0_results.md) |
| Qwen3-8B inference + activations, 6 datasets | ✅ done | Substitutes the originally-planned Qwen2.5-7B (better availability, same scale) |
| Qwen3-8B seed-0 training × 6 datasets | ✅ done | See [`results/seed0_qwen3_results.md`](results/seed0_qwen3_results.md) |
| SearchQA Multi-Layer probe | ↪️ obsolete | Multi-layer probe dropped from main grid (see below) |
| Multi-layer probe dropped from main baseline grid | ✅ done | Commit `fabf4de`. Concatenating all 16 lower-half layers overfit — consistently underperformed the single-layer linear probe at layer 22. Keep as an **ablation row**, not a main baseline. Paper must explain the drop and report numbers from seed-0. |
| Movies anomaly (contrastive < linear probe) | ↪️ obsolete | Movies excluded from experiment matrix (no train split) |
| Llama seeds {1, 2, 3, 4} × 6 datasets × learned methods | 🟡 in flight | Runner shells: `scripts/run_baseline_llama3_seed{1..4}.sh` |
| Qwen3-8B seed sweep, seeds {1, 2, 3, 4} × 6 datasets | 🟡 in flight | Runner shells: `scripts/run_baseline_qwen3_seed{1..4}.sh` |
| SAPLMA baseline (`SimpleHaluClassifier`) on full grid | ❌ not started | Code already in `activation_research/model.py:276` as `SimpleHaluClassifier` (~11M-param MLP on last-token activations) — wired into trainer but **absent from `configs/experiments/baseline_comparison_*.json`**. Forecloses the "linear probe is a strawman" reviewer concern. Activation-space; reuses cached zarr stores. Tracked in [#51](https://github.com/hyang0129/HalluLens/issues/51). |
| P(true) baseline | ❌ not started | Cheap; one extra prompt per example. Tracked in [#50](https://github.com/hyang0129/HalluLens/issues/50). Output-space only — does not need activation logging; can run concurrently with the seed sweeps. |
| Sampling-based baselines bundle (SE + SelfCheckGPT + SEP-SE + SEP-binary) | ❌ not started | One K=10 sampling pass produces 4 baselines. 5 free-form datasets {HotpotQA, NQ, PopQA, SciQ, SearchQA} for SE/SelfCheckGPT/SEP-SE; 6 datasets for SEP-binary (MMLU added — SEP-binary is activation-space and survives the NLI-clustering degeneracy on letter tokens that excludes MMLU from SE/SelfCheckGPT). SearchQA capped 10k test / 5k train. SEP-SE trained on 5k stratified train subset (Kossen-faithful regression on SE labels); SEP-binary trained on full train split with binary hallu labels — free byproduct, apples-to-apples with linear probe. Tracked in [#49](https://github.com/hyang0129/HalluLens/issues/49). Sampling pass is output-space only — can run concurrently with the seed sweeps on a separate GPU. |
| Cross-dataset transfer table | ❌ not started | Train→test pairs across datasets, no new training |
| AUPRC, ECE, bootstrap 95% CIs | ❌ not started | Pure analysis; zero compute |
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
   - **SEP-SE** (Kossen et al. 2024) — linear probe on activations trained to predict length-normalized SE; forecloses "didn't compare to the natural probe-based descendant of SE."
   - **SEP-binary** — free byproduct of SEP infrastructure: linear probe on the same activations but trained on the binary hallucination label over the full train split. Forecloses "you undertrained SEP relative to your linear probe" and gives a genuine compute-matched probe baseline.

   **Scope:** 5 free-form datasets (HotpotQA, NQ, PopQA, SciQ, SearchQA) for SE / SelfCheckGPT / SEP-SE; 6 datasets for SEP-binary (MMLU added — SEP-binary is activation-space and survives the NLI-clustering degeneracy on letter tokens that forces MMLU out of SE/SelfCheckGPT). SearchQA capped at 10k test / 5k train items to keep GPU bounded. SEP-SE trained on 5k stratified train subset per (dataset, model). Frame as **compute-matched comparison** (1 forward pass + linear probe vs. 10 samples + entailment clustering). Both models if budget allows; Llama-only is the firm fallback. **Sampling pass is output-space only — does not need activation logging**, so it can be dispatched independently of the seed sweeps and won't compete for zarr stores; SEP variants then read existing cached activations. Tracked in [#49](https://github.com/hyang0129/HalluLens/issues/49).
5. **Cross-dataset transfer table.** Forecloses "trained classifier may overfit per-dataset." For each (source, target) dataset pair, evaluate the source-trained checkpoint on the target test split. No new training. One supplementary table, one summary number in the body.
6. **AUPRC + ECE + bootstrap 95% CIs across seeds.** Forecloses "AUROC alone hides class imbalance / calibration issues / seed noise." Pure analysis; merge into the main table where space allows, push extras to appendix.
7. **Ablation: contrastive loss decomposition — writeup only.** Already tried: unsupervised SimCLR contrastive pretraining followed by a linear probe on the learned features → ~random (AUROC ≈ 0.5). Interpretation: the logprob-recon auxiliary loss carries the hallucination signal; SimCLR alone learns representations that are invariant to the right things for self-supervision but orthogonal to the hallucination axis. The logprob-recon-only variant is still worth running (1 retrain × seeds × 2 datasets, cached activations) to complete the decomposition. Either way, this ablation closes the "which part of the loss matters" question, and the negative SimCLR-only result is itself a useful contribution.
8. **Ablation: layer-pair sensitivity.** Sweep layer pairs on 1–2 datasets using cached layers 14–29 activations. Reuses cached activations. Connects directly to Justification 2.
9. **Multi-layer probe ablation + writeup.** The multi-layer probe (concatenation of layers 14–29) was dropped from the main baseline grid because it consistently underperformed the single-layer linear probe at layer 22 — likely overfitting from the ~16× parameter inflation with no inductive bias tying the layers together. We owe the reader: (a) one ablation row showing seed-0 multi-layer numbers next to single-layer, (b) a sentence in §Methods explaining why naive concatenation is the wrong baseline and why our learned contrastive compression is the right comparison, (c) connect this to the layer-pair-sensitivity ablation (item 8) — the contrastive method picks specific layer pairs rather than dumping all of them in. This is now part of the **motivation** for the method, not a hole.

## 5. Won't-add list (explicit cuts — read this before scoping creep)

- ❌ More models beyond {Llama-3.1-8B, Qwen2.5-7B}. No 70B, no Mistral, no Gemma.
- ❌ Long-form benchmarks (LongWiki, FactScore-style). Save for the journal extension.
- ❌ Asymmetric / BYOL-style encoders, K>2 contrastive views, FiLM layer conditioning, spectral analysis. All in the legacy idea pool; none are required for the claim.
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
**Trained methods (per cell × 5 seeds):** {Linear probe, SAPLMA (`SimpleHaluClassifier`), Contrastive+Logprob-recon}
**Ablation-only (seed-0, not in main grid):** Multi-layer probe (layers 14–29 concatenated) — reported once for motivation; see §4 item 9.
**Non-trained methods (per cell × 1):** {Logprob, Token entropy, P(true)}
**Sampling baselines** (5 free-form datasets only — MMLU excluded due to NLI-clustering degeneracy on letter tokens; SearchQA test capped at 10k items, seed=42):
- Semantic entropy — length-normalized (headline) + discrete (supplementary).
- SelfCheckGPT — NLI (headline) + BERTScore + n-gram (supplementary).
**Probe-style baselines from the same sampling pass (SEP, Kossen et al. 2024):**
- SEP-SE: `sklearn.linear_model.Ridge` on last-token activations at the linear-probe layer, target = length-normalized SE. Trained on 5k stratified train subset per (dataset, model). 5 datasets.
- SEP-binary: `sklearn.linear_model.LogisticRegression` on the same features, target = binary hallucination label, trained on full train split (free — no new sampling). 6 datasets (MMLU included). Apples-to-apples with the linear probe baseline.

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
| Class imbalance makes AUROC look strong but AUPRC weak on some datasets | medium | Already mitigated by reporting both. Disclose imbalance ratios in §3 dataset table. |
| Writing slips past deadline | low (8h budget × 2 contingency) | Hard cut: drop ablation §, push to appendix only |

---

## 10. Daily decision rules

- **Don't add experiments not on the must-add list** without first removing one. Scope creep is the failure mode here, not under-ambition.
- **Don't rerun completed seed-0 cells** to chase tiny improvements. Lock the seed-0 numbers; new methods get evaluated against the locked numbers.
- **Don't start writing before §6 ablations are at least sketched** — rewriting around new ablation results is more expensive than writing them up after.
- **Do flag any deviation** from this roadmap with a one-line entry in the file's "Last updated" section + a note in commits.
- **Do prefer producing the supplementary appendix early.** It's where reviewer concerns get parked cheaply.
