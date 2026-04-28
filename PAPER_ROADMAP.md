# Paper Roadmap — EMNLP Submission

**Last updated:** 2026-04-27
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

A learned contrastive compression of intermediate-layer activations detects intrinsic LLM hallucinations more reliably than (a) logprob/entropy from the same generation, (b) last-layer and multi-layer linear probes on the same activations, and (c) sampling-based output-uncertainty methods at matched compute. The signal is consistent across two model families (Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct), seven QA/reasoning benchmarks, and five training seeds; it transfers across datasets without retraining; and it concentrates in mid-to-late residual-stream layer pairs as predicted by the cross-layer-coherence theory.

If any of those clauses fails to hold up empirically, the framing changes — that's fine, but **the framing must match what the data actually shows**. Don't write the abstract until §2 is closed.

---

## 2. What's done vs. what's missing

| Component | Status | Notes |
|---|---|---|
| Llama-3.1-8B inference + activation logging, 7 datasets | ✅ done | HotpotQA, NQ, MMLU, Movies, PopQA, SciQ, SearchQA |
| Llama-3.1-8B seed-0 training × 7 datasets × {linear, multi-layer, contrastive×3 scorers} | ✅ done | See [`results/seed0_results.md`](results/seed0_results.md) |
| SearchQA Multi-Layer probe | ❌ hole | Backfill in week 1 |
| Movies anomaly (contrastive < linear probe) | ❌ unexplained | Root-cause in week 1 |
| Llama seeds {5, 26, 42, 63} × 7 datasets × 3 learned methods | 🟡 partial | ~84 training runs remaining |
| Qwen2.5-7B inference + activation logging, 7 datasets | ❌ not started | Highest-priority GPU job |
| Qwen2.5-7B seed sweep (5 seeds × 7 datasets × 3 methods) | ❌ not started | After inference completes |
| P(true) baseline | ❌ not started | Cheap; one extra prompt per example |
| Semantic-entropy baseline (Farquhar et al. 2024) on 3-dataset subset | ❌ not started | 10× inference cost; cap at HotpotQA, NQ, PopQA |
| Cross-dataset transfer table | ❌ not started | Train→test pairs across datasets, no new training |
| AUPRC, ECE, bootstrap 95% CIs | ❌ not started | Pure analysis; zero compute |
| Ablation: SimCLR-only vs +logprob-recon | ❌ not started | 1 retrain × seeds; reuses cached activations |
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

1. **Second model: Qwen2.5-7B-Instruct.** Forecloses "results may be Llama-specific." This is the single highest-leverage missing piece. The Qwen 72B qgen plumbing is partially built (legacy §8.6); the 7B inference path needs `model_map` + `--quantization` wiring or the equivalent for non-quantized 7B. Prefer non-quantized 7B unless VRAM forces otherwise.
2. **P(true) baseline** (Kadavath et al. 2022). Forecloses "didn't compare to prompt-based self-evaluation." Implementation cost: one templated follow-up prompt per generated answer. Run on all 7 datasets, both models.
3. **Semantic entropy** (Farquhar et al. *Nature* 2024) on 3-dataset subset. Forecloses "didn't compare to the strongest sampling-based method." Cap at HotpotQA, NQ, PopQA to keep GPU bounded; frame as **compute-matched comparison** (1 forward pass vs. 10 samples + entailment clustering). Both models if budget allows; Llama-only is acceptable as long as the framing is honest.
4. **Cross-dataset transfer table.** Forecloses "trained classifier may overfit per-dataset." For each (source, target) dataset pair, evaluate the source-trained checkpoint on the target test split. No new training. One supplementary table, one summary number in the body.
5. **AUPRC + ECE + bootstrap 95% CIs across seeds.** Forecloses "AUROC alone hides class imbalance / calibration issues / seed noise." Pure analysis; merge into the main table where space allows, push extras to appendix.
6. **Ablation: contrastive loss decomposition.** Train SimCLR-only and logprob-recon-only variants on 2 datasets × 5 seeds. Forecloses "which part of the loss matters." Reuses cached activations.
7. **Ablation: layer-pair sensitivity.** Sweep layer pairs on 1–2 datasets using cached layers 14–29 activations. Reuses cached activations. Connects directly to Justification 2.
8. **Movies root-cause.** Either fix or own it. Likely candidates: small N (1.5k), label-noise from entity-string matching, or a genuine regime where the contrastive prior hurts. Either result is publishable; an unexplained loss in the main table is not.
9. **Backfill SearchQA Multi-Layer probe.** No story choices here, just close the hole.

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
**Trained methods (per cell × 5 seeds):** {Linear probe, Multi-layer probe, Contrastive+Logprob-recon}
**Non-trained methods (per cell × 1):** {Logprob, Token entropy, P(true)}
**Sampling baselines:** Semantic entropy on {HotpotQA, NQ, PopQA} only.

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
| 4 | P(true) inference on both models × 7 datasets | Independent of training jobs; can interleave | ~1 day |
| 5 | Semantic entropy on {HotpotQA, NQ, PopQA} × both models (or Llama-only if budget tight) | 10× inference cost; deliberately the last expensive thing | ~2–3 days |
| 6 | Ablations (loss decomposition, layer-pair) | All on cached activations | <1 day |

Use `scripts/gpu_dispatch.py` for batch dispatch, `resume=True` everywhere. **Never submit or kill SLURM jobs without explicit approval** (per CLAUDE.md).

If GPU access slips by >3 days at any stage: cut semantic entropy first, then Qwen ablations, then drop semantic-entropy to Llama-only. Do not cut the Qwen full sweep — losing the second-model story is worse than losing semantic entropy.

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
- **Compute-matched comparison:** Bar of AUROC per forward-pass count (ours @ 1, semantic entropy @ 10) on the 3 covered datasets.

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Qwen results materially weaker than Llama | medium | Reframe as "model-family-dependent" honestly; Movies-style root-cause; this is a finding, not a failure — but it changes the abstract |
| Semantic entropy beats us on its 3 datasets at 10× compute | medium | Pivot framing to compute-matched: ours wins per-FLOP. Theory still holds. |
| GPU node reclamation interrupts long Qwen inference | medium | `resume=True` everywhere; Zarr checkpointing; dispatch with `gpu_dispatch.py` not raw SSH |
| Movies anomaly is genuine ("contrastive prior hurts on small entity-domain datasets") | medium | Own it in §6 limitations + a one-paragraph diagnosis. This actually strengthens the paper if framed as bounded scope. |
| Multi-seed CIs reveal that contrastive vs. multi-layer probe gap is not significant on some datasets | medium-high | Headline becomes "consistent improvement at matched parameter count + transfer + Movies-style insight," not "always wins." Pre-commit to AUROC + AUPRC + win-rate-across-seeds reporting. |
| Class imbalance makes AUROC look strong but AUPRC weak on some datasets | medium | Already mitigated by reporting both. Disclose imbalance ratios in §3 dataset table. |
| Writing slips past deadline | low (8h budget × 2 contingency) | Hard cut: drop ablation §, push to appendix only |

---

## 10. Daily decision rules

- **Don't add experiments not on the must-add list** without first removing one. Scope creep is the failure mode here, not under-ambition.
- **Don't rerun completed seed-0 cells** to chase tiny improvements. Lock the seed-0 numbers; new methods get evaluated against the locked numbers.
- **Don't start writing before §6 ablations are at least sketched** — rewriting around new ablation results is more expensive than writing them up after.
- **Do flag any deviation** from this roadmap with a one-line entry in the file's "Last updated" section + a note in commits.
- **Do prefer producing the supplementary appendix early.** It's where reviewer concerns get parked cheaply.
