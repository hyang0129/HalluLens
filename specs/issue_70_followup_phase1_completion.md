# ICR Probe — Phase 1 completion (follow-up to #70)

Issue #70 landed the ICR probe code (score formula, probe MLP, trainer, dataset, configs, dispatch, tests) and verified the implementation on the one capture we have locally: Llama-3.1-8B HotpotQA train, post-recompute, 5-fold CV AUROC = **0.6747 ± 0.0056**.

This issue closes out Phase 1 once the remaining captures land.

## Remaining work

### 1. Captures (Empire AI / GPU)
- [ ] `shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct` — produced by the same capture pipeline used for train; needed for the actual train→test split used by `scripts/run_experiment.py`
- [ ] `shared/icr_capture/hotpotqa_train_Qwen3-8B_0-50000`
- [ ] `shared/icr_capture/hotpotqa_test_Qwen3-8B`

All three captures must be **post-recompute** (full-sequence top-p effective-k, fix from commit `029af6c`). If any capture predates that fix, run `scripts/recompute_icr_scores.py` against it before use.

### 2. Phase 1 sweep (5 seeds × 2 models)
Per spec §13, once the captures are in place:
```bash
for s in 0 1 2 3 4; do
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa.json \
    --methods icr_probe --seeds $s
done

for s in 0 1 2 3 4; do
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa_qwen3.json \
    --methods icr_probe --seeds $s
done
```

Note: this is **not equivalent** to the 5-fold CV we already ran. The CV was a single-cell sanity check; this is the per-seed sweep that produces the artifacts `scripts/results_table.py` aggregates (`runs/baseline_comparison_hotpotqa/.../icr_probe/seed_<s>/{eval_metrics.json, predictions.csv, linear_probe_last.pt}`).

### 3. Fill in the remaining sanity-note checks
`notes/icr_probe_sanity.md` already has Checks 1, 2 (Llama), and 5 filled in from the current capture. After Phase 1 completes, fill in:
- [ ] Check 2 — Qwen3 per-layer single-feature AUROCs
- [ ] Check 3 — replace the CV proxy with the actual seed-0 Phase 1 test AUROC + run wall time
- [ ] Check 4 — ICR probe mean AUROC vs other baselines on Llama HotpotQA + Qwen3 HotpotQA (run `scripts/results_table.py`, pull the HotpotQA rows)
- [ ] Final results-summary table at the bottom of the notes
- [ ] Phase 2 decision checkbox

### 4. Phase 2 decision
After Phase 1 results land, decide per spec §14 (redefined):
- **Roll out to Phase 2** (popqa, mmlu, natural_questions, sciq, searchqa) if ICR probe shows clear signal and is reasonably positioned vs other baselines on both models
- **Document + proceed** (keep at HotpotQA only) if the result is mid-pack or weaker — the ablation argument is sufficient for the paper
- **Stop and diagnose** only if the probe collapses to chance on one of the two models

User's stated stance (2026-05-17): an ablation showing correct implementation + competitive comparison vs other baselines is sufficient — not chasing the paper's 0.798 Llama-3 number.

## Out of scope
- Score formula or trainer changes — that's #70, already merged.
- Phase 2 rollout if the decision lands on "document + proceed".
- Cross-model / cross-dataset transfer (Issue #62).

## Dependencies
- #70 merged (provides all code + the Llama HotpotQA train capture).
- `scripts/results_table.py` now recognizes `icr_probe` (one-line fix in #70 PR).
