# Probe wiring verification smoketest (slice of #70 for #72 final-gate)

**Status:** scope / proposal
**Branch:** `feat/issue-72-inference-capture-rewrite`
**Relation to #70:** this is the *minimal* probe slice needed to certify that the #72 capture path produces probe-trainable data end-to-end. It is **NOT** a reproduction of the paper's reported AUROC numbers — that's the full Phase 1 of #70, which requires hours of HotpotQA capture per model and is out of scope here.

## Goal

Prove the round-trip works:

```
capture (#72) → icr_scores.npy + eval_results.json
              → ICRDataset(mode="memmap")
              → ICRProbe (the paper's MLP)
              → BCE training loop
              → test-split AUROC
```

If AUROC > chance (>0.5 ± noise) on a reasonable capture size, the wiring is right and the next gate is the paper-numbers verification.

## In scope

1. **`activation_research/icr_probe.py`** — `ICRProbe(nn.Module)` matching the paper exactly:
   `Linear(L, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)`.
   ~10 LOC.
2. **`scripts/smoketest_probe_72.py`** — standalone CLI that:
   - Takes a `--capture-dir` (one out_dir produced by `capture_inference.py`).
   - Loads `ICRDataset(mode="memmap", capture_dir=...)`.
   - Builds train/val/test splits (stratified, seed=0; reuses the existing `_make_split_indices`).
   - Trains `ICRProbe` with BCE-with-logits + Adam (lr=1e-3, weight_decay=0, batch_size=256) for 50 epochs with early stopping on val AUROC (patience=10).
   - Prints test-split metrics: AUROC, AUPRC, accuracy, ECE.
   - Writes `<capture-dir>/probe_eval/eval_metrics.json` and `predictions.csv`.
   - No checkpoint saving (smoketest only).
3. **`tests/test_icr_probe.py`** — 5-test set:
   - `test_probe_forward_shape` — `(B, L) → (B,)`.
   - `test_probe_backward_pass` — loss has grad, weights update after a step.
   - `test_probe_param_count` — sanity-check it's the ~12K params the paper claims.
   - `test_probe_overfits_tiny_batch` — 16 random samples, 200 epochs → train loss < 0.01.
   - `test_smoketest_cli_dry_run` — invokes `smoketest_probe_72.py --help` and asserts exit 0.

## Out of scope (explicit non-goals)

- **Matching the paper's reported HotpotQA AUROC of ~0.84.** That gate is Phase 1 of #70. Requires HotpotQA capture × both models × 5 seeds.
- **Integration with `scripts/run_experiment.py`** dispatch or `configs/experiments/baseline_comparison_*.json`. Those are needed for the full paper comparison; for wiring verification a standalone script is enough.
- **`activation_research/icr_trainer.py` as a class.** Skip the class abstraction; the smoketest script keeps the training loop inline (~50 LOC). When Phase 1 of #70 lands, factor into a `LinearProbeTrainer`-style class then.
- **Cross-dataset transfer, per-module variants, ablations** — all #70 / #62 scope.
- **Multi-seed sweeps.** Smoketest is single-seed (seed=0).

## Inputs

The capture dirs produced by `scripts/capture_inference.py`:

```
shared/icr_capture/<task>_<logical_split>_<model_slug>/
  config.json           # has num_layers
  meta.jsonl
  icr_scores.npy        # (N, num_layers) fp32  ← THIS is the probe input
  eval_results.json     # {"halu_test_res": [bool, ...], "abstantion": [false, ...]}
```

The smoketest reads ONLY these four files (via `ICRDataset(mode="memmap")`). The rest of the capture layout (response_attention.npy, prompt_activations.npy, etc.) is not consumed — that's only for `mode="memmap-raw"` ablations.

## Data caveats for "real" results

A 50-sample smoketest is too small to give meaningful AUROC:
- Stratified 80/15/5 split → ~40/8/2 samples. Test-set AUROC ±0.2.
- The probe might "work" but you can't tell.

The minimum data size for a sensible signal:
- **SciQ test full (1000 samples)** — train/val/test ~800/150/50. AUROC ±0.05. Captures in ~3 h on H100, fits comfortably on disk (~150 GB).
- **HotpotQA test full (7400 samples)** — best smoketest target if you have the GPU time. Captures in ~20 h per model.

**Recommendation:** capture *full SciQ test* (Llama and Qwen3, no train splits, no smoketest tail) as the data input to this probe verification. Re-dispatch via the FS-claim-queue dispatcher (worker.sh) — one cell per (sciq_test, model). That's 2 cells, ~6 h total wall time on the H100/H200 pair.

## Files to create

| File | LOC | Purpose |
|---|---|---|
| `activation_research/icr_probe.py` | ~15 | Paper's 4-layer MLP |
| `scripts/smoketest_probe_72.py` | ~120 | Standalone train+eval CLI |
| `tests/test_icr_probe.py` | ~80 | 5 CPU-only tests |
| `notes/icr_probe_smoketest_results.md` | (added after run) | One-paragraph result writeup |

Total ~215 LOC new code; no existing files modified.

## Verification gate

The smoketest passes if:
1. All 5 tests in `tests/test_icr_probe.py` pass.
2. On a real full-SciQ-test capture (≥1000 samples), test-set AUROC > 0.55 (just better than chance — proves wiring + signal). Higher is bonus.
3. No NaN losses, no exploding gradients, training converges (loss decreases monotonically on train split).

Failing #2 means either (a) ICR scores from the capture are degenerate, (b) probe is mis-wired, or (c) the SciQ closed-book signal is too weak — diagnose before scaling to HotpotQA.

## Estimated wall time

- Probe implementation: ~30 min
- CLI training script: ~1 h
- Tests: ~30 min
- Full SciQ test capture (Llama + Qwen3): ~6 h GPU wall (dispatched, not active)
- Smoketest run + writeup: ~30 min (Sciq×2 + report)

**Total: half a day of active engineering, plus one overnight GPU dispatch.** Single sonnet can implement and run.

## Follow-up (not in this scope)

Full Phase 1 of #70 after this gate passes:
- HotpotQA capture × both models × full splits (~40 h GPU wall total, parallelizable across nodes via the dispatcher)
- 5 seeds × 2 models trained via `LinearProbeTrainer`-style class
- Compare to paper's ~0.84 AUROC; branch decision on Phase 2 rollout per #70 spec §14.
