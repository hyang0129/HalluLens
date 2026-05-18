# Hyperparameter Sweep Spec: K-Head Lambda Sweep (Issue #75 Follow-up)

**Status:** Draft — depends on issue-75 smoketest passing (`scripts/smoketest_issue_75.py`).

**Goal:** Identify the optimal weight `λ_attn` for the attention-reconstruction head (Mechanism K) relative to the SupCon contrastive objective (weight 1, fixed) and the logprob-reconstruction head (Mechanism F, weight `λ_lp = 1.0`). Also produce a K-only diagnostic to test whether K's signal is independent of F's. Direction and offset are fixed at safe defaults (see §3) so the sweep isolates the lambda dimension cleanly.

This is the first cluster-scale ablation run on issue #75's new model class, scoped to fit a 7-slice × 7-hour cluster budget.

## 1. Scope

### In scope (this sweep)
- 7 method configurations, all using `LogprobAttnReconProgressiveCompressor` with `attn_direction='both', attn_offset_k=1, attn_target='stats'`. Vary `(λ_lp, λ_attn)` only.
- Each method runs across the **6 Llama-3.1-8B-Instruct datasets** with `icr_capture` available: HotpotQA, PopQA, NQ, SciQ, SearchQA, MMLU.
- **Single seed (seed=0)** per cell — prior baselines have shown per-cell std < 0.5 AUROC, so seed CI is deferred.
- One slice per method; each slice chains its 6 datasets sequentially inside the 7-hour wallclock.

### Out of scope (follow-up)
- Cross-model generalization (Qwen3 captures exist but are deferred — easy to extend once the Llama sweep declares a winner).
- Cross-direction comparison (`forward` vs `backward` vs `both`) — fixed at `'both'` here; the spec's primary ablation matrix table in `issue_75_combined_logprob_attn_recon.md` §"Ablation matrix" covers that question separately.
- Offset-k sensitivity — fixed at `k=1`. Larger `k` introduces out-of-range attention layers at the deep end of `relevant_layers=14-29` (e.g. fwd k=4 NaN's layers 28 and 29). Keeping `k=1` keeps every view's attention target valid for both directions.
- Multi-seed CI runs.
- `attn_target ∈ {'coarse', 'full'}` decoder variants (not implemented in the model yet).

## 2. Dependencies

- Issue #75 scaffolding merged on `feat/issue-75-logprob-attn-recon` (commit `ce602b4`) ✓
- Smoketest pass on real capture data (this is `scripts/smoketest_issue_75.{py,sh}` — job `aebc290f9afc` in flight at draft time).
- `shared/icr_capture/{dataset}_test_Llama-3.1-8B-Instruct/` and `..._train_Llama-3.1-8B-Instruct_0-50000/` exist for all 6 datasets. Verified for HotpotQA (7405 test, 50k train); others follow the same naming and are listed in `shared/icr_capture/` per CLAUDE.md's data audit.

## 3. Sweep design

### Fixed across all 7 runs
| Hyperparameter | Value | Notes |
|---|---|---|
| `model_class` | `logprob_attn_recon_progressive_compressor` | issue #75 |
| `attn_direction` | `both` | fwd and bwd heads both active |
| `attn_offset_k` | `1` | keeps all attention targets in range for `relevant_layers=14-29` |
| `attn_target` | `stats` | only variant implemented |
| `attn_num_stat_features` | `3` | entropy / focal_frac / self_mass |
| `attn_recon_hidden_dim` | `256` | spec default |
| `attn_var_threshold` | `1e-5` | suppress collapsed-variance batches |
| `recon_seq_len` | `63` | matches `pad_length` for logprob aux |
| `recon_hidden_dim` | `256` | F head hidden width |
| `logprob_var_threshold` | `1e-4` | F suppression |
| `final_dim` | `512` | encoder output |
| `input_dropout` | `0.3` | matches `contrastive_logprob_recon.json` |
| Training | `max_epochs=100, batch_size=512, lr=1e-5, temperature=0.25, sub_batch_size=64, min_total_steps=3000` | mirrors prior contrastive runs |
| `relevant_layers` | `14-29` | mirrors `contrastive_logprob_attn_recon.json` |
| `target_layers` (eval) | `[22, 26]` | mirrors prior runs |
| `num_views` | `2` | standard |
| `pad_length` | `63` | one less than `max_response_len=64` |
| `response_logprobs_top_k` | `20` | stored top-K |
| `seed` | `0` | single-seed run |

### Varied — the 7 cells

| Slice | `λ_lp` | `λ_attn` | Effective K total¹ | Role |
|---|---:|---:|---:|---|
| 1 | 1.0 | 0.0 | 0.0 | **F-only baseline** — K decoders dormant; comparison floor for "does K add anything" |
| 2 | 1.0 | 0.1 | 0.2 | K very low — sanity that low-weight K doesn't perturb F |
| 3 | 1.0 | 0.3 | 0.6 | K below F |
| 4 | 1.0 | 1.0 | 2.0 | **Spec default** — K per-head matches F |
| 5 | 1.0 | 3.0 | 6.0 | K above F |
| 6 | 1.0 | 10.0 | 20.0 | K dominant — does over-weight hurt SupCon? |
| 7 | 0.0 | 1.0 | 2.0 | **K-only diagnostic** — F decoder dormant; tests K's standalone signal vs slice 1 |

¹ "Effective K total" = `2 × λ_attn` because `attn_direction='both'` instantiates two decoders and the trainer sums per-direction MSE with the same `λ_attn` weight. If we want absolute parity between the F and K objectives, slice 4's effective K weight is 2× F's. This is documented but not corrected — sweeping `λ_attn` (the user-facing knob) is the cleaner reporting axis.

## 4. Dataset coverage

Each slice runs its method across these 6 datasets, in this order:

| Order | Dataset | Capture train dir | Capture test dir | Approx N (train / test) |
|---|---|---|---|---|
| 1 | hotpotqa | `shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000` | `shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct` | 50000 / 7405 |
| 2 | popqa | `shared/icr_capture/popqa_train_Llama-3.1-8B-Instruct` | `shared/icr_capture/popqa_test_Llama-3.1-8B-Instruct` | TBD / TBD |
| 3 | nq | `shared/icr_capture/natural_questions_train_Llama-3.1-8B-Instruct` | `shared/icr_capture/natural_questions_test_Llama-3.1-8B-Instruct` | TBD / TBD |
| 4 | sciq | `shared/icr_capture/sciq_train_Llama-3.1-8B-Instruct` | `shared/icr_capture/sciq_test_Llama-3.1-8B-Instruct` | TBD / TBD |
| 5 | searchqa | `shared/icr_capture/searchqa_train_Llama-3.1-8B-Instruct_0-50000` | `shared/icr_capture/searchqa_test_Llama-3.1-8B-Instruct` | TBD / TBD |
| 6 | mmlu | `shared/icr_capture/mmlu_train_Llama-3.1-8B-Instruct_0-50000` | `shared/icr_capture/mmlu_test_Llama-3.1-8B-Instruct` | TBD / TBD |

(Confirm N's with `python scripts/audit_datasets.py --model meta-llama/Llama-3.1-8B-Instruct` before kickoff.)

Total cells: **7 methods × 6 datasets × 1 seed = 42**.

## 5. Resource budget

- **7 idle Jupyter-bound slices** on Empire AI (5× H100 80GB, 2× H200 140GB).
- **7 hours wallclock per slice.**
- Total budget: **49 GPU-hours**.

Per-cell cost estimate: F-only contrastive training on a 50k cache + eval is ~30-60 min on an H100 at the prior config (`max_epochs=100, batch_size=512`). 6 cells × ~1 hr ≈ **6-7 hr per slice**, leaving minimal headroom. If cells run long, drop SearchQA + MMLU (the two largest) for any slices that overrun.

## 6. Config and code setup

### 6.1 Method configs (7 new files)

Create 7 files under `configs/methods/` by templating from the existing
`configs/methods/contrastive_logprob_attn_recon.json`:

| File | `recon_lambda` | `attn_recon_lambda` |
|---|---:|---:|
| `contrastive_logprob_attn_recon_l10_a00.json` | 1.0 | 0.0 |
| `contrastive_logprob_attn_recon_l10_a01.json` | 1.0 | 0.1 |
| `contrastive_logprob_attn_recon_l10_a03.json` | 1.0 | 0.3 |
| `contrastive_logprob_attn_recon_l10_a10.json` | 1.0 | 1.0 |
| `contrastive_logprob_attn_recon_l10_a30.json` | 1.0 | 3.0 |
| `contrastive_logprob_attn_recon_l10_a100.json` | 1.0 | 10.0 |
| `contrastive_logprob_attn_recon_l00_a10.json` | 0.0 | 1.0 |

Naming convention: `_lXX_aYY.json` where `XX = int(λ_lp × 10)` and `YY = int(λ_attn × 10)` (so `_l10_a01` = `λ_lp=1.0, λ_attn=0.1`). Keeps lexical sort stable across 0.0/0.1/0.3/1.0/3.0/10.0.

The remaining fields stay identical to `contrastive_logprob_attn_recon.json` (training, data, evaluation blocks all unchanged); only `model_params.recon_lambda` and `model_params.attn_recon_lambda` differ per file.

**Open question (decide before kickoff):** keep `attn_direction='both'` or split into per-direction lambdas for slices 4-7 to allow asymmetric weighting? Keeping `both` is simpler and matches the spec; per-direction lambdas would require a model-class change.

### 6.2 Dataset configs (no new files)

Reuse the existing `configs/datasets/{hotpotqa,popqa,nq,sciq,searchqa,mmlu}.json`. Each already records the `icr_capture.train_dir` and `icr_capture.test_dir` paths the new `MemmapContrastiveDataset` consumes.

**Code wiring required:** the dataset configs were written for the zarr-backed `ActivationParser` (`backend: "zarr"`). The new memmap dataset reads from `icr_capture.{train,test}_dir`. Two paths:

- **(a) Add a `memmap_contrastive_dataset` branch to `scripts/run_experiment.py`'s dataset-class dispatch** (currently the F-only branch hard-codes `ActivationParser`). This is the proper integration — small change (~20 LOC) but a real code commit on top of `ce602b4`.
- **(b) Bypass `run_experiment.py` with a thin per-slice driver script** that constructs `MemmapContrastiveDataset` + `LogprobAttnReconProgressiveCompressor` + `train_contrastive_logprob_attn_recon` directly and reads `icr_capture.*` paths from the dataset config JSON. Mirrors the `scripts/smoketest_issue_75.py` pattern.

**Recommendation: (b)** for this sweep — keeps `run_experiment.py` clean of incomplete branches, and a sweep-specific driver lets us batch 6 datasets in one Python process (avoiding model-load overhead per cell). Promote to (a) only when the sweep declares a winner that we want to land in the canonical baseline matrix.

### 6.3 Sweep driver: `scripts/sweep_issue_75_lambda.py`

Single-slice entry point. Arguments:

```
--method-config CONFIG    one of the 7 method jsons
--datasets DS1,DS2,...    comma list of dataset config names (default: hotpotqa,popqa,nq,sciq,searchqa,mmlu)
--seed N                  default 0
--output-dir OUT          default: runs/issue_75_lambda_sweep/{method_name}/seed_{seed}
```

Behavior:
1. Load method config.
2. For each dataset name:
   - Load dataset config JSON, resolve `icr_capture.train_dir` and `icr_capture.test_dir`.
   - Build `MemmapContrastiveDataset` for train and test splits.
   - Build `LogprobAttnReconProgressiveCompressor` with lambdas from method config.
   - Run `train_contrastive_logprob_attn_recon` with the training block.
   - Run OOD evaluation (mirror `run_experiment.py` lines 328-380): `MultiMetricHallucinationEvaluator` over `target_layers` with `cosine`, `mds`, `knn`.
   - Write per-cell results to `{output_dir}/{dataset_name}/{results.json, artifacts/*}`.
3. Append summary row to `{output_dir}/summary.csv` after each dataset (so partial results survive timeouts).

Reuses the smoketest's import pattern (`activation_research.memmap_contrastive_dataset`, `activation_research.model`, `activation_research.training`).

### 6.4 Per-slice wrapper: `scripts/sweep_issue_75_lambda_{cell_name}.sh`

One shell per slice (7 total). Each sets the thread-cap env vars, picks its method config, logs to `/tmp/sweep_issue_75_{cell_name}.log`, and invokes the driver. Mirrors `scripts/smoketest_issue_75.sh` structure.

Examples:
- `sweep_issue_75_lambda_l10_a00.sh` → `--method-config configs/methods/contrastive_logprob_attn_recon_l10_a00.json`
- `sweep_issue_75_lambda_l00_a10.sh` → `--method-config configs/methods/contrastive_logprob_attn_recon_l00_a10.json`

### 6.5 Dispatch

```bash
# One dispatch per slice — issued from the login node
python scripts/gpu_dispatch.py run --jupyter --node alphagpu17-8883 -- bash scripts/sweep_issue_75_lambda_l10_a00.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu17-8881 -- bash scripts/sweep_issue_75_lambda_l10_a01.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu17-8888 -- bash scripts/sweep_issue_75_lambda_l10_a03.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu01-8885 -- bash scripts/sweep_issue_75_lambda_l10_a10.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu07-8887 -- bash scripts/sweep_issue_75_lambda_l10_a30.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu19-8889 -- bash scripts/sweep_issue_75_lambda_l10_a100.sh
python scripts/gpu_dispatch.py run --jupyter --node alphagpu23-8881 -- bash scripts/sweep_issue_75_lambda_l00_a10.sh
```

(alphagpu23-8888 is omitted — currently has 22.5 GB loaded from another session.)

Dispatch requires explicit user approval per CLAUDE.md (`gpu_dispatch.py run` is on the no-go list otherwise).

## 7. Outputs and analysis

### Per-cell artifacts (`runs/issue_75_lambda_sweep/{method}/seed_0/{dataset}/`)
- `results.json` — AUROC + AUPR per `target_layer` and per metric (cosine / mds / knn).
- `artifacts/final_weights.pt` — final encoder weights only (decoders discarded).
- `artifacts/{epoch_NNN.pt}` — periodic checkpoints (snapshot_every=10).
- `train.log` — loguru output from the trainer.
- `run_manifest.json` — git commit, hostname, command, torch/cuda versions.

### Sweep-level summary (`runs/issue_75_lambda_sweep/summary.csv`)

One row per cell:

```csv
method,dataset,seed,target_layer,metric,auroc,aupr,train_secs,n_train,n_test
contrastive_logprob_attn_recon_l10_a00,hotpotqa,0,22,knn,0.812,0.794,1843,50000,7405
...
```

### Decision rules (apply after all 42 cells complete)

1. **Headline:** if `max(slice 2-6) AUROC > slice 1 AUROC + 1.0pt` on ≥4 of 6 datasets at any `target_layer`, K helps. Promote the winning `λ_attn` to the canonical config.
2. **Independence:** if slice 7 (K-only) AUROC ≥ slice 1 (F-only) AUROC on ≥4 of 6 datasets, K carries independent signal — combined F+K may have meaningful additivity.
3. **Over-weighting harm:** if slice 6 (`λ_attn=10`) AUROC drops > 2pt below slice 4 (`λ_attn=1`) on ≥3 datasets, the K head is interfering with SupCon at high weight — bound the recommended `λ_attn` at ≤ 3.0.
4. **Null finding:** if no slice's mean AUROC beats slice 1 by > 0.5pt across datasets, K-as-MSE-on-summary-stats doesn't add signal beyond F. Document and consider promoting either `attn_target='coarse'` (richer target) or the direction sweep as the next probe — don't waste budget repeating this lambda dimension.

## 8. Risks

1. **Per-cell runtime over budget.** If F+K training is meaningfully slower than F-only (extra decoder forward + per-direction MSE), cells could push 90 min each → 9 hr per slice → bust the 7-hr budget. **Mitigation:** smoketest measures this. If single-cell on H100 > 75 min, drop one dataset (SearchQA — largest) from each slice.
2. **`λ_attn=10` numerically unstable.** Large attention-loss gradients can saturate the encoder's contrastive geometry. **Mitigation:** `grad_clip_norm=1.0` is already set; monitor `total_loss` in train.log for NaN/Inf.
3. **K-only (slice 7) collapses without F's regularization.** Without the logprob recon head shaping the encoder, the pure SupCon + attention-MSE objective may not converge meaningfully. **Mitigation:** treated as a diagnostic — null result is acceptable evidence.
4. **Variance suppression dominates K loss.** If `attn_var_threshold=1e-5` is too aggressive for `stats` summaries (which are bounded in [0, log r_max] for entropy, [0,1] for focal/self), most batches may be suppressed → K head never trains. **Mitigation:** smoketest reports `attn_var` per batch in the trainer's progress line; if mostly suppressed, lower threshold to `1e-7` before kickoff.
5. **Two-direction effective weighting confounds interpretation.** Per §3 footnote, slice 4's effective K total is 2× slice 4's nominal `λ_attn=1.0`. If the winner is, say, `λ_attn=0.3`, the "real" balance is 0.6 vs F's 1.0 — note this in the report.
6. **Sequential 6-dataset chain loses partial work on crash.** Each cell writes `summary.csv` rows incrementally, but a model crash mid-train loses that dataset's run. **Mitigation:** the driver wraps each dataset's run in a try/except so one failure doesn't abort the chain.

## 9. Open questions (decide before kickoff)

1. **Per-direction lambdas?** Currently shared. If yes, sweep changes to 2D (`λ_fwd`, `λ_bwd`) — not feasible in 7 slices. Recommend keeping shared.
2. **`recon_seq_len` vs `pad_length`.** `pad_length=63` and `recon_seq_len=63` (one less than `max_response_len=64`) — matches the smoketest. Confirm this is the intended setup, not `pad_length=64`.
3. **Slice 7's λ_lp=0 vs simply using `ProgressiveCompressor` (no aux at all).** They're not equivalent — slice 7 keeps the K decoder's gradient flow through the encoder. Worth keeping as-is so the F-vs-K comparison is apples-to-apples.
4. **Qwen3 follow-up timing.** If Llama sweep declares a winner, an equivalent Qwen3 sweep is 49 more GPU-hours — wait for free slices or pre-empt?

## 10. Related

- `specs/issue_75_combined_logprob_attn_recon.md` — model + trainer scaffolding spec; this sweep is the first cluster run on that scaffolding.
- Issue #75 — GitHub issue with the original ablation matrix proposal (5 seeds, multi-direction); this sweep deliberately narrows that scope for the 7-slice budget.
- `scripts/smoketest_issue_75.py` — single-cell smoketest the sweep depends on passing first.
