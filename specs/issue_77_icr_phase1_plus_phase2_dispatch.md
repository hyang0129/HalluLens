# Spec: ICR Probe Phase 1 + Phase 2 dispatch — full 6-dataset × 2-model results

**Scope.** Run the ICR probe (from #70) across all six datasets × {Llama-3.1-8B-Instruct, Qwen3-8B} × 5 seeds = **60 runs** so that the EMNLP results table has ICR-probe numbers in every cell.

> Note on branch name: this work was started in a worktree branched as `feat/issue-75-followup-runs`. Scope pivoted to ICR probe mid-stream; rename the branch to `feat/issue-77-icr-phase1-plus-phase2-dispatch` before opening a PR.

**Status of dependencies (verified 2026-05-18, Empire AI remote).**

| Dataset × model | capture dir present | `icr_scores.npy` present | meta count |
|---|:---:|:---:|---|
| hotpotqa_{train_0-50000,test} × Llama       | ✓ | ✓ | 49996 / 7405 |
| hotpotqa_{train_0-50000,test} × Qwen3        | ✓ | ✓ | 49996 / 7405 |
| popqa_{train,test} × Llama                   | ✓ | ✓ | 10595 / 2797 |
| popqa_{train,test} × Qwen3                   | ✓ | ✓ | 10595 / 2797 |
| mmlu_{train_0-50000,test} × Llama            | ✓ | ✓ | 49666 / 10271 |
| mmlu_{train_0-50000,test} × Qwen3            | ✓ | ✓ | 49666 / 10271 |
| natural_questions_{train,test} × Llama       | ✓ | ✓ | 16617 / 4155 |
| natural_questions_{train,test} × Qwen3       | ✓ | ✓ | 16617 / 4155 |
| sciq_{train,test} × Llama                    | ✓ | ✓ | 11609 / 1000 |
| sciq_{train,test} × Qwen3                    | ✓ | ✓ | 11609 / 1000 |
| searchqa_{train_0-50000,test} × Llama        | ✓ | ✓ | 49959 / 21609 |
| searchqa_{train_0-50000,test} × Qwen3        | ✓ | ✓ | 49959 / 21609 |

All capture data is in place. ICR scores already computed everywhere (`scripts/recompute_icr_scores.py` does not need to run). This means **no GPU inference work remains** for #77 — only the lightweight probe-training sweep on cached features.

## 1. What's missing right now

### 1a. `icr_capture` block on 5 dataset configs
The Llama dataset configs (`hotpotqa.json`, `popqa.json`, `mmlu.json`, `nq.json`, `sciq.json`, `searchqa.json`) all have an `icr_capture: {train_dir, test_dir}` block that `run_icr_probe` reads at [scripts/run_experiment.py:1714](scripts/run_experiment.py#L1714). `hotpotqa_qwen3.json` was also updated as part of #70. The remaining five Qwen3 configs were never touched and have no `icr_capture` block.

### 1b. `icr_probe` is missing from 10 experiment configs
`baseline_comparison_hotpotqa.json` and `baseline_comparison_hotpotqa_qwen3.json` already list `"icr_probe"`. The other 10 (`popqa{,_qwen3}`, `mmlu{,_qwen3}`, `nq{,_qwen3}`, `sciq{,_qwen3}`, `searchqa{,_qwen3}`) do not. Without the entry, the existing baseline_comparison runs skip ICR even when the data is there.

That's it. No model code, no trainer code, no dataset class code — those all landed in #70.

## 2. File diff

### 2a. Add `icr_capture` block to 5 Qwen3 dataset configs

Append to each file. Capture paths verified against the remote listing.

`configs/datasets/popqa_qwen3.json`:
```json
"icr_capture": {
  "train_dir": "shared/icr_capture/popqa_train_Qwen3-8B",
  "test_dir":  "shared/icr_capture/popqa_test_Qwen3-8B"
}
```

`configs/datasets/mmlu_qwen3.json`:
```json
"icr_capture": {
  "train_dir": "shared/icr_capture/mmlu_train_Qwen3-8B_0-50000",
  "test_dir":  "shared/icr_capture/mmlu_test_Qwen3-8B"
}
```

`configs/datasets/nq_qwen3.json`:
```json
"icr_capture": {
  "train_dir": "shared/icr_capture/natural_questions_train_Qwen3-8B",
  "test_dir":  "shared/icr_capture/natural_questions_test_Qwen3-8B"
}
```

`configs/datasets/sciq_qwen3.json`:
```json
"icr_capture": {
  "train_dir": "shared/icr_capture/sciq_train_Qwen3-8B",
  "test_dir":  "shared/icr_capture/sciq_test_Qwen3-8B"
}
```

`configs/datasets/searchqa_qwen3.json`:
```json
"icr_capture": {
  "train_dir": "shared/icr_capture/searchqa_train_Qwen3-8B_0-50000",
  "test_dir":  "shared/icr_capture/searchqa_test_Qwen3-8B"
}
```

(Naming asymmetry alert: `nq` is the project-internal short name but the icr_capture directory uses `natural_questions_*`. Same pattern as the existing Llama `nq.json` — no surprise.)

### 2b. Add `"icr_probe"` to 10 experiment configs

The 10 `baseline_comparison_{popqa,mmlu,nq,sciq,searchqa}{,_qwen3}.json` files. One-line addition to the `"methods"` list in each. SmolLM3 variants are explicitly out of scope.

## 3. Dispatch plan

ICR probe is cheap: 50 epochs of a tiny MLP on ~10k–50k cached feature vectors. Per [`notes/icr_probe_sanity.md`](notes/icr_probe_sanity.md), the seed-0 single-cell run was a few minutes on H100. Conservatively: **60 runs × ~5 min/run = ~5 H100-hours total**, trivially parallelizable across nodes.

Two dispatch strategies, pick one:

**A. One sweep per dataset, both models:**
```bash
for ds in hotpotqa popqa mmlu nq sciq searchqa; do
  for mv in "" "_qwen3"; do
    python scripts/gpu_dispatch.py run --jupyter --node alphagpu23-8881 -- \
      python scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${ds}${mv}.json \
        --methods icr_probe --seeds 0 1 2 3 4
  done
done
```

**B. Drop the loop into a single `run_experiment.py --experiment ...` call** since the runner already iterates seeds internally. Then submit 12 jobs through `gpu_dispatch.py`. Easier per-cell status tracking.

Both produce the same `runs/baseline_comparison_<ds>/.../icr_probe/seed_<s>/{eval_metrics.json, predictions.csv, linear_probe_last.pt}` artifacts. `scripts/results_table.py` picks them up automatically — [`scripts/results_table.py` was updated in #70 to recognize `icr_probe` runs](https://github.com/hyang0129/HalluLens/pull/74).

**Recommend B** — fewer dispatch handles, easier `gpu_dispatch.py jobs` to read. Use `--node` flags to spread across Jupyter-only nodes (`alphagpu23-8881` is idle; see also `gpu_dispatch.py sync-jupyter` before dispatching to refresh the registry).

## 4. Verification before dispatch

Before kicking off 60 runs, confirm the 5 modified Qwen3 dataset configs resolve correctly with a 1-seed smoketest:

```bash
python scripts/gpu_dispatch.py run --jupyter --node alphagpu23-8881 -- \
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_popqa_qwen3.json \
    --methods icr_probe --seeds 0
```

If that produces an `eval_metrics.json` with a non-NaN AUROC and the right `n_train` / `n_test`, fan out to the rest. If it fails on `FileNotFoundError` on the capture dir, fix the path string in the dataset config — the remote uses `popqa_train_Qwen3-8B` (no `_0-50000` suffix because the original capture covered the entire 10595-row split in one go).

## 5. After dispatch — sanity notes + Phase 2 decision

Per #77 §3, when results are in:
- Fill `notes/icr_probe_sanity.md` Check 2 (Qwen3 per-layer single-feature AUROCs) — generate with the `scripts/icr_sanity_check.py` already in tree.
- Replace Check 3's CV proxy with the actual seed-0 Phase 1 test AUROC + wall time.
- Fill Check 4 — ICR probe mean AUROC vs other baselines on each dataset (read from `scripts/results_table.py`).
- Per-dataset results table at the bottom of the notes — 6 datasets × 2 models × {mean ± 95% CI}.

Then make the Phase 2 decision per #77 §4. Since we're running all 6 datasets in this dispatch (i.e. Phase 1 + Phase 2 collapsed), the decision is purely write-up: which datasets get featured in §4 of the paper vs relegated to the appendix table.

## 6. Out of scope (explicit)

- ICR score formula or probe trainer changes — #70 territory, merged.
- `attn_target` higher-bandwidth variants — #82.
- Combined logprob + attention recon (issue #75 follow-up dispatch) — separate ticket.
- Cross-model / cross-dataset transfer — #62.
- SmolLM3 — never in scope for #77.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Qwen3 capture path string typo in a `*_qwen3.json` config | Smoketest one cell per dataset before fan-out (§4). |
| `icr_scores.npy` was written by an older code path that pre-dates commit `029af6c` (full-sequence top-p effective-k) | The remote files post-date that commit (verified by git log against the icr_capture creation dates). Spot-check one cell's icr_scores against a fresh `recompute_icr_scores.py` run before relying on the sweep numbers. |
| Some seed produces an outlier AUROC (cf. #55 for linear_probe) | 5-seed mean + CI is the standard. If one cell has a clear outlier, investigate via the same protocol used for #55. |
| Phase 2 ICR results are weaker than other baselines on most cells | Per #77 §4 user stance: "ablation showing correct implementation + competitive comparison is sufficient — not chasing the paper's 0.798 Llama-3 number." Document and proceed. |

## 8. File diff summary

| File | Change | LOC |
|---|---|---|
| `configs/datasets/popqa_qwen3.json`      | +icr_capture block | 4 |
| `configs/datasets/mmlu_qwen3.json`       | +icr_capture block | 4 |
| `configs/datasets/nq_qwen3.json`         | +icr_capture block | 4 |
| `configs/datasets/sciq_qwen3.json`       | +icr_capture block | 4 |
| `configs/datasets/searchqa_qwen3.json`   | +icr_capture block | 4 |
| `configs/experiments/baseline_comparison_popqa.json`         | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_popqa_qwen3.json`   | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_mmlu.json`          | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_mmlu_qwen3.json`    | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_nq.json`            | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_nq_qwen3.json`      | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_sciq.json`          | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_sciq_qwen3.json`    | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_searchqa.json`      | +"icr_probe" | 1 |
| `configs/experiments/baseline_comparison_searchqa_qwen3.json`| +"icr_probe" | 1 |

15 files, ~30 LOC of pure JSON. No Python.

## 9. Plan-of-record

1. Apply the 15 config edits in this worktree.
2. Smoketest one Qwen3 cell via gpu_dispatch (popqa_qwen3, seed 0) to validate the icr_capture path string.
3. Open PR (small, all-JSON).
4. After merge, dispatch the full 12-job sweep (6 datasets × 2 models). Track via `gpu_dispatch.py jobs --all`.
5. When all 60 runs land, run `scripts/results_table.py`, fill `notes/icr_probe_sanity.md`, close #77.
