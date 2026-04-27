# Dataset Migration Status

Last updated: 2026-04-15

## Overview

Each dataset needs three things to run experiments:
1. **Activations** — zarr store with model activations (`shared/{name}/activations.zarr`)
2. **Generation file** — `generation.jsonl` with prompts, responses, logprobs (`output/{name}/Llama-3.1-8B-Instruct/generation.jsonl`)
3. **Eval labels** — `eval_results_for_training.json` with hallucination labels (`output/{name}/Llama-3.1-8B-Instruct/eval_results_for_training.json`)

Learned methods (contrastive, probes) need activations + labels.
Baselines (token_entropy, logprob_baseline) also need the generation file.

## Dataset Sizes

| Dataset | Train samples | Train hallu % | Test samples | Test hallu % | Config format |
|---------|--------------|---------------|--------------|--------------|---------------|
| HotpotQA | 62,135 (of 77,669) | 61.9% | 7,405 | 66.9% | flat |
| NQ | 16,617 (of 20,772) | 70.2% | (internal val: 4,155) | — | flat |
| MMLU | ~94,200 (no cache) | — | 9,940 | 88.1% | unified |
| Movies | 6,006 (of 7,508) | 74.8% | none | — | flat |
| PopQA | 7,806 (of 9,758) | 63.8% | 2,532 | 65.6% | unified |
| SciQ | 9,056 (of 11,320) | 36.8% | 777 | 36.2% | unified |
| SearchQA | ~150,410 (no cache) | — | 2,777 | 48.5% | unified |

Train samples column shows `used (of total)` where the total is split into train/val internally.

## Per-Dataset Status

### PopQA — READY
- Config: `configs/datasets/popqa.json` (unified format)
- Train activations: `shared/popqa_train/activations.zarr` (9,758 samples)
- Test activations: `shared/popqa/activations.zarr` (2,532 samples)
- Generation files: `output/popqa_train/` and `output/popqa/` — all present
- Eval labels: present
- **No action needed.**

### SciQ — READY
- Config: `configs/datasets/sciq.json` (unified format)
- Train activations: `shared/sciq_train/activations.zarr` (11,320 samples)
- Test activations: `shared/sciq/activations.zarr` (777 samples)
- Generation files: `output/sciq_train/` and `output/sciq/` — all present
- Eval labels: present
- **No action needed.**

### NQ (Natural Questions) — READY
- Config: `configs/datasets/nq_test_hallu_cor.json` (flat format)
- Activations: `shared/natural_questions_logprob/activations.zarr` (20,772 samples, internal train/val split)
- Generation file: `shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct/generation.jsonl`
- Eval labels: present at same path
- **No action needed.** (uses flat format with single zarr, no separate test set)

### HotpotQA — FIXED
- Config: `configs/datasets/hotpotqa.json` (unified format, new)
- Train activations: `shared/hotpotqa_train/activations.zarr` (77,669 samples)
- Test activations: `shared/hotpotqa/activations.zarr` (7,405 samples)
- Generation files: copied from zarr `meta/index.jsonl`
- Eval labels: reconstructed from zarr memmap `labels.npy`

**Fixes applied:**
1. [x] Copy `meta/index.jsonl` → `output/.../generation.jsonl`
2. [x] Reconstruct `eval_results_for_training.json` from zarr memmap labels
3. [x] Create unified config `hotpotqa.json` with train/test sub-keys
4. [x] Update experiment config to use `hotpotqa` dataset
5. [ ] Verify experiment runner works end-to-end

### MMLU — FIXED
- Config: `configs/datasets/mmlu.json` (unified format)
- Train activations: `shared/mmlu_train/activations.zarr` (94,200 entries, no memmap cache yet)
- Test activations: `shared/mmlu/activations.zarr` (10,225 entries)
- Train eval: 94,200 samples, 81,897 hallucinated (86.9%)
- Test eval: 10,225 samples, 9,010 hallucinated (88.1%)

**Root cause:** Inference ran and logged activations to zarr, but the zarr index doesn't include the `answer` field needed for eval. Generation files in `output/` were never written.

**Fixes applied:**
1. [x] Reconstructed `generation.jsonl` for train and test by joining zarr `index.jsonl` (prompt+response) with HuggingFace MMLU dataset (answers)
2. [x] Ran eval (substring matching) to produce `eval_results.json` and `raw_eval_res.jsonl`
3. [x] Built `eval_results_for_training.json` for both splits
4. [x] Cleared stale test memmap cache (will rebuild on first run)
5. [ ] Train memmap cache needs to be built on first run (94k samples)

### Movies — NEEDS INFERENCE
- Config: `configs/datasets/movies.json` (flat format)
- Activations: `shared/movies/activations.zarr` (7,508 samples, cache exists)
- Generation files: **MISSING** — `output/movies/` doesn't exist
- Eval labels: **MISSING**
- No separate test set exists

**Migration steps:**
1. [ ] Run inference to generate `output/movies/` generation file
2. [ ] Build eval labels
3. [ ] Consider creating a separate test split or use internal val split only

### SearchQA — FIXED
- Config: `configs/datasets/searchqa.json` (unified format)
- Train activations: `shared/searchqa_train/activations.zarr` (150,410 entries, no memmap cache yet)
- Train generation: 151,140 lines, eval: 151,140 (matched, 730 extra vs zarr — hash-joined)
- Test activations: `shared/searchqa/activations.zarr` (42,783 entries in zarr index)
- Test generation: 43,227 lines
- **Root cause:** `eval_results_for_training.json` had only 3,000 entries from an earlier N=3000 test run, while `eval_results.json` had the full 43,227. Memmap cache was built with the stale 3k eval.

**Fixes applied:**
1. [x] Rebuilt `eval_results_for_training.json` from `eval_results.json` (43,227 entries)
2. [x] Cleared stale test memmap cache (was 2,777 samples, will rebuild with full data)
3. [ ] Train memmap cache needs to be built on first run (150k samples — may take a while)
