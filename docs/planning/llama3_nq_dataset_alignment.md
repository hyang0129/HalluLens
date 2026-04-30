# Llama3 Natural Questions Dataset — Layout Anomaly Report

**Date:** 2026-04-29
**Scope:** Investigate why the Llama-3.1-8B-Instruct Natural Questions dataset (~20k activations) does not match the canonical layout used by every other dataset (`hotpotqa`, `mmlu`, `popqa`, `sciq`, `searchqa`, and the Qwen3 variant of NQ itself), and propose a concrete migration plan.

---

## 1. Canonical layout (what every other dataset looks like)

Every well-formed dataset uses a strict three-place layout:

```
shared/<dataset>/activations.zarr/                # TEST split activations (zarr v3)
shared/<dataset>_train/activations.zarr/          # TRAIN split activations
output/<dataset>/<MODEL_NAME>/                    # TEST split generations + eval
    generation.jsonl
    generation.sanitized_for_eval.jsonl
    eval_results.json                  # summary (rates + counts)
    eval_results_for_training.json     # per-sample labels keyed by prompt_hash
    raw_eval_res.jsonl                 # per-sample raw evaluator output
output/<dataset>_train/<MODEL_NAME>/              # TRAIN split generations + eval (same files)
configs/datasets/<dataset>.json                   # ties test+train together
```

Reference example — Qwen3 NQ ([configs/datasets/nq_qwen3.json](configs/datasets/nq_qwen3.json)):

| Role | Path | Count |
|---|---|---|
| Test activations | `shared/natural_questions_qwen3_8b/activations.zarr` | 4,155 |
| Train activations | `shared/natural_questions_train_qwen3_8b/activations.zarr` | 16,617 |
| Test generations | `output/natural_questions/Qwen3-8B/generation.jsonl` | 4,155 |
| Train generations | `output/natural_questions_train/Qwen3-8B/generation.jsonl` | 16,617 |

All zarr stores share the same schema (`schema_version: zarr-v1`, 33 layers, 4096 hidden, fp16, identical chunking).

---

## 2. The Llama3 NQ artifact (what's actually on disk)

The Llama3 NQ data lives in a non-canonical directory and ignores the train/test split convention. Layout below.

### 2.1 Activations
```
shared/natural_questions_logprob/                           # NOTE: "_logprob" suffix
    activations.zarr/
        meta/index.jsonl   →  20,772 samples   (single combined dump)
        zarr.json          →  same schema as canonical (33 layers, 4096, fp16) ✓
```

### 2.2 Generations + eval (colocated INSIDE `shared/`)
```
shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct/
    generation.jsonl                       20,772 lines   (matches zarr)
    generation.sanitized_for_eval.jsonl    20,770 lines   (2 dropped)
    eval_results.json                      total_count: 20,770, halu_rate: 70.2%
    eval_results_for_training.json         per-sample labels
    # no raw_eval_res.jsonl                ← MISSING (config references it)
```

### 2.3 Config
```
configs/datasets/nq_test_hallu_cor.json    # only Llama3 NQ config; no _train sibling
configs/experiments/baseline_comparison_nq.json  → references "nq_test_hallu_cor"
```

There is no `output/natural_questions/Llama-3.1-8B-Instruct/`, no `output/natural_questions_train/Llama-3.1-8B-Instruct/`, and no `shared/natural_questions_train/activations.zarr/` for Llama3.

---

## 3. How it differs from the canonical layout

| Dimension | Canonical (e.g. hotpotqa, mmlu, nq_qwen3) | Llama3 NQ (current) |
|---|---|---|
| **Shared dir name** | `shared/natural_questions/` | `shared/natural_questions_logprob/` (legacy `_logprob` capture suffix) |
| **Generations live in** | `output/<dataset>/<model>/` | Nested *inside* `shared/.../natural_questions/Llama-3.1-8B-Instruct/` |
| **Train/test split** | Two separate zarrs (`<dataset>` + `<dataset>_train`) | One combined zarr of 20,772 samples — no split |
| **Config name** | `nq.json` / `nq_test.json` / `nq_train.json` | `nq_test_hallu_cor.json` (single, ad-hoc name) |
| **Config has `train`+`test` blocks** | Yes (see `nq_qwen3.json`) | No — flat single-section schema |
| **`raw_eval_res.jsonl`** | Present | **Missing** despite being referenced by config |
| **Sample-count consistency** | zarr ↔ generation ↔ sanitized ↔ eval all agree | zarr=20,772, generation=20,772, sanitized=20,770, eval=20,770 (off by 2) |
| **Output dir for Llama3 NQ** | Would be `output/natural_questions/Llama-3.1-8B-Instruct/` | **Does not exist** |

The schema of the zarr store itself is fine — same `zarr.json`, same arrays (`prompt_activations`, `response_activations`, `response_token_ids`, ...), same dtype/chunking. The anomaly is purely in **directory layout, naming, and the missing train/test partition**.

---

## 4. Why this happened (best read)

The `_logprob` suffix and the `nq_test_hallu_cor` config name are residue from an earlier "logprob capture / hallucination correlation" experiment (Mar 2026) that pre-dated the current train/test convention adopted in April. The directory structure was written by the older `run_with_server.py` flow, which dumped generations alongside activations under the same base dir and didn't yet enforce a train/test split. Every dataset added after April (Qwen3 NQ included) follows the canonical layout because the config schema was tightened.

Evidence: the older `output/natural_questions_logprob_n100/` smoke-test artifact uses the same nested layout, and the train/test refactor for NQ was completed only on the Qwen3 side (configs `nq_train_qwen3.json`, `nq_test_qwen3.json`).

---

## 5. Alignment plan

### Option A — Migrate in place (preferred, no recompute)

Treat the existing 20,772-sample zarr as the **train split** and re-run inference for a held-out **test split**. This mirrors the user's [feedback_train_test_split](.claude/projects/-mnt-home-hyang1-LLM-research-HalluLens/memory/feedback_train_test_split.md) preference and matches Qwen3 NQ's split sizes (16,617 train / 4,155 test).

Concrete steps:

1. **Rename + relocate the activations.**
   ```
   shared/natural_questions_logprob/activations.zarr/
       → shared/natural_questions_train/activations.zarr/
   ```
   (rename of the `_memmap_cache` is harmless; `meta/index.jsonl` and `arrays/` are layout-stable.)

2. **Move generations/evals into `output/`.**
   ```
   shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct/*
       → output/natural_questions_train/Llama-3.1-8B-Instruct/
   ```

3. **Regenerate `raw_eval_res.jsonl`** by re-running `--step eval` on the train split. This is the file the config already references but is missing from disk.

4. **Reconcile the 2-sample mismatch.** The zarr has 20,772 entries but `generation.sanitized_for_eval.jsonl` and `eval_results.json` only cover 20,770. Identify the 2 dropped samples (likely empty/refusal responses dropped during sanitization), then either:
   - drop them from the zarr by rebuilding `meta/index.jsonl` and the `arrays/sample_key` array, OR
   - tag them with `label=missing` in `eval_results_for_training.json` so the dataloader skips them.

5. **Run a true held-out test split.** Run inference + eval on the NQ test split using `scripts/run_with_server.py --task natural_questions --model meta-llama/Llama-3.1-8B-Instruct` writing to:
   ```
   shared/natural_questions/activations.zarr/
   output/natural_questions/Llama-3.1-8B-Instruct/
   ```
   (~4k samples, dispatched via `gpu_dispatch.py` per project convention.)

6. **Replace the config.** Delete `configs/datasets/nq_test_hallu_cor.json` after creating a Llama3 sibling of `nq_qwen3.json`:

   ```json
   // configs/datasets/nq.json
   {
     "name": "nq",
     "model_name": "Llama-3.1-8B-Instruct",
     "input_dim": 4096,
     "backend": "zarr",
     "label_source": "eval_json",
     "outlier_class": 1,
     "train": {
       "inference_json": "output/natural_questions_train/Llama-3.1-8B-Instruct/generation.jsonl",
       "activations_path": "shared/natural_questions_train/activations.zarr",
       "eval_json": "output/natural_questions_train/Llama-3.1-8B-Instruct/eval_results_for_training.json",
       "raw_eval_jsonl": "output/natural_questions_train/Llama-3.1-8B-Instruct/raw_eval_res.jsonl"
     },
     "test": {
       "inference_json": "output/natural_questions/Llama-3.1-8B-Instruct/generation.jsonl",
       "activations_path": "shared/natural_questions/activations.zarr",
       "eval_json": "output/natural_questions/Llama-3.1-8B-Instruct/eval_results_for_training.json",
       "raw_eval_jsonl": "output/natural_questions/Llama-3.1-8B-Instruct/raw_eval_res.jsonl"
     }
   }
   ```

7. **Update `configs/experiments/baseline_comparison_nq.json`** to use `"dataset": "nq"` instead of `"nq_test_hallu_cor"`.

8. **Cleanup.** Remove the now-empty `shared/natural_questions_logprob/` and the side artifact `output/natural_questions_logprob_n100/`.

### Option B — Adapter shim (no data motion)

If moving ~85 GB of activation files is undesirable, instead:
- Add the missing `raw_eval_res.jsonl` (re-run eval, write to the existing nested dir).
- Reconcile the 2-sample drift.
- Symlink:
  ```
  shared/natural_questions_train/activations.zarr → ../natural_questions_logprob/activations.zarr
  output/natural_questions_train/Llama-3.1-8B-Instruct → ../../shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct
  ```
- Treat the 20,772-sample dump as the train split and still run a fresh test split into the canonical `shared/natural_questions/`.

This avoids touching the zarr but leaves a permanent lie in the directory structure. **Not recommended** — the symlinks will rot.

### Option C — Recompute everything

Re-run inference on both NQ test and NQ train splits with Llama-3.1-8B-Instruct, writing directly to canonical paths, and discard `shared/natural_questions_logprob/` entirely. Cleanest result, but ~21k samples × 33 layers × fp16 of activation logging — significant GPU time.

---

## 6. Recommendation

**Option A.** It preserves the existing 20k activations, fits the existing convention without symlink hacks, and only requires re-running the (small) ~4k-sample test split + a one-shot eval pass for `raw_eval_res.jsonl`. Estimated cost: one H200-hour for the test split inference + a few minutes for eval.

Open questions for the user before executing:
1. Is the 20,772-sample dump actually the **train** half of NQ, or was it run on the test prompts? (Need to verify by sampling 5–10 prompts and checking against HF `natural_questions` split.) If it was test prompts, the migration target flips.
2. Are you OK deleting `shared/natural_questions_logprob/` after the move, or do you want it retained as a frozen snapshot for the original `baseline_comparison_nq` runs in `runs/baseline_comparison_nq/nq_test_hallu_cor/`?
