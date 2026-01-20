# Results Schema (Experiment Contract)

This document defines the **required on-disk artifacts** for HalluLens experiments so that we can:
- rerun experiments deterministically (multi-seed, multi-hparam),
- aggregate results into paper tables/plots,
- trace every reported metric back to (prompt, response, label, activations).

This is a **contract**: the experiment runner(s) should write these files, and the aggregator should only rely on these files.

---

## 1) Run Directory Layout

Each run is uniquely identified by:
- `method` (e.g., `last_layer_baseline`, `contrastive_v1`)
- `model` (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- `benchmark` (e.g., `precisewikiqa`, `triviaqa`)
- `split` (e.g., `train`, `dev`, `test`; may be encoded in config and/or filenames)
- `seed`

**Required path convention**

`runs/{date}/{method}/{model}/{benchmark}/seed_{seed}/`

Example:

`runs/2026-01-20/contrastive_v1/meta-llama__Llama-3.1-8B-Instruct/precisewikiqa/seed_42/`

Notes:
- Use a filesystem-safe model slug (`/` replaced with `__`).
- If you run multiple splits, either:
  - create separate directories per split, OR
  - store split in each per-example record (required regardless).

---

## 2) Required Files (Per Run)

### 2.1 `config.json`
The configuration used for this run. Must be complete enough to reproduce the run.

Required top-level fields (minimum):
- `benchmark`: string
- `split`: string (`train`/`dev`/`test`)
- `method`: string
- `seed`: int
- `model`: object
  - `id`: string
  - `backend`: string (e.g., `vllm`, `gguf`, `hf`)  
  - `dtype`: optional string
- `inference`: object
  - `max_tokens`: int
  - `temperature`: number
  - `top_p`: number
  - `n`: int (number of samples per prompt; typically 1)
- `activation_logging`: object
  - `enabled`: bool
  - `format`: string (`lmdb` | `json_npy` | `zarr`)
  - `path`: string (path to activations store)
  - `layers`: list[int] or string (e.g., `all`)
  - `token_selector`: string (e.g., `last_token`, `mean_pool`, `exact_answer_last_token`)
- `training`: object (for training runs)
  - `epochs`: int
  - `batch_size`: int
  - `sub_batch_size`: int
  - `lr`: number
  - `temperature`: optional number (contrastive loss temperature)

### 2.2 `run_manifest.json`
Metadata for provenance and safety.

Required fields:
- `created_at`: ISO8601 string
- `git`: object
  - `commit`: string
  - `dirty`: bool
- `environment`: object
  - `python_version`: string
  - `platform`: string
  - `cuda`: optional string
- `notes`: optional string

### 2.3 `generations.jsonl`
Line-delimited JSON; one record per generated example.

Each record must include:
- `example_id`: string (stable unique ID within the run; see ID rules below)
- `benchmark`: string
- `split`: string
- `seed`: int
- `model_id`: string
- `prompt`: string
- `response`: string
- `reference`: optional object (ground-truth answer(s), if available)
- `label`: object
  - `halu`: int (0/1) or bool
  - `label_source`: string (e.g., `exact_match`, `annotation`, `llm_judge`)
- `generation_params`: object (temperature/top_p/max_tokens)
- `timing`: optional object

Activation linkage fields (at least one is required):
- `activation_key`: string (the key used in LMDB/JSON index), OR
- `hashkey`: string (if the rest of the pipeline uses this), OR
- both.

### 2.4 `predictions.csv`
Per-example model predictions used for metrics.

Required columns:
- `example_id`
- `score_halu` (float; higher means “moreHS” / more hallucination)
- `label_halu` (0/1)
- `split`
- `benchmark`

Recommended columns:
- `method`
- `model_id`
- `seed`
- `threshold` (if you apply a fixed threshold)
- `pred_halu` (0/1; derived from `score_halu` + threshold)

### 2.5 `eval_metrics.json`
Aggregated metrics computed from `predictions.csv`.

Required fields:
- `benchmark`
- `split`
- `method`
- `model_id`
- `seed`
- `n_examples`: int
- `auroc`: number

Recommended fields:
- `auprc`: number
- `accuracy`: number
- `f1`: number
- `ece`: number
- `calibration_curve`: optional

### 2.6 `train_metrics.jsonl`
Line-delimited JSON of training logs (one line per step or per epoch).

Required fields (per record):
- `time`: ISO8601 string
- `epoch`: int
- `split`: string (`train` or `val`)
- `loss`: number

Recommended fields:
- `auroc`: number (if you evaluate during training)
- `lr`: number
- `cosine_sim`: number (if relevant)

---

## 3) Activation Store Requirements

The activation store may be LMDB / JSON+NPY / Zarr, but must satisfy:

- A deterministic mapping from `example_id` (or `activation_key`) → activation tensors.
- Sufficient metadata to recover:
  - which model produced the activations
  - which prompt/response they correspond to
  - which layers/tokens were recorded

Minimum required metadata per activation entry:
- `example_id` or `activation_key`
- `model_id`
- `prompt`
- `response`
- `layers`
- `token_indices` or `token_selector`

---

## 4) ID Rules (Critical)

### 4.1 `example_id`
Must be **stable and unique** across runs and resilient to path changes.

Recommended format:
- `sha256(benchmark + split + canonical_prompt + model_id + seed)`

If you already use a `hashkey` field in datasets, ensure:
- it is stable across processes
- it does not depend on Python’s randomized `hash()` (which changes across runs)

### 4.2 Linking generations ↔ activations ↔ labels
For every row in `predictions.csv`:
- `example_id` must exist in `generations.jsonl`
- the corresponding activation entry must exist in the activation store
- `label_halu` must match the `label.halu` from `generations.jsonl`

---

## 5) Aggregation Contract (Multi-Seed)

An aggregator should be able to scan `runs/**/eval_metrics.json` and produce:
- per-benchmark mean ± CI across seeds
- global summary tables

Therefore, `eval_metrics.json` must be self-contained and include the identifying fields:
- `benchmark`, `split`, `method`, `model_id`, `seed`

---

## 6) Compliance Checklist (Per Run)

- [ ] Run directory follows the naming convention
- [ ] `config.json` exists and includes all required fields
- [ ] `run_manifest.json` exists and includes git + environment provenance
- [ ] `generations.jsonl` contains `example_id`, `prompt`, `response`, `label.halu`
- [ ] `predictions.csv` contains `example_id`, `score_halu`, `label_halu`
- [ ] `eval_metrics.json` contains `auroc` and identifying fields
- [ ] Activations store can be joined via `example_id` or `activation_key`

