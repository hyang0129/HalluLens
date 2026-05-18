# Implementation Spec: LLMsKnow Probe Baseline (Issue #52)

**Goal:** Add a third probe baseline that replicates the Orgad et al. 2024 "LLMsKnow" methodology — sweep all (layer, token-position) pairs on a dev subset to pick the single best location, then train sklearn LogisticRegression at that location and evaluate over 5 seeds.

All code changes are offline (no GPU needed). Run them now; queue the 60-run grid on the cluster once data is ready.

---

## 1. Token Position Definitions

14 canonical positions, named as strings in the config:

| Name | Index formula | Notes |
|---|---|---|
| `response_0` | `input_length + 0` | first answer token |
| `response_1` | `input_length + 1` | second answer token |
| `response_2` | `input_length + 2` | |
| `response_3` | `input_length + 3` | |
| `response_4` | `input_length + 4` | |
| `exact_answer_last` | derived (see §3.1) | fallback: `response_0` |
| `last_1` | `response_end - 0` | actual last response token (not padded) |
| `last_2` | `response_end - 1` | |
| `last_3` | `response_end - 2` | |
| `last_4` | `response_end - 3` | |
| `last_5` | `response_end - 4` | |
| `last_6` | `response_end - 5` | |
| `last_7` | `response_end - 6` | |
| `last_8` | `response_end - 7` | |

`response_end` = `input_length + response_length - 1`.  
`response_length` must be loaded from zarr metadata (see §3.2 — do NOT use `pad_length` as a proxy since padding is random Gaussian noise that is indistinguishable from real activations).

---

## 2. Sweep Dimensions

- **Layers:** 14–29 (16 layers, the full cached range)
- **Tokens:** the 14 positions above
- **Grid:** 16 × 14 = **224 (layer, token) pairs** per sweep
- **Dev subset:** 1000 samples drawn from training data using `training_seed`
- **Classifier:** `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`
- **Dev metric:** AUROC (from `sklearn.metrics.roc_auc_score`)

---

## 3. New Module: `activation_research/llmsknow_probe.py`

### 3.1 `locate_exact_answer_span`

```python
def locate_exact_answer_span(
    response_text: str,
    gold_answer: str,
    tokenizer,          # HF tokenizer for the model
    response_offset: int = 0,   # token offset of response start in full sequence
) -> int | None:
    """Find the last token index of the longest common substring match.

    Returns absolute token index in the full padded sequence (response_offset + span_end).
    Returns None if no substring found (caller should fall back to response_0).
    """
```

Implementation notes:
- Find longest common substring of `gold_answer` in `response_text` (case-insensitive, strip punctuation).
- Tokenize only the matched substring portion to determine its token length.
- Return `response_offset + (start_token + span_token_length - 1)`.
- If `gold_answer` not found in `response_text`, return `None`.

### 3.2 `resolve_token_index`

```python
def resolve_token_index(
    token_spec: str,
    input_length: int,
    response_length: int,       # actual unpadded response token count
    exact_answer_idx: int | None = None,
) -> int | None:
    """Map a token_spec string to an absolute sequence index.

    Returns None if the position is out of bounds (e.g. response too short).
    The caller should skip this (layer, token_spec) pair for that sample.
    """
```

Rules:
- `response_k` → `input_length + k`; valid if `k < response_length`
- `exact_answer_last` → `exact_answer_idx` if not None, else `input_length + 0`
- `last_k` → `input_length + response_length - k`; valid if `response_length >= k`

### 3.3 `get_response_lengths`

The zarr dataset pads activations to `pad_length=63` with random noise, so we cannot read response length from the tensor shape. Load it from zarr metadata instead:

```python
def get_response_lengths(zarr_path: str, prompt_hashes: list[str]) -> dict[str, int]:
    """Load response_length (or total_length - input_length) from zarr metadata.

    Returns a dict mapping prompt_hash → response_length.
    Falls back to reading from generation.jsonl if zarr metadata is missing the field.
    """
```

Implementation: open the zarr store, read the metadata array for each hash. The zarr activations logger stores `input_length` per sample; verify whether `response_length` or `total_length` is also stored. If not present, fall back to `len(tokenizer(row["response"])["input_ids"])` from `generation.jsonl`.

> **Decision point during implementation:** Check `activation_logging/zarr_activations_logger.py` to see which metadata fields are written. If `response_length` is absent, add it to the writer as a one-line change, or load from generation.jsonl. Document the chosen approach in a comment.

### 3.4 `sweep_layers_tokens`

```python
def sweep_layers_tokens(
    activations_by_layer: dict[int, np.ndarray],  # layer → (N, pad_length, H)
    labels: np.ndarray,               # (N,) int
    input_lengths: np.ndarray,        # (N,) int
    response_lengths: np.ndarray,     # (N,) int
    token_specs: list[str],
    dev_indices: np.ndarray,          # indices into N for the 1000-sample dev set
    exact_answer_indices: np.ndarray | None = None,  # (N,) int or None per sample
) -> pd.DataFrame:
    """Run the full (layer × token) sweep on the dev subset.

    Returns DataFrame with columns: layer, token_spec, dev_auroc, n_valid_dev
    where n_valid_dev is the count of dev samples that had a valid position
    (not out-of-bounds, not in the noise-padded region).

    Rows where n_valid_dev < 50 should have dev_auroc = NaN (too few samples).
    """
```

Inner loop:
```
for layer in layers (14..29):
    X_layer = activations_by_layer[layer]  # (N, T, H)
    for token_spec in token_specs:
        # For each dev sample, resolve token index → extract activation
        X_dev, y_dev = [], []
        for i in dev_indices:
            idx = resolve_token_index(token_spec, input_lengths[i], response_lengths[i],
                                      exact_answer_indices[i] if exact_answer_indices else None)
            if idx is None or idx >= X_layer.shape[1]:
                continue
            X_dev.append(X_layer[i, idx, :])
            y_dev.append(labels[i])
        if len(set(y_dev)) < 2 or len(y_dev) < 50:
            auroc = float('nan')
        else:
            clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
            # Use 80% for fit, 20% for AUROC within the dev subset
            split = int(0.8 * len(X_dev))
            clf.fit(X_dev[:split], y_dev[:split])
            scores = clf.predict_proba(X_dev[split:])[:, 1]
            auroc = roc_auc_score(y_dev[split:], scores)
        record sweep row
```

### 3.5 `select_best`

```python
def select_best(sweep_df: pd.DataFrame) -> dict:
    """Return {'layer': int, 'token': str, 'dev_auroc': float}
    from the row with maximum dev_auroc (NaNs excluded).
    """
```

### 3.6 `fit_final_probe`

```python
def fit_final_probe(
    activations: np.ndarray,   # (N, pad_length, H) for the selected layer
    labels: np.ndarray,        # (N,)
    input_lengths: np.ndarray,
    response_lengths: np.ndarray,
    token_spec: str,
    exact_answer_indices: np.ndarray | None = None,
) -> tuple[LogisticRegression, int]:
    """Train final LogReg on all samples with a valid position.

    Returns (fitted_clf, n_train).
    """
```

### 3.7 `evaluate_probe`

```python
def evaluate_probe(
    clf: LogisticRegression,
    activations: np.ndarray,   # (N_test, pad_length, H) for selected layer
    labels: np.ndarray,        # (N_test,)
    input_lengths: np.ndarray,
    response_lengths: np.ndarray,
    token_spec: str,
    exact_answer_indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Evaluate probe on test set. Returns (auroc, scores_array)."""
```

---

## 4. Method Config: `configs/methods/llmsknow_probe.json`

```json
{
  "name": "llmsknow_probe",
  "routine": "llmsknow_probe",
  "model_class": null,
  "model_params": null,
  "training": null,
  "data": {
    "relevant_layers": "14-29",
    "pad_length": 63,
    "preload": true,
    "include_response_logprobs": false
  },
  "sweep": {
    "layers": "14-29",
    "token_specs": [
      "response_0", "response_1", "response_2", "response_3", "response_4",
      "exact_answer_last",
      "last_1", "last_2", "last_3", "last_4",
      "last_5", "last_6", "last_7", "last_8"
    ],
    "dev_subset_size": 1000
  },
  "final": {
    "classifier": "logreg_l2_C1"
  },
  "evaluation": {
    "metrics": ["auroc"]
  }
}
```

---

## 5. New Routine: `run_llmsknow_probe` in `scripts/run_experiment.py`

### Signature

```python
def run_llmsknow_probe(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """LLMsKnow probe: sweep (layer, token) on dev subset, train LogReg at best location."""
```

### Logic

```
1. Parse config
   - data_cfg = method_cfg["data"]
   - sweep_cfg = method_cfg["sweep"]
   - layers = parse_layer_range(sweep_cfg["layers"])          # [14..29]
   - token_specs = sweep_cfg["token_specs"]                    # list of 14 strings
   - dev_subset_size = sweep_cfg["dev_subset_size"]            # 1000

2. Load datasets
   - ds_kwargs = dict(relevant_layers=layers, pad_length=data_cfg["pad_length"],
                      preload=True, num_views=len(layers),
                      include_response_logprobs=False, check_ram=False)
   - train_ds = ap.get_dataset("train", **ds_kwargs)
   - eval_ap = test_ap if test_ap else ap
   - test_ds = eval_ap.get_dataset("test", **ds_kwargs)

3. Materialize activations to numpy
   - Iterate train_ds in DataLoader (batch_size=256, shuffle=False, num_workers=0)
   - Collect: activations_by_layer[layer_idx] → np.array (N, T, H)
   - Collect: labels (N,), input_lengths (N,), prompt_hashes (N,)
   - Similarly for test_ds → test_activations_by_layer, test_labels, etc.

   NOTE: views_activations shape from MultiLayerActivationDataset is (num_layers, T, H)
   per sample; index as views_activations[layer_position, :, :] where layer_position
   is the 0-based index into the relevant_layers list (NOT the absolute layer number).
   Map: layer_position = layers.index(abs_layer).

4. Load response_lengths
   - From zarr metadata or generation.jsonl for train + test
   - Use get_response_lengths() from activation_research.llmsknow_probe

5. Dev subset sampling
   - rng = np.random.default_rng(training_seed)
   - dev_indices = rng.choice(len(train_ds), size=min(dev_subset_size, len(train_ds)),
                              replace=False)

6. Optional: load exact_answer_indices
   - Load generation.jsonl + eval_results.json for train set
   - For each sample, call locate_exact_answer_span() to get exact_answer_idx
   - Store as np.array of int (use -1 for "not found", resolve_token_index treats
     exact_answer_idx=None as fallback to response_0)
   - This requires a tokenizer; load with AutoTokenizer.from_pretrained(dataset_cfg["model_name"])
     OR skip for v1 and treat exact_answer_last as fallback only (no span extraction)

   **Recommended for v1:** skip span extraction. Set all exact_answer_indices to None.
   The fallback in resolve_token_index() uses response_0 when exact_answer_idx is None,
   making exact_answer_last a duplicate of response_0 in v1. Document this. Add span
   extraction in a follow-up.

7. Sweep
   - sweep_df = sweep_layers_tokens(activations_by_layer, labels, input_lengths,
                                    response_lengths, token_specs, dev_indices)
   - Save sweep_df to os.path.join(output_dir, "sweep.csv")

8. Selection
   - selected = select_best(sweep_df)
   - Save to os.path.join(output_dir, "selected.json")
   - sel_layer = selected["layer"]
   - sel_token = selected["token"]

9. Final probe (train on all training data)
   - clf, n_train = fit_final_probe(activations_by_layer[sel_layer], labels,
                                    input_lengths, response_lengths, sel_token)
   - Save clf with joblib.dump to os.path.join(output_dir, "artifacts", "probe.joblib")

10. Evaluation
    - auroc, scores = evaluate_probe(clf, test_activations_by_layer[sel_layer],
                                     test_labels, test_input_lengths,
                                     test_response_lengths, sel_token)
    - eval_metrics = {
        "method": "llmsknow_probe",
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": n_train,
        "n_test": len(test_ds),
        "auroc": float(auroc),
        "selected_layer": sel_layer,
        "selected_token": sel_token,
        "dev_auroc": selected["dev_auroc"],
      }
    - predictions = [{"example_id": i, "score_halu": float(s), "label_halu": int(l)}
                     for i, (s, l) in enumerate(zip(scores, test_labels))]
    - Also save np.save(os.path.join(output_dir, "predictions.npy"), scores)

11. Return eval_metrics, predictions
```

### Dispatch Registration

Add to the dispatch block in `scripts/run_experiment.py` around line 1881 (after `logprob_baseline`):

```python
elif routine == "llmsknow_probe":
    eval_metrics, predictions = run_llmsknow_probe(
        ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
        test_ap=test_ap,
    )
```

The method IS learned (has seeds), so `is_learned = True` must be set in the experiment runner's method-meta lookup. Verify this: the runner classifies a method as learned if it has a non-null `"training"` key OR if the routine requires seeds. The `llmsknow_probe` config has `"training": null` but still uses seeds. Check `scripts/experiment_utils.py` to see how `is_learned` is determined and add `"llmsknow_probe"` to the learned-routine set if needed.

---

## 6. Experiment Config Updates (12 files)

Add `"llmsknow_probe"` to the `"methods"` list in each of these files.
The order should be after `"linear_probe"` and before `"token_entropy"`.

**Llama-3.1-8B-Instruct configs (6 files):**
- `configs/experiments/baseline_comparison_hotpotqa.json`
- `configs/experiments/baseline_comparison_mmlu.json`
- `configs/experiments/baseline_comparison_nq.json`
- `configs/experiments/baseline_comparison_popqa.json`
- `configs/experiments/baseline_comparison_sciq.json`
- `configs/experiments/baseline_comparison_searchqa.json`

**Qwen3-8B configs (6 files):**
- `configs/experiments/baseline_comparison_hotpotqa_qwen3.json`
- `configs/experiments/baseline_comparison_mmlu_qwen3.json`
- `configs/experiments/baseline_comparison_nq_qwen3.json`
- `configs/experiments/baseline_comparison_popqa_qwen3.json`
- `configs/experiments/baseline_comparison_sciq_qwen3.json`
- `configs/experiments/baseline_comparison_searchqa_qwen3.json`

Do NOT update the `_smollm3` configs.

---

## 7. `is_learned` Fix in experiment_utils.py

The runner determines whether to iterate over seeds by checking if a method is "learned". Current logic likely checks for `"training" in method_cfg and method_cfg["training"] is not None`. Since `llmsknow_probe` has `"training": null`, it may be incorrectly classified as non-learned (single run, no seeds).

Fix: add a check for `method_cfg.get("routine") == "llmsknow_probe"` in `is_learned` logic, OR change the check to also look for a `"sweep"` key, OR simply check `method_cfg.get("routine") not in {"token_entropy", "logprob_baseline"}`.

Check `scripts/experiment_utils.py` `RunSpec` / `enumerate_runs` to locate the exact condition.

---

## 8. Tests: `tests/test_llmsknow_probe.py`

```python
def test_resolve_token_index_response_tokens():
    # response_0 with input_length=10, response_length=20 → 10
    # response_4 → 14
    # response_5 with response_length=4 → None (out of bounds)

def test_resolve_token_index_last_tokens():
    # last_1 with input_length=10, response_length=5 → 14
    # last_5 with response_length=5 → 10
    # last_6 with response_length=5 → None (out of bounds)

def test_resolve_token_index_exact_answer():
    # with exact_answer_idx=17 → 17
    # with exact_answer_idx=None → falls back to response_0 → input_length

def test_locate_exact_answer_span_found():
    # gold = "Paris", response contains "Paris" → returns non-None int

def test_locate_exact_answer_span_not_found():
    # gold not in response → returns None

def test_sweep_synthetic():
    # Create synthetic activations: (N=200, T=63, H=64) for 3 layers
    # Create labels with signal only at (layer=1, response_0)
    # Run sweep_layers_tokens() with dev_indices
    # Assert: sweep_df has 3*14=42 rows, selected layer=1 and response_0

def test_fit_and_evaluate_probe():
    # Synthetic data, fit probe, evaluate, verify AUROC > 0.5 with planted signal
```

---

## 9. Output Artifacts Per Run

```
runs/baseline_comparison_{dataset}/{dataset}/{split_seed}/llmsknow_probe/seed_{seed}/
  config.json            # merged config (written by run_experiment.py)
  run_manifest.json      # git hash, hostname, etc.
  sweep.csv              # 224 rows: layer, token_spec, dev_auroc, n_valid_dev
  selected.json          # {"layer": int, "token": str, "dev_auroc": float}
  eval_metrics.json      # final metrics including selected_layer, selected_token
  predictions.csv        # example_id, score_halu, label_halu
  predictions.npy        # raw float scores array
  artifacts/
    probe.joblib         # fitted LogisticRegression
```

---

## 10. Report Script Update: `scripts/generate_seed0_report.py`

Add `llmsknow_probe` to the method list. For each cell, also show `selected_layer:selected_token` in a tooltip or parenthetical (so we can see what location was chosen per dataset).

---

## 11. Implementation Order

Do all of this locally (no GPU needed):

1. **`activation_research/llmsknow_probe.py`** — new module with all functions
2. **`tests/test_llmsknow_probe.py`** — unit tests; run with `pytest tests/test_llmsknow_probe.py`
3. **`configs/methods/llmsknow_probe.json`** — method config
4. **`scripts/run_experiment.py`** — add `run_llmsknow_probe()` function + dispatch case
5. **`scripts/experiment_utils.py`** — fix `is_learned` classification if needed
6. **12 experiment configs** — add `"llmsknow_probe"` to methods lists
7. **`scripts/generate_seed0_report.py`** — add column

Then queue on cluster:

```bash
# Smoketest first (1 dataset, 1 seed, to verify no crashes)
python scripts/run_experiment.py \
  --experiment configs/experiments/baseline_comparison_hotpotqa.json \
  --methods llmsknow_probe --seeds 0

# Full grid (all 12 configs × 5 seeds = 60 runs)
python scripts/gpu_dispatch.py run --node <node> \
  bash scripts/run_all_baselines.sh  # or dispatch per-experiment
```

---

## 12. Known Constraints / Out of Scope

- **Layer range:** Only layers 14–29 are swept (not full 0..N-1). Document in eval_metrics.json as `"sweep_layers": [14, 29]`.
- **exact_answer_last in v1:** No span extraction; falls back to `response_0`. Add tokenizer-based span extraction as a follow-up.
- **`last_q_token`:** Excluded — prompt activations are not cached.
- **Hyperparameter tuning:** Fixed `C=1.0`, no grid search.
- **smollm3 configs:** Not included in this issue.
