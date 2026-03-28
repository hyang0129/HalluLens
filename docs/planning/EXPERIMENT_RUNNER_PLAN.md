# Experiment Runner: Design Plan

## Goal

Replace the notebook-driven workflow (`d_baseline_comparison_hotpotqa.ipynb`) with a config-driven CLI that:
1. Runs all 5 methods (contrastive, logprob-recon, linear probe, token entropy, logprob baseline)
2. Evaluates with consistent OOD metrics
3. Produces structured, aggregatable artifacts
4. Supports multi-seed sweeps for publishable results

---

## 1. Config Architecture

Three levels of config, composed at runtime:

### 1a. Dataset Config (`configs/datasets/*.json`)

One file per dataset. Defines paths and label semantics.

```json
{
  "name": "hotpotqa_train",
  "inference_json": "output/hotpotqa_train/Llama-3.1-8B-Instruct/generation.jsonl",
  "activations_path": "shared/hotpotqa_train/activations.zarr",
  "eval_json": "output/hotpotqa_train/Llama-3.1-8B-Instruct/eval_results_for_training.json",
  "model_name": "Llama-3.1-8B-Instruct",
  "input_dim": 4096,
  "backend": "zarr",
  "label_source": "eval_json",
  "outlier_class": 1
}
```

**Why separate**: Dataset paths/semantics rarely change between experiments. One file per dataset avoids repetition across experiment configs.

**Datasets to define initially**: `hotpotqa_train`, `nq_test_hallu_cor`, `hotpotqa_hallu_cor`, `goodwiki_jsonv2`, `triviiqa_json`.

### 1b. Method Config (`configs/methods/*.json`)

One file per method. Defines model architecture, training routine, and evaluation strategy.

```json
// configs/methods/contrastive.json
{
  "name": "contrastive",
  "routine": "contrastive",
  "model_class": "progressive_compressor",
  "model_params": {
    "final_dim": 512,
    "input_dropout": 0.3
  },
  "training": {
    "max_epochs": 50,
    "batch_size": 512,
    "lr": 1e-5,
    "temperature": 0.25,
    "steps_per_epoch_override": 200,
    "use_labels": true,
    "ignore_label": 1,
    "use_infinite_index_stream": true,
    "use_infinite_index_stream_eval": true,
    "balanced_sampling": false
  },
  "data": {
    "relevant_layers": "14-29",
    "target_layers": [22, 26],
    "num_views": 2,
    "pad_length": 63,
    "preload": true,
    "include_response_logprobs": false
  },
  "evaluation": {
    "metrics": ["cosine", "mds", "knn"],
    "knn_params": {
      "k": 50,
      "metric": "euclidean",
      "calibrate_k": true,
      "k_candidates": [50, 100, 200, 500, 1000],
      "max_train_size": 200000
    },
    "eval_batch_size": 256,
    "sub_batch_size": 64
  }
}
```

```json
// configs/methods/linear_probe.json
{
  "name": "linear_probe",
  "routine": "linear_probe",
  "model_class": "linear_probe",
  "model_params": {
    "pooling": "mean"
  },
  "training": {
    "max_epochs": 30,
    "batch_size": 512,
    "lr": 1e-3,
    "steps_per_epoch_override": 100,
    "balanced_sampling": true
  },
  "data": {
    "probe_layer": 22,
    "preload": true,
    "include_response_logprobs": false
  },
  "evaluation": {
    "metrics": ["auroc"]
  }
}
```

```json
// configs/methods/token_entropy.json
{
  "name": "token_entropy",
  "routine": "token_entropy",
  "model_class": null,
  "training": null,
  "data": {
    "relevant_layers": "14-29",
    "num_views": 2,
    "preload": true,
    "include_response_logprobs": true,
    "response_logprobs_top_k": 20
  },
  "evaluation": {
    "metrics": ["mean_logprob", "min_logprob", "mean_entropy"]
  }
}
```

```json
// configs/methods/logprob_baseline.json
{
  "name": "logprob_baseline",
  "routine": "logprob_baseline",
  "model_class": null,
  "training": null,
  "data": {
    "relevant_layers": "14-29",
    "num_views": 2,
    "preload": true,
    "include_response_logprobs": true,
    "response_logprobs_top_k": 20
  },
  "evaluation": {
    "metrics": ["mean_logprob", "seq_logprob", "perplexity"]
  }
}
```

### 1c. Experiment Config (`configs/experiments/*.json`)

Composes dataset + methods + sweep parameters for a specific experiment run.

```json
{
  "experiment_name": "baseline_comparison_hotpotqa",
  "dataset": "hotpotqa_train",
  "methods": ["contrastive", "logprob_recon", "linear_probe", "token_entropy", "logprob_baseline"],
  "split_seed": 42,
  "training_seeds": [0, 5, 26, 42, 63],
  "device": "auto",
  "num_workers": 4,
  "persistent_workers": true,
  "output_dir": "runs"
}
```

**Multi-dataset variant** (for cross-benchmark comparison):
```json
{
  "experiment_name": "cross_benchmark_v1",
  "datasets": ["hotpotqa_train", "nq_test_hallu_cor"],
  "methods": ["contrastive", "linear_probe"],
  "split_seed": 42,
  "training_seeds": [0, 42],
  "device": "auto",
  "num_workers": 4,
  "output_dir": "runs"
}
```

---

## 2. CLI Interface

### 2a. Main runner

```bash
# Single experiment
python scripts/run_experiment.py \
  --experiment configs/experiments/baseline_comparison_hotpotqa.json

# Override seeds from CLI
python scripts/run_experiment.py \
  --experiment configs/experiments/baseline_comparison_hotpotqa.json \
  --seeds 0,42

# Single method + dataset (no experiment config needed)
python scripts/run_experiment.py \
  --dataset configs/datasets/hotpotqa_train.json \
  --method configs/methods/contrastive.json \
  --seed 42
```

### 2b. Aggregation

```bash
# Aggregate results from a runs directory
python scripts/aggregate_results.py \
  --runs-dir runs/baseline_comparison_hotpotqa \
  --output results/baseline_comparison_hotpotqa.csv
```

---

## 3. Output Directory Structure

```
runs/
  baseline_comparison_hotpotqa/
    hotpotqa_train/
      contrastive/
        seed_0/
          config.json              # Merged config (dataset + method + runtime)
          run_manifest.json        # Git hash, python version, timestamp, CUDA info
          train_metrics.jsonl      # Per-epoch: {epoch, loss, lr, cosine_sim, timestamp}
          eval_metrics.json        # {auroc, auprc, method, dataset, seed, n_train, n_test, ...}
          predictions.csv          # Per-example: example_id, score, label, split
          artifacts/
            final_weights.pt       # Model checkpoint
            snapshots/             # Training snapshots (epoch_10.pt, etc.)
        seed_42/
          ...
      linear_probe/
        seed_0/
          ...
      token_entropy/                # No seed subdirectory (deterministic)
        config.json
        eval_metrics.json
        predictions.csv
      logprob_baseline/             # No seed subdirectory (deterministic)
        ...
```

**Key decisions:**
- Non-learned methods (token_entropy, logprob_baseline) have no seed directory since they're deterministic
- `config.json` is the merged, fully-resolved config — everything needed to reproduce the run
- `predictions.csv` enables post-hoc analysis (custom thresholds, per-category breakdown, etc.)

---

## 4. run_manifest.json

Auto-generated at the start of each run:

```json
{
  "created_at": "2026-03-28T14:30:00Z",
  "git_commit": "a571c9c",
  "git_dirty": true,
  "python_version": "3.12.0",
  "torch_version": "2.1.0",
  "cuda_version": "12.1",
  "gpu_name": "NVIDIA H200",
  "hostname": "alphagpu23",
  "command": "python scripts/run_experiment.py --experiment ..."
}
```

---

## 5. Seeding Strategy

### Two-level seeding: split seed (fixed) + training seed (varied)

```
Split seed:    FIXED = 42  (never changes across methods or runs)
Training seeds: [0, 5, 26, 42, 63]  (same 5 seeds for every method on every dataset)
```

**Why**: Every method sees the same train/test split. The only thing that varies
is training randomness (weight init, batch order, dropout). This makes
method-to-method comparison fair — any AUROC difference is from the method, not
the data split.

**Non-learned methods** (token_entropy, logprob_baseline) are deterministic — they
run once per dataset with no seed loop. Their single result is replicated across
all seed columns in the summary table.

### What each seed controls

| Source of randomness | Controlled by | Current state |
|---|---|---|
| Train/test split | `ActivationParser(random_seed=42)` | Already seeded (fixed at 42) |
| PyTorch weight init | `torch.manual_seed(training_seed)` | **NOT seeded — must add** |
| CUDA ops | `torch.cuda.manual_seed_all(training_seed)` | **NOT seeded — must add** |
| Numpy | `np.random.seed(training_seed)` | **NOT seeded — must add** |
| Python random | `random.seed(training_seed)` | **NOT seeded — must add** |
| CuDNN | `torch.backends.cudnn.deterministic = True` | **NOT set — must add** |
| DataLoader workers | `worker_init_fn(worker_id)` | **NOT seeded — must add** |
| View sampling (layer selection per __getitem__) | Python `random` | **NOT seeded — must add** |
| InfiniteIndexStream shuffle | `seed` param | Passed but defaults to 0 |
| KNN subsampling | `sample_seed` param | Passed but defaults to 0 |

### Implementation

```python
def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int):
    """Deterministic worker seeding for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Called at the start of each `(method, seed)` run. The training seed also flows to:
- `InfiniteIndexStream(seed=training_seed)`
- `knn_ood_stats(sample_seed=training_seed)`
- Any view sampling RNG in ActivationDataset

### Config representation

```json
// In experiment config
{
  "split_seed": 42,
  "training_seeds": [0, 5, 26, 42, 63]
}
```

```json
// In per-run config.json (fully resolved)
{
  "split_seed": 42,
  "training_seed": 5,
  "is_deterministic_method": false
}
```

### Resume safety
- Runner checks if `eval_metrics.json` already exists in output dir
- If exists and `--force` not passed, skip that (method, seed) combination
- Enables interrupted sweeps to resume cleanly

---

## 6. Implementation: Script Structure

### `scripts/run_experiment.py` (new, ~300 lines)

```
main()
├── load_experiment_config()       # Merge dataset + method + experiment configs
├── for dataset in datasets:
│   ├── load_dataset_config()
│   ├── build_eval_json_if_needed()  # From notebook cell 4
│   └── for method in methods:
│       ├── load_method_config()
│       ├── seeds = [None] if method.training is None else experiment.seeds
│       └── for seed in seeds:
│           ├── seed_everything(seed)
│           ├── resolve_output_dir()
│           ├── skip if already complete (eval_metrics.json exists)
│           ├── write_run_manifest()
│           ├── write_config()
│           ├── build_datasets()     # ActivationParser.get_dataset()
│           ├── run_method()         # Dispatch to method-specific runner
│           │   ├── run_contrastive()
│           │   ├── run_linear_probe()
│           │   ├── run_token_entropy()
│           │   └── run_logprob_baseline()
│           ├── write_predictions()  # Per-example CSV
│           └── write_eval_metrics() # Aggregate JSON
└── print summary table
```

### Method runners (inside `scripts/run_experiment.py`)

Each method runner follows the same contract:
- **Input**: dataset(s), method config, device, output_dir
- **Output**: list of `{example_id, score, label}` dicts (predictions)

```python
def run_contrastive(train_dataset, test_dataset, method_config, device, output_dir):
    """Train ProgressiveCompressor, evaluate with OOD metrics. Returns predictions."""
    model = ProgressiveCompressor(...)
    trainer = ContrastiveTrainer(model, ContrastiveTrainerConfig(...))
    trainer.fit(train_dataset, test_dataset)
    # Save weights
    # Run OOD evaluation (cosine, mds, knn)
    # Return per-example predictions for each metric
    return predictions, eval_metrics

def run_token_entropy(test_dataset, method_config, device, output_dir):
    """Non-learned baseline. Returns predictions."""
    detector = TokenEntropyDetector(outlier_class=1)
    # Score all test examples
    return predictions, eval_metrics
```

### `scripts/aggregate_results.py` (new, ~100 lines)

```
main()
├── walk runs directory
├── collect eval_metrics.json from each run
├── build DataFrame: dataset × method × seed × metric → AUROC
├── compute mean ± 95% CI per (dataset, method, metric)
├── output:
│   ├── results.csv (raw per-seed results)
│   ├── summary.csv (mean ± CI)
│   └── comparison_table.md (paper-ready markdown table)
└── optionally: bar chart (matplotlib)
```

---

## 7. Reuse from Existing Code

| Notebook step | Existing code to call | Wrapper needed? |
|---|---|---|
| Build eval JSON | Cell 4 logic | Extract to `utils/eval_builder.py` |
| Load datasets | `ActivationParser.get_dataset()` | No |
| Train contrastive | `ContrastiveTrainer` | Thin wrapper for config |
| Train linear probe | `LinearProbeTrainer` | Thin wrapper for config |
| Train logprob-recon | `train_contrastive_logprob_recon()` | Thin wrapper for config |
| OOD evaluation | `MultiMetricHallucinationEvaluator` | No |
| Token entropy | `TokenEntropyDetector` | No |
| Logprob baseline | Cell 21 logic (~30 lines) | Extract to `activation_research/baselines.py` |

The logprob baseline scoring (mean logprob, seq logprob, perplexity) is currently inline in the notebook. This is the only piece that needs to be extracted into a reusable function.

---

## 8. What This Does NOT Change

- No changes to model architectures (`activation_research/model.py`)
- No changes to training loops (`activation_research/training.py`)
- No changes to evaluation/metrics (`activation_research/evaluation.py`, `metrics.py`)
- No changes to activation parsing (`activation_logging/activation_parser.py`)
- No changes to data generation pipeline (`scripts/run_with_server.py`)

This is purely an orchestration + artifact layer on top of existing code.

---

## 9. Migration Path

### Phase 1: Config files + single-run CLI
- Create `configs/` directory with dataset, method, and experiment configs
- Implement `scripts/run_experiment.py` with single (dataset, method, seed) support
- Validate: reproduce notebook results for hotpotqa contrastive + linear probe

### Phase 2: Multi-seed sweeps + aggregation
- Add seed loop to runner
- Implement `scripts/aggregate_results.py`
- Validate: 5-seed sweep on hotpotqa, check mean ± CI

### Phase 3: Multi-dataset + full comparison
- Add multi-dataset support to experiment config
- Run full comparison across hotpotqa + NQ
- Produce paper-ready tables

---

## 10. Example: Full Workflow

```bash
# 1. Run full baseline comparison on HotpotQA with 5 seeds
python scripts/run_experiment.py \
  --experiment configs/experiments/baseline_comparison_hotpotqa.json

# 2. Same for Natural Questions
python scripts/run_experiment.py \
  --experiment configs/experiments/baseline_comparison_nq.json

# 3. Aggregate all results
python scripts/aggregate_results.py \
  --runs-dir runs/ \
  --output results/cross_benchmark.csv

# 4. Output:
#   results/cross_benchmark.csv          — raw per-seed
#   results/cross_benchmark_summary.csv  — mean ± 95% CI
#   results/cross_benchmark_table.md     — paper table
```

Expected summary table output:

```
| Method              | Metric | HotpotQA (AUROC)  | NQ (AUROC)        |
|---------------------|--------|-------------------|-------------------|
| Contrastive (KNN)   | knn    | 0.742 ± 0.012     | 0.731 ± 0.015     |
| Contrastive (MDS)   | mds    | 0.728 ± 0.009     | 0.719 ± 0.011     |
| Linear Probe        | auroc  | 0.701 ± 0.008     | 0.694 ± 0.010     |
| Token Entropy       | mean   | 0.623             | 0.618             |
| Logprob (perplexity)| ppl    | 0.591             | 0.585             |
```
