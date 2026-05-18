# Implementation Spec: Cross-Dataset Transfer Matrix (Issue #62)

**Goal:** Evaluate whether hallucination detectors generalize across datasets by applying
source-trained checkpoints to target test sets, producing a 6×6 AUROC matrix per
(model, method). Zero new training; zero new inference. CPU-only (cluster GPU not needed,
but the existing zarr activation stores must be present on the cluster filesystem).

---

## 1. Scope

### Methods (3)
| Method | Checkpoint file | Eval mechanism |
|--------|----------------|----------------|
| `contrastive_logprob_recon` | `contrastive_last.pt` | Mahalanobis OOD + KNN OOD on embeddings |
| `linear_probe` | `linear_probe_last.pt` | Direct sigmoid → AUROC |
| `saplma` | `linear_probe_last.pt` | Direct sigmoid → AUROC |

### Datasets (6)
`hotpotqa`, `mmlu`, `nq`, `popqa`, `sciq`, `searchqa`

### Models (2)
`Llama-3.1-8B-Instruct`, `Qwen3-8B`

### Seeds
Use whatever seeds were actually trained in the existing experiment runs —
**do not assume [0,5,26,42,63] from the issue description**; the existing configs use
`training_seeds: [0, 1, 2, 3, 4]`. The script should discover available seeds from the
filesystem rather than hardcoding.

### Matrix cells
6 source datasets × 6 target datasets × 2 models × 3 methods × 5 seeds = **1,800 cells**
(including diagonal, which is in-distribution and used for reference).

---

## 2. Checkpoint & Data Path Conventions

### Checkpoint paths
```
runs/baseline_comparison_{source_dataset}/
  {source_dataset}/
    {method}/
      seed_{seed}/
        contrastive_last.pt      (contrastive_logprob_recon)
        linear_probe_last.pt     (linear_probe, saplma)
        eval_metrics.json        (marker: run is complete)
```

A run is considered **complete and loadable** only if `eval_metrics.json` exists alongside
the checkpoint file. Skip any run where either is missing.

### Dataset config paths
```
configs/datasets/{dataset}.json         (test split)
configs/datasets/{dataset}_train.json   (train split)
```

The dataset config provides:
- `input_dim`: 4096 for both Llama and Qwen3
- `test.activations_path`: zarr store for test split
- `test.eval_json`: eval results with `halu` labels
- `train.activations_path`: zarr store for train split (needed for Mahalanobis reference)
- `train.eval_json`: eval results for train split

Qwen3 dataset configs use `{dataset}_qwen3` naming. Resolve the correct dataset config
from the experiment config via its `"dataset"` field, which is already set correctly
(e.g., `"hotpotqa_qwen3"` for Qwen3 experiments).

---

## 3. Files to Create

### 3.1 `activation_research/transfer_eval.py`

Core transfer evaluation logic. No CLI entry point; imported by the script.

#### `load_checkpoint_model(method: str, checkpoint_path: str, dataset_cfg: dict) -> nn.Module`

```python
def load_checkpoint_model(
    method: str,
    checkpoint_path: str,
    dataset_cfg: dict,
) -> torch.nn.Module:
    """Load model from checkpoint. Returns model in eval mode on CPU."""
```

Implementation:
```
1. Determine model class from method:
   - "contrastive_logprob_recon" → LogprobReconProgressiveCompressor
   - "linear_probe"              → LinearProbe
   - "saplma"                    → SimpleHaluClassifier

2. Read model_params from the method config embedded in the run's config.json
   (path: checkpoint_path/../config.json). Fall back to these defaults if absent:
   - LogprobReconProgressiveCompressor:
       input_dim=dataset_cfg["input_dim"], final_dim=512, input_dropout=0.3,
       recon_seq_len=64, recon_hidden_dim=256, recon_lambda=1.0,
       logprob_var_threshold=1e-4
   - LinearProbe:
       input_dim=dataset_cfg["input_dim"], pooling="mean"
   - SimpleHaluClassifier:
       input_dim=dataset_cfg["input_dim"], hidden_dims=[2048, 1024, 512], dropout=0.1

3. ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
   model.load_state_dict(ckpt["model_state_dict"])
   model.eval()
   return model
```

#### `get_embeddings_contrastive(model, zarr_path, eval_json_path, relevant_layers, device="cpu") -> list[dict]`

```python
def get_embeddings_contrastive(
    model: torch.nn.Module,
    zarr_path: str,
    eval_json_path: str,
    relevant_layers: list[int],
    device: str = "cpu",
    batch_size: int = 128,
    num_workers: int = 4,
) -> list[dict]:
    """Run inference_embeddings on a zarr dataset split.

    Returns list of records: [{"hashkey": str, "z_views": Tensor(1, D), "halu": int}, ...]
    """
```

Implementation:
```
1. Build ActivationParser from zarr_path + eval_json_path
   (replicate the ap.get_dataset("test"/"train", ...) pattern from run_experiment.py)
2. ds = ap.get_dataset(
       split=split,
       relevant_layers=relevant_layers,
       num_views=1,              # 1 view — we only need forward-pass embeddings
       include_response_logprobs=False,
       preload=False,
       check_ram=False,
   )
3. dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, persistent_workers=False)
4. records = []
   with torch.no_grad():
       for batch in dl:
           x = batch["views_activations"][:, 0, :, :]   # (B, T, H) — first (only) view
           z = model(x.to(device)).cpu()                  # (B, D)
           for i in range(len(z)):
               records.append({
                   "hashkey": batch["hashkey"][i],
                   "z_views": z[i].unsqueeze(0),          # (1, D) — metrics.py expects (K, D)
                   "halu": int(batch["halu"][i]),
               })
5. return records
```

#### `get_scores_probe(model, zarr_path, eval_json_path, probe_layer, device="cpu") -> tuple[np.ndarray, np.ndarray]`

```python
def get_scores_probe(
    model: torch.nn.Module,
    zarr_path: str,
    eval_json_path: str,
    probe_layer: int,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass a LinearProbe or SimpleHaluClassifier on a dataset split.

    Returns (scores, labels): float32 arrays of shape (N,).
    scores are sigmoid probabilities in [0, 1].
    """
```

Implementation:
```
1. Build ActivationParser for zarr_path + eval_json_path
2. ds = ap.get_dataset(
       split=split,
       relevant_layers=[probe_layer],
       num_views=1,
       include_response_logprobs=False,
       preload=False,
       check_ram=False,
   )
3. dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, persistent_workers=False)
4. all_scores, all_labels = [], []
   with torch.no_grad():
       for batch in dl:
           x = batch["views_activations"][:, 0, :, :]   # (B, T, H)
           out = model(x.to(device)).squeeze(-1).cpu()   # (B,)
           all_scores.append(out.numpy())
           all_labels.append(batch["halu"].numpy())
5. return np.concatenate(all_scores), np.concatenate(all_labels)
```

#### `evaluate_transfer_cell(source_run_dir, source_dataset_cfg, target_test_cfg, method, relevant_layers, probe_layer, device="cpu") -> dict`

```python
def evaluate_transfer_cell(
    source_run_dir: str,
    source_dataset_cfg: dict,     # dataset config for the SOURCE (for Mahalanobis reference)
    target_test_cfg: dict,        # dataset config for TARGET (test split to evaluate on)
    method: str,
    relevant_layers: list[int],
    probe_layer: int,             # used only for linear_probe / saplma
    device: str = "cpu",
) -> dict:
    """Evaluate one transfer matrix cell. Returns metrics dict."""
```

Implementation:
```
1. Determine checkpoint file:
   if method == "contrastive_logprob_recon":
       ckpt_file = "contrastive_last.pt"
   else:   # linear_probe, saplma
       ckpt_file = "linear_probe_last.pt"
   checkpoint_path = os.path.join(source_run_dir, ckpt_file)

   if not os.path.exists(checkpoint_path):
       return {"status": "missing_checkpoint"}

2. model = load_checkpoint_model(method, checkpoint_path, source_dataset_cfg)
   model = model.to(device)

3a. If method == "contrastive_logprob_recon":
    # Reference distribution: source TRAIN embeddings
    src_train_records = get_embeddings_contrastive(
        model,
        zarr_path=source_dataset_cfg["train"]["activations_path"],
        eval_json_path=source_dataset_cfg["train"]["eval_json"],
        relevant_layers=relevant_layers,
        device=device,
    )
    # Target TEST embeddings
    tgt_test_records = get_embeddings_contrastive(
        model,
        zarr_path=target_test_cfg["test"]["activations_path"],
        eval_json_path=target_test_cfg["test"]["eval_json"],
        relevant_layers=relevant_layers,
        device=device,
    )
    # OOD stats — both Mahalanobis and KNN
    maha_stats = mahalanobis_ood_stats(src_train_records, tgt_test_records, outlier_class=1)
    knn_stats = knn_ood_stats(src_train_records, tgt_test_records, outlier_class=1,
                               k=50, metric="euclidean", calibrate_k=False)
    return {
        "status": "ok",
        "auroc": maha_stats["mahalanobis_auroc"],          # primary metric
        "mahalanobis_auroc": maha_stats["mahalanobis_auroc"],
        "mahalanobis_mean_id": maha_stats["mahalanobis_mean_id"],
        "mahalanobis_std_id": maha_stats["mahalanobis_std_id"],
        "mahalanobis_mean_ood": maha_stats["mahalanobis_mean_ood"],
        "mahalanobis_std_ood": maha_stats["mahalanobis_std_ood"],
        "knn_auroc": knn_stats["knn_auroc"],
        "n_src_train": len(src_train_records),
        "n_tgt_test": len(tgt_test_records),
    }

3b. Else (linear_probe, saplma):
    scores, labels = get_scores_probe(
        model,
        zarr_path=target_test_cfg["test"]["activations_path"],
        eval_json_path=target_test_cfg["test"]["eval_json"],
        probe_layer=probe_layer,
        device=device,
    )
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores))
    return {
        "status": "ok",
        "auroc": auroc,
        "n_tgt_test": len(labels),
    }
```

#### `discover_runs(runs_root, method) -> list[dict]`

```python
def discover_runs(
    runs_root: str,
    method: str,
) -> list[dict]:
    """Scan runs_root for completed runs of the given method.

    Returns list of dicts: [
        {
            "experiment_name": str,   # e.g. "baseline_comparison_hotpotqa"
            "dataset": str,           # e.g. "hotpotqa"
            "method": str,
            "seed": int,
            "run_dir": str,           # absolute path to seed dir
            "config": dict,           # parsed config.json
        },
        ...
    ]
    Only includes runs where both eval_metrics.json and the checkpoint file exist.
    """
```

Walk: `runs_root/{experiment_name}/{dataset}/{method}/seed_{seed}/`
Check: `eval_metrics.json` exists AND checkpoint file exists.
Parse: `config.json` in the same dir.

---

### 3.2 `scripts/eval_transfer_matrix.py`

CLI entry point. No GPU required; runs on CPU.

#### Arguments
```
--runs-dir            str, default "runs"         Root of existing training runs
--configs-dir         str, default "configs"      Root of configs/
--output-dir          str, default "runs/transfer_matrix"
--methods             str list, default all three methods
--model-slugs         str list, default ["llama", "qwen3"]
                      Used to select which experiment configs to load.
                      "llama"  → experiment configs without _qwen3 or _smollm3 suffix
                      "qwen3"  → experiment configs with _qwen3 suffix
--source-datasets     str list, default all 6
--target-datasets     str list, default all 6
--seeds               int list, default None (discover from filesystem)
--device              str, default "cpu"
--num-workers         int, default 4
--relevant-layers     str, default "14-29"  (parsed as range)
--resume              flag, skip cells where output JSON already exists
```

#### Logic

```
1. Parse args.

2. Load dataset configs for all target datasets:
   target_cfgs = {}
   for dataset in target_datasets:
       for slug in model_slugs:
           cfg_name = dataset if slug == "llama" else f"{dataset}_qwen3"
           cfg = json.load(open(f"configs/datasets/{cfg_name}.json"))
           target_cfgs[(dataset, slug)] = cfg

3. Discover source runs:
   all_runs = []
   for method in methods:
       all_runs += discover_runs(runs_dir, method)
   Filter all_runs by source_datasets and model_slugs.

4. relevant_layers = parse_layer_range(args.relevant_layers)   # [14..29]
   For each source run:
     - Determine model_slug from run's experiment_name
       (contains "_qwen3" → "qwen3", else "llama")
     - Load source dataset config for Mahalanobis reference:
         src_dataset_cfg = json.load(open(f"configs/datasets/{run['dataset']}.json"))
         if model_slug == "qwen3":
             src_dataset_cfg = json.load(open(f"configs/datasets/{run['dataset']}_qwen3.json"))
     - Get probe_layer from run's eval_metrics.json "selected_layer" field if present,
       else from run's config.json method_cfg["probe_layer"], else default to 26.

5. Enumerate cells:
   for run in filtered_source_runs:
       for target_dataset in target_datasets:
           if (run["dataset"], target_dataset) already has same-model check:
               # Only evaluate same-model-family transfers (Llama→Llama, Qwen→Qwen)
               # Cross-model transfer is out of scope for this issue.
               pass
           cell_id = f"{run['dataset']}__{target_dataset}__{run['method']}__{run['seed']}"
           output_path = f"{output_dir}/{model_slug}/{cell_id}.json"
           if args.resume and os.path.exists(output_path):
               continue

           result = evaluate_transfer_cell(
               source_run_dir=run["run_dir"],
               source_dataset_cfg=src_dataset_cfg,
               target_test_cfg=target_cfgs[(target_dataset, model_slug)],
               method=run["method"],
               relevant_layers=relevant_layers,
               probe_layer=probe_layer,
               device=args.device,
           )
           result.update({
               "source_dataset": run["dataset"],
               "target_dataset": target_dataset,
               "method": run["method"],
               "seed": run["seed"],
               "model_slug": model_slug,
               "experiment_name": run["experiment_name"],
           })
           os.makedirs(os.path.dirname(output_path), exist_ok=True)
           json.dump(result, open(output_path, "w"), indent=2)
           print(f"[{cell_id}] auroc={result.get('auroc', 'ERR'):.4f}")

6. After all cells, call aggregate_results(output_dir) to write transfer_matrix.csv
```

#### `aggregate_results(output_dir: str) -> None`

```python
def aggregate_results(output_dir: str) -> None:
    """Read all per-cell JSON files, aggregate into transfer_matrix.csv.

    Columns: source_dataset, target_dataset, method, model_slug, seed, auroc, status
    Also writes:
      transfer_matrix_mean.csv — mean AUROC over seeds per (source, target, method, model_slug)
      transfer_matrix_ci.csv   — same with 95% CI (mean ± 1.96*std/sqrt(n))
    """
```

---

## 4. Data Flow Summary

```
For each (source_run, target_dataset):

  [source_run_dir/contrastive_last.pt]
          │
          ▼
  load_checkpoint_model()
          │
          ▼
  ┌───────────────────────────────────────────┐
  │ contrastive_logprob_recon                 │
  │                                           │
  │  source train split zarr                  │
  │      → get_embeddings_contrastive()       │
  │      → src_train_records (N_src, D)       │
  │                                           │
  │  target test split zarr                   │
  │      → get_embeddings_contrastive()       │
  │      → tgt_test_records (N_tgt, D)        │
  │                                           │
  │  mahalanobis_ood_stats(src, tgt)          │
  │      → {"mahalanobis_auroc": float}       │
  │                                           │
  │  knn_ood_stats(src, tgt, k=50)            │
  │      → {"knn_auroc": float}               │
  └───────────────────────────────────────────┘

  ┌───────────────────────────────────────────┐
  │ linear_probe / saplma                     │
  │                                           │
  │  target test split zarr                   │
  │      → get_scores_probe()                 │
  │      → (scores, labels)                   │
  │                                           │
  │  roc_auc_score(labels, scores)            │
  │      → auroc: float                       │
  └───────────────────────────────────────────┘
          │
          ▼
  cell JSON → transfer_matrix.csv
```

---

## 5. probe_layer Resolution for linear_probe / saplma

When the source run used a specific probe layer (chosen during training), the transfer eval
must use the same layer on the target dataset. Resolution order:

1. Read `eval_metrics.json` in the source run dir; use `"selected_layer"` if present.
2. Read `config.json` in the source run dir; use `method_cfg["probe_layer"]` if present.
3. Default to **26** (the best-performing layer in baseline experiments).

This is a deliberate design choice: we use the source-optimized layer on the target dataset
without re-sweeping. This tests pure transfer, not re-tuned transfer.

---

## 6. Handling saplma Checkpoint Filename Ambiguity

Both `linear_probe` and `saplma` use `LinearProbeTrainer` and save `linear_probe_last.pt`.
They live in separate method subdirectories so there is no collision. The script resolves
the correct file by checking the method name, not the filename.

If a run dir contains neither `linear_probe_last.pt` nor `contrastive_last.pt`, also check
for `trainer_last.pt` as a fallback (some runs may use the generic Trainer checkpoint name).

---

## 7. Output Structure

```
runs/transfer_matrix/
  llama/
    hotpotqa__mmlu__contrastive_logprob_recon__0.json
    hotpotqa__mmlu__contrastive_logprob_recon__1.json
    ... (one file per cell)
  qwen3/
    hotpotqa__mmlu__contrastive_logprob_recon__0.json
    ...
  transfer_matrix.csv             ← flat table, all cells
  transfer_matrix_mean.csv        ← mean over seeds
  transfer_matrix_ci.csv          ← with 95% CI
```

### Per-cell JSON schema
```json
{
  "source_dataset": "hotpotqa",
  "target_dataset": "mmlu",
  "method": "contrastive_logprob_recon",
  "model_slug": "llama",
  "seed": 0,
  "experiment_name": "baseline_comparison_hotpotqa",
  "status": "ok",
  "auroc": 0.6234,
  "mahalanobis_auroc": 0.6234,
  "mahalanobis_mean_id": 12.3,
  "mahalanobis_std_id": 3.1,
  "mahalanobis_mean_ood": 15.7,
  "mahalanobis_std_ood": 4.2,
  "knn_auroc": 0.6105,
  "n_src_train": 4500,
  "n_tgt_test": 1000
}
```
`auroc` is always the primary scalar (Mahalanobis for contrastive, sigmoid AUROC for probes).
`mahalanobis_auroc` and `knn_auroc` are both present for contrastive cells only.

### transfer_matrix.csv columns
```
source_dataset, target_dataset, method, model_slug, seed, auroc, status,
mahalanobis_auroc, knn_auroc, n_src_train, n_tgt_test
```
(`mahalanobis_auroc` and `knn_auroc` are NaN for `linear_probe` / `saplma` rows.)

### transfer_matrix_mean.csv columns
```
source_dataset, target_dataset, method, model_slug,
auroc_mean, auroc_std, auroc_ci95_lo, auroc_ci95_hi, n_seeds
```

---

## 8. No New Config Files Needed

This evaluation reads from existing experiment configs and dataset configs. No new
`configs/experiments/transfer_matrix_*.json` files are required. The script is
self-contained.

---

## 9. Implementation Order

All steps are local (no GPU, no new inference):

1. **`activation_research/transfer_eval.py`**
   - `load_checkpoint_model`
   - `get_embeddings_contrastive`
   - `get_scores_probe`
   - `evaluate_transfer_cell`
   - `discover_runs`

2. **`scripts/eval_transfer_matrix.py`**
   - Argument parsing
   - Main loop over (source_run, target_dataset) cells
   - `aggregate_results`

3. **Smoke test** (1 cell, diagonal — fastest since source==target activations already loaded for training):
   ```bash
   python scripts/eval_transfer_matrix.py \
     --source-datasets hotpotqa \
     --target-datasets hotpotqa \
     --methods linear_probe \
     --model-slugs llama \
     --seeds 0 \
     --device cpu
   ```
   Verify output JSON has `status: ok` and `auroc` close to the in-distribution
   `eval_metrics.json` value (should match within floating-point noise for the diagonal).

4. **Full run** (all cells — CPU parallelizable across source_datasets or seeds):
   ```bash
   # Llama
   python scripts/eval_transfer_matrix.py \
     --model-slugs llama \
     --device cpu \
     --num-workers 4 \
     --resume

   # Qwen3
   python scripts/eval_transfer_matrix.py \
     --model-slugs qwen3 \
     --device cpu \
     --num-workers 4 \
     --resume
   ```

---

## 10. Notes and Known Constraints

### Mahalanobis on CPU
The Mahalanobis computation in `metrics.py` uses `np.linalg.inv` on the full covariance
matrix of dimension D=512. For N_src_train ~4,500 samples, this is fast (<1s per cell).
No GPU needed.

### Out of scope
- Cross-model transfer (Llama → Qwen3): excluded per issue #62.
- Target-train fine-tuning.
- New training or inference.
- SmolLM3 (not included in issue scope).

### Seed discrepancy with issue #62
Issue #62 lists seeds `[0, 5, 26, 42, 63]` but existing experiments use
`training_seeds: [0, 1, 2, 3, 4]`. The script discovers seeds from the filesystem, so
whichever seeds were actually trained will be used. Report this discrepancy in the PR.

### Diagonal interpretation
Diagonal cells (source == target) are in-distribution evaluation. Their AUROC should
closely match the `eval_metrics.json` stored in the source run. If there is a meaningful
discrepancy (>0.02 AUROC), it indicates a bug in the transfer eval pipeline — use this as
a sanity check.

### relevant_layers for contrastive
The `relevant_layers` arg to `get_dataset()` controls which layers are loaded from zarr.
For `contrastive_logprob_recon`, layers 14–29 were used during training. The model forward
pass uses all loaded layers as a sequence; passing the same range at eval time is correct.
Do NOT change the layer range between source-train and target-eval.
