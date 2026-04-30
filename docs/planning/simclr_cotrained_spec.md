# SimCLRCotrainedModel — Bring-Up-to-Spec Plan

**Goal:** make `simclr_cotrained` a first-class experiment alongside `contrastive_logprob_recon`,
with identical dataset coverage, comparable training config, and results visible in the seed0 report.

---

## What already works

- `SimCLRCotrainedModel` implemented in `activation_research/model.py`
- Full training/eval dispatch in `scripts/run_experiment.py` (`run_simclr_cotrained`)
- Method config at `configs/methods/simclr_cotrained.json`
- Added to `configs/experiments/baseline_comparison_nq.json`

## What the eval pipeline produces

`run_simclr_cotrained` outputs two kinds of metrics in `eval_metrics.json`:

| Key | Source | Notes |
|-----|--------|-------|
| `cosine_auroc` | OOD eval on embeddings | same scorer as contrastive |
| `mahalanobis_auroc` | OOD eval on embeddings | same scorer as contrastive |
| `knn_auroc` | OOD eval on embeddings | same scorer as contrastive |
| `auroc` | BCE head on test set | **new** — direct classifier score |

The contrastive baseline only has the first three. `simclr_cotrained` adds a fourth
signal from the supervised head, which is its main differentiator.

---

## Gaps to close

### 1. Method config — align training hyperparameters

`simclr_cotrained.json` is missing some guards that `contrastive_logprob_recon.json` has:

```json
"training": {
    "max_epochs": 100,        // currently 50 — match contrastive for fair comparison
    "min_total_steps": 3000,  // missing — prevents early exit on small datasets (e.g. SciQ)
    "steps_per_epoch_override": null  // not present — add for datasets where needed
}
```

Also consider whether `use_labels` / `ignore_label` semantics should be plumbed into
`SimCLRCotrainedTrainer` (contrastive uses `ignore_label: 1` to exclude uncertain samples
from the contrastive loss — SimCLR currently always uses all samples).

### 2. Experiment configs — add to all datasets

Currently only `baseline_comparison_nq.json` includes `simclr_cotrained`. Add to all:

- [ ] `baseline_comparison_hotpotqa.json`
- [ ] `baseline_comparison_hotpotqa_qwen3.json`
- [ ] `baseline_comparison_mmlu.json`
- [ ] `baseline_comparison_movies.json`
- [ ] `baseline_comparison_nq_qwen3.json`
- [ ] `baseline_comparison_popqa.json`
- [ ] `baseline_comparison_sciq.json`
- [ ] `baseline_comparison_searchqa.json`

### 3. Seed0 report — add SimCLR columns

`scripts/generate_seed0_report.py` currently has no `simclr_cotrained` entries.
Need to add two new columns to the results table:

```python
# In get_metrics(), add:
m = load_json(runs_dir / dataset / "simclr_cotrained" / seed_dir / "eval_metrics.json")
if m:
    result["simclr_knn"]  = fmt(m.get("knn_auroc"))   # best OOD scorer
    result["simclr_head"] = fmt(m.get("auroc"))         # direct BCE head

# In cols list, add:
("SimCLR KNN",  "simclr_knn"),
("SimCLR Head", "simclr_head"),
```

The KNN column allows apples-to-apples comparison with `Contr. KNN`.
The Head column shows whether the supervised signal adds value on top of the embeddings.

### 4. Validation that the model actually trains

Before running all datasets, do a smoke-test on NQ (seed 0 only, already in the config):

```bash
python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_nq.json \
    --methods simclr_cotrained \
    --seeds 0
```

Check that:
- Training loss decreases (both `simclr_loss` and `bce_loss` components)
- `intra_cos` increases (views of same sample becoming more similar)
- `eval_metrics.json` contains all four AUROC keys after completion
- Head AUROC (`auroc`) is above 0.5

### 5. Tune loss weights (optional, post-smoke-test)

The default `simclr_weight: 1.0, bce_weight: 1.0` is a reasonable starting point,
but the two losses have different scales. If either dominates, try:
- `bce_weight: 0.1` to let the contrastive objective lead
- `bce_weight: 10.0` to emphasize the supervised signal

Track with W&B once that integration lands (see `project_wandb_integration.md`).

---

## Order of operations

1. Fix method config (`max_epochs`, `min_total_steps`)
2. Smoke-test on NQ seed 0
3. If training looks healthy, add to remaining experiment configs
4. Update `generate_seed0_report.py` with new columns
5. Run full seed sweep across all datasets
6. Evaluate: does `simclr_head` AUROC beat `Contr. KNN`? If yes, the supervised
   co-training is helping. If `simclr_knn` is worse than `Contr. KNN`, the BCE
   loss may be hurting the embedding quality.
