# Issue #156: gated token-wise architecture sweep

## Execution gate

Implementation and validation belong in PR #152, but the Stage 1 training
cells must not be created until all nine cells in the Issue #154 mixed 50/50
sweep are complete.  After those results are analyzed, exactly one training
recipe is selected:

- `mixed`: use `tokenwise_causal_mixed_half` if it improves the predeclared
  three-dataset pilot aggregate.
- `v1`: otherwise retain `tokenwise_contrastive_first_anchored` as the
  fallback.

`scripts/dispatch/build_issue_156_architecture_cells.py` enforces this gate.
It requires an explicit recipe, verifies that the nine mixed cells are in the
`done` state, verifies the selected baseline outputs, and records the choice in
`issue156_recipe_selection.json`. It refuses to mix recipes in one queue.

No Issue #156 training cells are queued as part of implementation.

## Stage 0: existing projection-surface diagnostic

The trained Issue #153 v2 checkpoints already expose a normalized 128-d
projection. Scoring that projection requires no retraining and is complete for
all five datasets.

| Dataset | v2 trunk KNN AUROC | v2 projection KNN AUROC | Projection delta |
|---|---:|---:|---:|
| HotpotQA | 0.6850 | 0.7017 | +0.0167 |
| NQ | 0.6504 | 0.6618 | +0.0114 |
| PopQA | 0.7869 | 0.7408 | -0.0461 |
| SciQ | 0.7012 | 0.6597 | -0.0415 |
| SearchQA | 0.6500 | 0.6128 | -0.0372 |
| Macro | 0.6947 | 0.6754 | -0.0193 |

The projection helps two datasets but loses 1.9 AUROC points in five-dataset
macro. It is therefore a useful scoring surface to retain, not an architecture
to promote by itself.

## Stage 1: one factor at a time

Starting from the selected post-#154 recipe, train seed 0 on HotpotQA, NQ, and
PopQA with exactly one model change per arm:

1. Input LayerNorm only (`+8,192` parameters).
2. Pre-norm transformer blocks only (`+0` parameters).
3. One-query layer-attention pooling only (`+512` parameters).
4. Disposable `512 -> 512 -> 128` normalized projection only (`+328,320`
   parameters).

The projection arm reports both 512-d trunk and 128-d projection KNN/probe
metrics in the same cell. This avoids redundant training and avoids an
evaluation cell racing its source checkpoint.

All arms inherit the selected recipe's data construction, objective,
optimizer, step count, validation-KNN checkpoint selection, splits, and
scorers. The primary statistic is raw-Euclidean KNN AUROC, aggregated as an
unweighted macro over the three pilot datasets. Normalized-cosine KNN, frozen
linear probe, and AUPRC remain secondary diagnostics.

## Queue commands after the gate

Choose one command only after the mixed sweep is complete and its aggregate is
known:

```bash
# Mixed wins
python scripts/dispatch/build_issue_156_architecture_cells.py \
  --recipe mixed \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch

# Mixed does not win; corrected v1 remains the fixed recipe
python scripts/dispatch/build_issue_156_architecture_cells.py \
  --recipe v1 \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch
```

Either command creates twelve independently claimable cells: four model arms
times three datasets times one seed. Existing generic workers can consume the
cells without any worker-specific changes. MMLU is excluded.
