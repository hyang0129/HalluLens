# Issue #156: gated token-wise architecture sweep

## Execution gate

Implementation and validation belong in PR #152, but the Stage 1 training
cells must not be created until all nine cells in the Issue #154 mixed 50/50
sweep are complete. Those results rejected the mixed recipe as the primary
KNN baseline. The completed five-dataset controls then motivated a matched
comparison of two retained recipes:

- `v1` / tn: `tokenwise_contrastive_first_anchored`, pairing t0 with a random
  same-response later token under the original objective.
- `t0`: `tokenwise_causal_t0_dropout`, pairing two independently dropped-out
  t0 views under the causal-control objective.

The five-dataset KNN macro is 0.7790 for v1 and 0.7751 for t0, while the
linear-probe macro reverses the order (0.7590 for v1 and 0.7719 for t0). The
architecture sweep therefore retains both instead of choosing one from a
single scoring surface.

`scripts/dispatch/build_issue_156_architecture_cells.py` enforces this gate.
It verifies that the nine mixed cells are in the `done` state, verifies each
recipe's baseline outputs, and records the comparison in
`issue156_recipe_selection.json`. The v1 and t0 recipes may share one queue;
the rejected legacy mixed recipe cannot be combined with them.

No Issue #156 training cells are queued merely by importing the implementation.

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

Starting separately from each retained recipe, train seed 0 on HotpotQA, NQ,
PopQA, SciQ, and SearchQA with exactly one model change per arm:

1. Input LayerNorm only (`+8,192` parameters).
2. Pre-norm transformer blocks only (`+0` parameters).
3. One-query layer-attention pooling only (`+512` parameters).
4. Disposable `512 -> 512 -> 128` normalized projection only (`+328,320`
   parameters).

The projection arm reports both 512-d trunk and 128-d projection KNN/probe
metrics in the same cell. This avoids redundant training and avoids an
evaluation cell racing its source checkpoint.

All arms inherit their recipe's data construction, objective,
optimizer, step count, validation-KNN checkpoint selection, splits, and
scorers. The primary statistic is raw-Euclidean KNN AUROC, aggregated as an
unweighted macro over all five datasets. Normalized-cosine KNN, frozen linear
probe, and AUPRC remain secondary diagnostics. Each architecture arm is
compared with the baseline from the same training recipe before comparing
cross-recipe outcomes.

## Queue commands after the gate

Run both commands after the mixed sweep is complete:

```bash
# Original t0 + tn recipe
python scripts/dispatch/build_issue_156_architecture_cells.py \
  --recipe v1 \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch

# Two independently dropped-out t0 views
python scripts/dispatch/build_issue_156_architecture_cells.py \
  --recipe t0 \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch
```

Together the commands create forty independently claimable cells: two recipes
times four model arms times five datasets times one seed. Existing generic
workers can consume the cells without any worker-specific changes. MMLU is
excluded.
