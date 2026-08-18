# Issue 157: validation-selected token-wise scorer study

## Decision boundary

Every scorer is fitted from the training embedding bank. Candidate selection
uses validation labels only, aggregated as the mean across seeds within each
of the five HalluLens datasets and then the macro mean across datasets. One
scorer is locked for the entire training recipe. Test embeddings are opened
only by the finalization phase, which evaluates only that locked scorer.

The current token-wise runs also use validation KNN AUROC for checkpoint
selection. Reusing the same validation split for scorer selection is therefore
reported as exploratory. A later confirmation should reserve a distinct
scorer-calibration split.

## Candidate matrix

Selection-eligible candidates are fixed before test evaluation:

1. Raw Euclidean distance to the all-example bank, `k=50`.
2. Raw Euclidean distance to the all-example bank, `k=1000`.
3. Raw Euclidean distance to the truthful-only bank, `k=50`.
4. `k=50` neighbor vote with inverse train-class-frequency weighting.
5. Squared Mahalanobis distance from the truthful class using OAS covariance.
6. Shared-covariance shrinkage LDA.
7. Distance-to-truth centroid minus distance-to-hallucination centroid.
8. Fixed `C=1` logistic regression.
9. Standardized, class-balanced, fixed `C=1` logistic regression.

L2-normalized cosine KNN at `k=50` is emitted as a geometry diagnostic but is
not eligible for deployment-scorer selection. Projection-head scoring remains
an architecture diagnostic rather than a scorer candidate.

The all-example KNN bank asks whether a point lies in a dense region of the
overall training representation distribution; its distance score is
label-agnostic. The truthful-only bank instead asks how far a point lies from
known truthful behavior. It can provide a cleaner one-class anomaly score, but
it discards hallucination-bank geometry and can mistake legitimate rare
truthful examples for anomalies.

## Artifacts

Each run must first contain `embeddings/manifest.json` and isolated
`train_z.npy`, `val_z.npy`, and `test_z.npy` surfaces with labels, raw prompt
hashes, stable row IDs, and split provenance.

The validation preparation phase writes:

- metrics and per-example scores for every candidate;
- train and validation embedding norms;
- KNN neighbor indices and distances;
- fitted covariance, centroid, and linear-model parameters;
- source fingerprints and prompt-hash overlap audits;
- a manifest certifying that test arrays were not accessed.

The global lock records every candidate's dataset means and macro validation
AUROC. Finalization writes per-example raw, validation-CDF, and global
validation-Platt scores for the selected scorer only, plus AUROC, AUPRC,
normalized AP gain, TPR at 5%/10% FPR, calibration metrics, and universal
validation-derived alert thresholds. Aggregate ranking metrics remain macro;
test examples are never micro-pooled for the headline comparison.

## Commands

Backfill the completed v1 and t0-control sweeps without retraining:

```bash
python scripts/dispatch/build_issue_157_validation_backfill_cells.py \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch
```

Prepare one run without opening test arrays:

```bash
python scripts/run_tokenwise_scorer_study.py prepare --run-dir <run-dir>
```

After all five datasets are prepared, create one recipe-level lock, then run
the explicit finalization command:

```bash
python scripts/run_tokenwise_scorer_study.py select \
  --training-recipe <recipe> \
  --validation-manifest <five-dataset validation manifests...> \
  --output <selection-lock.json>

python scripts/run_tokenwise_scorer_study.py finalize \
  --selection <selection-lock.json> \
  --output-dir <final-output-dir>
```

MMLU is excluded from this study.
