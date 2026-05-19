"""LLMsKnow Probe Baseline.

Implements a location-sweep sklearn logistic regression probe that:
1. Finds the best (layer, token_position) pair on a dev subset.
2. Trains a final classifier on the full training set at that location.

The cache is *not* materialized eagerly. Each phase reads only the slice it
needs from the (possibly memmapped) full activation cache:
  - Sweep: a (D, L, T, H) dev subset where D = dev_size (typically 1000).
  - Final fit + eval: a single (N, H) column at the chosen (layer, token).
This keeps RAM usage constant w.r.t. N for the dominant terms; the alternative
is materializing every (N_split, L, T, H) row, which is hundreds of GB on the
big benchmarks.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _split_view(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a lazy view over a PreloadedActivationDataset's split.

    Returns:
        full_cache: (N_total, L, T, H) array — may be a numpy memmap. Not copied.
        row_indices: int64 array of shape (N_split,) — positions of this split's
            rows inside full_cache. If the dataset is already split-sized,
            row_indices is arange(N_split).
        labels: int32 array of shape (N_split,).
    """
    full_cache: np.ndarray = ds.cache
    labels: np.ndarray = np.asarray(ds.labels, dtype=np.int32)

    row_indices = getattr(ds, "_row_indices", None)
    if row_indices is None:
        row_indices = np.arange(full_cache.shape[0], dtype=np.int64)
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)

    return full_cache, row_indices, labels


def _take_rows(
    full_cache: np.ndarray,
    row_indices: np.ndarray,
    sub_idx: np.ndarray,
) -> np.ndarray:
    """Materialize rows at split-relative positions sub_idx as a contiguous array.

    Reads len(sub_idx) full (L, T, H) rows from full_cache.
    """
    global_idx = row_indices[sub_idx]
    return np.asarray(full_cache[global_idx])


def _take_column(
    full_cache: np.ndarray,
    row_indices: np.ndarray,
    layer_idx: int,
    token_pos: int,
) -> np.ndarray:
    """Materialize the (N_split, H) slice at one (layer, token) location.

    Reads N_split × H values from full_cache instead of the full (L, T, H) rows.
    """
    return np.asarray(full_cache[row_indices, layer_idx, token_pos, :])


def sweep_locations(
    full_cache: np.ndarray,
    row_indices: np.ndarray,
    labels: np.ndarray,
    relevant_layers: list[int],
    dev_size: int = 2000,
    val_size: int = 1000,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 100,
) -> tuple[np.ndarray, int, int]:
    """Sweep all (layer_idx, token_pos) pairs on a dev subset.

    Args:
        full_cache: (N_total, L, T, H) full cache (possibly memmap).
        row_indices: (N_split,) positions of this split inside full_cache.
        labels: (N_split,) int labels.
        relevant_layers: List of model layer indices (length L), e.g. [14, ..., 29].
        dev_size: Total dev pool size — split into (dev_size - val_size) sub-train
            and val_size sub-validation samples. Paper uses 2000 / 1000 val.
        val_size: Held-out validation subset size for ranking locations.
        seed: Random seed for stratified sampling and LogisticRegression.
        C: Regularisation strength for LogisticRegression.
        max_iter: Max iterations for LogisticRegression solver
            (sklearn default 100 to match Orgad et al.).

    Returns:
        sweep_auroc_matrix: (L, T) AUROC per pair (NaN for skipped/padded positions).
        best_layer_idx: Index into the L dimension (positional, not model layer id).
        best_token_pos: Index into the T dimension.
    """
    N = row_indices.shape[0]
    L, T = full_cache.shape[1], full_cache.shape[2]
    labels = np.asarray(labels, dtype=np.int32)

    actual_dev = min(dev_size, N)
    if actual_dev < N:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=actual_dev, random_state=seed)
        _, dev_idx = next(splitter.split(np.zeros(N), labels))
    else:
        dev_idx = np.arange(N)

    logger.info(
        f"sweep_locations: materialising {len(dev_idx)}-sample dev subset "
        f"({len(dev_idx)} × {L} × {T} × {full_cache.shape[3]} × 2 bytes "
        f"= {len(dev_idx) * L * T * full_cache.shape[3] * 2 / 1024**3:.2f} GB) ..."
    )
    dev_cache = _take_rows(full_cache, row_indices, dev_idx).astype(np.float32)
    dev_labels = labels[dev_idx]

    unique_classes = np.unique(dev_labels)
    if len(unique_classes) < 2:
        logger.warning(
            "sweep_locations: dev subset has only one class — "
            "AUROC is undefined. Falling back to (0, 0)."
        )
        return np.full((L, T), np.nan, dtype=np.float32), 0, 0

    # Single stratified holdout inside the dev subset (paper procedure).
    holdout = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    sub_tr_idx, sub_val_idx = next(holdout.split(np.zeros(len(dev_labels)), dev_labels))

    logger.info(
        f"sweep_locations: sweeping {L} layers × {T} token positions "
        f"= {L * T} pairs with single holdout "
        f"({len(sub_tr_idx)} train / {len(sub_val_idx)} val) ..."
    )

    sweep_auroc = np.full((L, T), np.nan, dtype=np.float32)
    sweep_start = time.monotonic()
    log_every = 50  # log every N completed locations
    locs_done = 0
    locs_skipped = 0
    total_pairs = L * T

    for li in range(L):
        for ti in range(T):
            col = dev_cache[:, li, ti, :]  # (D, H)
            if np.std(col) < 1e-6:
                locs_skipped += 1
                continue  # padded noise position — leave as NaN

            X_tr, y_tr = col[sub_tr_idx], dev_labels[sub_tr_idx]
            X_val, y_val = col[sub_val_idx], dev_labels[sub_val_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                locs_done += 1
                continue

            clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
            clf.fit(X_tr, y_tr)
            scores = clf.predict_proba(X_val)[:, 1]
            sweep_auroc[li, ti] = float(roc_auc_score(y_val, scores))
            locs_done += 1

            if locs_done % log_every == 0:
                elapsed = time.monotonic() - sweep_start
                per_loc = elapsed / locs_done
                pairs_seen = locs_done + locs_skipped
                remaining = (total_pairs - pairs_seen) * per_loc
                running_best = float(np.nanmax(sweep_auroc))
                logger.info(
                    f"sweep_locations: loc {locs_done} "
                    f"(pair {pairs_seen}/{total_pairs}, skipped={locs_skipped}) — "
                    f"elapsed {elapsed:.1f}s ({per_loc*1000:.0f}ms/loc), "
                    f"ETA {remaining:.0f}s, "
                    f"running best val AUROC={running_best:.4f}"
                )

    if np.all(np.isnan(sweep_auroc)):
        logger.warning(
            "sweep_locations: no valid (layer, token) pair found "
            "(all positions are NaN). Falling back to (layer_idx=0, token_pos=0)."
        )
        return sweep_auroc, 0, 0

    best_flat = int(np.nanargmax(sweep_auroc))
    best_layer_idx = best_flat // T
    best_token_pos = best_flat % T

    # Why: a length mismatch between relevant_layers (config-derived names) and
    # L=full_cache.shape[1] (cache dimension) is technically an upstream bug,
    # but it crashed sweep_locations *after* the sweep finished and the result
    # was valid — turning a logging issue into 37 lost cell completions. Guard
    # the layer-name lookup so the function can still return its result.
    if best_layer_idx < len(relevant_layers):
        best_layer_name = str(relevant_layers[best_layer_idx])
    else:
        best_layer_name = (
            f"<unknown — len(relevant_layers)={len(relevant_layers)} < L={L}>"
        )
    logger.info(
        f"sweep_locations: selected layer_idx={best_layer_idx} "
        f"(layer={best_layer_name}), "
        f"token_pos={best_token_pos}, "
        f"dev AUROC={sweep_auroc[best_layer_idx, best_token_pos]:.4f}"
    )

    return sweep_auroc, best_layer_idx, best_token_pos


def train_final_probe(
    full_cache: np.ndarray,
    row_indices: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    token_pos: int,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a LogisticRegression probe on the full training set at one location.

    Reads only the (N_split, H) column at (layer_idx, token_pos).
    """
    y = np.asarray(labels, dtype=np.int32)

    logger.info(
        f"train_final_probe: materialising ({len(y)}, {full_cache.shape[3]}) "
        f"column at layer_idx={layer_idx}, token_pos={token_pos} "
        f"({len(y) * full_cache.shape[3] * 2 / 1024**2:.1f} MB) ..."
    )
    X = _take_column(full_cache, row_indices, layer_idx, token_pos).astype(np.float32)

    logger.info(f"train_final_probe: training on {len(y)} samples ...")
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(X, y)
    logger.info("train_final_probe: done.")
    return clf


def eval_probe(
    probe: LogisticRegression,
    full_cache: np.ndarray,
    row_indices: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    token_pos: int,
    outlier_class: int = 1,
) -> tuple[float, np.ndarray]:
    """Evaluate a fitted probe on the test set at one (layer, token) location.

    Reads only the (N_test, H) column.
    """
    y = np.asarray(labels, dtype=np.int32)
    X = _take_column(full_cache, row_indices, layer_idx, token_pos).astype(np.float32)

    classes = list(probe.classes_)
    col = classes.index(outlier_class) if outlier_class in classes else 1

    proba = probe.predict_proba(X)
    scores = proba[:, col].astype(np.float32)

    if len(np.unique(y)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y, scores))

    logger.info(f"eval_probe: test AUROC = {auroc:.4f}")
    return auroc, scores
