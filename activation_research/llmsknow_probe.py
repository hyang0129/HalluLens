"""LLMsKnow Probe Baseline.

Implements a location-sweep sklearn logistic regression probe that:
1. Finds the best (layer, token_position) pair on a dev subset.
2. Trains a final classifier on the full training set at that location.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _get_split_cache(
    ds,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (cache, labels) from a PreloadedActivationDataset.

    Handles both the direct-indexing case (cache is already split-sized) and
    the memmap-indirection case (cache is full-sized, _row_indices addresses
    the split rows).

    Returns:
        cache: float16 numpy array of shape (N_split, L, T, H)
        labels: int numpy array of shape (N_split,)
    """
    full_cache: np.ndarray = ds.cache  # may be (N_total, L, T, H) or (N_split, L, T, H)
    labels: np.ndarray = np.asarray(ds.labels, dtype=np.int32)

    row_indices = getattr(ds, "_row_indices", None)
    if row_indices is not None:
        # Memmap path: cache is full; index down to split rows
        cache = full_cache[row_indices]  # (N_split, L, T, H)
    else:
        cache = full_cache  # already split-sized

    return cache, labels


def sweep_locations(
    train_cache: np.ndarray,
    train_labels: np.ndarray,
    relevant_layers: list[int],
    dev_size: int = 1000,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> tuple[np.ndarray, int, int]:
    """Sweep all (layer_idx, token_pos) pairs on a dev subset.

    Args:
        train_cache: Float16 array of shape (N, L, T, H).
        train_labels: Int/bool array of shape (N,).
        relevant_layers: List of model layer indices (length L), e.g. [14, ..., 29].
        dev_size: Number of samples to use in the dev subset.
        seed: Random seed for stratified sampling and LogisticRegression.
        C: Regularisation strength for LogisticRegression.
        max_iter: Max iterations for LogisticRegression solver.

    Returns:
        sweep_auroc_matrix: Float array of shape (L, T), AUROC per pair (NaN for
            skipped/padded positions).
        best_layer_idx: Index into the L dimension (positional, not model layer id).
        best_token_pos: Index into the T dimension.
    """
    N, L, T, H = train_cache.shape
    labels = np.asarray(train_labels, dtype=np.int32)

    # Stratified dev subset
    actual_dev = min(dev_size, N)
    if actual_dev < N:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=actual_dev, random_state=seed)
        _, dev_idx = next(splitter.split(np.zeros(N), labels))
    else:
        dev_idx = np.arange(N)

    dev_cache = train_cache[dev_idx].astype(np.float32)  # (D, L, T, H)
    dev_labels = labels[dev_idx]

    unique_classes = np.unique(dev_labels)
    if len(unique_classes) < 2:
        logger.warning(
            "sweep_locations: dev subset has only one class — "
            "AUROC is undefined. Falling back to (0, 0)."
        )
        return np.full((L, T), np.nan, dtype=np.float32), 0, 0

    logger.info(
        f"sweep_locations: sweeping {L} layers × {T} token positions "
        f"= {L * T} pairs on {len(dev_idx)}-sample dev subset ..."
    )

    sweep_auroc = np.full((L, T), np.nan, dtype=np.float32)

    for li in range(L):
        for ti in range(T):
            # Skip padded noise positions (activations are constant across samples)
            col = dev_cache[:, li, ti, :]  # (D, H)
            if np.std(col) < 1e-6:
                continue  # leave as NaN

            clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
            clf.fit(col, dev_labels)
            scores = clf.predict_proba(col)[:, 1]

            if len(np.unique(dev_labels)) < 2:
                auroc = float("nan")
            else:
                auroc = roc_auc_score(dev_labels, scores)

            sweep_auroc[li, ti] = float(auroc)

    # Select best pair
    if np.all(np.isnan(sweep_auroc)):
        logger.warning(
            "sweep_locations: no valid (layer, token) pair found "
            "(all positions are NaN). Falling back to (layer_idx=0, token_pos=0)."
        )
        return sweep_auroc, 0, 0

    best_flat = int(np.nanargmax(sweep_auroc))
    best_layer_idx = best_flat // T
    best_token_pos = best_flat % T

    logger.info(
        f"sweep_locations: selected layer_idx={best_layer_idx} "
        f"(layer={relevant_layers[best_layer_idx]}), "
        f"token_pos={best_token_pos}, "
        f"dev AUROC={sweep_auroc[best_layer_idx, best_token_pos]:.4f}"
    )

    return sweep_auroc, best_layer_idx, best_token_pos


def train_final_probe(
    train_cache: np.ndarray,
    train_labels: np.ndarray,
    layer_idx: int,
    token_pos: int,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a LogisticRegression probe on the full training set at one location.

    Args:
        train_cache: Float16 array of shape (N, L, T, H).
        train_labels: Int array of shape (N,).
        layer_idx: Index into the L dimension.
        token_pos: Index into the T dimension.
        seed: Random seed for LogisticRegression.
        C: Regularisation strength.
        max_iter: Max solver iterations.

    Returns:
        Fitted sklearn LogisticRegression.
    """
    X = train_cache[:, layer_idx, token_pos, :].astype(np.float32)  # (N, H)
    y = np.asarray(train_labels, dtype=np.int32)

    logger.info(
        f"train_final_probe: training on {len(y)} samples at "
        f"layer_idx={layer_idx}, token_pos={token_pos} ..."
    )

    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(X, y)

    logger.info("train_final_probe: done.")
    return clf


def eval_probe(
    probe: LogisticRegression,
    test_cache: np.ndarray,
    test_labels: np.ndarray,
    layer_idx: int,
    token_pos: int,
    outlier_class: int = 1,
) -> tuple[float, np.ndarray]:
    """Evaluate a fitted probe on the test set.

    Args:
        probe: Fitted sklearn LogisticRegression.
        test_cache: Float16 array of shape (N_test, L, T, H).
        test_labels: Int array of shape (N_test,).
        layer_idx: Index into the L dimension.
        token_pos: Index into the T dimension.
        outlier_class: Class index treated as the positive/hallucination class.

    Returns:
        auroc: AUROC score (float).
        scores: 1-D float32 array of shape (N_test,), probability of outlier_class.
    """
    X = test_cache[:, layer_idx, token_pos, :].astype(np.float32)  # (N_test, H)
    y = np.asarray(test_labels, dtype=np.int32)

    # Find the column corresponding to outlier_class in probe.classes_
    classes = list(probe.classes_)
    if outlier_class in classes:
        col = classes.index(outlier_class)
    else:
        col = 1  # fallback

    proba = probe.predict_proba(X)  # (N_test, n_classes)
    scores = proba[:, col].astype(np.float32)

    if len(np.unique(y)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y, scores))

    logger.info(f"eval_probe: test AUROC = {auroc:.4f}")
    return auroc, scores
