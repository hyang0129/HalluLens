"""Semantic Entropy Probes (SEP): linear probes on cached greedy activations."""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_last_token_features(
    generation_jsonl: str,
    zarr_path: str,
    layer_idx: int,
    row_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract last-response-token hidden state at layer_idx from greedy zarr store.

    Args:
        generation_jsonl: Path to generation.jsonl for prompt→hash mapping.
        zarr_path: Path to zarr activation store.
        layer_idx: Layer to read.
        row_indices: Subset of row indices to extract (None = all).

    Returns:
        (X, valid_row_indices) where X has shape (n_valid, hidden_dim).
        Rows with missing activations are silently dropped.
    """
    from activation_logging.zarr_activations_logger import ZarrActivationsLogger

    gendf = pd.read_json(generation_jsonl, lines=True)
    gendf["row_idx"] = gendf.index
    gendf["prompt_hash"] = gendf["prompt"].apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )

    if row_indices is not None:
        gendf = gendf[gendf["row_idx"].isin(set(row_indices))].reset_index(drop=True)

    zarr_logger = ZarrActivationsLogger(zarr_path=str(zarr_path), read_only=True, verbose=False)

    features = []
    valid_indices = []

    for _, row in gendf.iterrows():
        key = row["prompt_hash"]
        act = zarr_logger.get_layer_activation(key, layer_idx, sequence_mode="response")
        if act is None:
            continue
        # act: (1, seq_len, hidden) — take last token
        last_tok = act[0, -1, :].numpy().astype(np.float32)
        features.append(last_tok)
        valid_indices.append(int(row["row_idx"]))

    zarr_logger.close() if hasattr(zarr_logger, "close") else None

    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    return np.array(features, dtype=np.float32), np.array(valid_indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Probe fitting
# ---------------------------------------------------------------------------

def fit_sep_se(X_train: np.ndarray, y_se: np.ndarray, alpha: float = 1.0) -> Pipeline:
    """Ridge regression probe predicting length-normalized SE."""
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X_train, y_se)
    return pipe


def fit_sep_binary(X_train: np.ndarray, y_binary: np.ndarray) -> Pipeline:
    """Logistic regression probe predicting binary hallucination label."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    pipe.fit(X_train, y_binary.astype(int))
    return pipe


def eval_auroc(pipe: Pipeline, X_test: np.ndarray, y_binary: np.ndarray, is_regression: bool) -> float:
    """AUROC of a fitted probe against binary hallucination labels."""
    if is_regression:
        scores = pipe.predict(X_test)
    else:
        scores = pipe.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_binary.astype(int), scores))


# ---------------------------------------------------------------------------
# Layer sweep (for Qwen3-8B)
# ---------------------------------------------------------------------------

def layer_sweep_sep_binary(
    generation_jsonl: str,
    zarr_path: str,
    eval_json: str,
    layers: List[int],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Dict[int, float]:
    """Fit SEP-binary for each layer; return {layer: val_auroc}."""
    from sklearn.model_selection import train_test_split

    with open(eval_json) as f:
        eval_data = json.load(f)
    all_labels = np.array(eval_data["halu_test_res"], dtype=bool)

    results: Dict[int, float] = {}
    for layer in layers:
        print(f"  Layer {layer}...", end=" ")
        X, valid_idx = extract_last_token_features(generation_jsonl, zarr_path, layer)
        if len(X) == 0:
            print("no activations — skipped")
            continue

        y = all_labels[valid_idx]

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_frac, stratify=y, random_state=seed
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_frac, random_state=seed
            )

        pipe = fit_sep_binary(X_train, y_train)
        auroc = eval_auroc(pipe, X_val, y_val, is_regression=False)
        results[layer] = auroc
        print(f"AUROC={auroc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Full SEP run
# ---------------------------------------------------------------------------

def run_sep(
    dataset: str,
    model_id: str,
    layer_idx: int,
    train_subset_indices: Optional[List[int]],
    se_labels_train_path: Optional[str],
    output_path: str,
) -> dict:
    """Fit SEP-SE and SEP-binary probes; return result dict.

    Args:
        train_subset_indices: Row indices for SEP-SE training (5k stratified subset).
            If None, SEP-SE is skipped (e.g. MMLU).
        se_labels_train_path: Path to se_labels.jsonl for train split.
            Required when train_subset_indices is not None.
    """
    from tasks.sampling_baselines.paths import (
        generation_jsonl,
        eval_results_json,
        zarr_path as get_zarr_path,
    )

    gen_train = str(generation_jsonl(dataset, model_id, "train"))
    gen_test = str(generation_jsonl(dataset, model_id, "test"))
    eval_train = str(eval_results_json(dataset, model_id, "train"))
    eval_test = str(eval_results_json(dataset, model_id, "test"))
    zarr_train = str(get_zarr_path(dataset, model_id, "train"))
    zarr_test = str(get_zarr_path(dataset, model_id, "test"))

    # Load binary labels
    with open(eval_train) as f:
        train_eval = json.load(f)
    with open(eval_test) as f:
        test_eval = json.load(f)

    train_labels_full = np.array(train_eval["halu_test_res"], dtype=bool)
    test_labels = np.array(test_eval["halu_test_res"], dtype=bool)

    result = {
        "dataset": dataset,
        "model": model_id,
        "layer": layer_idx,
    }

    # --- SEP-binary (full train split) ---
    print(f"Extracting test features (layer {layer_idx})...")
    X_test, test_valid_idx = extract_last_token_features(gen_test, zarr_test, layer_idx)
    y_test = test_labels[test_valid_idx]

    print(f"Extracting train features for SEP-binary (layer {layer_idx})...")
    X_train_full, train_valid_idx_full = extract_last_token_features(gen_train, zarr_train, layer_idx)
    y_train_full = train_labels_full[train_valid_idx_full]

    pipe_binary = fit_sep_binary(X_train_full, y_train_full)
    auroc_binary = eval_auroc(pipe_binary, X_test, y_test, is_regression=False)

    result["sep_binary_auroc"] = auroc_binary
    result["train_size_sep_binary"] = len(X_train_full)
    print(f"SEP-binary AUROC: {auroc_binary:.4f} (n_train={len(X_train_full)})")

    # --- SEP-SE (5k stratified subset) ---
    if train_subset_indices is not None and se_labels_train_path is not None:
        se_by_row = _load_se_labels(se_labels_train_path)
        # Only keep rows present in both SE labels and valid activations
        subset_set = set(train_subset_indices)

        print(f"Extracting subset features for SEP-SE ({len(subset_set)} rows)...")
        X_subset, subset_valid_idx = extract_last_token_features(
            gen_train, zarr_train, layer_idx, row_indices=train_subset_indices
        )

        # Align SE labels with extracted features
        y_se = np.array(
            [se_by_row[r]["length_normalized_se"] for r in subset_valid_idx if r in se_by_row],
            dtype=np.float32,
        )
        X_se = X_subset[[i for i, r in enumerate(subset_valid_idx) if r in se_by_row]]
        se_test_valid = np.array(
            [r in se_by_row for r in subset_valid_idx], dtype=bool
        )
        kept = se_test_valid.sum()

        if kept > 0:
            pipe_se = fit_sep_se(X_se, y_se)
            auroc_se = eval_auroc(pipe_se, X_test, y_test, is_regression=True)
            result["sep_se_auroc"] = auroc_se
            result["train_size_sep_se"] = int(kept)
            print(f"SEP-SE AUROC: {auroc_se:.4f} (n_train={kept})")
        else:
            result["sep_se_auroc"] = None
            result["train_size_sep_se"] = 0
    else:
        result["sep_se_auroc"] = None
        result["train_size_sep_se"] = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"SEP results → {output_path}")

    return result


def _load_se_labels(se_labels_path: str) -> dict:
    by_row = {}
    with open(se_labels_path) as f:
        for line in f:
            rec = json.loads(line)
            by_row[rec["row_idx"]] = rec
    return by_row
