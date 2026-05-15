"""Semantic Entropy Probes (SEP-SE): Ridge probes on cached greedy activations.

Implements SEP-SE per Kossen et al. 2024: a single-pass linear probe that
predicts length-normalized semantic entropy from one greedy hidden state, then
is scored as a hallucination detector via AUROC against binary halu labels on
the test split. SEP-binary (a logistic probe on halu labels directly) is not
in the original paper and is intentionally omitted.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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

    features: List[np.ndarray] = []
    valid_indices: List[int] = []

    for _, row in gendf.iterrows():
        key = row["prompt_hash"]
        act = zarr_logger.get_layer_activation(key, layer_idx, sequence_mode="response")
        if act is None:
            continue
        # act: (1, seq_len, hidden) — take last token
        last_tok = act[0, -1, :].numpy().astype(np.float32)
        features.append(last_tok)
        valid_indices.append(int(row["row_idx"]))

    if hasattr(zarr_logger, "close"):
        zarr_logger.close()

    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int64)

    return np.array(features, dtype=np.float32), np.array(valid_indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Probe fit / eval
# ---------------------------------------------------------------------------

def fit_sep_se(X_train: np.ndarray, y_se: np.ndarray, alpha: float = 1.0) -> Pipeline:
    """Ridge probe predicting length-normalized SE."""
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X_train, y_se)
    return pipe


def eval_auroc(pipe: Pipeline, X: np.ndarray, y_binary: np.ndarray) -> float:
    """AUROC of the probe's regression score against binary hallucination labels."""
    scores = pipe.predict(X)
    return float(roc_auc_score(y_binary.astype(int), scores))


def _load_se_labels(se_labels_path: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with open(se_labels_path) as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["row_idx"])] = float(rec["length_normalized_se"])
    return out


# ---------------------------------------------------------------------------
# Layer sweep (SEP-SE, val AUROC on inner split of train subset)
# ---------------------------------------------------------------------------

def layer_sweep_sep_se(
    generation_jsonl_train: str,
    zarr_path_train: str,
    eval_json_train: str,
    se_labels_train_path: str,
    train_subset_indices: List[int],
    layers: List[int],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Dict[int, float]:
    """Fit SEP-SE per layer on 80% of the train subset; report val AUROC on 20%.

    Layer is selected by AUROC against binary halu labels on a held-out fold of
    the train subset (the test split is never touched here).
    """
    with open(eval_json_train) as f:
        train_labels_full = np.array(json.load(f)["halu_test_res"], dtype=bool)
    se_by_row = _load_se_labels(se_labels_train_path)

    results: Dict[int, float] = {}
    for layer in layers:
        print(f"  Layer {layer}...", end=" ", flush=True)
        X, valid_idx = extract_last_token_features(
            generation_jsonl_train,
            zarr_path_train,
            layer,
            row_indices=train_subset_indices,
        )
        if len(X) == 0:
            print("no activations — skipped")
            continue

        keep = np.array([r in se_by_row for r in valid_idx], dtype=bool)
        X = X[keep]
        valid_idx = valid_idx[keep]
        if len(X) == 0:
            print("no SE-labelled rows — skipped")
            continue

        y_se = np.array([se_by_row[int(r)] for r in valid_idx], dtype=np.float32)
        y_bin = train_labels_full[valid_idx]

        try:
            X_tr, X_va, ys_tr, _ys_va, _yb_tr, yb_va = train_test_split(
                X, y_se, y_bin, test_size=val_frac, stratify=y_bin, random_state=seed
            )
        except ValueError:
            X_tr, X_va, ys_tr, _ys_va, _yb_tr, yb_va = train_test_split(
                X, y_se, y_bin, test_size=val_frac, random_state=seed
            )

        pipe = fit_sep_se(X_tr, ys_tr)
        try:
            auroc = eval_auroc(pipe, X_va, yb_va)
        except ValueError:
            auroc = float("nan")
        results[layer] = auroc
        print(f"val AUROC={auroc:.4f}  (n_train={len(X_tr)}, n_val={len(X_va)})")

    return results


# ---------------------------------------------------------------------------
# Full SEP-SE run
# ---------------------------------------------------------------------------

def run_sep(
    dataset: str,
    model_id: str,
    layer_idx: int,
    train_subset_indices: List[int],
    se_labels_train_path: str,
    output_path: str,
    test_subset_indices: Optional[List[int]] = None,
    alpha: float = 1.0,
) -> dict:
    """Fit SEP-SE on the 5k train subset; evaluate AUROC on test split.

    Args:
        test_subset_indices: Restrict test extraction to these row indices
            (used for the searchqa 10k cap; None = all test rows).
        alpha: Ridge regularization strength.
    """
    from tasks.sampling_baselines.paths import (
        eval_results_json,
        generation_jsonl,
        zarr_path as get_zarr_path,
    )

    gen_train = str(generation_jsonl(dataset, model_id, "train"))
    gen_test = str(generation_jsonl(dataset, model_id, "test"))
    eval_test = str(eval_results_json(dataset, model_id, "test"))
    zarr_train = str(get_zarr_path(dataset, model_id, "train"))
    zarr_test = str(get_zarr_path(dataset, model_id, "test"))

    with open(eval_test) as f:
        test_labels = np.array(json.load(f)["halu_test_res"], dtype=bool)
    se_by_row = _load_se_labels(se_labels_train_path)

    # --- Train features (SE-labelled subset) ---
    print(
        f"Extracting SEP-SE train features "
        f"({len(train_subset_indices)} rows requested, layer {layer_idx})..."
    )
    X_train, train_valid_idx = extract_last_token_features(
        gen_train, zarr_train, layer_idx, row_indices=train_subset_indices
    )
    keep = np.array([int(r) in se_by_row for r in train_valid_idx], dtype=bool)
    X_train = X_train[keep]
    train_valid_idx = train_valid_idx[keep]
    y_se = np.array([se_by_row[int(r)] for r in train_valid_idx], dtype=np.float32)

    if len(X_train) == 0:
        raise RuntimeError(
            f"No SE-labelled rows with cached activations for {dataset}/{model_id} train"
        )

    # --- Test features ---
    n_test_req = len(test_subset_indices) if test_subset_indices is not None else None
    print(
        f"Extracting SEP-SE test features (layer {layer_idx}"
        + (f", {n_test_req} subset rows" if n_test_req is not None else "")
        + ")..."
    )
    X_test, test_valid_idx = extract_last_token_features(
        gen_test, zarr_test, layer_idx, row_indices=test_subset_indices
    )
    if len(X_test) == 0:
        raise RuntimeError(
            f"No cached test activations for {dataset}/{model_id} at layer {layer_idx}"
        )
    y_test = test_labels[test_valid_idx]

    pipe = fit_sep_se(X_train, y_se, alpha=alpha)
    auroc = eval_auroc(pipe, X_test, y_test)

    result = {
        "dataset": dataset,
        "model": model_id,
        "layer": layer_idx,
        "alpha": alpha,
        "sep_se_auroc": auroc,
        "train_size_sep_se": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_subset_size": n_test_req,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"SEP-SE AUROC: {auroc:.4f} (n_train={len(X_train)}, n_test={len(X_test)})")
    print(f"SEP results → {output_path}")

    return result
