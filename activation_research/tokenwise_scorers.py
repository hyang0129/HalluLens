"""Validation-selected scorers for token-wise hallucination embeddings.

The public workflow is intentionally split into three phases:

``prepare_validation_run``
    Fit every predeclared candidate on one run's *training* embeddings and
    score only that run's validation embeddings.  Test arrays are not opened.

``select_global_scorer``
    Select one scorer for an entire training recipe using the mean validation
    AUROC across datasets (dataset means are computed before the macro mean).

``finalize_locked_scorer``
    Refit and evaluate only the locked scorer on test embeddings.  A single
    validation-fitted Platt map and empirical-CDF map are shared by every run
    in the lock so thresholds are not selected per dataset or on test.

This module is deliberately independent from the training evaluator.  It
consumes the split-isolated artifacts described by
``activation_research.evaluation.write_embedding_dump_manifest``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = 1
DEFAULT_DATASETS = (
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
)

# Order is the deterministic tie breaker.  Cosine is retained as a geometry
# diagnostic but cannot win the deployment-scorer selection.
PRIMARY_SCORERS = (
    "euclidean_all_k50",
    "euclidean_all_k1000",
    "euclidean_truthful_k50",
    "prior_corrected_vote_k50",
    "truthful_oas_mahalanobis",
    "shrinkage_lda",
    "two_centroid_margin",
    "fixed_logistic",
    "standardized_balanced_logistic",
)
DIAGNOSTIC_SCORERS = ("normalized_cosine_k50",)
ALL_SCORERS = (*PRIMARY_SCORERS, *DIAGNOSTIC_SCORERS)


@dataclass(frozen=True)
class EmbeddingSplit:
    """One validated embedding surface."""

    name: str
    z: np.ndarray
    labels: np.ndarray
    hashkeys: tuple[str, ...]
    stable_ids: tuple[str, ...]
    meta: Mapping[str, Any]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _flatten_token_zero(z: np.ndarray, *, split_name: str) -> np.ndarray:
    if z.ndim == 3:
        if z.shape[1] != 1:
            raise ValueError(
                f"{split_name}: scorer study requires one token-zero view; "
                f"got shape {tuple(z.shape)}"
            )
        z = z[:, 0, :]
    elif z.ndim != 2:
        raise ValueError(
            f"{split_name}: expected (N, 1, D) or (N, D), got {tuple(z.shape)}"
        )
    if not np.isfinite(z).all():
        raise ValueError(f"{split_name}: embeddings contain non-finite values")
    return np.asarray(z, dtype=np.float32)


def load_embedding_split(embedding_dir: Path | str, split_name: str) -> EmbeddingSplit:
    """Load and validate one surface named by the run-level manifest."""
    embedding_dir = Path(embedding_dir)
    manifest_path = embedding_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError(f"{manifest_path}: incomplete train/val/test manifest")
    if split_name not in manifest["splits"]:
        raise ValueError(f"{manifest_path}: missing split {split_name!r}")

    meta_path = embedding_dir / manifest["splits"][split_name]["meta"]
    meta = _read_json(meta_path)
    if meta.get("split_name") != split_name:
        raise ValueError(f"{meta_path}: split_name mismatch")
    files = meta.get("files", {})
    required = {"z", "labels", "hashkeys", "stable_ids"}
    if not required.issubset(files):
        raise ValueError(f"{meta_path}: missing files {sorted(required - set(files))}")

    z = np.load(embedding_dir / files["z"], mmap_mode="r")
    labels = np.asarray(np.load(embedding_dir / files["labels"]), dtype=np.int8)
    hashkeys = tuple(str(value) for value in _read_json(embedding_dir / files["hashkeys"]))
    stable_ids = tuple(str(value) for value in _read_json(embedding_dir / files["stable_ids"]))
    n = int(meta["n"])
    if len(z) != n or len(labels) != n or len(hashkeys) != n or len(stable_ids) != n:
        raise ValueError(f"{meta_path}: inconsistent surface lengths")
    if len(set(stable_ids)) != n:
        raise ValueError(f"{meta_path}: stable IDs are not unique")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{meta_path}: labels must be binary hallucination labels")
    return EmbeddingSplit(
        name=split_name,
        z=_flatten_token_zero(z, split_name=split_name),
        labels=labels,
        hashkeys=hashkeys,
        stable_ids=stable_ids,
        meta=meta,
    )


def _effective_k(requested: int, n_reference: int) -> int:
    if n_reference <= 0:
        raise ValueError("nearest-neighbor reference bank is empty")
    return min(int(requested), int(n_reference))


def _neighbor_query(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    effective_k = _effective_k(k, len(reference))
    neighbors = NearestNeighbors(n_neighbors=effective_k, metric=metric, n_jobs=-1)
    neighbors.fit(reference)
    distances, indices = neighbors.kneighbors(target)
    return (
        np.asarray(distances, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        effective_k,
    )


def _prior_corrected_vote(neighbor_labels: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Return a class-balanced neighbor vote with a fixed 50/50 target prior.

    Each neighbor contributes the inverse frequency of its class in the train
    bank.  This removes the train-bank class prior without estimating anything
    from validation or test labels.
    """
    n_positive = int(np.sum(train_labels == 1))
    n_negative = int(np.sum(train_labels == 0))
    if n_positive == 0 or n_negative == 0:
        raise ValueError("prior-corrected vote requires both train classes")
    positive_mass = np.sum(neighbor_labels == 1, axis=1) / float(n_positive)
    negative_mass = np.sum(neighbor_labels == 0, axis=1) / float(n_negative)
    denominator = positive_mass + negative_mass
    return np.divide(
        positive_mass,
        denominator,
        out=np.full_like(positive_mass, 0.5, dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)


def _fit_parametric_scorers(train_z: np.ndarray, train_y: np.ndarray) -> dict[str, dict[str, Any]]:
    truth_z = train_z[train_y == 0]
    hallu_z = train_z[train_y == 1]
    if not len(truth_z) or not len(hallu_z):
        raise ValueError("scorer study requires both train classes")

    oas = OAS(store_precision=True).fit(truth_z)
    lda = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", store_covariance=True
    ).fit(train_z, train_y)
    fixed_logistic = LogisticRegression(
        C=1.0, max_iter=1000, class_weight=None, random_state=0
    ).fit(train_z, train_y)
    scaler = StandardScaler().fit(train_z)
    balanced_logistic = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=0
    ).fit(scaler.transform(train_z), train_y)

    return {
        "truthful_oas_mahalanobis": {
            "mean": np.asarray(oas.location_, dtype=np.float64),
            "precision": np.asarray(oas.precision_, dtype=np.float64),
            "shrinkage": float(oas.shrinkage_),
        },
        "shrinkage_lda": {
            "coef": np.asarray(lda.coef_[0], dtype=np.float64),
            "intercept": float(lda.intercept_[0]),
            "means": np.asarray(lda.means_, dtype=np.float64),
            "covariance": np.asarray(lda.covariance_, dtype=np.float64),
        },
        "two_centroid_margin": {
            "truth_centroid": np.asarray(truth_z.mean(axis=0), dtype=np.float64),
            "hallu_centroid": np.asarray(hallu_z.mean(axis=0), dtype=np.float64),
        },
        "fixed_logistic": {
            "coef": np.asarray(fixed_logistic.coef_[0], dtype=np.float64),
            "intercept": float(fixed_logistic.intercept_[0]),
        },
        "standardized_balanced_logistic": {
            "mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scale": np.asarray(scaler.scale_, dtype=np.float64),
            "coef": np.asarray(balanced_logistic.coef_[0], dtype=np.float64),
            "intercept": float(balanced_logistic.intercept_[0]),
        },
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    nonnegative = values >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    out[~nonnegative] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32)


def _score_parametric(
    name: str, params: Mapping[str, Any], target_z: np.ndarray
) -> np.ndarray:
    target64 = np.asarray(target_z, dtype=np.float64)
    if name == "truthful_oas_mahalanobis":
        delta = target64 - params["mean"]
        return np.einsum("ij,jk,ik->i", delta, params["precision"], delta).astype(
            np.float32
        )
    if name == "shrinkage_lda":
        return (target64 @ params["coef"] + params["intercept"]).astype(np.float32)
    if name == "two_centroid_margin":
        d_truth = np.linalg.norm(target64 - params["truth_centroid"], axis=1)
        d_hallu = np.linalg.norm(target64 - params["hallu_centroid"], axis=1)
        return (d_truth - d_hallu).astype(np.float32)
    if name == "fixed_logistic":
        return _sigmoid(target64 @ params["coef"] + params["intercept"])
    if name == "standardized_balanced_logistic":
        standardized = (target64 - params["mean"]) / np.maximum(params["scale"], 1e-12)
        return _sigmoid(standardized @ params["coef"] + params["intercept"])
    raise KeyError(name)


def score_all_candidates(
    train_z: np.ndarray,
    train_y: np.ndarray,
    target_z: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray | int | str]],
    dict[str, dict[str, Any]],
]:
    """Fit the predeclared matrix on train and score one non-train surface."""
    train_z = np.asarray(train_z, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int8)
    target_z = np.asarray(target_z, dtype=np.float32)
    if set(np.unique(train_y)) != {0, 1}:
        raise ValueError("candidate matrix requires both train classes")

    scores: dict[str, np.ndarray] = {}
    neighbors: dict[str, dict[str, np.ndarray | int | str]] = {}

    all_dist, all_idx, all_k = _neighbor_query(
        train_z, target_z, k=1000, metric="euclidean"
    )
    k50 = min(50, all_k)
    scores["euclidean_all_k50"] = all_dist[:, :k50].mean(axis=1)
    scores["euclidean_all_k1000"] = all_dist.mean(axis=1)
    scores["prior_corrected_vote_k50"] = _prior_corrected_vote(
        train_y[all_idx[:, :k50]], train_y
    )
    neighbors["euclidean_all_bank"] = {
        "distances": all_dist,
        "indices": all_idx,
        "effective_k": all_k,
        "metric": "euclidean",
    }

    truth_indices = np.flatnonzero(train_y == 0)
    truth_dist, truth_local_idx, truth_k = _neighbor_query(
        train_z[truth_indices], target_z, k=50, metric="euclidean"
    )
    scores["euclidean_truthful_k50"] = truth_dist.mean(axis=1)
    neighbors["euclidean_truthful_bank"] = {
        "distances": truth_dist,
        "indices": truth_indices[truth_local_idx].astype(np.int32),
        "effective_k": truth_k,
        "metric": "euclidean",
    }

    train_norm = train_z / np.maximum(
        np.linalg.norm(train_z, axis=1, keepdims=True), 1e-12
    )
    target_norm = target_z / np.maximum(
        np.linalg.norm(target_z, axis=1, keepdims=True), 1e-12
    )
    cosine_dist, cosine_idx, cosine_k = _neighbor_query(
        train_norm, target_norm, k=50, metric="cosine"
    )
    scores["normalized_cosine_k50"] = cosine_dist.mean(axis=1)
    neighbors["normalized_cosine_all_bank"] = {
        "distances": cosine_dist,
        "indices": cosine_idx,
        "effective_k": cosine_k,
        "metric": "cosine_after_l2_normalization",
    }

    params = _fit_parametric_scorers(train_z, train_y)
    for name, fitted in params.items():
        scores[name] = _score_parametric(name, fitted, target_z)
    if set(scores) != set(ALL_SCORERS):
        raise RuntimeError(f"candidate implementation mismatch: {sorted(scores)}")
    return scores, neighbors, params


def score_locked_candidate(
    name: str,
    train_z: np.ndarray,
    train_y: np.ndarray,
    target_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None, dict[str, Any]]:
    """Fit and score exactly one locked candidate."""
    if name not in PRIMARY_SCORERS:
        raise ValueError(f"cannot finalize ineligible scorer {name!r}")
    train_z = np.asarray(train_z, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int8)
    target_z = np.asarray(target_z, dtype=np.float32)

    if name in {"euclidean_all_k50", "euclidean_all_k1000", "prior_corrected_vote_k50"}:
        requested = 1000 if name == "euclidean_all_k1000" else 50
        distances, indices, effective_k = _neighbor_query(
            train_z, target_z, k=requested, metric="euclidean"
        )
        if name == "prior_corrected_vote_k50":
            scores = _prior_corrected_vote(train_y[indices], train_y)
        else:
            scores = distances.mean(axis=1)
        return scores, {
            "distances": distances,
            "indices": indices,
            "effective_k": effective_k,
            "metric": "euclidean",
        }, {"reference_bank": "all", "requested_k": requested}

    if name == "euclidean_truthful_k50":
        truth_indices = np.flatnonzero(train_y == 0)
        distances, local_indices, effective_k = _neighbor_query(
            train_z[truth_indices], target_z, k=50, metric="euclidean"
        )
        return distances.mean(axis=1), {
            "distances": distances,
            "indices": truth_indices[local_indices].astype(np.int32),
            "effective_k": effective_k,
            "metric": "euclidean",
        }, {"reference_bank": "truthful_only", "requested_k": 50}

    params = _fit_parametric_scorers(train_z, train_y)[name]
    return _score_parametric(name, params, target_z), None, params


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Paper-facing binary ranking metrics, always oriented high=hallucination."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError("labels and scores are not aligned")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("ranking metrics require both classes")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)

    def tpr_at_fpr(limit: float) -> float:
        eligible = tpr[fpr <= limit + 1e-12]
        return float(eligible.max()) if len(eligible) else 0.0

    normalized_ap = (
        (auprc - prevalence) / (1.0 - prevalence)
        if prevalence < 1.0
        else float("nan")
    )
    return {
        "n": int(len(labels)),
        "n_hallucinated": int(labels.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": auprc,
        "normalized_ap_gain": _json_number(normalized_ap),
        "tpr_at_fpr_0.05": tpr_at_fpr(0.05),
        "tpr_at_fpr_0.10": tpr_at_fpr(0.10),
    }


def _save_neighbors(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distances=np.asarray(payload["distances"], dtype=np.float32),
        indices=np.asarray(payload["indices"], dtype=np.int32),
        effective_k=np.asarray(int(payload["effective_k"]), dtype=np.int32),
        metric=np.asarray(str(payload["metric"])),
    )


def _save_params(directory: Path, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in params.items() if isinstance(value, np.ndarray)}
    scalars = {key: value for key, value in params.items() if key not in arrays}
    files: dict[str, Any] = {"metadata": scalars}
    if arrays:
        path = directory / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        files["arrays"] = path.name
    meta_path = directory / f"{name}.json"
    _write_json(meta_path, scalars)
    files["json"] = meta_path.name
    return files


def _write_scores_csv(
    path: Path,
    surface: EmbeddingSplit,
    scores: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stable_id", "prompt_hash", "label_halu", *scores]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(surface.labels)):
            row: dict[str, Any] = {
                "stable_id": surface.stable_ids[index],
                "prompt_hash": surface.hashkeys[index],
                "label_halu": int(surface.labels[index]),
            }
            row.update({name: float(values[index]) for name, values in scores.items()})
            writer.writerow(row)


def _source_fingerprints(embedding_dir: Path, splits: Iterable[str]) -> dict[str, str]:
    manifest = _read_json(embedding_dir / "manifest.json")
    paths = [embedding_dir / "manifest.json"]
    for split_name in splits:
        meta_path = embedding_dir / manifest["splits"][split_name]["meta"]
        meta = _read_json(meta_path)
        paths.append(meta_path)
        paths.extend(embedding_dir / value for value in meta["files"].values())
    return {path.name: _sha256(path) for path in paths}


def _hash_overlap(surfaces: Mapping[str, EmbeddingSplit]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        # Test is intentionally absent during prepare; report that explicitly.
        if left not in surfaces or right not in surfaces:
            out[f"{left}_{right}"] = {"audited": False, "count": None}
            continue
        overlap = sorted(set(surfaces[left].hashkeys) & set(surfaces[right].hashkeys))
        out[f"{left}_{right}"] = {
            "audited": True,
            "count": len(overlap),
            "sample": overlap[:20],
        }
    return out


def prepare_validation_run(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """Fit all candidates on train and score validation without reading test."""
    run_dir = Path(run_dir).resolve()
    embedding_dir = run_dir / "embeddings"
    output_dir = (
        run_dir / "scorer_study" if output_dir is None else Path(output_dir).resolve()
    )
    train = load_embedding_split(embedding_dir, "train")
    validation = load_embedding_split(embedding_dir, "val")
    manifest = _read_json(embedding_dir / "manifest.json")
    run_metadata = dict(manifest.get("run_metadata", {}))

    scores, neighbors, params = score_all_candidates(
        train.z, train.labels, validation.z
    )
    metrics = {
        name: {
            **ranking_metrics(validation.labels, values),
            "selection_eligible": name in PRIMARY_SCORERS,
        }
        for name, values in scores.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_scores_csv(output_dir / "validation_scores.csv", validation, scores)
    _write_json(output_dir / "validation_metrics.json", metrics)
    np.savez_compressed(
        output_dir / "embedding_norms_train_val.npz",
        train=np.linalg.norm(train.z, axis=1).astype(np.float32),
        val=np.linalg.norm(validation.z, axis=1).astype(np.float32),
    )
    neighbor_files = {}
    for name, payload in neighbors.items():
        path = output_dir / "validation_neighbors" / f"{name}.npz"
        _save_neighbors(path, payload)
        neighbor_files[name] = str(path.relative_to(output_dir))
    param_files = {
        name: _save_params(output_dir / "fitted_params", name, fitted)
        for name, fitted in params.items()
    }

    config_path = run_dir / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    method_config = config.get("method", {})
    training_recipe = (
        run_metadata.get("training_recipe")
        or method_config.get("training_recipe")
        or method_config.get("name")
        or run_metadata.get("method")
    )
    validation_reused = bool(
        method_config.get("training", {}).get("select_on_val", False)
    )
    validation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "validation_candidate_comparison",
        "test_artifacts_accessed": False,
        "selection_eligible_scorers": list(PRIMARY_SCORERS),
        "diagnostic_only_scorers": list(DIAGNOSTIC_SCORERS),
        "source": {
            "run_dir": str(run_dir),
            "embedding_dir": str(embedding_dir),
            "dataset": run_metadata.get("dataset"),
            "method": run_metadata.get("method"),
            "training_recipe": training_recipe,
            "training_seed": run_metadata.get("training_seed"),
            "split_seed": run_metadata.get("split_seed"),
            "embedding_surface": run_metadata.get("embedding_surface"),
            "fingerprints": _source_fingerprints(
                embedding_dir, ("train", "val")
            ),
        },
        "validation_policy": {
            "fit_surface": "train",
            "selection_surface": "val",
            "final_surface": "test_after_global_lock_only",
            "validation_reused_for_checkpoint_and_scorer_selection": validation_reused,
            "exploratory_due_to_validation_reuse": validation_reused,
        },
        "hash_overlap_audit": _hash_overlap({"train": train, "val": validation}),
        "artifacts": {
            "metrics": "validation_metrics.json",
            "scores": "validation_scores.csv",
            "embedding_norms": "embedding_norms_train_val.npz",
            "neighbors": neighbor_files,
            "fitted_params": param_files,
        },
    }
    output_path = output_dir / "validation_manifest.json"
    _write_json(output_path, validation_manifest)
    return output_path


def _load_validation_rows(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(manifest_path)
    if manifest.get("phase") != "validation_candidate_comparison":
        raise ValueError(f"{manifest_path}: not a validation candidate manifest")
    if manifest.get("test_artifacts_accessed") is not False:
        raise ValueError(f"{manifest_path}: test isolation is not certified")
    metrics_path = manifest_path.parent / manifest["artifacts"]["metrics"]
    return manifest, _read_json(metrics_path)


def select_global_scorer(
    validation_manifests: Sequence[Path | str],
    *,
    training_recipe: str,
    output_path: Path | str,
    expected_datasets: Sequence[str] = DEFAULT_DATASETS,
) -> Path:
    """Lock one primary scorer from macro validation AUROC."""
    paths = [Path(path).resolve() for path in validation_manifests]
    if not paths:
        raise ValueError("global selection requires validation manifests")
    records: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    identities: set[tuple[str, int]] = set()
    for path in paths:
        manifest, metrics = _load_validation_rows(path)
        source = manifest["source"]
        if source.get("training_recipe") != training_recipe:
            raise ValueError(
                f"{path}: recipe {source.get('training_recipe')!r} does not match "
                f"{training_recipe!r}"
            )
        identity = (str(source.get("dataset")), int(source.get("training_seed")))
        if identity in identities:
            raise ValueError(f"duplicate dataset/seed selection input: {identity}")
        identities.add(identity)
        records.append((path, manifest, metrics))

    expected = tuple(expected_datasets)
    observed = {dataset for dataset, _ in identities}
    if observed != set(expected):
        raise ValueError(
            f"global selection requires datasets {sorted(expected)}, got {sorted(observed)}"
        )

    candidate_summary: dict[str, Any] = {}
    for scorer in PRIMARY_SCORERS:
        dataset_means = {}
        for dataset in expected:
            values = [
                float(metrics[scorer]["auroc"])
                for _, manifest, metrics in records
                if manifest["source"]["dataset"] == dataset
            ]
            if not values:
                raise ValueError(f"{scorer}: missing validation values for {dataset}")
            dataset_means[dataset] = float(np.mean(values))
        candidate_summary[scorer] = {
            "dataset_mean_auroc": dataset_means,
            "macro_validation_auroc": float(np.mean(list(dataset_means.values()))),
        }
    selected = max(
        PRIMARY_SCORERS,
        key=lambda name: (
            candidate_summary[name]["macro_validation_auroc"],
            -PRIMARY_SCORERS.index(name),
        ),
    )
    output_path = Path(output_path).resolve()
    lock = {
        "schema_version": SCHEMA_VERSION,
        "phase": "global_validation_lock",
        "training_recipe": training_recipe,
        "selection_metric": "five_dataset_macro_validation_auroc",
        "aggregation": "mean_seeds_within_dataset_then_mean_datasets",
        "expected_datasets": list(expected),
        "selected_scorer": selected,
        "test_evaluated": False,
        "exploratory_due_to_checkpoint_validation_reuse": any(
            record[1]["validation_policy"].get(
                "validation_reused_for_checkpoint_and_scorer_selection", False
            )
            for record in records
        ),
        "candidates": candidate_summary,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path, _, _ in records
        ],
    }
    _write_json(output_path, lock)
    return output_path


def _read_selected_validation_scores(
    manifest_path: Path, scorer: str
) -> tuple[np.ndarray, np.ndarray]:
    manifest = _read_json(manifest_path)
    score_path = manifest_path.parent / manifest["artifacts"]["scores"]
    labels: list[int] = []
    scores: list[float] = []
    with score_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if scorer not in (reader.fieldnames or []):
            raise ValueError(f"{score_path}: missing selected scorer {scorer}")
        for row in reader:
            labels.append(int(row["label_halu"]))
            scores.append(float(row[scorer]))
    return np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def _empirical_cdf(reference_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference_scores, dtype=np.float64))
    return (
        np.searchsorted(ordered, np.asarray(values), side="right")
        / float(len(ordered) + 1)
    ).astype(np.float32)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for index in range(bins):
        mask = which == index
        if mask.any():
            total += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(total)


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, max_iter=1000).fit(logits, labels)
    return {
        "brier": float(brier_score_loss(labels, clipped)),
        "nll": float(log_loss(labels, clipped, labels=[0, 1])),
        "ece_15_equal_width": _ece(labels, clipped, bins=15),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def finalize_locked_scorer(
    selection_path: Path | str,
    *,
    output_dir: Path | str,
) -> Path:
    """Evaluate only the validation-locked scorer on every test surface."""
    selection_path = Path(selection_path).resolve()
    lock = _read_json(selection_path)
    if lock.get("phase") != "global_validation_lock" or lock.get("test_evaluated") is not False:
        raise ValueError(f"{selection_path}: invalid or already-finalized selection lock")
    scorer = str(lock["selected_scorer"])
    if scorer not in PRIMARY_SCORERS:
        raise ValueError(f"{selection_path}: selected scorer is not eligible")

    validation_labels = []
    validation_scores = []
    input_paths = []
    for item in lock["inputs"]:
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"selection input changed after lock: {path}")
        labels, scores = _read_selected_validation_scores(path, scorer)
        validation_labels.append(labels)
        validation_scores.append(scores)
        input_paths.append(path)
    pooled_val_y = np.concatenate(validation_labels)
    pooled_val_scores = np.concatenate(validation_scores)
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(
        pooled_val_scores.reshape(-1, 1), pooled_val_y
    )
    quantile_thresholds = {
        "top_10pct": float(np.quantile(pooled_val_scores, 0.90)),
        "top_5pct": float(np.quantile(pooled_val_scores, 0.95)),
    }

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    dataset_metrics: dict[str, list[dict[str, Any]]] = {}
    for validation_manifest_path in input_paths:
        validation_manifest = _read_json(validation_manifest_path)
        source = validation_manifest["source"]
        run_dir = Path(source["run_dir"])
        embedding_dir = run_dir / "embeddings"
        train = load_embedding_split(embedding_dir, "train")
        test = load_embedding_split(embedding_dir, "test")
        raw_scores, neighbors, params = score_locked_candidate(
            scorer, train.z, train.labels, test.z
        )
        cdf_scores = _empirical_cdf(pooled_val_scores, raw_scores)
        platt_scores = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        metrics = ranking_metrics(test.labels, raw_scores)
        metrics["platt_calibration"] = _calibration_metrics(test.labels, platt_scores)
        for threshold_name, threshold in quantile_thresholds.items():
            alerts = raw_scores >= threshold
            positives = test.labels == 1
            negatives = ~positives
            metrics[f"{threshold_name}_test_tpr"] = float(
                alerts[positives].mean()
            )
            metrics[f"{threshold_name}_test_fpr"] = float(
                alerts[negatives].mean()
            )

        identity = f"{source['dataset']}__seed_{source['training_seed']}"
        run_output = output_dir / "runs" / identity
        _write_scores_csv(
            run_output / "test_scores.csv",
            test,
            {
                scorer: raw_scores,
                "validation_empirical_cdf": cdf_scores,
                "validation_platt_probability": platt_scores,
            },
        )
        np.savez_compressed(
            run_output / "embedding_norms_test.npz",
            test=np.linalg.norm(test.z, axis=1).astype(np.float32),
        )
        if neighbors is not None:
            _save_neighbors(run_output / "test_neighbors.npz", neighbors)
        fitted_files = _save_params(run_output / "fitted_params", scorer, params)
        hash_audit = _hash_overlap({"train": train, "test": test})
        result = {
            "dataset": source["dataset"],
            "training_seed": source["training_seed"],
            "split_seed": source["split_seed"],
            "scorer": scorer,
            "metrics": metrics,
            "hash_overlap_audit": hash_audit,
            "artifacts": {
                "scores": "test_scores.csv",
                "neighbors": "test_neighbors.npz" if neighbors is not None else None,
                "fitted_params": fitted_files,
                "embedding_norms": "embedding_norms_test.npz",
            },
        }
        _write_json(run_output / "test_metrics.json", result)
        per_run.append(result)
        dataset_metrics.setdefault(str(source["dataset"]), []).append(metrics)

    dataset_means = {
        dataset: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("auroc", "auprc", "normalized_ap_gain")
        }
        for dataset, rows in dataset_metrics.items()
    }
    finalization = {
        "schema_version": SCHEMA_VERSION,
        "phase": "locked_test_evaluation",
        "selection_lock": str(selection_path),
        "selection_lock_sha256": _sha256(selection_path),
        "training_recipe": lock["training_recipe"],
        "selected_scorer": scorer,
        "test_candidates_evaluated": [scorer],
        "validation_calibration": {
            "scope": "one_global_map_across_all_locked_validation_runs",
            "n": int(len(pooled_val_y)),
            "platt_C": 1.0,
            "platt_coef": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "raw_score_quantile_thresholds": quantile_thresholds,
        },
        "dataset_seed_results": per_run,
        "dataset_mean_metrics": dataset_means,
        "macro_test_metrics": {
            metric: float(np.mean([values[metric] for values in dataset_means.values()]))
            for metric in ("auroc", "auprc", "normalized_ap_gain")
        },
    }
    output_path = output_dir / "finalization_manifest.json"
    _write_json(output_path, finalization)
    return output_path
