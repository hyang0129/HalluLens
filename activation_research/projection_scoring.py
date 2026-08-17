"""Scoring-only diagnostics for a trained disposable contrastive projection.

Issue #153 evaluates the 512-dimensional deployment trunk, while SupCon trains
the model's normalized 128-dimensional projection.  This module projects the
already dumped trunk embeddings through the saved head and reuses the canonical
KNN and frozen-probe scorers.  No encoder retraining or activation I/O is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import frozen_linear_probe_stats, knn_ood_stats


def _load_projection_tensors(checkpoint_path: Path) -> tuple[torch.Tensor, ...]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch releases
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)

    def tensor_for(suffix: str) -> torch.Tensor:
        matches = [value for key, value in state.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(
                f"expected exactly one checkpoint tensor ending in {suffix!r}; "
                f"found {len(matches)}"
            )
        return matches[0].detach().float()

    return (
        tensor_for("projection_head.0.weight"),
        tensor_for("projection_head.0.bias"),
        tensor_for("projection_head.2.weight"),
        tensor_for("projection_head.2.bias"),
    )


def _load_single_view_dump(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError(
            f"projection scoring requires a one-view trunk dump (N, 1, D); "
            f"got {values.shape} from {path}"
        )
    return values[:, 0, :]


def _apply_projection(
    trunk: np.ndarray,
    tensors: tuple[torch.Tensor, ...],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    w1, b1, w2, b2 = (tensor.to(device) for tensor in tensors)
    output = np.empty((len(trunk), int(w2.shape[0])), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(trunk), int(batch_size)):
            stop = min(len(trunk), start + int(batch_size))
            # The source is a read-only memmap. Copy the small chunk before
            # wrapping it so PyTorch never receives a non-writable array.
            z = torch.as_tensor(
                np.asarray(trunk[start:stop]).copy(),
                dtype=torch.float32,
                device=device,
            )
            projected = F.linear(F.gelu(F.linear(z, w1, b1)), w2, b2)
            projected = F.normalize(projected, dim=-1)
            output[start:stop] = projected.cpu().numpy()
    return output


def _records(embeddings: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "z_views": torch.from_numpy(embeddings[index]).unsqueeze(0),
            "halu": int(labels[index]),
        }
        for index in range(len(labels))
    ]


def score_saved_projection(
    source_run_dir: str | Path,
    *,
    evaluation_cfg: dict,
    outlier_class: int = 1,
    sample_seed: int = 0,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score the saved projection head on dumped token-zero trunk embeddings."""
    source_run_dir = Path(source_run_dir)
    embeddings_dir = source_run_dir / "embeddings"
    checkpoint_path = source_run_dir / "artifacts" / "final_weights.pt"

    required = (
        checkpoint_path,
        embeddings_dir / "train_z.npy",
        embeddings_dir / "train_labels.npy",
        embeddings_dir / "test_z.npy",
        embeddings_dir / "test_labels.npy",
        embeddings_dir / "test_hashkeys.json",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"projection-scoring inputs missing or empty: {missing}")

    train_trunk = _load_single_view_dump(embeddings_dir / "train_z.npy")
    test_trunk = _load_single_view_dump(embeddings_dir / "test_z.npy")
    train_labels = np.asarray(np.load(embeddings_dir / "train_labels.npy"), dtype=np.int32)
    test_labels = np.asarray(np.load(embeddings_dir / "test_labels.npy"), dtype=np.int32)
    test_hashkeys = json.loads((embeddings_dir / "test_hashkeys.json").read_text())
    if len(train_trunk) != len(train_labels):
        raise ValueError("train projection dump and labels have different lengths")
    if len(test_trunk) != len(test_labels) or len(test_trunk) != len(test_hashkeys):
        raise ValueError("test projection dump, labels, and hashkeys have different lengths")

    projection_tensors = _load_projection_tensors(checkpoint_path)
    projection_batch_size = int(evaluation_cfg.get("projection_batch_size", 4096))
    train_projection = _apply_projection(
        train_trunk,
        projection_tensors,
        device=device,
        batch_size=projection_batch_size,
    )
    test_projection = _apply_projection(
        test_trunk,
        projection_tensors,
        device=device,
        batch_size=projection_batch_size,
    )
    train_records = _records(train_projection, train_labels)
    test_records = _records(test_projection, test_labels)

    knn_params = dict(evaluation_cfg.get("knn_params", {}))
    knn_params.update({"sample_seed": int(sample_seed), "include_per_sample": True})
    knn_stats = knn_ood_stats(
        train_records,
        test_records,
        outlier_class=int(outlier_class),
        **knn_params,
    )

    cosine_params = dict(evaluation_cfg.get("cosine_knn_params", knn_params))
    cosine_params.update(
        {
            "metric": "cosine",
            "l2_normalize": True,
            "sample_seed": int(sample_seed),
            "include_per_sample": True,
        }
    )
    cosine_stats_raw = knn_ood_stats(
        train_records,
        test_records,
        outlier_class=int(outlier_class),
        **cosine_params,
    )
    cosine_stats = {
        f"cosine_{key}": value for key, value in cosine_stats_raw.items()
    }

    probe_params = dict(evaluation_cfg.get("linear_probe_params", {}))
    probe_params.update({"sample_seed": int(sample_seed), "include_per_sample": True})
    probe_stats = frozen_linear_probe_stats(
        train_records,
        test_records,
        outlier_class=int(outlier_class),
        **probe_params,
    )

    knn_scores = np.asarray(knn_stats.pop("knn_scores"), dtype=np.float32)
    knn_stats.pop("knn_labels", None)
    cosine_scores = np.asarray(
        cosine_stats.pop("cosine_knn_scores"), dtype=np.float32
    )
    cosine_stats.pop("cosine_knn_labels", None)
    probe_scores = np.asarray(probe_stats.pop("linear_probe_scores"), dtype=np.float32)
    probe_stats.pop("linear_probe_labels", None)

    source_metrics_path = source_run_dir / "eval_metrics.json"
    source_metrics = (
        json.loads(source_metrics_path.read_text()) if source_metrics_path.is_file() else {}
    )
    metrics: dict[str, Any] = {
        "embedding_surface": "normalized_contrastive_projection",
        "projection_dim": int(train_projection.shape[1]),
        "projection_l2_normalized": True,
        "n_train": int(len(train_projection)),
        "n_test": int(len(test_projection)),
        "source_run_dir": str(source_run_dir),
        "source_trunk_knn_auroc": source_metrics.get("knn_auroc"),
        "source_trunk_cosine_knn_auroc": source_metrics.get("cosine_knn_auroc"),
        "source_trunk_linear_probe_auroc": source_metrics.get("linear_probe_auroc"),
    }
    metrics.update(knn_stats)
    metrics.update(cosine_stats)
    metrics.update(probe_stats)

    predictions = [
        {
            "hashkey": str(test_hashkeys[index]),
            "halu": int(test_labels[index]),
            "score_halu": float(knn_scores[index]),
            "score_halu_projection_knn": float(knn_scores[index]),
            "score_halu_projection_cosine_knn": float(cosine_scores[index]),
            "score_halu_projection_linear_probe": float(probe_scores[index]),
        }
        for index in range(len(test_labels))
    ]
    return metrics, predictions
