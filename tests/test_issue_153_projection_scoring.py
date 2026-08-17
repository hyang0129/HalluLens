"""Contracts for Issue #153 projection-space rescoring cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from activation_research.projection_scoring import score_saved_projection
from scripts.dispatch.build_issue_153_projection_eval_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_METHOD = "tokenwise_contrastive_v2_depthnorm_projection"
_METHOD = "tokenwise_contrastive_v2_projection_scoring"
_TARGETS = (
    ("hotpotqa_memmap", "issue153_v2_hotpotqa"),
    ("nq_memmap", "issue153_v2_nq"),
    ("popqa_memmap", "issue153_v2_popqa"),
    ("sciq_memmap", "issue153_v2_sciq"),
    ("searchqa_memmap", "issue153_v2_searchqa"),
)


def _write_ready_source(runs_root: Path, dataset: str, experiment: str) -> None:
    source = runs_root / experiment / dataset / _SOURCE_METHOD / "seed_0"
    required = (
        source / "artifacts" / "final_weights.pt",
        source / "eval_metrics.json",
        source / "embeddings" / "train_z.npy",
        source / "embeddings" / "train_labels.npy",
        source / "embeddings" / "test_z.npy",
        source / "embeddings" / "test_labels.npy",
        source / "embeddings" / "test_hashkeys.json",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ready")


def test_projection_rescore_uses_saved_head_and_emits_predictions(tmp_path):
    source = tmp_path / "source"
    embeddings = source / "embeddings"
    artifacts = source / "artifacts"
    embeddings.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    rng = np.random.default_rng(153)
    train_z = rng.normal(size=(20, 1, 4)).astype(np.float16)
    test_z = rng.normal(size=(8, 1, 4)).astype(np.float16)
    train_labels = np.asarray([0, 1] * 10, dtype=np.int8)
    test_labels = np.asarray([0, 1] * 4, dtype=np.int8)
    np.save(embeddings / "train_z.npy", train_z)
    np.save(embeddings / "test_z.npy", test_z)
    np.save(embeddings / "train_labels.npy", train_labels)
    np.save(embeddings / "test_labels.npy", test_labels)
    (embeddings / "test_hashkeys.json").write_text(
        json.dumps([f"test-{index}" for index in range(len(test_z))])
    )
    (source / "eval_metrics.json").write_text(
        json.dumps({"knn_auroc": 0.6, "cosine_knn_auroc": 0.61, "linear_probe_auroc": 0.62})
    )

    generator = torch.Generator().manual_seed(153)
    state = {
        "projection_head.0.weight": torch.randn(5, 4, generator=generator),
        "projection_head.0.bias": torch.randn(5, generator=generator),
        "projection_head.2.weight": torch.randn(3, 5, generator=generator),
        "projection_head.2.bias": torch.randn(3, generator=generator),
    }
    torch.save({"model_state_dict": state}, artifacts / "final_weights.pt")

    metrics, predictions = score_saved_projection(
        source,
        evaluation_cfg={
            "projection_batch_size": 4,
            "knn_params": {"k": 2, "calibrate_k": False},
            "cosine_knn_params": {"k": 2, "calibrate_k": False},
            "linear_probe_params": {"C": 1.0, "max_iter": 100},
        },
        sample_seed=0,
        device="cpu",
    )

    assert metrics["embedding_surface"] == "normalized_contrastive_projection"
    assert metrics["projection_dim"] == 3
    assert metrics["source_trunk_knn_auroc"] == 0.6
    assert 0.0 <= metrics["knn_auroc"] <= 1.0
    assert 0.0 <= metrics["cosine_knn_auroc"] <= 1.0
    assert 0.0 <= metrics["linear_probe_auroc"] <= 1.0
    assert len(predictions) == len(test_z)
    assert predictions[0]["hashkey"] == "test-0"
    assert "score_halu_projection_knn" in predictions[0]
    assert "score_halu_projection_cosine_knn" in predictions[0]
    assert "score_halu_projection_linear_probe" in predictions[0]


def test_projection_builder_defers_unready_sources_and_queues_highest_priority(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    for dataset, experiment in _TARGETS[:3]:
        _write_ready_source(runs_root, dataset, experiment)

    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 3
    first_cells = [
        json.loads(path.read_text())
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(first_cells) == 3
    assert all(cell["priority"] == "highest" for cell in first_cells)
    assert all(cell["method"] == _METHOD for cell in first_cells)
    assert all(cell["cell_id"].startswith("0_high_") for cell in first_cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in first_cells)

    for dataset, experiment in _TARGETS[3:]:
        _write_ready_source(runs_root, dataset, experiment)
    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 2
    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 0
    assert len(list((dispatch_root / "pending").glob("*.json"))) == 5


def test_all_issue153_experiments_expose_projection_scoring_method():
    for _, experiment in _TARGETS:
        payload = json.loads(
            (_ROOT / "configs" / "experiments" / f"{experiment}.json").read_text()
        )
        assert _SOURCE_METHOD in payload["methods"]
        assert _METHOD in payload["methods"]
