"""Contracts for the issue #151 one-seed KNN-validation rerun."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.dispatch.build_issue_151_knnval_rerun_cells import (
    build,
    build_tokenwise_full_sweep,
)


_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
}
_METHODS = {
    "dual_convention_contrastive_classifier_prefix_mixed_lowk",
    "tokenwise_contrastive_first_anchored",
    "contrastive_logprob_recon_prefix_mixed_lowk",
}


def test_knnval_rerun_builder_creates_exact_fresh_matrix(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 15
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 15
    assert {cell["dataset"] for cell in payloads} == _DATASETS
    assert {cell["method"] for cell in payloads} == _METHODS
    assert {cell["seed"] for cell in payloads} == {0}
    assert len(
        {(cell["dataset"], cell["method"], cell["seed"]) for cell in payloads}
    ) == 15
    assert all(cell["priority"] == "high" for cell in payloads)
    assert all(cell["rerun"] == "validation_knn_auroc" for cell in payloads)
    assert all("issue151_knnval_" in cell["output_check"] for cell in payloads)
    assert all(cell["output_check"].endswith("predictions.csv") for cell in payloads)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in payloads)


def test_tokenwise_full_sweep_reuses_seed_zero_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 15
    assert build_tokenwise_full_sweep(dispatch_root, project_root=_ROOT) == 20
    assert build_tokenwise_full_sweep(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    tokenwise = [
        cell
        for cell in payloads
        if cell["method"] == "tokenwise_contrastive_first_anchored"
    ]
    assert len(tokenwise) == 25
    assert {cell["dataset"] for cell in tokenwise} == _DATASETS
    assert {cell["seed"] for cell in tokenwise} == {0, 1, 2, 3, 4}
    assert len({(cell["dataset"], cell["seed"]) for cell in tokenwise}) == 25
    assert all(cell["priority"] == "high" for cell in tokenwise)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in tokenwise)


def test_knnval_experiments_cover_full_seed_sweep_and_non_mmlu():
    configs = sorted(
        (_ROOT / "configs/experiments").glob("issue151_knnval_*.json")
    )
    assert len(configs) == 5
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
    assert {payload["dataset"] for payload in payloads} == _DATASETS
    assert all(payload["training_seeds"] == [0, 1, 2, 3, 4] for payload in payloads)
    assert all(payload["split_seeds"] == [42, 1, 2, 3, 4] for payload in payloads)
    assert all(set(payload["methods"]) == _METHODS for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)
