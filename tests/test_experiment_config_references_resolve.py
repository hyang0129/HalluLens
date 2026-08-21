"""Sanity check: every configs/experiments/*.json's dataset/method references
resolve to files that actually exist in configs/datasets/ and configs/methods/.

Repo-wide (not scoped to the new chatv1_* configs) so it also catches
pre-existing drift, but the motivating case is the new chatv1 matrix added
alongside scripts/dispatch/build_chatv1_matrix_cells.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _ROOT / "configs/experiments"
_DATASETS_DIR = _ROOT / "configs/datasets"
_METHODS_DIR = _ROOT / "configs/methods"

_EXPERIMENT_PATHS = sorted(_EXPERIMENTS_DIR.glob("*.json"))


def _dataset_names(payload: dict) -> list[str]:
    value = payload.get("dataset") or payload.get("datasets")
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


@pytest.mark.parametrize("path", _EXPERIMENT_PATHS, ids=lambda p: p.stem)
def test_experiment_config_dataset_and_method_references_resolve(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    for dataset_name in _dataset_names(payload):
        dataset_path = _DATASETS_DIR / f"{dataset_name}.json"
        assert dataset_path.is_file(), (
            f"{path.name}: dataset {dataset_name!r} -> missing {dataset_path}"
        )

    for method_name in payload.get("methods", []):
        method_path = _METHODS_DIR / f"{method_name}.json"
        assert method_path.is_file(), (
            f"{path.name}: method {method_name!r} -> missing {method_path}"
        )


def test_chatv1_matrix_configs_are_present():
    tasks = ["hotpotqa", "nq", "popqa", "sciq", "searchqa"]
    expected = [f"chatv1_{t}.json" for t in tasks] + [f"chatv1_qwen3_{t}.json" for t in tasks]
    for name in expected:
        assert (_EXPERIMENTS_DIR / name).is_file(), name
