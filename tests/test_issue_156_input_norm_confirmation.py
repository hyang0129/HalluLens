"""Contracts for the Issue #156 v1 input-normalization confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dispatch.build_issue_156_input_norm_confirmation_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {
    "hotpotqa_memmap": "hotpotqa",
    "nq_memmap": "nq",
    "popqa_memmap": "popqa",
    "sciq_memmap": "sciq",
    "searchqa_memmap": "searchqa",
}
_METHOD = "tokenwise_arch_v1_input_norm_only"
_BASELINE = "tokenwise_contrastive_first_anchored"


def _write_baselines(runs_root: Path, *, omit: tuple[str, int] | None = None) -> None:
    for dataset, slug in _DATASETS.items():
        for seed in (1, 2, 3, 4):
            if omit == (dataset, seed):
                continue
            path = (
                runs_root
                / f"issue151_knnval_{slug}"
                / dataset
                / _BASELINE
                / f"seed_{seed}"
                / "eval_metrics.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"knn_auroc": 0.5}))


def test_confirmation_experiments_are_isolated_to_four_missing_seeds():
    payloads = []
    for slug in _DATASETS.values():
        path = (
            _ROOT
            / "configs"
            / "experiments"
            / f"issue156_inputnorm_confirm_{slug}.json"
        )
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    assert len(payloads) == 5
    assert {payload["dataset"] for payload in payloads} == set(_DATASETS)
    assert all(payload["training_seeds"] == [1, 2, 3, 4] for payload in payloads)
    assert all(payload["split_seeds"] == [1, 2, 3, 4] for payload in payloads)
    assert all(payload["methods"] == [_METHOD] for payload in payloads)
    assert all(
        payload["experiment_name"] == f"issue156_arch_v1_{slug}"
        for payload, slug in zip(payloads, _DATASETS.values())
    )
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_builder_adds_exact_twenty_matched_high_priority_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _write_baselines(runs_root)

    assert build(dispatch_root, project_root=_ROOT, runs_root=runs_root) == 20
    assert build(dispatch_root, project_root=_ROOT, runs_root=runs_root) == 0

    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(cells) == 20
    assert {cell["dataset"] for cell in cells} == set(_DATASETS)
    assert {cell["method"] for cell in cells} == {_METHOD}
    assert {cell["seed"] for cell in cells} == {1, 2, 3, 4}
    assert {(cell["seed"], cell["split_seed"]) for cell in cells} == {
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    }
    assert all(cell["priority"] == "highest" for cell in cells)
    assert all(cell["stage"] == 2 for cell in cells)
    assert all(cell["issue"] == 156 for cell in cells)
    assert all(cell["comparison_baseline_method"] == _BASELINE for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    source = next((dispatch_root / "pending").glob("*.json"))
    claimed = dispatch_root / "claimed" / "worker-0" / source.name
    claimed.parent.mkdir(parents=True)
    source.rename(claimed)
    assert build(dispatch_root, project_root=_ROOT, runs_root=runs_root) == 0
    assert claimed.is_file()


def test_builder_refuses_when_a_matched_baseline_is_missing(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _write_baselines(runs_root, omit=("nq_memmap", 4))

    with pytest.raises(FileNotFoundError, match="seed_4"):
        build(dispatch_root, project_root=_ROOT, runs_root=runs_root)
    assert not dispatch_root.exists()
