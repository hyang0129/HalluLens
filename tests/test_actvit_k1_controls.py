"""Contracts for the Issue #151 ACT-ViT k=1 comparison cells."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.dispatch.build_issue_151_actvit_k1_control_cells import build
from scripts.experiment_utils import load_method_config

_ROOT = Path(__file__).resolve().parent.parent
_SEEDS = (0, 1, 2, 3, 4)
_K64_DATASETS = {
    "nq_memmap": "nq",
    "popqa_memmap": "popqa",
    "searchqa_memmap": "searchqa",
}
_K1_DATASETS = {
    "hotpotqa_memmap": "hotpotqa",
    "nq_memmap": "nq",
    "popqa_memmap": "popqa",
    "sciq_memmap": "sciq",
    "searchqa_memmap": "searchqa",
}


def _write_canonical_checkpoints(runs_root: Path) -> None:
    for dataset, slug in _K64_DATASETS.items():
        for seed in _SEEDS:
            artifacts = (
                runs_root
                / f"baseline_comparison_{slug}_memmap"
                / dataset
                / "act_vit"
                / f"seed_{seed}"
                / "artifacts"
            )
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "best_checkpoint.pt").write_bytes(b"best")
            (artifacts / "final_weights.pt").write_bytes(b"final")


def test_method_configs_distinguish_k64_deployment_from_fixed_k1_training():
    k64_eval = load_method_config(
        "act_vit_k64_eval_k1", project_root=str(_ROOT)
    )
    assert k64_eval["routine"] == "act_vit"
    assert "fixed_prefix_length" not in k64_eval["training"]
    assert k64_eval["evaluation"]["eval_prefix_lengths"] == [1]

    k1_train = load_method_config("act_vit_k1", project_root=str(_ROOT))
    assert k1_train["routine"] == "act_vit"
    assert k1_train["training"]["fixed_prefix_length"] == 1
    assert k1_train["training"].get("prefix_training", False) is False
    assert k1_train["evaluation"]["eval_prefix_curve"] is False


def test_experiments_use_all_and_only_paired_in_scope_datasets():
    for dataset, slug in _K64_DATASETS.items():
        path = (
            _ROOT
            / "configs/experiments"
            / f"issue151_actvit_k64_to_k1_{slug}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["dataset"] == dataset
        assert payload["methods"] == ["act_vit_k64_eval_k1"]
        assert payload["training_seeds"] == [0, 1, 2, 3, 4]
        assert payload["split_seeds"] == [42, 1, 2, 3, 4]
        assert "mmlu" not in json.dumps(payload).lower()

    for dataset, slug in _K1_DATASETS.items():
        path = (
            _ROOT
            / "configs/experiments"
            / f"issue151_actvit_k1_{slug}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["dataset"] == dataset
        assert payload["methods"] == ["act_vit_k1"]
        assert payload["training_seeds"] == [0, 1, 2, 3, 4]
        assert payload["split_seeds"] == [42, 1, 2, 3, 4]
        assert "mmlu" not in json.dumps(payload).lower()


def test_builder_links_checkpoints_and_adds_15_eval_plus_25_train_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _write_canonical_checkpoints(runs_root)

    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 40
    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 0

    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    eval_cells = [cell for cell in cells if cell.get("eval_only")]
    train_cells = [cell for cell in cells if not cell.get("eval_only")]
    assert len(eval_cells) == 15
    assert len(train_cells) == 25
    assert {cell["dataset"] for cell in eval_cells} == set(_K64_DATASETS)
    assert {cell["dataset"] for cell in train_cells} == set(_K1_DATASETS)
    assert {cell["method"] for cell in eval_cells} == {"act_vit_k64_eval_k1"}
    assert {cell["method"] for cell in train_cells} == {"act_vit_k1"}
    assert all(cell["training_prefix_length"] == 64 for cell in eval_cells)
    assert all(cell["evaluation_prefix_length"] == 1 for cell in eval_cells)
    assert all(cell["training_prefix_length"] == 1 for cell in train_cells)
    assert all(cell["evaluation_prefix_length"] == 1 for cell in train_cells)
    assert all(
        cell["checkpoint_selection_metric"] == "validation_auroc_at_k1"
        for cell in train_cells
    )
    assert {(cell["seed"], cell["split_seed"]) for cell in cells} == {
        (0, 42),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    }
    assert all(cell["priority"] == "normal" for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    for dataset, slug in _K64_DATASETS.items():
        for seed in _SEEDS:
            source = (
                runs_root
                / f"baseline_comparison_{slug}_memmap"
                / dataset
                / "act_vit"
                / f"seed_{seed}"
                / "artifacts"
            )
            target = (
                runs_root
                / f"issue151_actvit_k64_to_k1_{slug}"
                / dataset
                / "act_vit_k64_eval_k1"
                / f"seed_{seed}"
                / "artifacts"
            )
            for filename in ("best_checkpoint.pt", "final_weights.pt"):
                assert (target / filename).is_symlink()
                assert (target / filename).resolve() == (source / filename).resolve()
