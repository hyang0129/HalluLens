"""Contracts for the Issue #151 Qwen3-8B first-token comparison matrix."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.dispatch.build_issue_151_qwen3_k1_cells import build
from scripts.experiment_utils import load_method_config

_ROOT = Path(__file__).resolve().parent.parent
_TOKENWISE = "tokenwise_contrastive_first_anchored_qwen3"
_TOKENWISE_INPUTNORM = "tokenwise_arch_v1_input_norm_only_qwen3"
_ACTVIT_K1 = "act_vit_k1"
_ACTVIT_EVAL = "act_vit_k64_eval_k1"
_DATASETS = {
    "hotpotqa_qwen3_memmap": "hotpotqa",
    "nq_qwen3_memmap": "nq",
    "popqa_qwen3_memmap": "popqa",
    "sciq_qwen3_memmap": "sciq",
    "searchqa_qwen3_memmap": "searchqa",
}
_SEEDS = (0, 1, 2, 3, 4)


def _write_qwen_actvit_checkpoints(runs_root: Path) -> None:
    for dataset, slug in _DATASETS.items():
        for seed in _SEEDS:
            artifacts = (
                runs_root
                / f"baseline_comparison_{slug}_qwen3_memmap"
                / dataset
                / "act_vit"
                / f"seed_{seed}"
                / "artifacts"
            )
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "best_checkpoint.pt").write_bytes(b"best")
            (artifacts / "final_weights.pt").write_bytes(b"final")


def test_qwen_tokenwise_pilot_uses_full_36_layer_depth_trajectory():
    methods = {
        name: load_method_config(name, project_root=str(_ROOT))
        for name in (_TOKENWISE, _TOKENWISE_INPUTNORM)
    }
    for method in methods.values():
        assert method["routine"] == "tokenwise_contrastive_logprob_recon"
        assert method["data"]["relevant_layers"] == "1-36"
        assert method["data"]["token_pair_mode"] == "first_anchored"
        assert method["training"]["checkpoint_selection_metric"] == "knn_auroc"
    assert methods[_TOKENWISE].get("model_params", {}).get(
        "normalize_input", False
    ) is False
    assert methods[_TOKENWISE_INPUTNORM]["model_params"]["normalize_input"] is True


def test_qwen_experiments_cover_five_paired_datasets_without_mmlu():
    expected_control_methods = [_ACTVIT_K1, _ACTVIT_EVAL]
    expected_pilot_methods = [_TOKENWISE, _TOKENWISE_INPUTNORM]
    payloads = []
    for dataset, slug in _DATASETS.items():
        path = (
            _ROOT
            / "configs/experiments"
            / f"issue151_qwen3_k1_controls_{slug}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        assert payload["dataset"] == dataset
        assert payload["methods"] == expected_control_methods
        assert payload["training_seeds"] == [0, 1, 2, 3, 4]
        assert payload["split_seeds"] == [42, 1, 2, 3, 4]
        assert "mmlu" not in json.dumps(payload).lower()
        pilot_path = (
            _ROOT
            / "configs/experiments"
            / f"issue156_qwen3_inputnorm_pilot_{slug}.json"
        )
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        payloads.append(pilot)
        assert pilot["dataset"] == dataset
        assert pilot["methods"] == expected_pilot_methods
        assert pilot["training_seeds"] == [0]
        assert pilot["split_seeds"] == [42]
        assert "mmlu" not in json.dumps(pilot).lower()
    assert len(payloads) == 10


def test_builder_adds_60_cells_and_links_all_qwen_checkpoints(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _write_qwen_actvit_checkpoints(runs_root)

    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 60
    assert build(
        dispatch_root, project_root=_ROOT, runs_root=runs_root
    ) == 0

    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(cells) == 60
    by_method = {
        method: [cell for cell in cells if cell["method"] == method]
        for method in (
            _TOKENWISE,
            _TOKENWISE_INPUTNORM,
            _ACTVIT_K1,
            _ACTVIT_EVAL,
        )
    }
    assert len(by_method[_TOKENWISE]) == 5
    assert len(by_method[_TOKENWISE_INPUTNORM]) == 5
    assert len(by_method[_ACTVIT_K1]) == 25
    assert len(by_method[_ACTVIT_EVAL]) == 25
    assert all(
        {cell["dataset"] for cell in method_cells} == set(_DATASETS)
        for method_cells in by_method.values()
    )
    actvit_cells = by_method[_ACTVIT_K1] + by_method[_ACTVIT_EVAL]
    tokenwise_cells = by_method[_TOKENWISE] + by_method[_TOKENWISE_INPUTNORM]
    assert {(cell["seed"], cell["split_seed"]) for cell in actvit_cells} == {
        (0, 42),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    }
    assert {(cell["seed"], cell["split_seed"]) for cell in tokenwise_cells} == {
        (0, 42)
    }
    assert all(cell["backbone"] == "Qwen3-8B" for cell in cells)
    assert all(cell["evaluation_prefix_length"] == 1 for cell in cells)
    assert all(cell["priority"] == "normal" for cell in actvit_cells)
    assert all(cell["priority"] == "high" for cell in tokenwise_cells)
    assert all(cell.get("eval_only") is True for cell in by_method[_ACTVIT_EVAL])
    assert all(
        cell.get("training_prefix_length") == 64
        for cell in by_method[_ACTVIT_EVAL]
    )
    assert all(
        cell.get("training_prefix_length") == 1
        for cell in by_method[_ACTVIT_K1]
    )
    assert all(
        cell.get("relevant_layers") == "1-36"
        for cell in tokenwise_cells
    )
    assert all(
        cell["normalize_input"] is False for cell in by_method[_TOKENWISE]
    )
    assert all(
        cell["normalize_input"] is True
        for cell in by_method[_TOKENWISE_INPUTNORM]
    )
    assert all(
        "expanding_seeds" in cell["decision_gate"] for cell in tokenwise_cells
    )
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    for dataset, slug in _DATASETS.items():
        for seed in _SEEDS:
            source = (
                runs_root
                / f"baseline_comparison_{slug}_qwen3_memmap"
                / dataset
                / "act_vit"
                / f"seed_{seed}"
                / "artifacts"
            )
            target = (
                runs_root
                / f"issue151_qwen3_k1_controls_{slug}"
                / dataset
                / _ACTVIT_EVAL
                / f"seed_{seed}"
                / "artifacts"
            )
            for filename in ("best_checkpoint.pt", "final_weights.pt"):
                assert (target / filename).is_symlink()
                assert (target / filename).resolve() == (source / filename).resolve()
