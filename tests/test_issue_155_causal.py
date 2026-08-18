"""Contracts for the Issue #154 claim-3 three-arm causal test."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from activation_research.training import TokenwiseCausalContrastiveLoss
from scripts.dispatch.build_issue_155_causal_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
}
_METHODS = {
    "tokenwise_causal_temporal_positive",
    "tokenwise_causal_t0_dropout",
    "tokenwise_causal_shuffled_later",
}
_PAIR_MODES = {"first_anchored", "first_same", "shuffled_later"}


def test_causal_loss_class_component_depends_only_on_token_zero():
    torch.manual_seed(0)
    loss_fn = TokenwiseCausalContrastiveLoss(
        temperature=0.25,
        ignore_label=1,
    )
    token_zero = torch.randn(6, 1, 16)
    later_a = torch.randn(6, 1, 16)
    later_b = torch.randn(6, 1, 16)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)

    temporal_a, class_a = loss_fn.components(
        torch.cat([token_zero, later_a], dim=1), labels=labels
    )
    temporal_b, class_b = loss_fn.components(
        torch.cat([token_zero, later_b], dim=1), labels=labels
    )

    torch.testing.assert_close(class_a, class_b, rtol=0, atol=0)
    assert not torch.isclose(temporal_a, temporal_b)


def test_causal_loss_updates_both_designated_views():
    torch.manual_seed(1)
    features = torch.randn(8, 2, 12, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    loss = TokenwiseCausalContrastiveLoss(temperature=0.25)(
        features, labels=labels
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None
    assert features.grad[:, 0].abs().sum() > 0
    assert features.grad[:, 1].abs().sum() > 0


def test_issue155_method_configs_are_matched_except_view_construction():
    configs = []
    for method_name in sorted(_METHODS):
        path = _ROOT / "configs" / "methods" / f"{method_name}.json"
        configs.append(json.loads(path.read_text(encoding="utf-8")))

    assert {config["data"]["token_pair_mode"] for config in configs} == _PAIR_MODES
    for config in configs:
        assert config["training"]["contrastive_objective"] == (
            "tokenwise_causal_control"
        )
        assert config["data"]["emit_view_logprob_targets"] is True
        assert config["training"]["checkpoint_selection_metric"] == "knn_auroc"
        assert config["training"]["validation_knn"] == {
            "k": 50,
            "metric": "euclidean",
            "calibrate_k": False,
            "train_selection": "all",
            "max_train_size": 200000,
        }

    normalized = []
    for config in configs:
        payload = json.loads(json.dumps(config))
        payload["name"] = "<arm>"
        payload["data"]["token_pair_mode"] = "<view-construction>"
        normalized.append(payload)
    assert normalized[0] == normalized[1] == normalized[2]


def test_issue155_experiments_cover_five_datasets_and_matched_seeds():
    paths = sorted(
        (_ROOT / "configs" / "experiments").glob("issue155_causal_*.json")
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 5
    assert {payload["dataset"] for payload in payloads} == _DATASETS
    assert all(set(payload["methods"]) == _METHODS for payload in payloads)
    assert all(
        payload["training_seeds"] == [0, 1, 2, 3, 4]
        for payload in payloads
    )
    assert all(
        payload["split_seeds"] == [42, 1, 2, 3, 4]
        for payload in payloads
    )
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_issue155_builder_appends_exact_75_cell_matrix(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 75
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 75
    assert {cell["dataset"] for cell in payloads} == _DATASETS
    assert {cell["method"] for cell in payloads} == _METHODS
    assert {cell["seed"] for cell in payloads} == {0, 1, 2, 3, 4}
    assert len(
        {(cell["dataset"], cell["method"], cell["seed"]) for cell in payloads}
    ) == 75
    assert all(cell["issue"] == 154 for cell in payloads)
    assert all(
        cell["experiment"] == "claim3_same_response_temporal_transfer"
        for cell in payloads
    )
    assert all(cell["kind"] == "experiment" for cell in payloads)
    assert all(
        cell["worker_script"] == "scripts/dispatch/worker_experiment.sh"
        for cell in payloads
    )
    assert all(path.name.startswith("1_high_") for path in paths)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in payloads)
