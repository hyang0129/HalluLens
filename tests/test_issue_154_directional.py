"""Contracts for Issue #154 Stage-A directional temporal training."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from activation_research.training import TokenwiseCausalContrastiveLoss
from scripts.dispatch.build_issue_154_directional_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {"hotpotqa_memmap", "nq_memmap", "popqa_memmap"}
_NEW_METHODS = {
    "tokenwise_causal_t0_to_later",
    "tokenwise_causal_t0_to_later_stopgrad",
}
_BASELINE_METHOD = "tokenwise_causal_temporal_positive"


def _temporal_gradients(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(154)
    features = torch.randn(8, 2, 12, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    temporal, _ = TokenwiseCausalContrastiveLoss(
        temperature=0.25,
        temporal_mode=mode,
    ).components(features, labels=labels)
    temporal.backward()
    assert features.grad is not None
    return features.grad[:, 0], features.grad[:, 1]


def test_directional_temporal_loss_updates_query_and_later_key():
    grad_t0, grad_later = _temporal_gradients("t0_to_later")
    assert grad_t0.abs().sum() > 0
    assert grad_later.abs().sum() > 0


def test_stopgrad_temporal_loss_updates_only_token_zero_path():
    grad_t0, grad_later = _temporal_gradients("t0_to_later_stopgrad")
    assert grad_t0.abs().sum() > 0
    torch.testing.assert_close(grad_later, torch.zeros_like(grad_later))


def test_symmetric_mode_remains_the_issue155_default():
    loss_fn = TokenwiseCausalContrastiveLoss(temperature=0.25)
    assert loss_fn.temporal_mode == "symmetric"
    assert loss_fn.temporal_loss.contrast_mode == "all"


def test_issue154_configs_change_only_temporal_gradient_mode():
    names = {_BASELINE_METHOD, *_NEW_METHODS}
    configs = []
    for method_name in sorted(names):
        path = _ROOT / "configs" / "methods" / f"{method_name}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["training"]["contrastive_objective"] == (
            "tokenwise_causal_control"
        )
        assert config["data"]["token_pair_mode"] == "first_anchored"
        assert config["data"]["emit_view_logprob_targets"] is True
        configs.append(config)

    assert {
        config["training"]["causal_temporal_mode"] for config in configs
    } == {"symmetric", "t0_to_later", "t0_to_later_stopgrad"}

    normalized = []
    for config in configs:
        payload = json.loads(json.dumps(config))
        payload["name"] = "<arm>"
        payload["training"]["causal_temporal_mode"] = "<gradient-mode>"
        normalized.append(payload)
    assert normalized[0] == normalized[1] == normalized[2]


def test_issue154_experiments_cover_three_datasets_at_seed_zero():
    paths = sorted(
        (_ROOT / "configs" / "experiments").glob(
            "issue154_directional_*.json"
        )
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 3
    assert {payload["dataset"] for payload in payloads} == _DATASETS
    assert all(set(payload["methods"]) == _NEW_METHODS for payload in payloads)
    assert all(payload["training_seeds"] == [0] for payload in payloads)
    assert all(payload["split_seeds"] == [42] for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_issue154_builder_appends_six_new_cells_and_records_baseline(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 6
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 6
    assert {cell["dataset"] for cell in payloads} == _DATASETS
    assert {cell["method"] for cell in payloads} == _NEW_METHODS
    assert {cell["seed"] for cell in payloads} == {0}
    assert all(cell["issue"] == 154 and cell["stage"] == "A" for cell in payloads)
    assert all(cell["kind"] == "experiment" for cell in payloads)
    assert all(
        cell["worker_script"] == "scripts/dispatch/worker_experiment.sh"
        for cell in payloads
    )
    assert all(
        cell["comparison_baseline_method"] == _BASELINE_METHOD
        for cell in payloads
    )
    assert all(path.name.startswith("2_high_") for path in paths)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in payloads)
