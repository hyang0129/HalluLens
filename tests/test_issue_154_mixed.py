"""Contracts for the Issue #154 50/50 mixed-view ablation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from activation_research.memmap_contrastive_dataset import (
    MemmapContrastiveDataset,
)
from activation_research.tokenwise_contrastive_dataset import (
    TokenwiseContrastiveDataset,
)
from scripts.dispatch.build_issue_154_mixed_cells import build
from tests.test_memmap_contrastive_dataset import _make_full_capture_dir

_ROOT = Path(__file__).resolve().parent.parent
_METHOD = "tokenwise_causal_mixed_half"
_DATASETS = {"hotpotqa_memmap", "nq_memmap", "popqa_memmap"}


def _base_dataset(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=8)
    return MemmapContrastiveDataset(
        capture,
        split="all",
        num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_logprobs=True,
        pad_length=12,
    )


def test_first_mixed_selects_both_view_types_at_half_probability(tmp_path):
    dataset = TokenwiseContrastiveDataset(
        _base_dataset(tmp_path),
        layer_positions=[1, 2, 3, 4],
        num_views=2,
        token_pair_mode="first_mixed",
        later_view_probability=0.5,
        min_response_tokens=2,
        emit_view_logprob_targets=True,
    )

    with patch(
        "activation_research.tokenwise_contrastive_dataset.random.random",
        side_effect=[0.75, 0.25],
    ):
        token_zero_pair = dataset[0]
        temporal_pair = dataset[0]

    assert token_zero_pair["view_token_indices"].tolist() == [0, 0]
    assert temporal_pair["view_token_indices"][0].item() == 0
    assert 1 <= temporal_pair["view_token_indices"][1].item() < 8
    assert token_zero_pair["view_source_indices"].tolist() == [0, 0]
    assert temporal_pair["view_source_indices"].tolist() == [0, 0]


def test_first_mixed_validates_later_view_probability(tmp_path):
    with pytest.raises(ValueError, match="later_view_probability"):
        TokenwiseContrastiveDataset(
            _base_dataset(tmp_path),
            layer_positions=[1, 2, 3, 4],
            num_views=2,
            token_pair_mode="first_mixed",
            later_view_probability=1.1,
            min_response_tokens=2,
        )


def test_mixed_method_matches_causal_endpoints_except_view_distribution():
    mixed = json.loads(
        (_ROOT / "configs/methods/tokenwise_causal_mixed_half.json").read_text()
    )
    temporal = json.loads(
        (
            _ROOT
            / "configs/methods/tokenwise_causal_temporal_positive.json"
        ).read_text()
    )

    assert mixed["data"]["token_pair_mode"] == "first_mixed"
    assert mixed["data"]["later_view_probability"] == 0.5
    normalized_mixed = json.loads(json.dumps(mixed))
    normalized_temporal = json.loads(json.dumps(temporal))
    normalized_mixed["name"] = normalized_temporal["name"] = "<method>"
    normalized_mixed["data"]["token_pair_mode"] = "first_anchored"
    normalized_mixed["data"].pop("later_view_probability")
    assert normalized_mixed == normalized_temporal


def test_mixed_experiments_cover_three_datasets_and_three_matched_seeds():
    paths = sorted(
        (_ROOT / "configs/experiments").glob("issue154_mixed_*.json")
    )
    payloads = [json.loads(path.read_text()) for path in paths]
    assert len(payloads) == 3
    assert {payload["dataset"] for payload in payloads} == _DATASETS
    assert all(payload["methods"] == [_METHOD] for payload in payloads)
    assert all(payload["training_seeds"] == [0, 1, 2] for payload in payloads)
    assert all(payload["split_seeds"] == [42, 1, 2] for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_mixed_builder_appends_exact_nine_cell_matrix(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 9
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text()) for path in paths]
    assert len(payloads) == 9
    assert {cell["dataset"] for cell in payloads} == _DATASETS
    assert {cell["method"] for cell in payloads} == {_METHOD}
    assert {cell["seed"] for cell in payloads} == {0, 1, 2}
    assert all(cell["issue"] == 154 for cell in payloads)
    assert all(cell["factor_later_view_probability"] == 0.5 for cell in payloads)
    assert all(cell["kind"] == "experiment" for cell in payloads)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in payloads)
