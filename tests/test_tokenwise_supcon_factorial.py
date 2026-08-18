"""Contracts for the matched token-wise standard-SupCon factorial."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from activation_research.model import LogprobReconProgressiveCompressor
from scripts.dispatch.build_tokenwise_supcon_factorial_cells import build
from scripts.experiment_utils import load_method_config

_ROOT = Path(__file__).resolve().parent.parent
_BASELINE = "tokenwise_contrastive_first_anchored"
_METHODS = {
    "tokenwise_supcon_t0_full_recon",
    "tokenwise_supcon_tn_no_recon",
    "tokenwise_supcon_t0_no_recon",
}
_DATASETS = {
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
}


def _normalize(config: dict) -> dict:
    payload = deepcopy(config)
    for key in (
        "name",
        "extends",
        "study",
        "factor_views",
        "factor_reconstruction",
    ):
        payload.pop(key, None)
    payload["model_params"].pop("expected_total_params", None)
    payload["model_params"].pop("reference_total_params", None)
    payload["model_params"]["recon_lambda"] = "<factor>"
    payload["training"]["contrastive_objective"] = "<legacy-supcon>"
    payload["data"]["token_pair_mode"] = "<factor>"
    return payload


def test_factorial_changes_only_views_and_reconstruction_weight():
    baseline = load_method_config(_BASELINE, project_root=str(_ROOT))
    configs = {
        method: load_method_config(method, project_root=str(_ROOT))
        for method in _METHODS
    }

    assert baseline["data"]["min_response_tokens"] == 2
    assert all(_normalize(config) == _normalize(baseline) for config in configs.values())
    assert all(
        config["training"]["contrastive_objective"] == "legacy_supcon"
        for config in configs.values()
    )
    assert {
        (config["factor_views"], config["factor_reconstruction"])
        for config in configs.values()
    } == {
        ("t0_plus_t0_dropout", "full_response"),
        ("t0_plus_random_same_response_tn", "none"),
        ("t0_plus_t0_dropout", "none"),
    }
    assert {
        (
            "t0_plus_random_same_response_tn",
            "full_response",
        ),
        *{
            (config["factor_views"], config["factor_reconstruction"])
            for config in configs.values()
        },
    } == {
        ("t0_plus_random_same_response_tn", "full_response"),
        ("t0_plus_t0_dropout", "full_response"),
        ("t0_plus_random_same_response_tn", "none"),
        ("t0_plus_t0_dropout", "none"),
    }


def test_all_three_arms_keep_original_v1_parameter_count():
    with torch.device("meta"):
        for method in _METHODS:
            config = load_method_config(method, project_root=str(_ROOT))
            params = config["model_params"]
            model = LogprobReconProgressiveCompressor(
                input_dim=4096,
                final_dim=params["final_dim"],
                input_dropout=params["input_dropout"],
                recon_seq_len=params["recon_seq_len"],
                recon_hidden_dim=params["recon_hidden_dim"],
                recon_lambda=params["recon_lambda"],
                logprob_var_threshold=params["logprob_var_threshold"],
            )
            count = sum(parameter.numel() for parameter in model.parameters())
            assert count == 77_538_112
            assert params["expected_total_params"] == count
            assert params["reference_total_params"] == count


def test_experiments_cover_five_datasets_at_one_matched_seed():
    paths = sorted(
        (_ROOT / "configs" / "experiments").glob(
            "issue151_supcon_factorial_*.json"
        )
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(payloads) == 5
    assert {payload["dataset"] for payload in payloads} == _DATASETS
    assert all(set(payload["methods"]) == _METHODS for payload in payloads)
    assert all(payload["training_seeds"] == [0] for payload in payloads)
    assert all(payload["split_seeds"] == [42] for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_builder_appends_exactly_15_highest_priority_resumable_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 15
    assert build(dispatch_root, project_root=_ROOT) == 0

    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(cells) == 15
    assert {cell["dataset"] for cell in cells} == _DATASETS
    assert {cell["method"] for cell in cells} == _METHODS
    assert {cell["seed"] for cell in cells} == {0}
    assert all(cell["priority"] == "highest" for cell in cells)
    assert all(cell["kind"] == "experiment" for cell in cells)
    assert all(cell["contrastive_objective"] == "legacy_supcon" for cell in cells)
    assert all(cell["matched_min_response_tokens"] == 2 for cell in cells)
    assert all(cell["expected_total_params"] == 77_538_112 for cell in cells)
    assert all(cell["cell_id"].startswith("0_high_") for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    # A claimed cell remains part of the resumable queue and must not be
    # recreated as pending on a subsequent builder invocation.
    source = next((dispatch_root / "pending").glob("*.json"))
    claimed = dispatch_root / "claimed" / "worker-0" / source.name
    claimed.parent.mkdir(parents=True)
    source.rename(claimed)
    assert build(dispatch_root, project_root=_ROOT) == 0
    assert claimed.is_file()
    assert not (dispatch_root / "pending" / claimed.name).exists()
