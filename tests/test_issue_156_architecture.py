"""Contracts for the gated Issue #156 one-factor architecture sweep."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from activation_research.model import (
    LogprobReconAttentionPoolProgressiveCompressor,
    LogprobReconProgressiveCompressor,
    LogprobReconProjectedProgressiveCompressor,
)
from scripts.dispatch.build_issue_156_architecture_cells import build
from scripts.experiment_utils import load_method_config

_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {
    "hotpotqa_memmap": "hotpotqa",
    "nq_memmap": "nq",
    "popqa_memmap": "popqa",
    "sciq_memmap": "sciq",
    "searchqa_memmap": "searchqa",
}
_MIXED_DATASETS = dict(list(_DATASETS.items())[:3])
_ARMS = {
    "input_layernorm_only",
    "transformer_prenorm_only",
    "single_query_attention_pool_only",
    "disposable_projection_128_only",
}


def _method_names(recipe: str) -> list[str]:
    return [
        f"tokenwise_arch_{recipe}_input_norm_only",
        f"tokenwise_arch_{recipe}_prenorm_only",
        f"tokenwise_arch_{recipe}_attention_pool_only",
        f"tokenwise_arch_{recipe}_projection128_only",
    ]


def _complete_mixed_gate(root: Path, *, n_done: int = 9) -> None:
    for state in ("pending", "claimed", "done", "failed", "cancelled"):
        (root / state).mkdir(parents=True, exist_ok=True)
    for index in range(n_done):
        (root / "done" / f"cell_{index}_issue154_mixed.json").write_text("{}")


def _write_baselines(runs_root: Path, recipe: str) -> None:
    if recipe == "v1":
        experiment_prefix = "issue151_knnval"
        method = "tokenwise_contrastive_first_anchored"
        datasets = _DATASETS
    elif recipe == "t0":
        experiment_prefix = "issue155_causal"
        method = "tokenwise_causal_t0_dropout"
        datasets = _DATASETS
    else:
        experiment_prefix = "issue154_mixed"
        method = "tokenwise_causal_mixed_half"
        datasets = _MIXED_DATASETS
    for dataset, slug in datasets.items():
        path = (
            runs_root
            / f"{experiment_prefix}_{slug}"
            / dataset
            / method
            / "seed_0"
            / "eval_metrics.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"knn_auroc": 0.5}))


def _model_from_config(config: dict):
    params = config["model_params"]
    common = {
        "input_dim": 4096,
        "final_dim": params["final_dim"],
        "input_dropout": params["input_dropout"],
        "normalize_input": params.get("normalize_input", False),
        "pre_norm": params.get("pre_norm", False),
        "recon_seq_len": params["recon_seq_len"],
        "recon_hidden_dim": params["recon_hidden_dim"],
        "recon_lambda": params["recon_lambda"],
        "logprob_var_threshold": params["logprob_var_threshold"],
    }
    model_class = config["model_class"]
    if model_class == "logprob_recon_attn_pool_progressive_compressor":
        return LogprobReconAttentionPoolProgressiveCompressor(
            **common, pool_num_queries=params["pool_num_queries"]
        )
    if model_class == "logprob_recon_projected_progressive_compressor":
        return LogprobReconProjectedProgressiveCompressor(
            **common,
            projection_hidden_dim=params["projection_hidden_dim"],
            projection_dim=params["projection_dim"],
            projection_l2_normalize=params["projection_l2_normalize"],
            depth_pooling=params["depth_pooling"],
            pool_num_queries=params["pool_num_queries"],
        )
    return LogprobReconProgressiveCompressor(**common)


@pytest.mark.parametrize(
    ("recipe", "base_method", "training_recipe"),
    [
        ("v1", "tokenwise_contrastive_first_anchored", "v1_first_anchored"),
        ("t0", "tokenwise_causal_t0_dropout", "t0_same_dropout"),
        ("mixed", "tokenwise_causal_mixed_half", "mixed_half"),
    ],
)
def test_architecture_overlays_change_one_model_factor_only(
    recipe, base_method, training_recipe
):
    base = load_method_config(base_method, project_root=str(_ROOT))
    configs = [
        load_method_config(name, project_root=str(_ROOT))
        for name in _method_names(recipe)
    ]

    assert {config["architecture_arm"] for config in configs} == _ARMS
    assert all(config["training_recipe"] == training_recipe for config in configs)
    assert all(config["training"] == base["training"] for config in configs)
    assert all(config["data"] == base["data"] for config in configs)
    for config in configs:
        evaluation = dict(config["evaluation"])
        evaluation.pop("score_projection_surface", None)
        evaluation.pop("projection_batch_size", None)
        assert evaluation == base["evaluation"]

    projection = configs[-1]
    assert projection["evaluation"]["score_projection_surface"] is True
    assert projection["evaluation"]["dump_embeddings"] is True
    assert projection["model_params"]["depth_pooling"] == "mean"
    assert projection["model_params"]["normalize_input"] is False
    assert projection["model_params"]["pre_norm"] is False


@pytest.mark.parametrize("recipe", ["v1", "t0", "mixed"])
def test_architecture_parameter_guards_match_constructed_models(recipe):
    with torch.device("meta"):
        configs = [
            load_method_config(name, project_root=str(_ROOT))
            for name in _method_names(recipe)
        ]
        counts = {
            config["architecture_arm"]: sum(
                parameter.numel() for parameter in _model_from_config(config).parameters()
            )
            for config in configs
        }
    assert counts == {
        "input_layernorm_only": 77_546_304,
        "transformer_prenorm_only": 77_538_112,
        "single_query_attention_pool_only": 77_538_624,
        "disposable_projection_128_only": 77_866_432,
    }
    for config in configs:
        assert (
            counts[config["architecture_arm"]]
            == config["model_params"]["expected_total_params"]
        )
        assert config["model_params"]["reference_total_params"] == 77_538_112


def test_builder_refuses_before_all_nine_mixed_cells_finish(tmp_path):
    mixed_root = tmp_path / "mixed"
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _complete_mixed_gate(mixed_root, n_done=8)
    (mixed_root / "pending" / "cell_8_issue154_mixed.json").write_text("{}")
    _write_baselines(runs_root, "v1")

    with pytest.raises(RuntimeError, match="8/9 done"):
        build(
            dispatch_root,
            recipe="v1",
            mixed_sweep_root=mixed_root,
            project_root=_ROOT,
            runs_root=runs_root,
        )
    assert not dispatch_root.exists()


@pytest.mark.parametrize("recipe", ["v1", "t0", "mixed"])
def test_builder_queues_exact_seed_zero_matrix_after_gate(tmp_path, recipe):
    mixed_root = tmp_path / "mixed"
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _complete_mixed_gate(mixed_root)
    _write_baselines(runs_root, recipe)

    expected = 12 if recipe == "mixed" else 20
    expected_datasets = _MIXED_DATASETS if recipe == "mixed" else _DATASETS
    assert (
        build(
            dispatch_root,
            recipe=recipe,
            mixed_sweep_root=mixed_root,
            project_root=_ROOT,
            runs_root=runs_root,
        )
        == expected
    )
    assert (
        build(
            dispatch_root,
            recipe=recipe,
            mixed_sweep_root=mixed_root,
            project_root=_ROOT,
            runs_root=runs_root,
        )
        == 0
    )

    cells = [
        json.loads(path.read_text())
        for path in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(cells) == expected
    assert {cell["dataset"] for cell in cells} == set(expected_datasets)
    assert {cell["method"] for cell in cells} == set(_method_names(recipe))
    assert {cell["seed"] for cell in cells} == {0}
    assert all(cell["mixed_sweep_gate"] == "complete" for cell in cells)
    assert all(cell["issue"] == 156 for cell in cells)
    assert all(cell["kind"] == "experiment" for cell in cells)
    assert all(cell["cell_id"].startswith("0_high_") for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    selection = json.loads(
        (dispatch_root / "issue156_recipe_selection.json").read_text()
    )
    assert set(selection["recipes"]) == {recipe}
    assert selection["recipes"][recipe]["datasets"] == list(expected_datasets)
    assert len(selection["completed_mixed_cells"]) == 9


def test_recorded_recipe_cannot_be_changed_in_same_queue(tmp_path):
    mixed_root = tmp_path / "mixed"
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _complete_mixed_gate(mixed_root)
    _write_baselines(runs_root, "v1")
    _write_baselines(runs_root, "mixed")
    build(
        dispatch_root,
        recipe="v1",
        mixed_sweep_root=mixed_root,
        project_root=_ROOT,
        runs_root=runs_root,
    )

    with pytest.raises(RuntimeError, match="already recorded"):
        build(
            dispatch_root,
            recipe="mixed",
            mixed_sweep_root=mixed_root,
            project_root=_ROOT,
            runs_root=runs_root,
        )


def test_v1_and_t0_recipes_can_share_the_matched_comparison_queue(tmp_path):
    mixed_root = tmp_path / "mixed"
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _complete_mixed_gate(mixed_root)
    _write_baselines(runs_root, "v1")
    _write_baselines(runs_root, "t0")

    assert build(
        dispatch_root,
        recipe="v1",
        mixed_sweep_root=mixed_root,
        project_root=_ROOT,
        runs_root=runs_root,
    ) == 20
    assert build(
        dispatch_root,
        recipe="t0",
        mixed_sweep_root=mixed_root,
        project_root=_ROOT,
        runs_root=runs_root,
    ) == 20

    selection = json.loads(
        (dispatch_root / "issue156_recipe_selection.json").read_text()
    )
    assert set(selection["recipes"]) == {"v1", "t0"}
    assert len(list((dispatch_root / "pending").glob("*.json"))) == 40


def test_builder_migrates_original_scalar_v1_selection_record(tmp_path):
    mixed_root = tmp_path / "mixed"
    dispatch_root = tmp_path / "dispatch"
    runs_root = tmp_path / "runs"
    _complete_mixed_gate(mixed_root)
    _write_baselines(runs_root, "v1")
    _write_baselines(runs_root, "t0")
    dispatch_root.mkdir()
    completed = sorted(
        path.stem for path in (mixed_root / "done").glob("*issue154_mixed*.json")
    )
    (dispatch_root / "issue156_recipe_selection.json").write_text(
        json.dumps(
            {
                "issue": 156,
                "recipe": "v1",
                "training_recipe": "v1_first_anchored",
                "mixed_sweep_gate": "complete",
                "completed_mixed_cells": completed,
                "methods": _method_names("v1"),
            }
        )
    )

    assert build(
        dispatch_root,
        recipe="v1",
        mixed_sweep_root=mixed_root,
        project_root=_ROOT,
        runs_root=runs_root,
    ) == 20
    assert build(
        dispatch_root,
        recipe="t0",
        mixed_sweep_root=mixed_root,
        project_root=_ROOT,
        runs_root=runs_root,
    ) == 20
    selection = json.loads(
        (dispatch_root / "issue156_recipe_selection.json").read_text()
    )
    assert "recipe" not in selection
    assert set(selection["recipes"]) == {"v1", "t0"}


def test_experiment_configs_are_gated_matched_seed_zero_pilots():
    paths = sorted((_ROOT / "configs/experiments").glob("issue156_arch_*.json"))
    assert len(paths) == 13
    payloads = [json.loads(path.read_text()) for path in paths]
    assert {payload["dataset"] for payload in payloads} == set(_DATASETS)
    assert all(payload["selection_gate"] == "issue154_mixed_3x3_complete" for payload in payloads)
    assert all(payload["training_seeds"] == [0] for payload in payloads)
    assert all(payload["split_seeds"] == [42] for payload in payloads)
    assert all(len(payload["methods"]) == 4 for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)
    assert sum("arch_v1" in payload["experiment_name"] for payload in payloads) == 5
    assert sum("arch_t0" in payload["experiment_name"] for payload in payloads) == 5
    assert sum("arch_mixed" in payload["experiment_name"] for payload in payloads) == 3
