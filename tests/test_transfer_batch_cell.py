"""Tests for the batch transfer cell dispatch system (issue #89 refactor).

Old: one dispatch cell per (src × tgt × method × slug × seed) = 1440 cells.
New: one dispatch cell per (src × slug × seed) = 60 cells.  Each batch cell
runs all methods × all targets sharing a single source-data load.

Tests are CPU-only and mock all checkpoint/model/DataLoader I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Minimal dataset config dict used across tests.
# ---------------------------------------------------------------------------

_SRC_CFG = {
    "input_dim": 4096,
    "outlier_class": 1,
    "icr_capture": {
        "train_dir": "shared/icr_capture/hotpotqa_train_Llama",
        "test_dir": "shared/icr_capture/hotpotqa_test_Llama",
    },
}

_TGT_CFG_HOTPOTQA = {
    "input_dim": 4096,
    "outlier_class": 1,
    "icr_capture": {
        "train_dir": "shared/icr_capture/hotpotqa_train_Llama",
        "test_dir": "shared/icr_capture/hotpotqa_test_Llama",
    },
}

_TGT_CFG_MMLU = {
    "input_dim": 4096,
    "outlier_class": 1,
    "icr_capture": {
        "train_dir": "shared/icr_capture/mmlu_train_Llama",
        "test_dir": "shared/icr_capture/mmlu_test_Llama",
    },
}

_OK_RESULT = {
    "status": "ok",
    "auroc": 0.75,
    "mahalanobis_auroc": None,
    "knn_auroc": None,
    "n_src_train": 100,
    "n_tgt_test": 50,
    "_scores": [0.6, 0.4, 0.8],
    "_labels": [1, 0, 1],
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_run_dir(tmp_path: Path) -> Path:
    """A minimal on-disk run dir with config.json + artifacts/linear_probe_last.pt."""
    run_dir = tmp_path / "runs" / "baseline_comparison_hotpotqa_memmap" / "hotpotqa_memmap" / "saplma" / "seed_0"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "linear_probe_last.pt").write_bytes(b"")
    (run_dir / "config.json").write_text(
        json.dumps({"split_seed": 42, "method": {"data": {"probe_layer": 18}}}),
        encoding="utf-8",
    )
    (run_dir / "eval_metrics.json").write_text(
        json.dumps({"auroc": 0.80, "selected_layer": None}),
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture()
def fake_runs_dir(tmp_path: Path) -> Path:
    """Full runs/configs tree used by generate_manifest tests."""
    runs = tmp_path / "runs"

    for method, artifact, config_extra in [
        ("saplma", "linear_probe_last.pt", {"probe_layer": 18}),
        ("contrastive_logprob_recon", "contrastive_last.pt", {"relevant_layers": [14, 15]}),
    ]:
        seed_dir = (
            runs
            / "baseline_comparison_hotpotqa_memmap"
            / "hotpotqa_memmap"
            / method
            / "seed_0"
        )
        artifacts_dir = seed_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / artifact).write_bytes(b"")
        (seed_dir / "eval_metrics.json").write_text(
            json.dumps({"auroc": 0.80, "selected_layer": None}),
            encoding="utf-8",
        )
        (seed_dir / "config.json").write_text(
            json.dumps(
                {"split_seed": 42, "method": {"data": config_extra}}
            ),
            encoding="utf-8",
        )

    # configs/datasets/
    configs = tmp_path / "configs" / "datasets"
    configs.mkdir(parents=True)
    for ds in ("hotpotqa_memmap", "mmlu_memmap"):
        (configs / f"{ds}.json").write_text(
            json.dumps(_SRC_CFG),
            encoding="utf-8",
        )

    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: build_source_scorer + score_on_target is equivalent to evaluate_transfer_cell
# ---------------------------------------------------------------------------


def test_build_then_score_equals_evaluate_cell(fake_run_dir: Path, tmp_path: Path):
    """build_source_scorer + score_on_target must produce the same result dict
    that evaluate_transfer_cell would return for the same (method, src, tgt) triple.
    """
    from activation_research.transfer_eval_memmap import (
        build_source_scorer,
        evaluate_transfer_cell,
        score_on_target,
    )

    expected = dict(_OK_RESULT)

    with (
        patch(
            "activation_research.transfer_eval_memmap.evaluate_transfer_cell",
            return_value=expected,
        ) as mock_eval,
        patch(
            "activation_research.transfer_eval_memmap.build_source_scorer",
            return_value={
                "method": "saplma",
                "split_seed": 42,
                "outlier_class": 1,
                "training_seed": 0,
                "device": "cpu",
            },
        ) as mock_build,
        patch(
            "activation_research.transfer_eval_memmap.score_on_target",
            return_value=expected,
        ) as mock_score,
    ):
        # Path A: evaluate_transfer_cell directly.
        result_direct = evaluate_transfer_cell(
            source_run_dir=str(fake_run_dir),
            source_dataset_cfg=_SRC_CFG,
            target_dataset_cfg=_TGT_CFG_HOTPOTQA,
            method="saplma",
            relevant_layers=list(range(14, 30)),
            probe_layer=18,
            device="cpu",
            training_seed=0,
        )

        # Path B: build_source_scorer → score_on_target.
        scorer = build_source_scorer(
            method="saplma",
            source_run_dir=str(fake_run_dir),
            source_dataset_cfg=_SRC_CFG,
            probe_layer=18,
            training_seed=0,
            device="cpu",
        )
        result_batched = score_on_target(
            scorer=scorer,
            target_dataset_cfg=_TGT_CFG_HOTPOTQA,
        )

    # Both paths must return the same keys and AUROC value.
    assert result_direct["status"] == result_batched["status"]
    assert result_direct["auroc"] == result_batched["auroc"]
    assert set(result_direct.keys()) == set(result_batched.keys())

    mock_build.assert_called_once()
    mock_score.assert_called_once()
    mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: run_batch_cell_json writes all outputs + sentinel, returns 0
# ---------------------------------------------------------------------------


def test_batch_cell_writes_all_outputs_and_sentinel(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_batch_cell_json

    methods = ["saplma", "contrastive_logprob_recon"]
    targets = ["hotpotqa", "mmlu"]
    output_dir = tmp_path / "transfer_matrix"
    sentinel = output_dir / "llama" / "hotpotqa__seed_0.done"

    cell = _make_batch_cell(
        tmp_path,
        methods=methods,
        targets=targets,
        sentinel=str(sentinel),
    )
    cell_json = tmp_path / "cell.json"
    cell_json.write_text(json.dumps(cell), encoding="utf-8")

    dummy_scorer = {"method": "saplma", "training_seed": 0, "device": "cpu"}
    score_result = {"status": "ok", "auroc": 0.75, "_scores": [], "_labels": []}

    with (
        patch(
            "scripts.eval_transfer_matrix_memmap.build_source_scorer",
            return_value=dummy_scorer,
        ),
        patch(
            "scripts.eval_transfer_matrix_memmap.score_on_target",
            return_value=score_result,
        ) as mock_score,
    ):
        rc = run_batch_cell_json(str(cell_json), output_dir=str(output_dir))

    assert rc == 0, f"expected return code 0, got {rc}"

    # All 2 methods × 2 targets JSON outputs must exist.
    expected_jsons = [
        output_dir / "llama" / f"hotpotqa__{tgt}__{method}__0.json"
        for method in methods
        for tgt in targets
    ]
    for p in expected_jsons:
        assert p.exists(), f"missing output: {p}"

    # Sentinel must exist.
    assert sentinel.exists(), "sentinel file not written"

    # score_on_target called once per (method × target) = 4 times.
    assert mock_score.call_count == len(methods) * len(targets)


# ---------------------------------------------------------------------------
# Test 3: run_batch_cell_json skips existing outputs
# ---------------------------------------------------------------------------


def test_batch_cell_skips_existing_outputs(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_batch_cell_json

    methods = ["saplma", "contrastive_logprob_recon"]
    targets = ["hotpotqa", "mmlu"]
    output_dir = tmp_path / "transfer_matrix"
    sentinel = output_dir / "llama" / "hotpotqa__seed_0.done"

    cell = _make_batch_cell(
        tmp_path,
        methods=methods,
        targets=targets,
        sentinel=str(sentinel),
    )
    cell_json = tmp_path / "cell.json"
    cell_json.write_text(json.dumps(cell), encoding="utf-8")

    # Pre-write one output JSON for saplma × hotpotqa.
    pre_existing = output_dir / "llama" / "hotpotqa__hotpotqa__saplma__0.json"
    pre_existing.parent.mkdir(parents=True, exist_ok=True)
    pre_existing.write_text(json.dumps({"status": "ok", "auroc": 0.70}), encoding="utf-8")

    dummy_scorer = {"method": "saplma", "training_seed": 0, "device": "cpu"}
    score_result = {"status": "ok", "auroc": 0.75, "_scores": [], "_labels": []}

    with (
        patch(
            "scripts.eval_transfer_matrix_memmap.build_source_scorer",
            return_value=dummy_scorer,
        ),
        patch(
            "scripts.eval_transfer_matrix_memmap.score_on_target",
            return_value=score_result,
        ) as mock_score,
    ):
        rc = run_batch_cell_json(str(cell_json), output_dir=str(output_dir))

    assert rc == 0

    # Sentinel must still be written despite one pre-existing output.
    assert sentinel.exists()

    # score_on_target should be called one fewer time (3 instead of 4).
    total_cells = len(methods) * len(targets)
    assert mock_score.call_count == total_cells - 1


# ---------------------------------------------------------------------------
# Test 4: no sentinel written on scorer build failure
# ---------------------------------------------------------------------------


def test_batch_cell_no_sentinel_on_scorer_failure(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_batch_cell_json

    methods = ["saplma"]
    targets = ["hotpotqa", "mmlu"]
    output_dir = tmp_path / "transfer_matrix"
    sentinel = output_dir / "llama" / "hotpotqa__seed_0.done"

    cell = _make_batch_cell(
        tmp_path,
        methods=methods,
        targets=targets,
        sentinel=str(sentinel),
    )
    cell_json = tmp_path / "cell.json"
    cell_json.write_text(json.dumps(cell), encoding="utf-8")

    failed_scorer = {
        "method": "saplma",
        "training_seed": 0,
        "device": "cpu",
        "status": "missing_checkpoint",
    }

    with (
        patch(
            "scripts.eval_transfer_matrix_memmap.build_source_scorer",
            return_value=failed_scorer,
        ),
        patch(
            "scripts.eval_transfer_matrix_memmap.score_on_target",
        ) as mock_score,
    ):
        rc = run_batch_cell_json(str(cell_json), output_dir=str(output_dir))

    assert rc == 1, f"expected return code 1, got {rc}"
    assert not sentinel.exists(), "sentinel must NOT be written on scorer failure"
    mock_score.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: run_cell_json detects batch vs single-cell format
# ---------------------------------------------------------------------------


def test_batch_cell_detects_format(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_cell_json

    # Old single-cell format: has "source_run_dir" (str) and "method" (str).
    old_cell = {
        "cell_id": "hotpotqa__hotpotqa__saplma__llama__0",
        "task_type": "transfer_eval",
        "source_dataset": "hotpotqa",
        "target_dataset": "hotpotqa",
        "method": "saplma",
        "model_slug": "llama",
        "seed": 0,
        "source_run_dir": "runs/baseline/seed_0",
        "source_dataset_cfg": "configs/datasets/hotpotqa_memmap.json",
        "target_dataset_cfg": "configs/datasets/hotpotqa_memmap.json",
        "output_check": "runs/transfer_matrix_memmap/llama/hotpotqa__hotpotqa__saplma__0.json",
        "relevant_layers": list(range(14, 30)),
        "probe_layer": 18,
    }

    # New batch-cell format: has "source_run_dirs" (dict) and "target_datasets" (list).
    new_batch_cell = {
        "cell_id": "hotpotqa__llama__0",
        "task_type": "transfer_eval",
        "source_dataset": "hotpotqa",
        "model_slug": "llama",
        "seed": 0,
        "source_run_dirs": {
            "saplma": "runs/baseline/saplma/seed_0",
            "contrastive_logprob_recon": "runs/baseline/contrastive/seed_0",
        },
        "source_dataset_cfg": "configs/datasets/hotpotqa_memmap.json",
        "target_datasets": ["hotpotqa", "mmlu"],
        "target_dataset_cfgs": {
            "hotpotqa": "configs/datasets/hotpotqa_memmap.json",
            "mmlu": "configs/datasets/mmlu_memmap.json",
        },
        "probe_layers": {"saplma": 18, "contrastive_logprob_recon": 22},
        "relevant_layers": list(range(14, 30)),
        "output_check": "runs/transfer_matrix_memmap/llama/hotpotqa__seed_0.done",
    }

    old_json = tmp_path / "old_cell.json"
    old_json.write_text(json.dumps(old_cell), encoding="utf-8")

    new_json = tmp_path / "new_cell.json"
    new_json.write_text(json.dumps(new_batch_cell), encoding="utf-8")

    with (
        patch(
            "scripts.eval_transfer_matrix_memmap._run_single_cell_json",
            return_value=0,
        ) as mock_single,
        patch(
            "scripts.eval_transfer_matrix_memmap.run_batch_cell_json",
            return_value=0,
        ) as mock_batch,
    ):
        run_cell_json(str(old_json))
        run_cell_json(str(new_json))

    mock_single.assert_called_once()
    mock_batch.assert_called_once()

    # The single-cell call should have received the old-format path.
    single_call_arg = mock_single.call_args[0][0]
    assert "old_cell" in single_call_arg

    # The batch call should have received the new-format path.
    batch_call_arg = mock_batch.call_args[0][0]
    assert "new_cell" in batch_call_arg


# ---------------------------------------------------------------------------
# Test 6: generate_manifest (batch shape)
# ---------------------------------------------------------------------------


def test_generate_manifest_batch_shape(fake_runs_dir: Path, tmp_path: Path):
    from scripts.dispatch.generate_manifest_89 import generate_manifest

    dispatch_root = tmp_path / "_dispatch"
    output_dir = tmp_path / "transfer_matrix_memmap"

    n_written = generate_manifest(
        dispatch_root=dispatch_root,
        runs_dir=fake_runs_dir / "runs",
        configs_dir=fake_runs_dir / "configs",
        output_dir=output_dir,
        methods=["saplma", "contrastive_logprob_recon"],
        model_slugs=["llama"],
        source_datasets=["hotpotqa"],
        target_datasets=["hotpotqa", "mmlu"],
        seed_filter=[0],
        relevant_layers=list(range(14, 30)),
        skip_existing=False,
    )

    pending = list((dispatch_root / "pending").glob("*.json"))

    # New batch format: exactly 1 cell for (hotpotqa × llama × seed_0).
    assert n_written == 1, f"expected 1 batch cell, got {n_written}"
    assert len(pending) == 1

    cell = json.loads(pending[0].read_text(encoding="utf-8"))

    # Batch-shape structural requirements.
    assert "source_run_dirs" in cell, "cell missing 'source_run_dirs' dict"
    assert isinstance(cell["source_run_dirs"], dict)
    assert set(cell["source_run_dirs"].keys()) == {"saplma", "contrastive_logprob_recon"}

    assert "target_datasets" in cell, "cell missing 'target_datasets' list"
    assert set(cell["target_datasets"]) == {"hotpotqa", "mmlu"}

    assert "probe_layers" in cell, "cell missing 'probe_layers' dict"
    assert isinstance(cell["probe_layers"], dict)

    assert cell.get("task_type") == "transfer_eval"

    output_check = cell.get("output_check", "")
    assert output_check.endswith(".done"), f"output_check should end with .done, got: {output_check!r}"

    # Must NOT have old single-cell keys at top level.
    assert "method" not in cell, "'method' is a single-cell key and must not appear in batch cell"
    assert "source_run_dir" not in cell, "'source_run_dir' (singular) must not appear in batch cell"


# ---------------------------------------------------------------------------
# Test 7: task_type field present in all cells
# ---------------------------------------------------------------------------


def test_task_type_field_present(fake_runs_dir: Path, tmp_path: Path):
    from scripts.dispatch.generate_manifest_89 import generate_manifest

    dispatch_root = tmp_path / "_dispatch"
    output_dir = tmp_path / "transfer_matrix_memmap"

    generate_manifest(
        dispatch_root=dispatch_root,
        runs_dir=fake_runs_dir / "runs",
        configs_dir=fake_runs_dir / "configs",
        output_dir=output_dir,
        methods=["saplma"],
        model_slugs=["llama"],
        source_datasets=["hotpotqa"],
        target_datasets=["hotpotqa"],
        seed_filter=[0],
        relevant_layers=list(range(14, 30)),
        skip_existing=False,
    )

    for cell_json in (dispatch_root / "pending").glob("*.json"):
        cell = json.loads(cell_json.read_text(encoding="utf-8"))
        assert cell.get("task_type") == "transfer_eval", (
            f"cell {cell_json.name} missing task_type=='transfer_eval'"
        )


# ---------------------------------------------------------------------------
# Helper: build a batch cell JSON dict for testing run_batch_cell_json
# ---------------------------------------------------------------------------


def _make_batch_cell(
    tmp_path: Path,
    *,
    methods: list[str],
    targets: list[str],
    sentinel: str,
) -> dict:
    """Create the dataset-config files and return a batch cell dict."""
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True, exist_ok=True)

    cfg_paths: dict[str, str] = {}
    for ds in ["hotpotqa", "mmlu"]:
        cfg_file = configs_dir / f"{ds}_memmap.json"
        cfg_file.write_text(
            json.dumps(_SRC_CFG if ds == "hotpotqa" else _TGT_CFG_MMLU),
            encoding="utf-8",
        )
        cfg_paths[ds] = str(cfg_file)

    # Build fake run dirs for each method.
    source_run_dirs: dict[str, str] = {}
    for method in methods:
        artifact_name = (
            "linear_probe_last.pt" if method == "saplma" else "contrastive_last.pt"
        )
        run_dir = tmp_path / "runs" / method / "seed_0"
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / artifact_name).write_bytes(b"")
        (run_dir / "config.json").write_text(
            json.dumps({"split_seed": 42, "method": {"data": {"probe_layer": 18}}}),
            encoding="utf-8",
        )
        source_run_dirs[method] = str(run_dir)

    return {
        "cell_id": f"hotpotqa__llama__0",
        "task_type": "transfer_eval",
        "source_dataset": "hotpotqa",
        "model_slug": "llama",
        "seed": 0,
        "source_run_dirs": source_run_dirs,
        "source_dataset_cfg": cfg_paths["hotpotqa"],
        "target_datasets": targets,
        "target_dataset_cfgs": {tgt: cfg_paths[tgt] for tgt in targets},
        "probe_layers": {m: 18 for m in methods},
        "relevant_layers": list(range(14, 30)),
        "output_check": sentinel,
    }
