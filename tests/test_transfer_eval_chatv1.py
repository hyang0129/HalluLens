"""Tests for activation_research/transfer_eval_chatv1.py and the --suite chatv1
mode of scripts/eval_transfer_matrix_memmap.py (issue #151/#156 chat-template
transfer matrix).

CPU-only, no GPU, no cluster access. Uses tiny synthetic icr_capture
directories (same minimal-fields pattern as
tests/test_memmap_contrastive_dataset.py's _make_full_capture_dir) and
freshly-initialized (untrained) tiny checkpoints -- these tests exercise
wiring, scorer-selection logic, run discovery, and resume/skip behavior, not
learned accuracy.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_research.model import LogprobReconProgressiveCompressor, TokenZeroMLPProbe
from activation_research.transfer_eval_chatv1 import (
    MLP_METHODS,
    TOKENWISE_METHODS,
    _method_family,
    build_source_scorer_chatv1,
    discover_runs_chatv1,
    evaluate_transfer_cell_chatv1,
    score_on_target_chatv1,
)

# ---------------------------------------------------------------------------
# Synthetic icr_capture builder (minimal fields — include_response_logprobs
# is always False on the chatv1 transfer path, so only the activation/length
# memmaps + meta.jsonl are needed; see memmap_contrastive_dataset.py:340-365).
# ---------------------------------------------------------------------------

_TEST_CFG = {
    "model_name": "fake-model",
    # hidden_dim must be >=128: ProgressiveCompressor's TransformerBlock
    # clamps num_heads to d_model // 64, so smaller values construct 0 heads.
    "num_layers": 4,
    "hidden_dim": 128,
    "r_max": 4,
    "dtype": "float16",
    "response_logprobs_top_k": 5,
    "max_prompt_len": 8,
    "max_response_len": 12,
}

def _make_capture_dir(tmp_path: Path, name: str, n_samples: int, *, seed: int = 0) -> Path:
    """Build a minimal synthetic icr_capture directory."""
    cfg = dict(_TEST_CFG)
    cfg["n_samples"] = n_samples

    out = tmp_path / name
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg))

    nl = cfg["num_layers"]
    hd = cfg["hidden_dim"]
    mr = cfg["max_response_len"]

    rng = np.random.default_rng(seed=seed)
    resp_act = rng.standard_normal(size=(n_samples, nl + 1, mr, hd)).astype(np.float16)
    mm = np.memmap(str(out / "response_activations.npy"), dtype=np.float16, mode="w+", shape=resp_act.shape)
    mm[:] = resp_act
    mm.flush()
    del mm

    response_lengths = np.full((n_samples,), 8, dtype=np.int32)
    mm = np.memmap(str(out / "response_len.npy"), dtype=np.int32, mode="w+", shape=(n_samples,))
    mm[:] = response_lengths
    mm.flush()
    del mm

    mm = np.memmap(str(out / "prompt_len.npy"), dtype=np.int32, mode="w+", shape=(n_samples,))
    mm[:] = 6
    mm.flush()
    del mm

    halu_arr = np.array([bool(i % 2) for i in range(n_samples)])
    with (out / "meta.jsonl").open("w") as fh:
        for i in range(n_samples):
            prompt = f"{name} fake prompt {i} seed{seed}"
            ph = hashlib.sha256(prompt.encode()).hexdigest()
            row = {
                "sample_index": i,
                "key": f"{name}_{i}",
                "prompt_hash": ph,
                "prompt_len": 6,
                "response_len": int(response_lengths[i]),
                "hallucinated": bool(halu_arr[i]),
                "wrote_at": "2026-01-01T00:00:00",
            }
            fh.write(json.dumps(row) + "\n")

    return out


def _write_run_dir(tmp_path: Path, name: str, model: torch.nn.Module, method_cfg: dict, *, split_seed: int = 42) -> Path:
    run_dir = tmp_path / name
    (run_dir / "artifacts").mkdir(parents=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "training_summary": {}},
        run_dir / "artifacts" / "final_weights.pt",
    )
    (run_dir / "config.json").write_text(
        json.dumps({"split_seed": split_seed, "method": method_cfg, "training_seed": 0})
    )
    return run_dir


def _dataset_cfg(capture_dir: Path) -> dict:
    return {
        "input_dim": _TEST_CFG["hidden_dim"],
        "outlier_class": 1,
        "icr_capture": {
            "train_dir": str(capture_dir),
            "test_dir": str(capture_dir),
        },
    }


# ---------------------------------------------------------------------------
# _method_family
# ---------------------------------------------------------------------------


def test_method_family_classifies_tokenwise():
    for m in TOKENWISE_METHODS:
        assert _method_family(m) == "tokenwise_contrastive"


def test_method_family_classifies_mlp():
    for m in MLP_METHODS:
        assert _method_family(m) == "token_zero_mlp_probe"


def test_method_family_rejects_unknown_method():
    with pytest.raises(ValueError):
        _method_family("act_vit")


# ---------------------------------------------------------------------------
# discover_runs_chatv1
# ---------------------------------------------------------------------------


def test_discover_runs_chatv1_finds_ready_and_not_ready(tmp_path: Path):
    runs = tmp_path / "runs"

    # llama, ready (checkpoint present).
    ready_dir = (
        runs / "chatv1_hotpotqa" / "hotpotqa_chat_memmap"
        / "tokenwise_contrastive_first_anchored" / "seed_0"
    )
    (ready_dir / "artifacts").mkdir(parents=True)
    (ready_dir / "config.json").write_text("{}")
    (ready_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    # qwen3, not ready (config written, still training -- no checkpoint yet).
    training_dir = (
        runs / "chatv1_qwen3_nq" / "nq_qwen3_chat_memmap"
        / "token_zero_mlp_probe_qwen3" / "seed_0"
    )
    (training_dir / "artifacts").mkdir(parents=True)
    (training_dir / "config.json").write_text("{}")

    # non-chatv1 experiment dir -- must be ignored entirely.
    legacy_dir = (
        runs / "baseline_comparison_hotpotqa_memmap" / "hotpotqa_memmap"
        / "saplma" / "seed_0"
    )
    (legacy_dir / "artifacts").mkdir(parents=True)
    (legacy_dir / "config.json").write_text("{}")
    (legacy_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    # method not in CHATV1_METHODS -- must be ignored.
    unknown_method_dir = (
        runs / "chatv1_hotpotqa" / "hotpotqa_chat_memmap" / "act_vit" / "seed_0"
    )
    (unknown_method_dir / "artifacts").mkdir(parents=True)
    (unknown_method_dir / "config.json").write_text("{}")
    (unknown_method_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    results = discover_runs_chatv1(str(runs))
    by_key = {(r["method"], r["dataset"], r["model_slug"]): r for r in results}

    key_ready = ("tokenwise_contrastive_first_anchored", "hotpotqa", "llama")
    assert key_ready in by_key
    assert by_key[key_ready]["ready"] is True
    assert by_key[key_ready]["run_dir"] == str(ready_dir)

    key_training = ("token_zero_mlp_probe_qwen3", "nq", "qwen3")
    assert key_training in by_key
    assert by_key[key_training]["ready"] is False

    assert ("saplma", "hotpotqa", "llama") not in by_key
    assert ("act_vit", "hotpotqa", "llama") not in by_key


def test_discover_runs_chatv1_missing_root_returns_empty(tmp_path: Path):
    assert discover_runs_chatv1(str(tmp_path / "does_not_exist")) == []


def test_discover_runs_chatv1_ignores_run_dir_without_config(tmp_path: Path):
    runs = tmp_path / "runs"
    seed_dir = (
        runs / "chatv1_sciq" / "sciq_chat_memmap"
        / "tokenwise_arch_v1_input_norm_only" / "seed_0"
    )
    seed_dir.mkdir(parents=True)  # directory pre-created by dispatch, nothing written yet

    results = discover_runs_chatv1(str(runs))
    assert results == []


# ---------------------------------------------------------------------------
# build_source_scorer_chatv1 / score_on_target_chatv1 status handling
# ---------------------------------------------------------------------------


def test_build_scorer_missing_config_is_missing_artifact(tmp_path: Path):
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    scorer = build_source_scorer_chatv1(
        method="token_zero_mlp_probe",
        source_run_dir=str(run_dir),
        source_dataset_cfg={"input_dim": 128, "outlier_class": 1,
                             "icr_capture": {"train_dir": "x", "test_dir": "y"}},
        training_seed=0,
    )
    assert scorer["status"] == "missing_artifact"


def test_build_scorer_missing_checkpoint_is_missing_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"split_seed": 42, "method": {"data": {"relevant_layers": "1-4"}}})
    )
    scorer = build_source_scorer_chatv1(
        method="token_zero_mlp_probe",
        source_run_dir=str(run_dir),
        source_dataset_cfg={"input_dim": 128, "outlier_class": 1,
                             "icr_capture": {"train_dir": "x", "test_dir": "y"}},
        training_seed=0,
    )
    assert scorer["status"] == "missing_checkpoint"


def test_score_on_target_propagates_scorer_status_without_touching_data():
    scorer = {"status": "missing_checkpoint", "n_src_train": None}
    result = score_on_target_chatv1(scorer, {"icr_capture": {"test_dir": "/nonexistent"}})
    assert result["status"] == "missing_checkpoint"
    assert result["auroc"] is None


# ---------------------------------------------------------------------------
# End-to-end scoring: tiny synthetic checkpoints + captures.
# ---------------------------------------------------------------------------


def test_tokenwise_family_end_to_end_transfer(tmp_path: Path):
    src_capture = _make_capture_dir(tmp_path, "src_capture", n_samples=24, seed=0)
    tgt_capture = _make_capture_dir(tmp_path, "tgt_capture", n_samples=16, seed=1)

    model = LogprobReconProgressiveCompressor(
        input_dim=_TEST_CFG["hidden_dim"], final_dim=64, recon_seq_len=12,
    )
    method_cfg = {
        "model_params": {"final_dim": 64, "recon_seq_len": 12},
        "data": {
            "relevant_layers": "1-4",
            "pad_length": _TEST_CFG["max_response_len"],
            "token_pair_mode": "first_anchored",
        },
        "evaluation": {
            "cosine_knn_params": {
                "k": 3, "metric": "cosine", "l2_normalize": True,
                "calibrate_k": False, "max_train_size": 200000,
            },
        },
    }
    run_dir = _write_run_dir(tmp_path, "tokenwise_run", model, method_cfg)

    result = evaluate_transfer_cell_chatv1(
        method="tokenwise_contrastive_first_anchored",
        source_run_dir=str(run_dir),
        source_dataset_cfg=_dataset_cfg(src_capture),
        target_dataset_cfg=_dataset_cfg(tgt_capture),
        training_seed=0,
        device="cpu",
    )

    assert result["status"] in ("ok", "single_class")
    if result["status"] == "ok":
        assert 0.0 <= result["auroc"] <= 1.0
    assert result["n_test"] == 16
    assert result["n_src_train"] is not None and result["n_src_train"] > 0


def test_tokenwise_family_diagonal_uses_own_train_split_as_bank(tmp_path: Path):
    """source == target (the matrix diagonal) must still route the KNN bank
    through the SOURCE dataset's train split, not the test split being scored
    -- i.e. n_src_train should reflect the ~90% three-way train subset, not
    the full capture count."""
    capture = _make_capture_dir(tmp_path, "diag_capture", n_samples=24, seed=0)

    model = LogprobReconProgressiveCompressor(
        input_dim=_TEST_CFG["hidden_dim"], final_dim=64, recon_seq_len=12,
    )
    method_cfg = {
        "model_params": {"final_dim": 64, "recon_seq_len": 12},
        "data": {"relevant_layers": "1-4", "pad_length": _TEST_CFG["max_response_len"]},
        "evaluation": {
            "cosine_knn_params": {
                "k": 3, "metric": "cosine", "l2_normalize": True,
                "calibrate_k": False, "max_train_size": 200000,
            },
        },
    }
    run_dir = _write_run_dir(tmp_path, "diag_run", model, method_cfg)
    cfg = _dataset_cfg(capture)

    result = evaluate_transfer_cell_chatv1(
        method="tokenwise_contrastive_first_anchored",
        source_run_dir=str(run_dir),
        source_dataset_cfg=cfg,
        target_dataset_cfg=cfg,
        training_seed=0,
        device="cpu",
    )
    assert result["status"] in ("ok", "single_class")
    # 90/10 three-way split of 24 rows -> bank should be strictly smaller than
    # the full 24-row capture (never trained on the rows it evaluates twice).
    assert 0 < result["n_src_train"] < 24


def test_mlp_family_end_to_end_transfer(tmp_path: Path):
    tgt_capture = _make_capture_dir(tmp_path, "mlp_tgt_capture", n_samples=16, seed=2)

    model = TokenZeroMLPProbe(
        input_dim=_TEST_CFG["hidden_dim"], num_layers=4, hidden_dim=32, output_dim=16,
        normalize_input=True,
    )
    method_cfg = {
        "data": {"relevant_layers": "1-4", "pad_length": _TEST_CFG["max_response_len"]},
        "model_params": {"hidden_dim": 32, "output_dim": 16, "dropout": 0.1, "normalize_input": True},
    }
    run_dir = _write_run_dir(tmp_path, "mlp_run", model, method_cfg)
    cfg = _dataset_cfg(tgt_capture)

    result = evaluate_transfer_cell_chatv1(
        method="token_zero_mlp_probe",
        source_run_dir=str(run_dir),
        source_dataset_cfg=cfg,
        target_dataset_cfg=cfg,
        training_seed=0,
        device="cpu",
    )

    assert result["status"] in ("ok", "single_class")
    assert result["n_test"] == 16
    assert result["n_src_train"] is None  # supervised probe, no bank


# ---------------------------------------------------------------------------
# scripts/eval_transfer_matrix_memmap.py --suite chatv1 (run_chatv1_suite)
# ---------------------------------------------------------------------------


def _write_chatv1_dataset_configs(configs_dir: Path, datasets: list[str]) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        cfg = {
            "input_dim": 4096, "outlier_class": 1,
            "icr_capture": {"train_dir": f"shared/{ds}_train", "test_dir": f"shared/{ds}_test"},
        }
        (configs_dir / f"{ds}_chat_memmap.json").write_text(json.dumps(cfg))


def test_run_chatv1_suite_writes_csv_json_and_reuses_scorer(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_chatv1_suite

    runs = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    output_dir = tmp_path / "out"

    _write_chatv1_dataset_configs(configs_root / "datasets", ["hotpotqa", "nq"])

    run_dir = (
        runs / "chatv1_hotpotqa" / "hotpotqa_chat_memmap"
        / "tokenwise_contrastive_first_anchored" / "seed_0"
    )
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"split_seed": 42, "method": {}}))
    (run_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    dummy_scorer = {
        "method": "tokenwise_contrastive_first_anchored", "family": "tokenwise_contrastive",
        "training_seed": 0, "device": "cpu", "n_src_train": 100,
    }
    ok_result = {"status": "ok", "auroc": 0.65, "n_test": 50, "n_src_train": 100}

    with (
        patch(
            "scripts.eval_transfer_matrix_memmap.build_source_scorer_chatv1",
            return_value=dummy_scorer,
        ) as mock_build,
        patch(
            "scripts.eval_transfer_matrix_memmap.score_on_target_chatv1",
            return_value=ok_result,
        ) as mock_score,
    ):
        run_chatv1_suite(
            runs_dir=str(runs),
            configs_dir=str(configs_root),
            output_dir=str(output_dir),
            source_datasets=["hotpotqa"],
            target_datasets=["hotpotqa", "nq"],
            model_slugs=["llama"],
            methods=["tokenwise_contrastive_first_anchored"],
            resume=False,
            device="cpu",
        )

    # One run x two targets: the scorer is built once and reused (lazy build
    # inside the target loop), score_on_target_chatv1 called once per target.
    assert mock_build.call_count == 1
    assert mock_score.call_count == 2

    csv_path = output_dir / "transfer_matrix_chatv1.csv"
    json_path = output_dir / "transfer_matrix_chatv1.json"
    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert list(df.columns) == [
        "model", "method", "source", "target", "seed", "auroc", "n_test", "n_src_train", "status",
    ]
    assert len(df) == 2
    assert set(df["target"]) == {"hotpotqa", "nq"}
    assert set(df["source"]) == {"hotpotqa"}
    assert (df["status"] == "ok").all()

    with open(json_path) as f:
        records = json.load(f)
    assert len(records) == 2


def test_run_chatv1_suite_resume_skips_existing_cells(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_chatv1_suite

    runs = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    output_dir = tmp_path / "out"

    _write_chatv1_dataset_configs(configs_root / "datasets", ["hotpotqa"])

    run_dir = (
        runs / "chatv1_hotpotqa" / "hotpotqa_chat_memmap"
        / "tokenwise_contrastive_first_anchored" / "seed_0"
    )
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"split_seed": 42, "method": {}}))
    (run_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    dummy_scorer = {
        "method": "tokenwise_contrastive_first_anchored", "family": "tokenwise_contrastive",
        "training_seed": 0, "device": "cpu", "n_src_train": 100,
    }
    ok_result = {"status": "ok", "auroc": 0.5, "n_test": 10, "n_src_train": 100}

    common_kwargs = dict(
        runs_dir=str(runs),
        configs_dir=str(configs_root),
        output_dir=str(output_dir),
        source_datasets=["hotpotqa"],
        target_datasets=["hotpotqa"],
        model_slugs=["llama"],
        methods=["tokenwise_contrastive_first_anchored"],
        device="cpu",
    )

    with (
        patch("scripts.eval_transfer_matrix_memmap.build_source_scorer_chatv1", return_value=dummy_scorer),
        patch("scripts.eval_transfer_matrix_memmap.score_on_target_chatv1", return_value=ok_result) as mock_score,
    ):
        run_chatv1_suite(resume=False, **common_kwargs)
    assert mock_score.call_count == 1

    # Second invocation with resume=True must skip the already-written cell.
    with (
        patch("scripts.eval_transfer_matrix_memmap.build_source_scorer_chatv1") as mock_build2,
        patch("scripts.eval_transfer_matrix_memmap.score_on_target_chatv1") as mock_score2,
    ):
        run_chatv1_suite(resume=True, **common_kwargs)
    mock_build2.assert_not_called()
    mock_score2.assert_not_called()


def test_run_chatv1_suite_skips_not_ready_runs_without_error(tmp_path: Path):
    from scripts.eval_transfer_matrix_memmap import run_chatv1_suite

    runs = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    output_dir = tmp_path / "out"

    _write_chatv1_dataset_configs(configs_root / "datasets", ["hotpotqa"])

    # Run dir has config.json but no checkpoint -- still training.
    run_dir = (
        runs / "chatv1_hotpotqa" / "hotpotqa_chat_memmap"
        / "tokenwise_contrastive_first_anchored" / "seed_0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"split_seed": 42, "method": {}}))

    with (
        patch("scripts.eval_transfer_matrix_memmap.build_source_scorer_chatv1") as mock_build,
        patch("scripts.eval_transfer_matrix_memmap.score_on_target_chatv1") as mock_score,
    ):
        # Must not raise -- missing checkpoints are skipped with a warning.
        run_chatv1_suite(
            runs_dir=str(runs),
            configs_dir=str(configs_root),
            output_dir=str(output_dir),
            source_datasets=["hotpotqa"],
            target_datasets=["hotpotqa"],
            model_slugs=["llama"],
            methods=["tokenwise_contrastive_first_anchored"],
            resume=False,
            device="cpu",
        )

    mock_build.assert_not_called()
    mock_score.assert_not_called()


def test_run_chatv1_suite_within_model_only(tmp_path: Path):
    """A qwen3 run must never be scored against a llama-only target request
    (and vice versa) -- transfer is within-model only (32 vs 36 captured
    layers are architecture-incompatible)."""
    from scripts.eval_transfer_matrix_memmap import run_chatv1_suite

    runs = tmp_path / "runs"
    configs_root = tmp_path / "configs"
    output_dir = tmp_path / "out"
    _write_chatv1_dataset_configs(configs_root / "datasets", ["hotpotqa"])

    qwen_run_dir = (
        runs / "chatv1_qwen3_hotpotqa" / "hotpotqa_qwen3_chat_memmap"
        / "tokenwise_contrastive_first_anchored_qwen3" / "seed_0"
    )
    (qwen_run_dir / "artifacts").mkdir(parents=True)
    (qwen_run_dir / "config.json").write_text(json.dumps({"split_seed": 42, "method": {}}))
    (qwen_run_dir / "artifacts" / "final_weights.pt").write_bytes(b"x")

    with (
        patch("scripts.eval_transfer_matrix_memmap.build_source_scorer_chatv1") as mock_build,
        patch("scripts.eval_transfer_matrix_memmap.score_on_target_chatv1") as mock_score,
    ):
        # Only llama requested -- the qwen3 run must be filtered out entirely.
        run_chatv1_suite(
            runs_dir=str(runs),
            configs_dir=str(configs_root),
            output_dir=str(output_dir),
            source_datasets=["hotpotqa"],
            target_datasets=["hotpotqa"],
            model_slugs=["llama"],
            methods=["tokenwise_contrastive_first_anchored", "tokenwise_contrastive_first_anchored_qwen3"],
            resume=False,
            device="cpu",
        )

    mock_build.assert_not_called()
    mock_score.assert_not_called()
