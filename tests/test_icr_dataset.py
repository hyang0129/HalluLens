"""
tests/test_icr_dataset.py

Tests for both modes of activation_research.icr_dataset.ICRDataset.

Coverage:
  mode="icr"  — fast-path loading from icr_scores.npy + sidecar + eval_results.json.
                Tests cover basic load/shape, path defaulting from attention_zarr_path,
                missing-path errors, stratified split label distribution, relevant_layers
                warning, out-of-bounds sample_index in meta, and no-common-keys error.
  mode="raw"  — ablations path loading via AttentionParser.get_paired().
                Tests cover returned dict structure/shapes and required-parameter
                validation (attention_zarr_path=None, relevant_layers=None).

All tests are CPU-only; no real model weights or GPU required.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import zarr

from activation_logging.attention_memmap_writer import AttentionMemmapWriter
from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from activation_research.icr_dataset import ICRDataset
from activation_research.icr_scores_writer import ICRScoresWriter


# ---------------------------------------------------------------------------
# Constants shared across fixtures
# ---------------------------------------------------------------------------

NUM_SAMPLES = 30        # large enough for stratified splits
NUM_BLOCKS = 4          # number of transformer blocks / ICR score columns
NUM_LAYERS = 2          # attention layers stored in attention dir
ACT_LAYERS = NUM_LAYERS + 1  # L+1 layers in activations.zarr
R_MAX = 8
H = 16
MODEL_NAME = "test-model/icr-v1"

SMALL_SAMPLES = 20      # must be large enough for stratified splits (val_fraction=0.15 → test=3)
SMALL_BLOCKS = 4


# ---------------------------------------------------------------------------
# Shared helpers (copied from test_attention_parser.py pattern)
# ---------------------------------------------------------------------------

def _make_keys(n: int) -> list[str]:
    return [f"key_{i:04d}" for i in range(n)]


def _make_attn_data(rng: np.random.Generator, num_layers: int, r_max: int) -> np.ndarray:
    return rng.random((num_layers, r_max, r_max)).astype(np.float16)


def _build_activations_zarr(
    zarr_path: Path,
    rng: np.random.Generator,
    sample_keys: list[str],
    response_lens: list[int],
    prompt_lens: list[int],
    act_layers: int,
    r_max: int,
    h: int,
) -> np.ndarray:
    """
    Create a synthetic activations.zarr.  Mirrors the helper in
    test_attention_parser.py.  Returns resp_acts_data for shape checks.
    """
    n = len(sample_keys)

    root = zarr.open_group(str(zarr_path), mode="w")
    arrays = root.require_group("arrays")

    prompt_acts_data = rng.random((n, act_layers, r_max, h)).astype(np.float16)
    resp_acts_data = rng.random((n, act_layers, r_max, h)).astype(np.float16)

    arrays.create_dataset("prompt_activations", data=prompt_acts_data,
                          chunks=(1, 1, r_max, h), dtype=np.float16)
    arrays.create_dataset("response_activations", data=resp_acts_data,
                          chunks=(1, 1, r_max, h), dtype=np.float16)
    arrays.create_dataset("prompt_len", data=np.array(prompt_lens, dtype=np.int32),
                          chunks=(n,), dtype=np.int32)
    arrays.create_dataset("response_len", data=np.array(response_lens, dtype=np.int32),
                          chunks=(n,), dtype=np.int32)
    sample_key_arr = arrays.require_dataset(
        "sample_key", shape=(n,), chunks=(n,), dtype=str,
        fill_value="", compressor=None, overwrite=True,
    )
    for i, k in enumerate(sample_keys):
        sample_key_arr[i] = k

    meta_dir = zarr_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "index.jsonl", "w", encoding="utf-8") as fh:
        for i, key in enumerate(sample_keys):
            entry = {"key": key, "sample_index": i,
                     "prompt_len": prompt_lens[i], "response_len": response_lens[i]}
            fh.write(json.dumps(entry) + "\n")

    act_config = {
        "model_name": MODEL_NAME,
        "num_layers": act_layers - 1,
        "r_max": r_max,
    }
    (meta_dir / "config.json").write_text(json.dumps(act_config, indent=2), encoding="utf-8")

    return resp_acts_data


def _build_icr_scores(
    out_path: Path,
    rng: np.random.Generator,
    keys: list[str],
    num_blocks: int,
) -> np.ndarray:
    """Write icr_scores.npy + sidecar; returns the data array."""
    n = len(keys)
    scores = rng.random((n, num_blocks)).astype(np.float32)
    with ICRScoresWriter(str(out_path), mode="w", n_samples=n, num_blocks=num_blocks) as w:
        for i, key in enumerate(keys):
            w.write(sample_index=i, sample_key=key, icr_vector=scores[i])
    return scores


def _build_eval_results(path: Path, keys: list[str], labels: list[int]) -> None:
    """Write a per-key dict eval_results.json (Format 1b)."""
    data = {k: int(v) for k, v in zip(keys, labels)}
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_attention_dir(
    attn_dir: Path,
    rng: np.random.Generator,
    keys: list[str],
    response_lens: list[int],
    prompt_lens: list[int],
    act_path: Path,
) -> list[np.ndarray]:
    """Write an attention dir via AttentionMemmapWriter; returns per-sample attn arrays."""
    n = len(keys)
    config_dict = {
        "source_activations_zarr": str(act_path),
        "model_name": MODEL_NAME,
        "num_layers": NUM_LAYERS,
        "r_max": R_MAX,
        "attention_region": "response_to_response",
    }
    attn_data = []
    with AttentionMemmapWriter(
        out_dir=str(attn_dir), mode="w", n_samples=n,
        num_layers=NUM_LAYERS, r_max=R_MAX,
        config_dict=config_dict, dtype="float16",
    ) as attn_w:
        for i, key in enumerate(keys):
            arr = _make_attn_data(rng, NUM_LAYERS, R_MAX)
            attn_data.append(arr)
            attn_w.write(
                sample_index=i, sample_key=key,
                response_attn=arr,
                response_len=response_lens[i],
                prompt_len=prompt_lens[i],
            )
    return attn_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_icr_dataset(tmp_path: Path):
    """
    Full fixture: icr_scores.npy, attention dir, activations.zarr, eval_results.json.
    Uses NUM_SAMPLES samples with balanced labels.
    """
    rng = np.random.default_rng(42)
    keys = _make_keys(NUM_SAMPLES)
    # Balanced labels: first half halu=0, second half halu=1.
    labels = [0] * (NUM_SAMPLES // 2) + [1] * (NUM_SAMPLES // 2)
    response_lens = [min(4 + (i % 5), R_MAX) for i in range(NUM_SAMPLES)]
    prompt_lens = [3] * NUM_SAMPLES

    act_path = tmp_path / "activations.zarr"
    _build_activations_zarr(
        zarr_path=act_path, rng=rng,
        sample_keys=keys, response_lens=response_lens, prompt_lens=prompt_lens,
        act_layers=ACT_LAYERS, r_max=R_MAX, h=H,
    )
    act_logger = ZarrActivationsLogger(zarr_path=str(act_path), read_only=True, verbose=False)

    attn_dir = tmp_path / "attention"
    _build_attention_dir(attn_dir, rng, keys, response_lens, prompt_lens, act_path)

    icr_scores_path = tmp_path / "icr_scores.npy"
    _build_icr_scores(icr_scores_path, rng, keys, NUM_BLOCKS)

    eval_path = tmp_path / "eval_results.json"
    _build_eval_results(eval_path, keys, labels)

    return {
        "tmp_path": tmp_path,
        "attention_dir": attn_dir,
        "icr_scores_path": icr_scores_path,
        "icr_scores_meta_path": tmp_path / "icr_scores_meta.jsonl",
        "eval_results_path": eval_path,
        "activations_zarr_path": act_path,
        "act_logger": act_logger,
        "keys": keys,
        "labels": labels,
    }


@pytest.fixture
def tmp_small_icr(tmp_path: Path):
    """Smaller fixture (SMALL_SAMPLES samples) for targeted icr-mode tests."""
    rng = np.random.default_rng(7)
    n = SMALL_SAMPLES
    keys = _make_keys(n)
    labels = [i % 2 for i in range(n)]

    icr_scores_path = tmp_path / "icr_scores.npy"
    _build_icr_scores(icr_scores_path, rng, keys, SMALL_BLOCKS)

    eval_path = tmp_path / "eval_results.json"
    _build_eval_results(eval_path, keys, labels)

    return {
        "tmp_path": tmp_path,
        "icr_scores_path": icr_scores_path,
        "icr_scores_meta_path": tmp_path / "icr_scores_meta.jsonl",
        "eval_results_path": eval_path,
        "keys": keys,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# mode="icr" tests
# ---------------------------------------------------------------------------

class TestIcrModeLoadsScoresAndReturnsDict:
    def test_icr_mode_loads_scores_and_returns_dict(self, tmp_small_icr: dict[str, Any]):
        """dataset[0] returns exactly {"hashkey","halu","icr_score"} with correct shape/dtype."""
        f = tmp_small_icr
        dataset = ICRDataset(
            attention_zarr_path=None,
            eval_results_path=str(f["eval_results_path"]),
            split="train",
            mode="icr",
            icr_scores_path=str(f["icr_scores_path"]),
        )
        assert len(dataset) > 0
        item = dataset[0]
        assert set(item.keys()) == {"hashkey", "halu", "icr_score"}
        assert item["icr_score"].shape == (SMALL_BLOCKS,)
        assert item["icr_score"].dtype == torch.float32
        assert item["halu"] in (0, 1)
        assert isinstance(item["hashkey"], str)


class TestIcrModePathDefaultFromAttentionZarrPath:
    def test_icr_mode_path_default_from_attention_zarr_path(self, tmp_path: Path):
        """When icr_scores_path=None, path is derived as attention_zarr_path/../icr_scores.npy."""
        rng = np.random.default_rng(11)
        n = SMALL_SAMPLES
        keys = _make_keys(n)
        labels = [i % 2 for i in range(n)]

        # Place attention dir inside a subdirectory; icr_scores.npy beside it.
        attn_dir = tmp_path / "attention"
        # Derived path: Path(attention_zarr_path).parent / "icr_scores.npy"
        # = tmp_path / "icr_scores.npy"
        icr_scores_path = tmp_path / "icr_scores.npy"
        _build_icr_scores(icr_scores_path, rng, keys, SMALL_BLOCKS)

        eval_path = tmp_path / "eval_results.json"
        _build_eval_results(eval_path, keys, labels)

        # Construct without icr_scores_path — must derive from attention_zarr_path parent.
        dataset = ICRDataset(
            attention_zarr_path=str(attn_dir),
            eval_results_path=str(eval_path),
            split="train",
            mode="icr",
            icr_scores_path=None,
        )
        assert len(dataset) > 0
        item = dataset[0]
        assert item["icr_score"].shape == (SMALL_BLOCKS,)


class TestIcrModeRequiresSomePath:
    def test_icr_mode_requires_some_path(self, tmp_path: Path):
        """Passing both attention_zarr_path=None and icr_scores_path=None raises ValueError."""
        eval_path = tmp_path / "eval_results.json"
        eval_path.write_text(json.dumps({"k": 0}), encoding="utf-8")

        with pytest.raises(ValueError, match="icr_scores_path or attention_zarr_path"):
            ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(eval_path),
                split="train",
                mode="icr",
                icr_scores_path=None,
            )


class TestIcrModeSplitPreservesLabelDistribution:
    def test_icr_mode_split_preserves_label_distribution(self, tmp_icr_dataset: dict[str, Any]):
        """All three splits should maintain roughly balanced labels (within ±20%)."""
        f = tmp_icr_dataset
        for split in ("train", "val", "test"):
            ds = ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(f["eval_results_path"]),
                split=split,
                mode="icr",
                icr_scores_path=str(f["icr_scores_path"]),
            )
            assert len(ds) > 0, f"Split '{split}' is empty"
            halos = [ds[i]["halu"] for i in range(len(ds))]
            frac_halu = sum(halos) / len(halos)
            assert 0.3 <= frac_halu <= 0.7, (
                f"Split '{split}' label distribution out of bounds: frac_halu={frac_halu:.3f}"
            )


class TestIcrModeRelevantLayersIgnoredWithWarning:
    def test_icr_mode_relevant_layers_ignored_with_warning(self, tmp_small_icr: dict[str, Any]):
        """Passing relevant_layers in mode='icr' emits UserWarning but succeeds."""
        f = tmp_small_icr
        with pytest.warns(UserWarning, match="relevant_layers is ignored"):
            dataset = ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(f["eval_results_path"]),
                split="train",
                mode="icr",
                icr_scores_path=str(f["icr_scores_path"]),
                relevant_layers=[0, 1],
            )
        assert len(dataset) > 0


class TestIcrModeMetaSampleIndexOutOfBoundsRaises:
    def test_icr_mode_meta_sample_index_out_of_bounds_raises(self, tmp_path: Path):
        """icr_scores_meta.jsonl referencing sample_index > n_rows raises ValueError at init."""
        rng = np.random.default_rng(3)

        # Write icr_scores.npy with 5 rows.
        icr_scores_path = tmp_path / "icr_scores.npy"
        with ICRScoresWriter(str(icr_scores_path), mode="w", n_samples=5, num_blocks=SMALL_BLOCKS) as w:
            for i in range(5):
                w.write(sample_index=i, sample_key=f"k_{i}",
                        icr_vector=rng.random(SMALL_BLOCKS).astype(np.float32))

        # Overwrite meta sidecar with a sample_index that is out of range (10 > 4).
        meta_path = tmp_path / "icr_scores_meta.jsonl"
        meta_path.write_text(
            json.dumps({"key": "k_0", "sample_index": 10}) + "\n",
            encoding="utf-8",
        )

        # eval_results.json needs at least one key in common.
        eval_path = tmp_path / "eval_results.json"
        _build_eval_results(eval_path, ["k_0"], [0])

        with pytest.raises(ValueError):
            ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(eval_path),
                split="train",
                mode="icr",
                icr_scores_path=str(icr_scores_path),
                icr_scores_meta_path=str(meta_path),
            )


class TestIcrModeNoCommonKeysRaises:
    def test_icr_mode_no_common_keys_raises(self, tmp_path: Path):
        """Keys in meta.jsonl and eval_results.json that share no overlap raise ValueError."""
        rng = np.random.default_rng(5)

        icr_scores_path = tmp_path / "icr_scores.npy"
        meta_keys = ["a", "b", "c"]
        with ICRScoresWriter(str(icr_scores_path), mode="w", n_samples=3, num_blocks=SMALL_BLOCKS) as w:
            for i, key in enumerate(meta_keys):
                w.write(sample_index=i, sample_key=key,
                        icr_vector=rng.random(SMALL_BLOCKS).astype(np.float32))

        # eval_results.json has completely disjoint keys.
        eval_path = tmp_path / "eval_results.json"
        _build_eval_results(eval_path, ["x", "y", "z"], [0, 1, 0])

        with pytest.raises(ValueError, match="No keys are shared"):
            ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(eval_path),
                split="train",
                mode="icr",
                icr_scores_path=str(icr_scores_path),
            )


# ---------------------------------------------------------------------------
# mode="raw" tests
# ---------------------------------------------------------------------------

class TestRawModeReturnsLegacyDict:
    def test_raw_mode_returns_legacy_dict(self, tmp_icr_dataset: dict[str, Any]):
        """dataset[0] returns exactly the six-key legacy dict with correct tensor shapes."""
        f = tmp_icr_dataset
        relevant_layers = [0, 1]
        dataset = ICRDataset(
            attention_zarr_path=str(f["attention_dir"]),
            eval_results_path=str(f["eval_results_path"]),
            relevant_layers=relevant_layers,
            split="train",
            mode="raw",
            activations_parser=f["act_logger"],
        )
        assert len(dataset) > 0
        item = dataset[0]
        assert set(item.keys()) == {
            "hashkey", "halu", "response_attn", "h_block_input", "delta_h", "response_len"
        }

        L = len(relevant_layers)
        # response_attn: (L, R_max, R_max)
        assert item["response_attn"].shape == (L, R_MAX, R_MAX)
        assert item["response_attn"].dtype == torch.float32
        # h_block_input and delta_h: (L, R_max, H)
        assert item["h_block_input"].shape == (L, R_MAX, H)
        assert item["h_block_input"].dtype == torch.float32
        assert item["delta_h"].shape == (L, R_MAX, H)
        assert item["delta_h"].dtype == torch.float32
        # response_len is a plain int
        assert isinstance(item["response_len"], int)
        assert item["halu"] in (0, 1)


class TestRawModeRequiresAttentionAndLayers:
    def test_raw_mode_requires_attention_zarr_path(self, tmp_icr_dataset: dict[str, Any]):
        """mode='raw' with attention_zarr_path=None raises ValueError."""
        f = tmp_icr_dataset
        with pytest.raises(ValueError, match="attention_zarr_path"):
            ICRDataset(
                attention_zarr_path=None,
                eval_results_path=str(f["eval_results_path"]),
                relevant_layers=[0, 1],
                split="train",
                mode="raw",
            )

    def test_raw_mode_requires_relevant_layers(self, tmp_icr_dataset: dict[str, Any]):
        """mode='raw' with relevant_layers=None raises ValueError."""
        f = tmp_icr_dataset
        with pytest.raises(ValueError, match="relevant_layers"):
            ICRDataset(
                attention_zarr_path=str(f["attention_dir"]),
                eval_results_path=str(f["eval_results_path"]),
                relevant_layers=None,
                split="train",
                mode="raw",
            )
