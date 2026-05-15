"""
tests/test_attention_parser.py

Integration-style tests for AttentionParser (activation_logging/attention_parser.py).

Fixture ``tmp_stores`` builds a synthetic attention.zarr (via AttentionZarrLogger) and
a synthetic activations.zarr (hand-built zarr arrays + ZarrActivationsLogger opened
read-only) so no real model weights or GPU are needed.

Zarr layout summary:
  attention.zarr  — written by AttentionZarrLogger
    arrays/response_attn  (3, 2, 8, 8)  float16
    arrays/sample_key     (3,)
    arrays/response_len   (3,)   int32
    arrays/prompt_len     (3,)   int32
    meta/config.json
    meta/index.jsonl

  activations.zarr  — hand-built, then opened via ZarrActivationsLogger(read_only=True)
    arrays/prompt_activations     (3, 3, 8, 16)  float16   (L+1 = 3 layers)
    arrays/response_activations   (3, 3, 8, 16)  float16
    arrays/prompt_len             (3,)            int32
    arrays/response_len           (3,)            int32
    arrays/sample_key             (3,)            str
    meta/index.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import zarr

from activation_logging.attention_zarr_logger import AttentionZarrLogger
from activation_logging.attention_parser import AttentionParser
from activation_logging.zarr_activations_logger import ZarrActivationsLogger

# ---------------------------------------------------------------------------
# Constants shared by all tests
# ---------------------------------------------------------------------------
NUM_SAMPLES = 3
NUM_LAYERS = 2       # transformer blocks (attention.zarr axis 1)
ACT_LAYERS = NUM_LAYERS + 1   # L+1 layers stored in activations.zarr
R_MAX = 8
H = 16
MODEL_NAME = "test-model/v1"

# Deterministic per-sample lengths (all <= R_MAX)
SAMPLE_KEYS = ["key_alpha", "key_beta", "key_gamma"]
RESPONSE_LENS = [4, 6, 8]   # last sample fills r_max exactly
PROMPT_LENS = [3, 5, 4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attn_data(
    rng: np.random.Generator, num_layers: int, r_max: int
) -> np.ndarray:
    """Return a random (num_layers, r_max, r_max) float16 attention array."""
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
    Create a synthetic activations.zarr by directly writing zarr arrays.

    Stores shape (N, act_layers, r_max, h) for both prompt and response activations.
    Returns the response_activations array so tests can compute expected delta_h.

    The store's meta/index.jsonl is written so ZarrActivationsLogger._load_index
    can populate _index correctly.
    """
    n = len(sample_keys)

    # Zarr group
    root = zarr.open_group(str(zarr_path), mode="w")
    arrays = root.require_group("arrays")

    # Activation arrays — float16 matching real stores
    prompt_acts_data = rng.random((n, act_layers, r_max, h)).astype(np.float16)
    resp_acts_data = rng.random((n, act_layers, r_max, h)).astype(np.float16)

    arrays.create_dataset(
        "prompt_activations",
        data=prompt_acts_data,
        chunks=(1, 1, r_max, h),
        dtype=np.float16,
    )
    arrays.create_dataset(
        "response_activations",
        data=resp_acts_data,
        chunks=(1, 1, r_max, h),
        dtype=np.float16,
    )
    arrays.create_dataset(
        "prompt_len",
        data=np.array(prompt_lens, dtype=np.int32),
        chunks=(n,),
        dtype=np.int32,
    )
    arrays.create_dataset(
        "response_len",
        data=np.array(response_lens, dtype=np.int32),
        chunks=(n,),
        dtype=np.int32,
    )
    # sample_key must use zarr's variable-length string support (like ZarrActivationsLogger)
    sample_key_arr = arrays.require_dataset(
        "sample_key",
        shape=(n,),
        chunks=(n,),
        dtype=str,
        fill_value="",
        compressor=None,
        overwrite=True,
    )
    for i, k in enumerate(sample_keys):
        sample_key_arr[i] = k

    # Meta directory + index.jsonl (required by ZarrActivationsLogger._load_index)
    meta_dir = zarr_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "index.jsonl", "w", encoding="utf-8") as fh:
        for i, key in enumerate(sample_keys):
            entry = {
                "key": key,
                "sample_index": i,
                "prompt_len": prompt_lens[i],
                "response_len": response_lens[i],
            }
            fh.write(json.dumps(entry) + "\n")

    return resp_acts_data


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_stores(tmp_path: Path):
    """
    Build synthetic attention.zarr + activations.zarr, return a dict with:
      - "attention_path":   Path to attention.zarr
      - "activations_path": Path to activations.zarr
      - "act_logger":       ZarrActivationsLogger opened read-only
      - "resp_acts_data":   np.ndarray (3, 3, 8, 16) float16 — ground-truth activations
      - "attn_data":        list of 3 np.ndarray (2, 8, 8) float16 — ground-truth attn
      - "keys":             list[str]
      - "response_lens":    list[int]
      - "prompt_lens":      list[int]
    """
    rng = np.random.default_rng(42)

    attn_path = tmp_path / "attention.zarr"
    act_path = tmp_path / "activations.zarr"

    # ------------------------------------------------------------------ #
    # 1. Build activations.zarr by hand                                   #
    # ------------------------------------------------------------------ #
    resp_acts_data = _build_activations_zarr(
        zarr_path=act_path,
        rng=rng,
        sample_keys=SAMPLE_KEYS,
        response_lens=RESPONSE_LENS,
        prompt_lens=PROMPT_LENS,
        act_layers=ACT_LAYERS,
        r_max=R_MAX,
        h=H,
    )

    # Open via ZarrActivationsLogger in read-only mode
    act_logger = ZarrActivationsLogger(
        zarr_path=str(act_path),
        read_only=True,
        verbose=False,
    )

    # ------------------------------------------------------------------ #
    # 2. Build attention.zarr via AttentionZarrLogger                     #
    # ------------------------------------------------------------------ #
    config_dict = {
        "source_activations_zarr": str(act_path),
        "model_name": MODEL_NAME,
        "num_layers": NUM_LAYERS,
        "r_max": R_MAX,
        "attention_region": "response_to_response",
    }

    attn_data_list = []
    with AttentionZarrLogger(
        zarr_path=str(attn_path),
        mode="w",
        num_layers=NUM_LAYERS,
        r_max=R_MAX,
        config_dict=config_dict,
        expected_samples=NUM_SAMPLES,
        dtype="float16",
    ) as attn_logger:
        for i, key in enumerate(SAMPLE_KEYS):
            attn_arr = _make_attn_data(rng, NUM_LAYERS, R_MAX)
            attn_data_list.append(attn_arr)
            attn_logger.write(
                sample_key=key,
                response_attn=attn_arr,
                response_len=RESPONSE_LENS[i],
                prompt_len=PROMPT_LENS[i],
            )

    return {
        "attention_path": attn_path,
        "activations_path": act_path,
        "act_logger": act_logger,
        "resp_acts_data": resp_acts_data,
        "attn_data": attn_data_list,
        "keys": SAMPLE_KEYS,
        "response_lens": RESPONSE_LENS,
        "prompt_lens": PROMPT_LENS,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetAttentionShape:
    def test_get_attention_shape(self, tmp_stores: dict[str, Any]):
        """get_attention() must return tensors of the correct shapes."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        result = parser.get_attention(SAMPLE_KEYS[0])
        assert "response_attn" in result
        assert "response_len" in result
        assert "prompt_len" in result
        # Shape must be (L, R_max, R_max)
        assert result["response_attn"].shape == (NUM_LAYERS, R_MAX, R_MAX), (
            f"Expected ({NUM_LAYERS}, {R_MAX}, {R_MAX}), got {result['response_attn'].shape}"
        )
        # dtype must be float32
        assert result["response_attn"].dtype == torch.float32


class TestGetAttentionValues:
    def test_get_attention_values(self, tmp_stores: dict[str, Any]):
        """Values round-trip through float16 storage within fp16 tolerance (1e-3)."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        for i, key in enumerate(SAMPLE_KEYS):
            result = parser.get_attention(key)
            written = torch.from_numpy(tmp_stores["attn_data"][i].astype(np.float32))
            diff = (result["response_attn"] - written).abs().max().item()
            assert diff < 1e-3, (
                f"Round-trip error for key={key}: max_abs_diff={diff:.6f} > 1e-3 (fp16 tolerance)"
            )


class TestGetAttentionResponseLen:
    def test_get_attention_response_len(self, tmp_stores: dict[str, Any]):
        """response_len in returned dict must match the value written to the store."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        for i, key in enumerate(SAMPLE_KEYS):
            result = parser.get_attention(key)
            assert result["response_len"] == RESPONSE_LENS[i], (
                f"key={key}: expected response_len={RESPONSE_LENS[i]}, got {result['response_len']}"
            )
            assert result["prompt_len"] == PROMPT_LENS[i], (
                f"key={key}: expected prompt_len={PROMPT_LENS[i]}, got {result['prompt_len']}"
            )


class TestGetPairedShapes:
    def test_get_paired_shapes(self, tmp_stores: dict[str, Any]):
        """get_paired() returns correctly-shaped tensors for each relevant layer."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        relevant_layers = [0, 1]
        key = SAMPLE_KEYS[0]
        response_len = RESPONSE_LENS[0]

        result = parser.get_paired(key, relevant_layers=relevant_layers)

        assert set(result.keys()) == {
            "h_block_input", "delta_h", "response_attn", "response_len", "prompt_len"
        }

        for b in relevant_layers:
            h_in = result["h_block_input"][b]
            dh = result["delta_h"][b]
            ra = result["response_attn"][b]

            assert h_in.shape == (response_len, H), (
                f"h_block_input[{b}] shape {h_in.shape} != ({response_len}, {H})"
            )
            assert dh.shape == (response_len, H), (
                f"delta_h[{b}] shape {dh.shape} != ({response_len}, {H})"
            )
            assert ra.shape == (response_len, response_len), (
                f"response_attn[{b}] shape {ra.shape} != ({response_len}, {response_len})"
            )

            # All float32
            assert h_in.dtype == torch.float32
            assert dh.dtype == torch.float32
            assert ra.dtype == torch.float32


class TestGetPairedDeltaH:
    def test_get_paired_delta_h(self, tmp_stores: dict[str, Any]):
        """delta_h[b] == activations[b+1] - activations[b] within fp16 round-trip tolerance."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        relevant_layers = [0, 1]

        for sample_idx, key in enumerate(SAMPLE_KEYS):
            rl = RESPONSE_LENS[sample_idx]
            result = parser.get_paired(key, relevant_layers=relevant_layers)

            resp_acts = tmp_stores["resp_acts_data"]  # (3, 3, 8, 16) float16

            for b in relevant_layers:
                # Ground-truth delta: activations[b+1] - activations[b] (float32)
                h_in_gt = torch.from_numpy(
                    resp_acts[sample_idx, b, :rl, :].astype(np.float32)
                )
                h_out_gt = torch.from_numpy(
                    resp_acts[sample_idx, b + 1, :rl, :].astype(np.float32)
                )
                expected_dh = h_out_gt - h_in_gt

                actual_dh = result["delta_h"][b]
                diff = (actual_dh - expected_dh).abs().max().item()
                assert diff < 1e-4, (
                    f"delta_h mismatch for key={key}, block={b}: "
                    f"max_abs_diff={diff:.6f} > 1e-4"
                )


class TestGetPairedLayerAlignment:
    def test_get_paired_layer_alignment(self, tmp_stores: dict[str, Any]):
        """h_block_input[0] == activations[:, 0, :response_len, :] (embedding output = block-0 input)."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        # Check all samples for block 0
        for sample_idx, key in enumerate(SAMPLE_KEYS):
            rl = RESPONSE_LENS[sample_idx]
            result = parser.get_paired(key, relevant_layers=[0])

            resp_acts = tmp_stores["resp_acts_data"]  # (3, 3, 8, 16) float16

            # activations index 0 = embedding output = block-0 input per spec §3 item 3
            expected = torch.from_numpy(
                resp_acts[sample_idx, 0, :rl, :].astype(np.float32)
            )
            actual = result["h_block_input"][0]
            diff = (actual - expected).abs().max().item()
            assert diff < 1e-4, (
                f"h_block_input[0] mismatch for key={key}: "
                f"max_abs_diff={diff:.6f} > 1e-4 (fp16 round-trip)"
            )


class TestListKeys:
    def test_list_keys(self, tmp_stores: dict[str, Any]):
        """list_keys() returns exactly all 3 written keys (order independent)."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        keys = parser.list_keys()
        assert isinstance(keys, list)
        assert sorted(keys) == sorted(SAMPLE_KEYS), (
            f"list_keys() returned {sorted(keys)}, expected {sorted(SAMPLE_KEYS)}"
        )
        assert len(parser) == NUM_SAMPLES


class TestMissingKeyRaises:
    def test_missing_key_raises(self, tmp_stores: dict[str, Any]):
        """get_attention('bad_key') must raise KeyError."""
        parser = AttentionParser(
            attention_zarr_path=str(tmp_stores["attention_path"]),
            activations_parser=tmp_stores["act_logger"],
        )
        with pytest.raises(KeyError):
            parser.get_attention("bad_key")


class TestConfigMismatchRaises:
    def test_config_mismatch_raises(self, tmp_stores: dict[str, Any], tmp_path: Path):
        """
        AttentionParser raises ValueError at __init__ when the attention store's
        model_name does not match the activations logger's reported model_name.

        The parser checks: config.get("model_name") against
        act_logger._config.get("model_name") when _config exists.
        Since ZarrActivationsLogger has no _config attribute, we monkey-patch
        a _config dict onto a fresh logger to trigger the validation path.
        """
        # Build a second attention store with a different model name
        mismatch_attn_path = tmp_path / "mismatch_attention.zarr"
        mismatch_config = {
            "source_activations_zarr": str(tmp_stores["activations_path"]),
            "model_name": "some-other-model/v2",   # intentionally wrong
            "num_layers": NUM_LAYERS,
            "r_max": R_MAX,
        }
        with AttentionZarrLogger(
            zarr_path=str(mismatch_attn_path),
            mode="w",
            num_layers=NUM_LAYERS,
            r_max=R_MAX,
            config_dict=mismatch_config,
        ) as az:
            # Write one sample so the store is valid
            az.write(
                sample_key=SAMPLE_KEYS[0],
                response_attn=np.zeros((NUM_LAYERS, R_MAX, R_MAX), dtype=np.float16),
                response_len=RESPONSE_LENS[0],
                prompt_len=PROMPT_LENS[0],
            )

        # Monkey-patch _config onto the activations logger so the parser can
        # detect the mismatch (the parser checks hasattr(act_logger, "_config"))
        act_logger = tmp_stores["act_logger"]
        act_logger._config = {"model_name": MODEL_NAME}   # matches the real model, not "some-other-model"

        with pytest.raises(ValueError, match="model_name"):
            AttentionParser(
                attention_zarr_path=str(mismatch_attn_path),
                activations_parser=act_logger,
            )
