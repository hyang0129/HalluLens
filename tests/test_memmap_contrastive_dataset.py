"""Tests for MemmapContrastiveDataset.

All tests are CPU-only and synthesise fake Issue #72 capture directories
in tmp_path. No real model, GPU, or actual #72 captures needed.

Run with:
    pytest tests/test_memmap_contrastive_dataset.py -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from activation_research.memmap_contrastive_dataset import (
    MemmapContrastiveDataset,
    _ATTN_STAT_DIM,
    _compute_attn_stats,
)


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

_SMALL_CFG = {
    "model_name": "fake-model",
    "num_layers": 4,
    # hidden_dim must be ≥ 128 because ProgressiveCompressor's
    # TransformerBlock clamps num_heads to d_model // 64, so anything below
    # 64 produces 0 heads and torch errors out at construction. We pick 128
    # so the model tests can use input_dim=128, final_dim=64.
    "hidden_dim": 128,
    "r_max": 6,
    "dtype": "float16",
    "response_logprobs_top_k": 5,
    "max_prompt_len": 16,
    "max_response_len": 12,
}


def _make_full_capture_dir(
    tmp_path: Path,
    n_samples: int,
    *,
    n_meta: int | None = None,
    halu_pattern: np.ndarray | None = None,
    response_lengths: np.ndarray | None = None,
) -> Path:
    """Build a synthetic InferenceCaptureWriter output directory with ALL
    fields the contrastive dataset can request (activations, attention,
    logprobs, lengths)."""
    if n_meta is None:
        n_meta = n_samples

    cfg = dict(_SMALL_CFG)
    cfg["n_samples"] = n_samples

    out = tmp_path / "capture"
    out.mkdir()

    (out / "config.json").write_text(json.dumps(cfg))

    nl = cfg["num_layers"]
    hd = cfg["hidden_dim"]
    rm = cfg["r_max"]
    mr = cfg["max_response_len"]
    mp = cfg["max_prompt_len"]
    tk = cfg["response_logprobs_top_k"]

    rng = np.random.default_rng(seed=0)

    # response_activations: deterministic but non-zero, fp16 representable.
    resp_act = rng.standard_normal(
        size=(n_samples, nl + 1, mr, hd), dtype=np.float32
    ).astype(np.float16)
    mm = np.memmap(
        str(out / "response_activations.npy"), dtype=np.float16, mode="w+",
        shape=resp_act.shape,
    )
    mm[:] = resp_act
    mm.flush()
    del mm

    # response_attention: row-normalised so the stats computation is meaningful.
    raw_attn = rng.uniform(0.01, 1.0, size=(n_samples, nl, rm, rm)).astype(np.float32)
    raw_attn = raw_attn / raw_attn.sum(axis=-1, keepdims=True)
    raw_attn = raw_attn.astype(np.float16)
    mm = np.memmap(
        str(out / "response_attention.npy"), dtype=np.float16, mode="w+",
        shape=raw_attn.shape,
    )
    mm[:] = raw_attn
    mm.flush()
    del mm

    # prompt_activations
    prompt_act = rng.standard_normal(
        size=(n_samples, nl + 1, mp, hd), dtype=np.float32
    ).astype(np.float16)
    mm = np.memmap(
        str(out / "prompt_activations.npy"), dtype=np.float16, mode="w+",
        shape=prompt_act.shape,
    )
    mm[:] = prompt_act
    mm.flush()
    del mm

    # lengths
    if response_lengths is None:
        response_lengths = np.full((n_samples,), 8, dtype=np.int32)
    else:
        response_lengths = np.asarray(response_lengths, dtype=np.int32)
        assert response_lengths.shape == (n_samples,)
    mm = np.memmap(str(out / "response_len.npy"), dtype=np.int32, mode="w+",
                   shape=(n_samples,))
    mm[:] = response_lengths
    mm.flush()
    del mm

    mm = np.memmap(str(out / "prompt_len.npy"), dtype=np.int32, mode="w+",
                   shape=(n_samples,))
    mm[:] = 10
    mm.flush()
    del mm

    # token IDs / logprobs / topk — NaN/-1 padded past response_lengths
    token_ids = np.full((n_samples, mr), -1, dtype=np.int32)
    token_lps = np.full((n_samples, mr), np.nan, dtype=np.float32)
    topk_ids = np.full((n_samples, mr, tk), -1, dtype=np.int32)
    topk_lps = np.full((n_samples, mr, tk), np.nan, dtype=np.float32)
    for i in range(n_samples):
        rl = int(response_lengths[i])
        if rl <= 0:
            continue
        token_ids[i, :rl] = rng.integers(0, 1000, size=rl, dtype=np.int32)
        token_lps[i, :rl] = rng.uniform(-5.0, 0.0, size=rl).astype(np.float32)
        topk_ids[i, :rl, :] = rng.integers(0, 1000, size=(rl, tk), dtype=np.int32)
        topk_lps[i, :rl, :] = rng.uniform(-10.0, 0.0, size=(rl, tk)).astype(np.float32)

    for name, arr in [
        ("response_token_ids.npy", token_ids),
        ("response_token_logprobs.npy", token_lps),
        ("response_topk_token_ids.npy", topk_ids),
        ("response_topk_logprobs.npy", topk_lps),
    ]:
        mm = np.memmap(str(out / name), dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        del mm

    # icr_scores.npy
    icr = rng.standard_normal(size=(n_samples, nl)).astype(np.float32)
    np.save(str(out / "icr_scores.npy"), icr)

    # meta.jsonl
    if halu_pattern is None:
        halu_arr = np.array([bool(i % 2) for i in range(n_meta)])
    else:
        halu_arr = np.asarray(halu_pattern, dtype=bool)
    assert len(halu_arr) == n_meta

    with (out / "meta.jsonl").open("w") as fh:
        for i in range(n_meta):
            prompt = f"fake prompt {i}"
            ph = hashlib.sha256(prompt.encode()).hexdigest()
            row = {
                "sample_index": i,
                "key": f"fake_{i}",
                "prompt_hash": ph,
                "prompt_len": 10,
                "response_len": int(response_lengths[i]),
                "hallucinated": bool(halu_arr[i]),
                "wrote_at": "2026-01-01T00:00:00",
            }
            fh.write(json.dumps(row) + "\n")

    # eval_results.json
    eval_res = {
        "halu_test_res": [bool(x) for x in halu_arr],
        "abstantion": [False] * n_meta,
    }
    (out / "eval_results.json").write_text(json.dumps(eval_res))

    return out


# ---------------------------------------------------------------------------
# Attention stats helper
# ---------------------------------------------------------------------------

def test_attn_stats_zero_response_returns_nan():
    block = np.zeros((6, 6), dtype=np.float16)
    out = _compute_attn_stats(block, response_len=0)
    assert out.shape == (_ATTN_STAT_DIM,)
    assert np.all(np.isnan(out))


def test_attn_stats_uniform_attention_high_entropy():
    # Uniform attention over a 4-token response → entropy = ln(4).
    block = np.zeros((6, 6), dtype=np.float32)
    block[:4, :4] = 0.25
    out = _compute_attn_stats(block.astype(np.float16), response_len=4)
    expected_entropy = np.log(4.0)
    assert abs(out[0] - expected_entropy) < 1e-2
    # focal_frac for a uniform row = 0.25
    assert abs(out[1] - 0.25) < 1e-2
    # self-mass on diagonal = 0.25
    assert abs(out[2] - 0.25) < 1e-2


def test_attn_stats_focal_attention_low_entropy():
    # Each query attends entirely to position 0 → entropy ≈ 0, focal_frac=1.
    block = np.zeros((6, 6), dtype=np.float32)
    block[:4, 0] = 1.0
    out = _compute_attn_stats(block.astype(np.float16), response_len=4)
    assert out[0] < 0.5
    assert abs(out[1] - 1.0) < 1e-2
    # self-mass is the diagonal — only position 0 has self=1 here,
    # so the mean over 4 rows is 0.25.
    assert abs(out[2] - 0.25) < 1e-2


# ---------------------------------------------------------------------------
# Dataset construction + dict shape
# ---------------------------------------------------------------------------

def test_dataset_minimal_construction(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=20)
    ds = MemmapContrastiveDataset(capture, split="all", num_views=2)
    assert len(ds) == 20

    sample = ds[0]
    assert set(sample.keys()) >= {
        "hashkey", "halu", "views_activations", "view_indices", "input_length",
    }
    assert isinstance(sample["hashkey"], str)
    assert sample["halu"].dtype == torch.float32
    assert sample["views_activations"].shape == (2, _SMALL_CFG["max_response_len"], _SMALL_CFG["hidden_dim"])
    assert sample["views_activations"].dtype == torch.float32
    assert sample["view_indices"].shape == (2,)
    assert sample["view_indices"].dtype == torch.long
    assert isinstance(sample["input_length"], int)


def test_dataset_emits_logprob_fields(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        include_response_logprobs=True, response_logprobs_top_k=5, pad_length=12,
    )
    sample = ds[0]

    assert "response_token_logprobs" in sample
    assert sample["response_token_logprobs"].shape == (12,)
    assert sample["response_token_logprobs"].dtype == torch.float32
    # First 8 positions (default response_len=8) should be finite; rest NaN.
    lp = sample["response_token_logprobs"]
    assert torch.isfinite(lp[:8]).all()
    assert torch.isnan(lp[8:]).all()

    assert sample["response_token_ids"].shape == (12,)
    assert sample["response_topk_token_ids"].shape == (12, 5)
    assert sample["response_topk_logprobs"].shape == (12, 5)
    assert sample["response_logprob_mask"].sum() == 8


def test_dataset_emits_attention_backward_field(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    # relevant_layers covers indices 1..4 in the (num_layers+1=5) axis.
    # backward offset=2 → for view at layer 3, target attention layer = 1 (valid).
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[2, 3, 4],
        include_response_attention=True,
        attention_target_layer_offset_backward=2,
    )
    sample = ds[0]

    assert "attention_backward" in sample
    assert "attention_forward" not in sample
    assert sample["attention_backward"].shape == (2, _ATTN_STAT_DIM)
    # For these layer choices, all targets fall within [0, num_layers-1]=[0,3]
    # so no NaN rows.
    assert torch.isfinite(sample["attention_backward"]).all()


def test_dataset_attention_out_of_range_yields_nan_rows(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    # fixed_layer=0 with backward offset=2 → target attention layer = -2 (out of range).
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=1,
        relevant_layers=[0],
        fixed_layer=0,
        include_response_attention=True,
        attention_target_layer_offset_backward=2,
    )
    sample = ds[0]
    assert torch.isnan(sample["attention_backward"]).all()


def test_dataset_attention_both_directions_emitted(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3],
        include_response_attention=True,
        attention_target_layer_offset_forward=0,
        attention_target_layer_offset_backward=0,
    )
    sample = ds[0]
    assert "attention_forward" in sample
    assert "attention_backward" in sample
    # offset=0 → forward and backward target the same model layer, so the
    # stats arrays must be identical for the same view layers.
    torch.testing.assert_close(sample["attention_forward"], sample["attention_backward"])


def test_dataset_field_omission(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    ds = MemmapContrastiveDataset(capture, split="all", num_views=2)
    sample = ds[0]
    assert "response_token_logprobs" not in sample
    assert "attention_forward" not in sample
    assert "attention_backward" not in sample


# ---------------------------------------------------------------------------
# Split logic (delegated to shared helper — sanity check only)
# ---------------------------------------------------------------------------

def test_dataset_split_disjoint(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=40)
    sizes = {}
    indices = {}
    for s in ("train", "val", "test"):
        ds = MemmapContrastiveDataset(capture, split=s, num_views=2, random_seed=42)
        sizes[s] = len(ds)
        indices[s] = set(int(ds._split_indices[i]) for i in range(len(ds)))
    assert indices["train"].isdisjoint(indices["val"])
    assert indices["train"].isdisjoint(indices["test"])
    assert indices["val"].isdisjoint(indices["test"])
    assert sum(sizes.values()) == 40


# ---------------------------------------------------------------------------
# View determinism
# ---------------------------------------------------------------------------

def test_view_determinism_with_fixed_layer(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        fixed_layer=2,
    )
    # First view should always be the position of fixed_layer in relevant_layers.
    fixed_pos = ds._relevant_layers.index(2)
    for i in range(len(ds)):
        sample = ds[i]
        assert int(sample["view_indices"][0]) == fixed_pos


# ---------------------------------------------------------------------------
# ValueError on invalid summary mode
# ---------------------------------------------------------------------------

def test_invalid_attention_summary_raises(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=5)
    with pytest.raises(ValueError):
        MemmapContrastiveDataset(
            capture, split="all", num_views=2,
            include_response_attention=True, attention_summary="hexgrid",
        )


# ---------------------------------------------------------------------------
# Coarse attention summary
# ---------------------------------------------------------------------------

def test_coarse_attention_summary_shape(tmp_path):
    """Coarse target: (num_views, num_layers, 8, 8) per direction."""
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    nl = _SMALL_CFG["num_layers"]
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_attention=True,
        attention_summary="coarse",
        attention_target_layer_offset_forward=1,
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert "attention_forward" in sample
    assert "attention_backward" in sample
    assert sample["attention_forward"].shape == (2, nl, 8, 8)
    assert sample["attention_backward"].shape == (2, nl, 8, 8)
    assert sample["attention_forward"].dtype == torch.float32


def test_coarse_backward_only_emitted(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    nl = _SMALL_CFG["num_layers"]
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_attention=True,
        attention_summary="coarse",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert "attention_backward" in sample
    assert "attention_forward" not in sample
    assert sample["attention_backward"].shape == (2, nl, 8, 8)


def test_coarse_no_nan_in_full_response(tmp_path):
    """When all response tokens fill r_max, no NaN cells should appear in coarse."""
    n = 6
    r_max = _SMALL_CFG["r_max"]
    # Set response_len = r_max so all cells are valid.
    rls = np.full(n, r_max, dtype=np.int32)
    capture = _make_full_capture_dir(tmp_path, n_samples=n, response_lengths=rls)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_attention=True,
        attention_summary="coarse",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert torch.isfinite(sample["attention_backward"]).all()


def test_coarse_zero_response_yields_nan(tmp_path):
    """response_len=0 → all NaN output for coarse."""
    n = 4
    rls = np.zeros(n, dtype=np.int32)
    capture = _make_full_capture_dir(tmp_path, n_samples=n, response_lengths=rls)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=1,
        relevant_layers=[1],
        fixed_layer=1,
        include_response_attention=True,
        attention_summary="coarse",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert torch.isnan(sample["attention_backward"]).all()


# ---------------------------------------------------------------------------
# Full attention summary
# ---------------------------------------------------------------------------

def test_full_attention_summary_shape(tmp_path):
    """Full target: (num_views, num_layers, r_max, r_max) per direction."""
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    nl = _SMALL_CFG["num_layers"]
    rm = _SMALL_CFG["r_max"]
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_attention=True,
        attention_summary="full",
        attention_target_layer_offset_forward=1,
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert "attention_forward" in sample
    assert "attention_backward" in sample
    assert sample["attention_forward"].shape == (2, nl, rm, rm)
    assert sample["attention_backward"].shape == (2, nl, rm, rm)
    assert sample["attention_forward"].dtype == torch.float32


def test_full_backward_only_emitted(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=10)
    nl = _SMALL_CFG["num_layers"]
    rm = _SMALL_CFG["r_max"]
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_attention=True,
        attention_summary="full",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert "attention_backward" in sample
    assert "attention_forward" not in sample
    assert sample["attention_backward"].shape == (2, nl, rm, rm)


def test_full_out_of_response_cells_are_nan(tmp_path):
    """Cells beyond r_eff must be NaN; cells within r_eff must be finite."""
    n = 6
    r_eff = 3  # shorter than r_max=6
    rls = np.full(n, r_eff, dtype=np.int32)
    capture = _make_full_capture_dir(tmp_path, n_samples=n, response_lengths=rls)
    rm = _SMALL_CFG["r_max"]
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=1,
        relevant_layers=[1],
        fixed_layer=1,
        include_response_attention=True,
        attention_summary="full",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    attn = sample["attention_backward"]  # (1, num_layers, r_max, r_max)
    # All cells within r_eff × r_eff block must be finite.
    assert torch.isfinite(attn[:, :, :r_eff, :r_eff]).all()
    # Cells outside the block must be NaN.
    assert torch.isnan(attn[:, :, r_eff:, :]).all()
    assert torch.isnan(attn[:, :, :, r_eff:]).all()


def test_full_zero_response_yields_nan(tmp_path):
    n = 4
    rls = np.zeros(n, dtype=np.int32)
    capture = _make_full_capture_dir(tmp_path, n_samples=n, response_lengths=rls)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=1,
        relevant_layers=[1],
        fixed_layer=1,
        include_response_attention=True,
        attention_summary="full",
        attention_target_layer_offset_backward=1,
    )
    sample = ds[0]
    assert torch.isnan(sample["attention_backward"]).all()
