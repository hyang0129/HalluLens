"""
Tests for ICRDataset mode="memmap" and mode="memmap-raw".

All tests are CPU-only and synthesise fake capture directories in tmp_path.
No real model or GPU is needed.

Run with:
    pytest tests/test_icr_dataset_memmap.py -v
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from activation_research.icr_dataset import ICRDataset, _make_split_indices


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = {
    "model_name": "fake-model",
    "num_layers": 4,
    "hidden_dim": 8,
    "r_max": 6,
    "dtype": "float16",
    "response_logprobs_top_k": 5,
    "max_prompt_len": 16,
    "max_response_len": 12,
}


def _make_capture_dir(
    tmp_path: Path,
    n_samples: int,
    n_meta: int | None = None,
    halu_pattern: np.ndarray | None = None,
) -> Path:
    """Build a synthetic InferenceCaptureWriter output directory.

    Parameters
    ----------
    n_samples : int
        Pre-allocated size of the memmap arrays.
    n_meta : int or None
        How many rows to write into meta.jsonl.  Defaults to n_samples.
        Use n_meta < n_samples to test partial-meta validity contract.
    halu_pattern : array-like or None
        Boolean/int array of length n_meta giving hallucination labels.
        Defaults to alternating True/False.
    """
    if n_meta is None:
        n_meta = n_samples

    cfg = dict(_SMALL_CFG)
    cfg["n_samples"] = n_samples

    out = tmp_path / "capture"
    out.mkdir()

    # config.json
    (out / "config.json").write_text(json.dumps(cfg))

    nl = cfg["num_layers"]
    hd = cfg["hidden_dim"]
    rm = cfg["r_max"]
    mr = cfg["max_response_len"]
    mp = cfg["max_prompt_len"]

    # Allocate memmap arrays
    for name, shape, dtype in [
        ("response_activations.npy", (n_samples, nl + 1, mr, hd), np.float16),
        ("response_attention.npy",   (n_samples, nl, rm, rm),      np.float16),
        ("prompt_activations.npy",   (n_samples, nl + 1, mp, hd),  np.float16),
        ("response_len.npy",         (n_samples,),                  np.int32),
        ("prompt_len.npy",           (n_samples,),                  np.int32),
    ]:
        mm = np.memmap(str(out / name), dtype=dtype, mode="w+", shape=shape)
        if name == "response_len.npy":
            mm[:] = 8  # non-zero response lengths
        elif name == "prompt_len.npy":
            mm[:] = 10
        else:
            # Fill with a recognisable pattern so readback can verify non-zero
            mm[:] = np.arange(mm.size, dtype=dtype).reshape(shape)
        mm.flush()
        del mm

    # icr_scores.npy — regular np.save
    icr = np.arange(n_samples * nl, dtype=np.float32).reshape(n_samples, nl)
    np.save(str(out / "icr_scores.npy"), icr)

    # halu labels
    if halu_pattern is None:
        halu_arr = np.array([bool(i % 2) for i in range(n_meta)])
    else:
        halu_arr = np.asarray(halu_pattern, dtype=bool)
    assert len(halu_arr) == n_meta

    # meta.jsonl — only n_meta rows
    with (out / "meta.jsonl").open("w") as fh:
        for i in range(n_meta):
            prompt = f"fake prompt number {i}"
            ph = hashlib.sha256(prompt.encode()).hexdigest()
            row = {
                "sample_index": i,
                "key": f"fake_{i}",
                "prompt_hash": ph,
                "prompt_len": 10,
                "response_len": 8,
                "hallucinated": bool(halu_arr[i]),
                "wrote_at": "2026-01-01T00:00:00",
            }
            fh.write(json.dumps(row) + "\n")

    # eval_results.json (synthesised, compat format)
    eval_res = {
        "halu_test_res": [bool(x) for x in halu_arr],
        "abstantion": [False] * n_meta,
    }
    (out / "eval_results.json").write_text(json.dumps(eval_res))

    return out


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------

def test_memmap_mode_constructs_from_synthetic_capture_dir(tmp_path):
    capture_dir = _make_capture_dir(tmp_path, n_samples=20)
    ds = ICRDataset(capture_dir=capture_dir, mode="memmap", split="train")
    # Dataset must expose samples — combined train+val+test should cover all 20
    assert len(ds) > 0


def test_memmap_mode_total_len_equals_n_meta(tmp_path):
    """All three splits union to exactly the number of meta.jsonl rows."""
    capture_dir = _make_capture_dir(tmp_path, n_samples=20)
    total = sum(
        len(ICRDataset(capture_dir=capture_dir, mode="memmap", split=s))
        for s in ("train", "val", "test")
    )
    assert total == 20


# ---------------------------------------------------------------------------
# 2. __getitem__ dict shape
# ---------------------------------------------------------------------------

def test_memmap_mode_returns_icr_dict_shape(tmp_path):
    capture_dir = _make_capture_dir(tmp_path, n_samples=20)
    ds = ICRDataset(capture_dir=capture_dir, mode="memmap", split="train")
    sample = ds[0]
    assert set(sample.keys()) == {"hashkey", "halu", "icr_score"}
    assert isinstance(sample["hashkey"], str)
    assert isinstance(sample["halu"], torch.Tensor)
    assert isinstance(sample["icr_score"], torch.Tensor)
    assert sample["icr_score"].shape == (4,)  # num_layers=4
    assert sample["icr_score"].dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. Split consistency
# ---------------------------------------------------------------------------

def test_memmap_mode_split_consistency(tmp_path):
    """train, val, test indices are disjoint and their union covers all samples."""
    capture_dir = _make_capture_dir(tmp_path, n_samples=40)

    splits = {}
    for s in ("train", "val", "test"):
        ds = ICRDataset(capture_dir=capture_dir, mode="memmap", split=s,
                        random_seed=42)
        splits[s] = set(int(ds._split_indices[i]) for i in range(len(ds)))

    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"]

    # disjoint
    assert train_set.isdisjoint(val_set), "train and val overlap"
    assert train_set.isdisjoint(test_set), "train and test overlap"
    assert val_set.isdisjoint(test_set), "val and test overlap"

    # union covers exactly 40
    assert len(train_set | val_set | test_set) == 40


# ---------------------------------------------------------------------------
# 4. Stratification balance
# ---------------------------------------------------------------------------

def test_memmap_mode_stratification_balance(tmp_path):
    """For a balanced dataset, each split's halu rate is within 15% of 0.5."""
    n = 60
    halu = np.array([i % 2 for i in range(n)], dtype=bool)
    capture_dir = _make_capture_dir(tmp_path, n_samples=n, halu_pattern=halu)

    for s in ("train", "val", "test"):
        ds = ICRDataset(capture_dir=capture_dir, mode="memmap", split=s,
                        random_seed=42)
        halu_vals = [float(ds[i]["halu"]) for i in range(len(ds))]
        rate = sum(halu_vals) / len(halu_vals)
        assert abs(rate - 0.5) < 0.15, (
            f"Split {s!r}: halu rate {rate:.2f} deviates from 0.5 by more than 0.15"
        )


# ---------------------------------------------------------------------------
# 5. Partial meta.jsonl (validity contract)
# ---------------------------------------------------------------------------

def test_memmap_mode_partial_meta_jsonl(tmp_path):
    """Pre-allocate N=20 rows but only register 12 in meta.jsonl.
    Dataset must expose exactly 12, not 20.
    """
    capture_dir = _make_capture_dir(tmp_path, n_samples=20, n_meta=12)
    total = sum(
        len(ICRDataset(capture_dir=capture_dir, mode="memmap", split=s))
        for s in ("train", "val", "test")
    )
    assert total == 12, f"Expected 12 (from meta.jsonl), got {total}"


# ---------------------------------------------------------------------------
# 6. mode="memmap-raw" dict shape
# ---------------------------------------------------------------------------

def test_memmap_raw_mode_returns_raw_dict_shape(tmp_path):
    capture_dir = _make_capture_dir(tmp_path, n_samples=20)
    ds = ICRDataset(capture_dir=capture_dir, mode="memmap-raw", split="train")
    sample = ds[0]
    expected_keys = {"hashkey", "halu", "response_attn", "h_block_input",
                     "delta_h", "response_len"}
    assert set(sample.keys()) == expected_keys
    assert isinstance(sample["hashkey"], str)
    assert sample["response_attn"].shape == (4, 6, 6)   # (nl, r_max, r_max)
    assert sample["h_block_input"].shape == (4, 8)       # (nl, hidden_dim)
    assert sample["delta_h"].shape == (4, 8)             # (nl, hidden_dim)
    assert isinstance(sample["response_len"], int)


# ---------------------------------------------------------------------------
# 7. mode="memmap" matches mode="icr" on same underlying data
# ---------------------------------------------------------------------------

def _make_legacy_icr_dir(tmp_path: Path, capture_dir: Path) -> Path:
    """Build a legacy mode="icr" dir from the same meta.jsonl + icr_scores."""
    legacy = tmp_path / "legacy_icr"
    legacy.mkdir()

    # Copy meta.jsonl and icr_scores.npy verbatim
    import shutil
    shutil.copy(capture_dir / "meta.jsonl", legacy / "meta.jsonl")
    shutil.copy(capture_dir / "icr_scores.npy", legacy / "icr_scores.npy")
    return legacy


def test_memmap_mode_matches_icr_mode_on_same_data(tmp_path):
    """mode='memmap' and mode='icr' produce the same samples in the same order
    when reading from equivalent data sources.
    """
    n = 30
    capture_dir = _make_capture_dir(tmp_path, n_samples=n)
    legacy_dir = _make_legacy_icr_dir(tmp_path, capture_dir)

    for split in ("train", "val", "test"):
        ds_mm = ICRDataset(capture_dir=capture_dir, mode="memmap",
                           split=split, random_seed=42)
        ds_icr = ICRDataset(capture_dir=legacy_dir, mode="icr",
                            split=split, random_seed=42)

        assert len(ds_mm) == len(ds_icr), (
            f"split={split}: memmap has {len(ds_mm)}, icr has {len(ds_icr)}"
        )
        for i in range(len(ds_mm)):
            item_mm = ds_mm[i]
            item_icr = ds_icr[i]
            assert item_mm["hashkey"] == item_icr["hashkey"], (
                f"split={split}[{i}] hashkey mismatch"
            )
            assert float(item_mm["halu"]) == float(item_icr["halu"]), (
                f"split={split}[{i}] halu mismatch"
            )
            assert torch.allclose(item_mm["icr_score"], item_icr["icr_score"]), (
                f"split={split}[{i}] icr_score mismatch"
            )
