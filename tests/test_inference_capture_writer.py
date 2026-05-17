"""
Tests for InferenceCaptureWriter.

All tests are CPU-only and use tmp_path for isolation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from activation_logging.inference_capture_writer import InferenceCaptureWriter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 10
NUM_LAYERS = 4
HIDDEN_DIM = 8
R_MAX = 6
TOP_K = 3
MAX_PROMPT_LEN = 12
MAX_RESPONSE_LEN = 8


@pytest.fixture()
def cfg() -> dict:
    return {
        "model_name": "test-model",
        "num_layers": NUM_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "r_max": R_MAX,
        "dtype": "float16",
        "response_logprobs_top_k": TOP_K,
        "max_prompt_len": MAX_PROMPT_LEN,
        "max_response_len": MAX_RESPONSE_LEN,
    }


def _make_writer(out_dir: Path, mode: str, cfg: dict, n: int = N_SAMPLES) -> InferenceCaptureWriter:
    return InferenceCaptureWriter(out_dir, mode, n, cfg)


def _synthetic_arrays(prompt_len: int = 5, response_len: int = 3) -> dict:
    """Return a dict of numpy arrays matching the writer's expected shapes."""
    rng = np.random.default_rng(42)
    return {
        "prompt_activations": rng.random(
            (NUM_LAYERS + 1, prompt_len, HIDDEN_DIM), dtype=np.float32
        ).astype(np.float16),
        "response_activations": rng.random(
            (NUM_LAYERS + 1, response_len, HIDDEN_DIM), dtype=np.float32
        ).astype(np.float16),
        "response_attention": rng.random(
            (NUM_LAYERS, R_MAX, R_MAX), dtype=np.float32
        ).astype(np.float16),
        "prompt_token_ids": np.arange(prompt_len, dtype=np.int32),
        "response_token_ids": np.arange(response_len, dtype=np.int32),
        "response_token_logprobs": rng.random(response_len, dtype=np.float32),
        "response_topk_token_ids": rng.integers(0, 1000, (response_len, TOP_K), dtype=np.int32),
        "response_topk_logprobs": rng.random((response_len, TOP_K), dtype=np.float32),
        "icr_score_per_layer": rng.random(NUM_LAYERS, dtype=np.float32),
    }


def _append_sample(
    writer: InferenceCaptureWriter,
    sample_index: int,
    prompt_hash: str = None,
    hallucinated: bool = False,
    prompt_len: int = 5,
    response_len: int = 3,
) -> dict:
    if prompt_hash is None:
        prompt_hash = f"hash_{sample_index:04x}"
    arrays = _synthetic_arrays(prompt_len, response_len)
    writer.append(
        sample_index=sample_index,
        prompt_hash=prompt_hash,
        key=f"key_{sample_index}",
        prompt_len=prompt_len,
        response_len=response_len,
        hallucinated=hallucinated,
        generation_record={
            "prompt": "q?",
            "generation": "a",
            "answer": "a",
            "question": "q?",
            "hallucinated": hallucinated,
        },
        **arrays,
    )
    return arrays


# ---------------------------------------------------------------------------
# 1. Pre-allocation creates all expected files at correct sizes
# ---------------------------------------------------------------------------

def test_pre_allocation_creates_all_files(tmp_path, cfg):
    with _make_writer(tmp_path, "w", cfg):
        pass  # finalize() called by __exit__

    expected_stems = [
        "response_activations",
        "response_attention",
        "prompt_activations",
        "prompt_token_ids",
        "response_token_ids",
        "response_token_logprobs",
        "response_topk_token_ids",
        "response_topk_logprobs",
        "prompt_len",
        "response_len",
    ]
    for stem in expected_stems:
        path = tmp_path / f"{stem}.npy"
        assert path.exists(), f"Missing {stem}.npy"

    expected_shapes = {
        "response_activations": (N_SAMPLES, NUM_LAYERS + 1, MAX_RESPONSE_LEN, HIDDEN_DIM),
        "response_attention":   (N_SAMPLES, NUM_LAYERS, R_MAX, R_MAX),
        "prompt_activations":   (N_SAMPLES, NUM_LAYERS + 1, MAX_PROMPT_LEN, HIDDEN_DIM),
        "prompt_token_ids":     (N_SAMPLES, MAX_PROMPT_LEN),
        "response_token_ids":   (N_SAMPLES, MAX_RESPONSE_LEN),
        "response_token_logprobs": (N_SAMPLES, MAX_RESPONSE_LEN),
        "response_topk_token_ids": (N_SAMPLES, MAX_RESPONSE_LEN, TOP_K),
        "response_topk_logprobs":  (N_SAMPLES, MAX_RESPONSE_LEN, TOP_K),
        "prompt_len":           (N_SAMPLES,),
        "response_len":         (N_SAMPLES,),
    }
    expected_dtypes = {
        "response_activations": np.dtype("float16"),
        "response_attention":   np.dtype("float16"),
        "prompt_activations":   np.dtype("float16"),
        "prompt_token_ids":     np.dtype("int32"),
        "response_token_ids":   np.dtype("int32"),
        "response_token_logprobs": np.dtype("float32"),
        "response_topk_token_ids": np.dtype("int32"),
        "response_topk_logprobs":  np.dtype("float32"),
        "prompt_len":           np.dtype("int32"),
        "response_len":         np.dtype("int32"),
    }
    for stem, shape in expected_shapes.items():
        mm = np.memmap(str(tmp_path / f"{stem}.npy"), dtype=expected_dtypes[stem], mode="r", shape=shape)
        assert mm.shape == shape, f"{stem}.npy has wrong shape: {mm.shape} != {shape}"

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "meta.jsonl").exists()
    assert (tmp_path / "generation.jsonl").exists()


# ---------------------------------------------------------------------------
# 2. Pre-allocation fills correct padding values
# ---------------------------------------------------------------------------

def test_pre_allocation_padding_values(tmp_path, cfg):
    with _make_writer(tmp_path, "w", cfg):
        pass

    int32 = np.dtype("int32")
    fp32 = np.dtype("float32")
    fp16 = np.dtype("float16")

    # Token-ID arrays must be -1 everywhere.
    for stem, dtype in [("prompt_token_ids", int32), ("response_token_ids", int32),
                        ("response_topk_token_ids", int32)]:
        shape = {
            "prompt_token_ids": (N_SAMPLES, MAX_PROMPT_LEN),
            "response_token_ids": (N_SAMPLES, MAX_RESPONSE_LEN),
            "response_topk_token_ids": (N_SAMPLES, MAX_RESPONSE_LEN, TOP_K),
        }[stem]
        mm = np.memmap(str(tmp_path / f"{stem}.npy"), dtype=dtype, mode="r", shape=shape)
        assert np.all(mm == -1), f"{stem}.npy should be all -1; found other values"

    # Logprob arrays must be NaN everywhere.
    for stem, dtype, shape in [
        ("response_token_logprobs", fp32, (N_SAMPLES, MAX_RESPONSE_LEN)),
        ("response_topk_logprobs", fp32, (N_SAMPLES, MAX_RESPONSE_LEN, TOP_K)),
    ]:
        mm = np.memmap(str(tmp_path / f"{stem}.npy"), dtype=dtype, mode="r", shape=shape)
        assert np.all(np.isnan(mm)), f"{stem}.npy should be all NaN; found non-NaN values"

    # Activation and attention arrays must be zero.
    for stem, dtype, shape in [
        ("response_activations", fp16, (N_SAMPLES, NUM_LAYERS + 1, MAX_RESPONSE_LEN, HIDDEN_DIM)),
        ("response_attention", fp16, (N_SAMPLES, NUM_LAYERS, R_MAX, R_MAX)),
        ("prompt_activations", fp16, (N_SAMPLES, NUM_LAYERS + 1, MAX_PROMPT_LEN, HIDDEN_DIM)),
    ]:
        mm = np.memmap(str(tmp_path / f"{stem}.npy"), dtype=dtype, mode="r", shape=shape)
        assert np.all(mm == 0), f"{stem}.npy should be all zeros; found non-zero values"


# ---------------------------------------------------------------------------
# 3. Append round-trip: write 1 sample, re-open, read back, assert equality
# ---------------------------------------------------------------------------

def test_append_round_trip(tmp_path, cfg):
    writer = _make_writer(tmp_path, "w", cfg)
    arrays = _append_sample(writer, sample_index=0, prompt_hash="abc123",
                             hallucinated=True, prompt_len=5, response_len=3)
    writer.finalize()

    # Re-open and read back.
    fp16 = np.dtype("float16")
    int32 = np.dtype("int32")
    fp32 = np.dtype("float32")

    def load(stem, dtype, shape):
        return np.memmap(str(tmp_path / f"{stem}.npy"), dtype=dtype, mode="r", shape=shape)

    # Response activations: row 0 should match the written data (zero-padded to full shape).
    ra = load("response_activations", fp16, (N_SAMPLES, NUM_LAYERS + 1, MAX_RESPONSE_LEN, HIDDEN_DIM))
    src = arrays["response_activations"].astype(fp16)
    assert np.array_equal(ra[0, :, :src.shape[1], :], src), "response_activations round-trip failed"

    # Response token IDs: first response_len positions.
    rt = load("response_token_ids", int32, (N_SAMPLES, MAX_RESPONSE_LEN))
    assert np.array_equal(rt[0, :3], arrays["response_token_ids"]), "response_token_ids round-trip failed"
    # Padding positions must still be -1.
    assert np.all(rt[0, 3:] == -1), "response_token_ids padding not preserved"

    # Logprobs: first response_len positions match; rest are NaN.
    rl = load("response_token_logprobs", fp32, (N_SAMPLES, MAX_RESPONSE_LEN))
    assert np.allclose(rl[0, :3], arrays["response_token_logprobs"], atol=1e-5), \
        "response_token_logprobs round-trip failed"
    assert np.all(np.isnan(rl[0, 3:])), "response_token_logprobs NaN padding not preserved"

    # Scalar lengths.
    pl = load("prompt_len", int32, (N_SAMPLES,))
    rlen = load("response_len", int32, (N_SAMPLES,))
    assert pl[0] == 5
    assert rlen[0] == 3

    # meta.jsonl has exactly one line.
    lines = [l for l in (tmp_path / "meta.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    meta = json.loads(lines[0])
    assert meta["sample_index"] == 0
    assert meta["prompt_hash"] == "abc123"
    assert meta["hallucinated"] is True


# ---------------------------------------------------------------------------
# 4. Resume semantics
# ---------------------------------------------------------------------------

def test_resume_skips_written_samples(tmp_path, cfg):
    hashes = [f"hash_{i}" for i in range(3)]

    writer = _make_writer(tmp_path, "w", cfg)
    for i in range(3):
        _append_sample(writer, sample_index=i, prompt_hash=hashes[i])
    writer.finalize()

    writer2 = _make_writer(tmp_path, "a", cfg)
    assert writer2.next_index() == 3
    for h in hashes:
        assert writer2.is_written(h), f"is_written({h!r}) should be True after resume"
    writer2.finalize()


# ---------------------------------------------------------------------------
# 5. Crash safety: meta.jsonl line missing => sample not counted
# ---------------------------------------------------------------------------

def test_crash_safety_meta_not_appended(tmp_path, cfg):
    """Simulate a crash between generation.jsonl write and meta.jsonl write."""
    writer = _make_writer(tmp_path, "w", cfg)
    # Write sample 0 normally.
    _append_sample(writer, sample_index=0, prompt_hash="real_hash")
    # Do NOT call append for sample 1; instead manually write to generation.jsonl
    # but omit the meta.jsonl line (simulates crash after step 2 but before step 3).
    ghost_record = json.dumps({"sample_index": 1, "prompt_hash": "ghost_hash", "hallucinated": False})
    with open(tmp_path / "generation.jsonl", "a", encoding="utf-8") as fh:
        fh.write(ghost_record + "\n")
        fh.flush()
    # Close writer normally (meta.jsonl has only 1 line).
    writer.finalize()

    writer2 = _make_writer(tmp_path, "a", cfg)
    assert writer2.is_written("real_hash"), "real_hash should be written"
    assert not writer2.is_written("ghost_hash"), "ghost_hash must NOT be reported written"
    assert writer2.next_index() == 1, "next_index should be 1, not 2"
    writer2.finalize()


# ---------------------------------------------------------------------------
# 6. finalize() synthesizes correct eval_results.json
# ---------------------------------------------------------------------------

def test_finalize_writes_eval_results_json(tmp_path, cfg):
    halu_values = [True, False, True]
    writer = _make_writer(tmp_path, "w", cfg)
    for i, halu in enumerate(halu_values):
        _append_sample(writer, sample_index=i, hallucinated=halu)
    writer.finalize()

    er = json.loads((tmp_path / "eval_results.json").read_text())
    assert er["halu_test_res"] == halu_values, \
        f"halu_test_res mismatch: {er['halu_test_res']} != {halu_values}"
    assert er["abstantion"] == [False, False, False]


# ---------------------------------------------------------------------------
# 7. finalize() writes icr_scores.npy stacked in sample_index order
# ---------------------------------------------------------------------------

def test_finalize_writes_icr_scores_npy(tmp_path, cfg):
    rng = np.random.default_rng(7)
    icr_scores = [rng.random(NUM_LAYERS, dtype=np.float32) for _ in range(3)]

    writer = _make_writer(tmp_path, "w", cfg)
    for i in range(3):
        arrays = _synthetic_arrays()
        arrays["icr_score_per_layer"] = icr_scores[i]
        writer.append(
            sample_index=i,
            prompt_hash=f"h{i}",
            key=f"k{i}",
            prompt_len=5,
            response_len=3,
            hallucinated=False,
            generation_record={"prompt": "q", "generation": "a", "answer": "a",
                                "question": "q", "hallucinated": False},
            **{k: v for k, v in arrays.items() if k != "icr_score_per_layer"},
            icr_score_per_layer=icr_scores[i],
        )
    writer.finalize()

    saved = np.load(str(tmp_path / "icr_scores.npy"))
    assert saved.shape == (3, NUM_LAYERS)
    for i in range(3):
        assert np.allclose(saved[i], icr_scores[i]), f"ICR score row {i} mismatch"


# ---------------------------------------------------------------------------
# 8. Config mismatch raises ValueError on resume
# ---------------------------------------------------------------------------

def test_config_mismatch_raises_on_resume(tmp_path, cfg):
    with _make_writer(tmp_path, "w", cfg):
        pass

    # Tamper with model_name.
    config_path = tmp_path / "config.json"
    stored = json.loads(config_path.read_text())
    stored["model_name"] = "different-model"
    config_path.write_text(json.dumps(stored))

    bad_cfg = dict(cfg)
    bad_cfg["model_name"] = "test-model"  # original — now mismatches stored

    with pytest.raises(ValueError, match="model_name"):
        _make_writer(tmp_path, "a", bad_cfg)


# ---------------------------------------------------------------------------
# 9. Idempotent finalize
# ---------------------------------------------------------------------------

def test_idempotent_finalize(tmp_path, cfg):
    writer = _make_writer(tmp_path, "w", cfg)
    _append_sample(writer, sample_index=0)
    writer.finalize()

    # Second finalize must not raise or corrupt.
    writer.finalize()

    er = json.loads((tmp_path / "eval_results.json").read_text())
    assert len(er["halu_test_res"]) == 1
