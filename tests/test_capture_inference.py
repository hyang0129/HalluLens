"""
tests/test_capture_inference.py — CPU-only unit tests for capture_inference.py.

All tests avoid real model loads (no GPU required).

Test inventory:
  1. test_main_module_imports         — import without error.
  2. test_compute_icr_per_layer_shape — shape + dtype contract.
  3. test_pad_to_and_pad_2d           — padding helper correctness.
  4. test_sha256_stable               — deterministic hash.
  5. test_step_eval_only_rewrites_eval_results — eval-only mode e2e.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Import smoke-test
# ---------------------------------------------------------------------------

def test_main_module_imports():
    """capture_inference imports cleanly without heavy dependencies."""
    # Remove any cached import to test fresh each run
    mod_name = "scripts.capture_inference"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    # The module imports argparse, hashlib, importlib, json, logging, numpy — all stdlib/available.
    # Heavy deps (torch, transformers) are only imported inside functions, not at module level.
    import scripts.capture_inference  # noqa: F401


# ---------------------------------------------------------------------------
# 2. compute_icr_per_layer shape + dtype
# ---------------------------------------------------------------------------

def _make_fake_icr_score_module() -> types.ModuleType:
    """Return a fake icr_score module whose compute_icr_score returns a deterministic float."""
    m = types.ModuleType("activation_research.icr_score")

    def compute_icr_score(response_attn, h_block_input, delta_h, response_len, top_p=0.1):
        # Mirror real compute_icr_score: return 0.0 for empty response.
        if response_len == 0:
            return 0.0
        return float(np.mean(np.abs(response_attn[:response_len, :response_len])))

    m.compute_icr_score = compute_icr_score
    return m


def test_compute_icr_per_layer_shape():
    """compute_icr_per_layer must return (L,) float32 for valid inputs."""
    from scripts.capture_inference import compute_icr_per_layer

    L = 4
    R = 8
    H = 16

    rng = np.random.default_rng(42)
    resp_attn = rng.random((L, R, R), dtype=np.float32).astype(np.float16)
    resp_hs = rng.random((L + 1, R, H), dtype=np.float32).astype(np.float16)

    fake_module = _make_fake_icr_score_module()

    with patch.dict(sys.modules, {"activation_research.icr_score": fake_module}):
        scores = compute_icr_per_layer(resp_attn, resp_hs, response_len=5, top_p=0.1)

    assert scores.shape == (L,), f"Expected ({L},), got {scores.shape}"
    assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"


def test_compute_icr_per_layer_zero_response():
    """response_len=0 must return all-zero scores without crashing."""
    from scripts.capture_inference import compute_icr_per_layer

    L = 3
    R = 4
    H = 8

    rng = np.random.default_rng(0)
    resp_attn = rng.random((L, R, R)).astype(np.float16)
    resp_hs = rng.random((L + 1, R, H)).astype(np.float16)

    fake_module = _make_fake_icr_score_module()

    with patch.dict(sys.modules, {"activation_research.icr_score": fake_module}):
        scores = compute_icr_per_layer(resp_attn, resp_hs, response_len=0, top_p=0.1)

    assert scores.shape == (L,)
    assert np.all(scores == 0.0), "Expected all-zero scores for response_len=0"


# ---------------------------------------------------------------------------
# 3. pad_to and pad_2d
# ---------------------------------------------------------------------------

def test_pad_to_shorter():
    from scripts.capture_inference import pad_to

    arr = np.array([1, 2, 3], dtype=np.int32)
    result = pad_to(arr, 6, -1)
    expected = np.array([1, 2, 3, -1, -1, -1], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_pad_to_exact():
    from scripts.capture_inference import pad_to

    arr = np.array([4, 5], dtype=np.float32)
    result = pad_to(arr, 2, np.nan)
    np.testing.assert_array_equal(result, arr)


def test_pad_to_truncate():
    from scripts.capture_inference import pad_to

    arr = np.arange(10, dtype=np.int32)
    result = pad_to(arr, 5, -1)
    np.testing.assert_array_equal(result, arr[:5])


def test_pad_2d_shorter_both():
    from scripts.capture_inference import pad_2d

    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    result = pad_2d(arr, 4, 3, -1)
    assert result.shape == (4, 3)
    # Top-left 2×2 block must be preserved
    np.testing.assert_array_equal(result[:2, :2], arr)
    # Padded positions must be -1
    assert result[2, 0] == -1
    assert result[0, 2] == -1


def test_pad_2d_truncate():
    from scripts.capture_inference import pad_2d

    arr = np.ones((6, 6), dtype=np.int32)
    result = pad_2d(arr, 3, 4, -1)
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, np.ones((3, 4), dtype=np.int32))


def test_pad_2d_nan_fill():
    from scripts.capture_inference import pad_2d

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    result = pad_2d(arr, 2, 3, np.nan)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result[0, :2], [1.0, 2.0])
    assert np.isnan(result[0, 2])
    assert np.isnan(result[1, 0])


# ---------------------------------------------------------------------------
# 4. sha256 stability
# ---------------------------------------------------------------------------

def test_sha256_stable():
    from scripts.capture_inference import sha256

    h1 = sha256("hello world")
    h2 = sha256("hello world")
    assert h1 == h2, "sha256 must be deterministic"


def test_sha256_different_inputs_differ():
    from scripts.capture_inference import sha256

    assert sha256("foo") != sha256("bar"), "Different inputs must produce different hashes"


def test_sha256_known_value():
    from scripts.capture_inference import sha256
    import hashlib

    s = "test string for sha256"
    expected = hashlib.sha256(s.encode("utf-8")).hexdigest()
    assert sha256(s) == expected


# ---------------------------------------------------------------------------
# 5. --step eval-only rewrites eval_results.json
# ---------------------------------------------------------------------------

def _make_fake_hotpotqa_module() -> types.ModuleType:
    """Return a minimal fake tasks.llmsknow.hotpotqa module.

    Avoids importing the real module, which imports fcntl (Unix-only) at module
    level and would fail on Windows.
    """
    m = types.ModuleType("tasks.llmsknow.hotpotqa")

    def is_correct(generation: str, answer: str) -> bool:
        return answer.lower().strip() in generation.lower().strip()

    def format_prompt(question: str) -> str:
        return f"Answer the question concisely.\n\nQ: {question}\nA:"

    m.is_correct = is_correct
    m.format_prompt = format_prompt
    return m


def _build_hotpotqa_gen_jsonl(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake generation.jsonl + meta.jsonl for hotpotqa eval-only test."""
    records = [
        # Sample 0: correct answer present in generation → hallucinated=False
        {
            "prompt": "Answer the question concisely.\n\nQ: What is the capital of France?\nA:",
            "generation": "The capital of France is Paris.",
            "answer": "Paris",
            "question": "What is the capital of France?",
            "hallucinated": True,  # intentionally wrong — eval-only should fix this
            "prompt_hash": "aaa",
            "sample_index": 0,
        },
        # Sample 1: answer in generation → hallucinated=False
        {
            "prompt": "Answer the question concisely.\n\nQ: Who invented the telephone?\nA:",
            "generation": "Alexander Graham Bell invented the telephone.",
            "answer": "Graham Bell",
            "question": "Who invented the telephone?",
            "hallucinated": True,  # intentionally wrong — eval-only should fix this
            "prompt_hash": "bbb",
            "sample_index": 1,
        },
    ]

    gen_path = tmp_path / "generation.jsonl"
    with gen_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    meta_records = [
        {"sample_index": 0, "key": "q0", "prompt_hash": "aaa",
         "prompt_len": 20, "response_len": 7, "hallucinated": True},
        {"sample_index": 1, "key": "q1", "prompt_hash": "bbb",
         "prompt_len": 18, "response_len": 6, "hallucinated": True},
    ]
    meta_path = tmp_path / "meta.jsonl"
    with meta_path.open("w") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")

    return gen_path, meta_path


def test_step_eval_only_rewrites_eval_results(tmp_path):
    """eval-only mode must re-apply is_correct and write correct eval_results.json."""
    gen_path, meta_path = _build_hotpotqa_gen_jsonl(tmp_path)

    fake_task_mod = _make_fake_hotpotqa_module()

    test_args = [
        "capture_inference.py",
        "--task", "hotpotqa",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--out-dir", str(tmp_path),
        "--step", "eval-only",
    ]

    import importlib
    import scripts.capture_inference as ci_mod

    with patch.object(sys, "argv", test_args), \
         patch.dict(sys.modules, {"tasks.llmsknow.hotpotqa": fake_task_mod}):
        importlib.reload(ci_mod)
        ret = ci_mod.main()

    assert ret == 0, "main() must return 0 on success"

    eval_path = tmp_path / "eval_results.json"
    assert eval_path.exists(), "eval_results.json must be written"

    result = json.loads(eval_path.read_text())
    assert "halu_test_res" in result
    assert "abstantion" in result
    assert len(result["halu_test_res"]) == 2
    assert len(result["abstantion"]) == 2

    # Sample 0: "Paris" IS in "The capital of France is Paris." → correct → hallucinated=False
    assert result["halu_test_res"][0] is False, (
        "Sample 0 answer 'Paris' is in generation — hallucinated must be False"
    )
    # Sample 1: "Graham Bell" IS in "Alexander Graham Bell invented..." → correct → hallucinated=False
    assert result["halu_test_res"][1] is False, (
        "Sample 1 answer 'Graham Bell' is in generation — hallucinated must be False"
    )

    # abstantion must all be False (inline eval never abstains)
    assert all(v is False for v in result["abstantion"])


def test_step_eval_only_patches_meta_jsonl(tmp_path):
    """eval-only must patch hallucinated field in meta.jsonl to match re-evaluated labels."""
    _build_hotpotqa_gen_jsonl(tmp_path)

    fake_task_mod = _make_fake_hotpotqa_module()

    test_args = [
        "capture_inference.py",
        "--task", "hotpotqa",
        "--model", "fake-model",
        "--out-dir", str(tmp_path),
        "--step", "eval-only",
    ]

    import importlib
    import scripts.capture_inference as ci_mod

    with patch.object(sys, "argv", test_args), \
         patch.dict(sys.modules, {"tasks.llmsknow.hotpotqa": fake_task_mod}):
        importlib.reload(ci_mod)
        ci_mod.main()

    meta_path = tmp_path / "meta.jsonl"
    lines = [json.loads(ln) for ln in meta_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2

    # Both samples: answers appear in generations → hallucinated=False
    meta0 = next(m for m in lines if m["sample_index"] == 0)
    assert meta0["hallucinated"] is False

    meta1 = next(m for m in lines if m["sample_index"] == 1)
    assert meta1["hallucinated"] is False


def test_step_eval_only_missing_gen_jsonl_returns_nonzero(tmp_path):
    """eval-only must return non-zero when generation.jsonl does not exist."""
    fake_task_mod = _make_fake_hotpotqa_module()

    test_args = [
        "capture_inference.py",
        "--task", "hotpotqa",
        "--model", "fake-model",
        "--out-dir", str(tmp_path),
        "--step", "eval-only",
    ]

    import importlib
    import scripts.capture_inference as ci_mod

    with patch.object(sys, "argv", test_args), \
         patch.dict(sys.modules, {"tasks.llmsknow.hotpotqa": fake_task_mod}):
        importlib.reload(ci_mod)
        ret = ci_mod.main()

    assert ret != 0
