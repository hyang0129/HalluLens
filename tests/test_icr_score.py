"""
tests/test_icr_score.py

Spec-first unit tests for activation_research/icr_score.py.

Tests are written against the public interface spec in
specs/issue_69_wave4_fused_refactor.md §4.1 and the formula cited from
notes/icr_probe_paper_notes.md §3, §5, §9, §10. No real model weights or GPU
required.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from activation_research.icr_score import compute_icr_score


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _reference_icr_score(
    response_attn: np.ndarray,
    h_block_input: np.ndarray,
    delta_h: np.ndarray,
    response_len: int,
    top_p: float = 0.1,
) -> float:
    """Independent numpy reimplementation of compute_icr_score for cross-checking.

    Follows the formula in notes/icr_probe_paper_notes.md §10 and
    icr_score.py:217-267 (upstream) verbatim.  Intentionally duplicates the
    formula so that test_numerical_regression_4x4 catches any silent deviation
    in the production implementation.
    """
    if response_len == 0:
        return 0.0

    def _zscore(x: np.ndarray) -> np.ndarray:
        std = float(x.std())
        if std < 1e-8:
            std = 1e-8
        return (x - x.mean()) / std

    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    def _jsd(a_topk: np.ndarray, w_topk: np.ndarray) -> float:
        a_norm = _softmax(_zscore(a_topk))
        w_norm = _softmax(_zscore(w_topk))
        m = 0.5 * (a_norm + w_norm)
        return 0.5 * _kl(a_norm, m) + 0.5 * _kl(w_norm, m)

    jsd_per_q: list[float] = []
    for q in range(response_len):
        attn_row = response_attn[q, :response_len]
        k = max(1, int(top_p * response_len))
        idx = np.argsort(attn_row)[::-1][:k]
        a_topk = attn_row[idx]
        w_topk = np.array([
            (np.dot(delta_h[q], h_block_input[idx[j]])
             / (float(np.linalg.norm(h_block_input[idx[j]])) + 1e-8))
            for j in range(k)
        ], dtype=np.float64)
        jsd_per_q.append(_jsd(a_topk.astype(np.float64), w_topk))

    return float(np.mean(jsd_per_q))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_4x4_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical (response_attn, h_block_input, delta_h) for the 4x4 tests."""
    response_attn = np.array(
        [[0.5, 0.3, 0.1, 0.1],
         [0.1, 0.5, 0.3, 0.1],
         [0.1, 0.1, 0.5, 0.3],
         [0.1, 0.1, 0.1, 0.7]],
        dtype=np.float32,
    )
    h_block_input = np.eye(4, dtype=np.float32) * 2.0
    delta_h = np.eye(4, dtype=np.float32)
    return response_attn, h_block_input, delta_h


def _make_basic_inputs(R: int = 4, H: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return small random-but-reproducible (response_attn, h_block_input, delta_h)."""
    rng = np.random.default_rng(42)
    # Build a valid lower-triangular attention matrix (causal) and row-normalise.
    raw = rng.random((R, R)).astype(np.float32)
    raw = np.tril(raw)
    raw /= raw.sum(axis=1, keepdims=True) + 1e-8
    h_block_input = rng.standard_normal((R, H)).astype(np.float32)
    delta_h = rng.standard_normal((R, H)).astype(np.float32)
    return raw, h_block_input, delta_h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_response_len_returns_zero():
    response_attn, h_block_input, delta_h = _make_basic_inputs()
    result = compute_icr_score(response_attn, h_block_input, delta_h, response_len=0)
    assert result == 0.0


def test_returns_finite_python_float():
    response_attn, h_block_input, delta_h = _make_basic_inputs(R=4, H=8)
    result = compute_icr_score(response_attn, h_block_input, delta_h, response_len=4)
    assert isinstance(result, float), f"Expected Python float, got {type(result)}"
    assert math.isfinite(result), f"Expected finite result, got {result}"


def test_padding_beyond_response_len_does_not_affect_score():
    R = 4
    response_attn, h_block_input, delta_h = _make_basic_inputs(R=R, H=8)

    score_unpadded = compute_icr_score(
        response_attn, h_block_input, delta_h, response_len=R
    )

    # Pad all arrays out to 2R with garbage in the extra rows/cols.
    rng = np.random.default_rng(99)
    R2 = 2 * R
    padded_attn = np.full((R2, R2), fill_value=999.0, dtype=np.float32)
    padded_attn[:R, :R] = response_attn

    padded_h = np.full((R2, 8), fill_value=999.0, dtype=np.float32)
    padded_h[:R, :] = h_block_input

    padded_delta = np.full((R2, 8), fill_value=999.0, dtype=np.float32)
    padded_delta[:R, :] = delta_h

    score_padded = compute_icr_score(
        padded_attn, padded_h, padded_delta, response_len=R
    )
    assert score_unpadded == score_padded, (
        f"Padding beyond response_len changed score: {score_unpadded} vs {score_padded}"
    )


def test_topk_uses_top_p_fraction():
    """top_p=0.5, R=4 → k=2. Verify that only top-2 positions per row drive the score."""
    # Construct attention so that positions 0 and 1 dominate rows 0 and 1,
    # but delta_h projects more onto position 1 than position 0.
    # This gives non-trivial (non-zero) JSD when k=2 captures both.
    R, H = 4, 4
    response_attn = np.array(
        [[0.5, 0.3, 0.1, 0.1],
         [0.1, 0.5, 0.3, 0.1],
         [0.1, 0.1, 0.5, 0.3],
         [0.1, 0.1, 0.1, 0.7]],
        dtype=np.float32,
    )
    h_block_input = np.eye(R, dtype=np.float32) * 2.0
    # delta_h inverts the top-k rank relative to attention: positions with lower
    # attention get higher projection weight.
    delta_h = np.array(
        [[0.2, 0.8, 0.0, 0.0],
         [0.0, 0.2, 0.8, 0.0],
         [0.0, 0.0, 0.2, 0.8],
         [0.0, 0.0, 0.8, 0.2]],
        dtype=np.float32,
    )

    score = compute_icr_score(
        response_attn, h_block_input, delta_h,
        response_len=R, top_p=0.5,
    )
    reference = _reference_icr_score(
        response_attn, h_block_input, delta_h,
        response_len=R, top_p=0.5,
    )
    assert abs(score - reference) < 1e-6, (
        f"score={score:.8f} reference={reference:.8f}"
    )
    assert score > 0.0, "Expected non-zero JSD with inverted-rank delta_h"


def test_numerical_regression_4x4():
    """Full hand-computed example; both implementations must agree to 1e-6.

    Inputs are the spec-mandated 4x4 case (response_attn diagonal-dominant,
    h_block_input = 2*I, delta_h = I).  With these inputs the top-2 attention
    positions for each query token q coincide with the top-2 projection weights
    from delta_h, making a_norm == w_norm after zscore+softmax, and therefore
    JSD == 0.0 for every token.  Both the production implementation and the
    reference must reproduce this degenerate-but-correct value.
    """
    response_attn, h_block_input, delta_h = _make_4x4_inputs()
    response_len = 4
    top_p = 0.5

    result = compute_icr_score(
        response_attn, h_block_input, delta_h,
        response_len=response_len, top_p=top_p,
    )
    expected = _reference_icr_score(
        response_attn, h_block_input, delta_h,
        response_len=response_len, top_p=top_p,
    )
    # expected = 0.0 (see docstring above for derivation)
    assert abs(expected) < 1e-10, (
        f"Reference unexpectedly non-zero: {expected}"
    )
    assert abs(result - expected) < 1e-6, (
        f"compute_icr_score={result:.8f}, reference={expected:.8f}"
    )


def test_top_p_clamps_to_one():
    """response_len=1, top_p=0.1 → int(0.1*1)==0, impl must clamp to k=1."""
    rng = np.random.default_rng(7)
    response_attn = rng.random((4, 4)).astype(np.float32)
    h_block_input = rng.standard_normal((4, 8)).astype(np.float32)
    delta_h = rng.standard_normal((4, 8)).astype(np.float32)

    result = compute_icr_score(
        response_attn, h_block_input, delta_h,
        response_len=1, top_p=0.1,
    )
    assert math.isfinite(result), f"Expected finite result, got {result}"


def test_invalid_top_p_raises():
    response_attn, h_block_input, delta_h = _make_basic_inputs()
    with pytest.raises(ValueError):
        compute_icr_score(
            response_attn, h_block_input, delta_h,
            response_len=4, top_p=0.0,
        )
    with pytest.raises(ValueError):
        compute_icr_score(
            response_attn, h_block_input, delta_h,
            response_len=4, top_p=1.5,
        )


def test_fp16_input_accepted():
    """float16 response_attn should produce a result within 1e-3 of the float32 call."""
    R, H = 4, 8
    response_attn, h_block_input, delta_h = _make_basic_inputs(R=R, H=H)

    score_f32 = compute_icr_score(
        response_attn, h_block_input, delta_h, response_len=R
    )
    score_f16 = compute_icr_score(
        response_attn.astype(np.float16), h_block_input, delta_h, response_len=R
    )
    assert math.isfinite(score_f16), f"fp16 result is not finite: {score_f16}"
    assert abs(score_f16 - score_f32) < 1e-3, (
        f"fp16 vs fp32 diff too large: f16={score_f16}, f32={score_f32}"
    )


def test_shape_mismatch_raises():
    """delta_h with wrong second dimension must raise ValueError."""
    R, H = 4, 8
    response_attn, h_block_input, delta_h = _make_basic_inputs(R=R, H=H)

    bad_delta_h = delta_h[:, :H - 1]  # (R, H-1) — mismatches h_block_input's H
    with pytest.raises(ValueError):
        compute_icr_score(
            response_attn, h_block_input, bad_delta_h, response_len=R
        )
