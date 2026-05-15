"""
tests/test_attention_recompute.py

Spec-first unit tests for activation_logging/attention_recompute.py.

Tests are written against the public interface spec, not the implementation.
A minimal FakeBlock is used so no real HF model weights or GPU are required.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from activation_logging.attention_recompute import (
    recompute_block_attention,
    _head_average_resp_to_resp,
)


# ---------------------------------------------------------------------------
# Minimal fake attention block
# ---------------------------------------------------------------------------

class _FakeSelfAttn(nn.Module):
    """Tiny multi-head attention that applies a real causal softmax.

    Returns ``(attn_output, attn_weights, None)`` when ``output_attentions=True``,
    matching the HF eager attention contract.  Position IDs are accepted and
    ignored (no RoPE — not needed for shape / value correctness tests).
    """

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        bsz, T, H = hidden_states.shape
        n_heads, head_dim = self.n_heads, self.head_dim

        def _proj(linear: nn.Linear) -> torch.Tensor:
            return (
                linear(hidden_states)
                .view(bsz, T, n_heads, head_dim)
                .transpose(1, 2)
            )  # (B, n_heads, T, head_dim)

        q, k, v = _proj(self.q_proj), _proj(self.k_proj), _proj(self.v_proj)

        scale = head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        if attention_mask is not None:
            scores = scores + attention_mask  # additive mask (-inf → 0 after softmax)

        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)

        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(bsz, T, H)
        out = self.o_proj(out)

        if output_attentions:
            return out, attn_weights, None
        return out, None, None


class FakeBlock(nn.Module):
    """Minimal transformer block stub with the subset of attributes used by
    recompute_block_attention: ``input_layernorm`` (identity) and ``self_attn``.
    """

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _FakeSelfAttn(hidden_size, n_heads)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_block(n_heads: int = 4, head_dim: int = 16) -> FakeBlock:
    return FakeBlock(hidden_size=n_heads * head_dim, n_heads=n_heads)


def _run(
    prompt_len: int,
    response_len: int,
    n_heads: int = 4,
    head_dim: int = 16,
) -> torch.Tensor:
    hidden_size = n_heads * head_dim
    block = _make_block(n_heads, head_dim)
    T = prompt_len + response_len
    h_prev = torch.randn(T, hidden_size)
    return recompute_block_attention(h_prev, block, prompt_len, response_len)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shape_standard():
    """Standard config: prompt_len=32, response_len=10 → (10, 10)."""
    result = _run(prompt_len=32, response_len=10, n_heads=4, head_dim=16)
    assert result.shape == (10, 10), result.shape


def test_output_shape_single_response_token():
    """Edge case: response_len=1 → (1, 1)."""
    result = _run(prompt_len=8, response_len=1, n_heads=4, head_dim=16)
    assert result.shape == (1, 1), result.shape


def test_output_shape_r_max():
    """r_max edge case: response_len=64 → (64, 64)."""
    result = _run(prompt_len=16, response_len=64, n_heads=4, head_dim=16)
    assert result.shape == (64, 64), result.shape


def test_output_dtype():
    """Returned tensor must be float32 regardless of internal computation dtype."""
    result = _run(prompt_len=8, response_len=4)
    assert result.dtype == torch.float32, result.dtype


# ---------------------------------------------------------------------------
# Head-average helper test
# ---------------------------------------------------------------------------

def test_head_average_resp_to_resp():
    """_head_average_resp_to_resp slices rows/cols [prompt_len:end] and averages heads."""
    n_heads, T, prompt_len, response_len = 4, 10, 3, 7
    # Uniform attention: all heads, all positions = constant c.
    c = 0.25
    attn = torch.full((n_heads, T, T), c)
    result = _head_average_resp_to_resp(attn, prompt_len, response_len)

    assert result.shape == (response_len, response_len), result.shape
    # Slicing any sub-block of a uniform tensor and averaging over heads = c.
    expected = torch.full((response_len, response_len), c)
    assert torch.allclose(result, expected), f"max_diff={((result - expected).abs()).max()}"


def test_head_average_resp_to_resp_different_heads():
    """Averaging across heads produces the mean, not just one head's values."""
    n_heads, T = 3, 6
    prompt_len, response_len = 2, 4
    attn = torch.zeros(n_heads, T, T)
    # Head 0: resp-to-resp block all 1s; Head 1: 2s; Head 2: 3s.
    attn[0, prompt_len:, prompt_len:] = 1.0
    attn[1, prompt_len:, prompt_len:] = 2.0
    attn[2, prompt_len:, prompt_len:] = 3.0

    result = _head_average_resp_to_resp(attn, prompt_len, response_len)
    expected = torch.full((response_len, response_len), 2.0)  # (1+2+3)/3
    assert torch.allclose(result, expected), f"max_diff={((result - expected).abs()).max()}"


# ---------------------------------------------------------------------------
# Cross-region discarded
# ---------------------------------------------------------------------------

def test_cross_region_discarded():
    """The returned tensor has shape (response_len, response_len) — prompt rows absent."""
    prompt_len, response_len = 20, 8
    result = _run(prompt_len=prompt_len, response_len=response_len)
    # Shape itself proves that prompt rows are not in the output.
    assert result.shape == (response_len, response_len), result.shape
    # All values should be non-negative (softmax outputs).
    assert (result >= 0).all(), "attention values must be non-negative"


# ---------------------------------------------------------------------------
# Rows sum to 1
# ---------------------------------------------------------------------------

def test_rows_sum_to_one():
    """With prompt_len=0 the response IS the full sequence; rows must sum to 1."""
    prompt_len, response_len = 0, 12
    result = _run(prompt_len=prompt_len, response_len=response_len)
    row_sums = result.sum(dim=-1)  # (response_len,)
    assert torch.allclose(row_sums, torch.ones(response_len), atol=1e-5), (
        f"Row sums: {row_sums}"
    )


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def test_causal_mask_lower_triangular():
    """Future-token columns must be zero: result[i, j] == 0 for j > i."""
    prompt_len, response_len = 0, 8
    result = _run(prompt_len=prompt_len, response_len=response_len)

    for i in range(response_len):
        for j in range(i + 1, response_len):
            val = result[i, j].item()
            assert abs(val) < 1e-6, (
                f"Expected 0 at [{i}, {j}] (future token), got {val:.6f}"
            )


def test_causal_mask_lower_triangular_with_prompt():
    """Causal property holds even when prompt_len > 0."""
    prompt_len, response_len = 16, 10
    result = _run(prompt_len=prompt_len, response_len=response_len)
    # Upper triangle of the response-to-response block must be zero.
    for i in range(response_len):
        for j in range(i + 1, response_len):
            val = result[i, j].item()
            assert abs(val) < 1e-6, (
                f"Expected 0 at [{i}, {j}] (future response token), got {val:.6f}"
            )


# ---------------------------------------------------------------------------
# Edge case: prompt_len=0 with various response lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response_len", [1, 2, 16, 64])
def test_shape_prompt_len_zero(response_len: int):
    result = _run(prompt_len=0, response_len=response_len)
    assert result.shape == (response_len, response_len)


# ---------------------------------------------------------------------------
# ValueError on bad inputs
# ---------------------------------------------------------------------------

def test_bad_shape_raises():
    block = _make_block()
    h_prev = torch.randn(5, 64)  # T=5, but prompt_len+response_len=10
    with pytest.raises(ValueError):
        recompute_block_attention(h_prev, block, prompt_len=6, response_len=4)


def test_zero_response_len_raises():
    block = _make_block()
    h_prev = torch.randn(8, 64)
    with pytest.raises(ValueError):
        recompute_block_attention(h_prev, block, prompt_len=8, response_len=0)
