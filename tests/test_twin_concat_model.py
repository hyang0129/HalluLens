"""Tests for TwinConcatModel (issue #99).

CPU-only; no real data or GPU.

Run with:
    pytest tests/test_twin_concat_model.py -v
"""

from __future__ import annotations

import torch

from activation_research.model import LogprobReconProgressiveCompressor, TwinConcatModel


def test_twin_concat_model_output_shape():
    """Output shape must be (B, final_dim_a + final_dim_b) = (B, 128)."""
    torch.manual_seed(0)
    # Use input_dim=128, final_dim=64 so TransformerBlock nhead=64//64=1 (valid).
    head_a = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    head_b = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    model = TwinConcatModel(head_a=head_a, head_b=head_b)
    model.eval()

    B = 4
    seq_len = 5
    x = torch.randn(B, seq_len, 128)
    out = model(x)

    assert out.shape == (B, 128), f"Expected (4, 128), got {out.shape}"


def test_twin_concat_model_gradients_independent():
    """head_a and head_b must share no parameters."""
    head_a = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    head_b = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    model = TwinConcatModel(head_a=head_a, head_b=head_b)

    ids_a = set(id(p) for p in model.head_a.parameters())
    ids_b = set(id(p) for p in model.head_b.parameters())

    assert ids_a.isdisjoint(ids_b), (
        "head_a and head_b share parameter objects — they must be independent"
    )
