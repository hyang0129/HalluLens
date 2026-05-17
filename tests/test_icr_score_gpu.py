"""
tests/test_icr_score_gpu.py — CPU-only equivalence tests for the batched GPU
ICR kernel in activation_research/icr_score_gpu.py.

All 6 tests required by specs/issue_72_gpu_icr.md §3. No CUDA required — the
kernel is tested on CPU with device='cpu'.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensors(
    B: int = 3,
    L: int = 4,
    r_max: int = 8,
    hidden_dim: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return synthetic (response_attn, h_block_input, delta_h) of canonical test shape."""
    rng = np.random.default_rng(seed)
    raw_attn = rng.random((B, L, r_max, r_max)).astype(np.float32)
    raw_attn = raw_attn / (raw_attn.sum(axis=-1, keepdims=True) + 1e-8)
    h = rng.standard_normal((B, L, r_max, hidden_dim)).astype(np.float32)
    dh = rng.standard_normal((B, L, r_max, hidden_dim)).astype(np.float32)
    return (
        torch.from_numpy(raw_attn).to(device),
        torch.from_numpy(h).to(device),
        torch.from_numpy(dh).to(device),
    )


def _numpy_reference(
    response_attn: torch.Tensor,   # (B, L, r_max, r_max)
    h_block_input: torch.Tensor,   # (B, L, r_max, hidden_dim)
    delta_h: torch.Tensor,         # (B, L, r_max, hidden_dim)
    response_lens: torch.Tensor,   # (B,)
    top_p: float = 0.1,
) -> np.ndarray:
    """Per-sample numpy loop matching the reference spec §3 contract."""
    from activation_research.icr_score import compute_icr_score

    B, L = response_attn.shape[:2]
    attn_np = response_attn.cpu().numpy()
    h_np = h_block_input.cpu().numpy()
    dh_np = delta_h.cpu().numpy()
    rlens = response_lens.cpu().numpy()

    return np.stack([
        np.array([
            compute_icr_score(
                attn_np[b, l], h_np[b, l], dh_np[b, l], int(rlens[b]), top_p=top_p
            )
            for l in range(L)
        ])
        for b in range(B)
    ])  # (B, L)


# ---------------------------------------------------------------------------
# 1. Single-sample equivalence (fp32)
# ---------------------------------------------------------------------------

def test_gpu_matches_numpy_single_sample():
    """Single sample (B=1): GPU result must match numpy reference within 1e-5."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 1, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([8], dtype=torch.int64)

    gpu_out = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()
    numpy_out = _numpy_reference(attn, h, dh, rlens)

    max_diff = float(np.max(np.abs(gpu_out - numpy_out)))
    assert max_diff < 1e-5, (
        f"Single-sample GPU vs numpy max|diff|={max_diff:.2e} >= 1e-5"
    )
    assert np.all(np.isfinite(gpu_out)), "GPU output must be finite"


# ---------------------------------------------------------------------------
# 2. Batched equivalence with different response_lens (fp32)
# ---------------------------------------------------------------------------

def test_gpu_matches_numpy_batched():
    """B=3 with distinct response_lens (2, 5, 8): each row must match numpy within 1e-5."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 3, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([2, 5, 8], dtype=torch.int64)

    gpu_out = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()
    numpy_out = _numpy_reference(attn, h, dh, rlens)

    for b in range(B):
        max_diff = float(np.max(np.abs(gpu_out[b] - numpy_out[b])))
        assert max_diff < 1e-5, (
            f"Batched b={b} (response_len={rlens[b]}): max|diff|={max_diff:.2e} >= 1e-5"
        )
    assert np.all(np.isfinite(gpu_out)), "GPU output must be finite"


# ---------------------------------------------------------------------------
# 3. response_len == 0 produces 0 (no NaN)
# ---------------------------------------------------------------------------

def test_gpu_handles_response_len_zero():
    """response_lens[b]=0 must produce 0.0 for that row, not NaN."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 3, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([0, 4, 8], dtype=torch.int64)

    gpu_out = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()
    numpy_out = _numpy_reference(attn, h, dh, rlens)

    assert np.all(np.isfinite(gpu_out)), f"GPU output contains NaN/Inf: {gpu_out}"
    np.testing.assert_array_equal(
        gpu_out[0], np.zeros(L, dtype=np.float32),
        err_msg="response_len=0 row must be all zeros",
    )
    # numpy reference also returns 0 for response_len=0
    np.testing.assert_array_equal(
        numpy_out[0], np.zeros(L, dtype=np.float32),
        err_msg="numpy reference must return 0 for response_len=0",
    )
    # Non-zero-len samples must still match
    for b in [1, 2]:
        max_diff = float(np.max(np.abs(gpu_out[b] - numpy_out[b])))
        assert max_diff < 1e-5, (
            f"b={b} (response_len={rlens[b]}): max|diff|={max_diff:.2e} >= 1e-5"
        )


# ---------------------------------------------------------------------------
# 4. response_len == r_max (full window, no padding to mask)
# ---------------------------------------------------------------------------

def test_gpu_handles_response_len_equals_r_max():
    """response_lens[b] == r_max: no padding; must match numpy within 1e-5."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 3, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([r_max, r_max, r_max], dtype=torch.int64)

    gpu_out = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()
    numpy_out = _numpy_reference(attn, h, dh, rlens)

    max_diff = float(np.max(np.abs(gpu_out - numpy_out)))
    assert max_diff < 1e-5, (
        f"response_len=r_max: max|diff|={max_diff:.2e} >= 1e-5"
    )
    assert np.all(np.isfinite(gpu_out)), "GPU output must be finite"


# ---------------------------------------------------------------------------
# 5. fp16 attention input (kernel must upcast internally)
# ---------------------------------------------------------------------------

def test_gpu_fp16_attention_input():
    """fp16 response_attn must be handled; result within 5e-4 of fp32 path."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 3, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([3, 6, 8], dtype=torch.int64)

    # fp32 path (reference)
    out_f32 = compute_icr_per_layer_batched_gpu(attn.float(), h, dh, rlens).numpy()
    # fp16 attention path (what the real pipeline stores)
    out_f16 = compute_icr_per_layer_batched_gpu(attn.half(), h, dh, rlens).numpy()

    assert np.all(np.isfinite(out_f16)), "fp16-attn output contains NaN/Inf"
    max_diff = float(np.max(np.abs(out_f32 - out_f16)))
    assert max_diff < 5e-4, (
        f"fp16 vs fp32 attention: max|diff|={max_diff:.2e} >= 5e-4"
    )


# ---------------------------------------------------------------------------
# 5b. Realistic r_max → k > 1 (regression for h_in.unsqueeze axis bug)
# ---------------------------------------------------------------------------

def test_gpu_matches_numpy_realistic_r_max():
    """At r_max=64 (production size), top_p=0.1 gives k=6 — the regime where
    h_in gather axis matters. With the wrong unsqueeze axis, all k positions
    collapse to the same h_in[q], degenerating the JSD; this test catches it.

    Smaller fixtures (r_max=8) silently pass because k=int(0.1*8)=0→clamped to
    1, and JSD with a single-element distribution is identically 0 either way.
    """
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 2, 3, 64, 128
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D, seed=7)
    rlens = torch.tensor([r_max, r_max], dtype=torch.int64)

    gpu_out = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()
    numpy_out = _numpy_reference(attn, h, dh, rlens)

    max_diff = float(np.max(np.abs(gpu_out - numpy_out)))
    assert max_diff < 1e-5, (
        f"r_max=64 GPU vs numpy max|diff|={max_diff:.2e} >= 1e-5"
    )


# ---------------------------------------------------------------------------
# 6. CPU fallback (monkeypatch cuda unavailable)
# ---------------------------------------------------------------------------

def test_cpu_fallback(monkeypatch):
    """Function must produce identical output whether CUDA is reported available or not."""
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu

    B, L, r_max, D = 3, 4, 8, 32
    attn, h, dh = _make_tensors(B=B, L=L, r_max=r_max, hidden_dim=D)
    rlens = torch.tensor([2, 5, 8], dtype=torch.int64)

    # Run once on CPU normally
    out_normal = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()

    # Force CUDA unavailable and run again
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out_no_cuda = compute_icr_per_layer_batched_gpu(attn, h, dh, rlens).numpy()

    assert np.all(np.isfinite(out_no_cuda)), "CPU fallback output contains NaN/Inf"
    max_diff = float(np.max(np.abs(out_normal - out_no_cuda)))
    assert max_diff < 1e-5, (
        f"CPU fallback vs normal: max|diff|={max_diff:.2e} >= 1e-5"
    )
