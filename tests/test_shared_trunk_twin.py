"""Tests for SharedTrunkSplitOutputCompressor (D1) and SharedTrunkProjectionHeadCompressor (D2).

CPU-only; no real data or GPU.

Run with:
    pytest tests/test_shared_trunk_twin.py -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

from activation_research.model import (
    SharedTrunkSplitOutputCompressor,
    SharedTrunkProjectionHeadCompressor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small input_dim that keeps the ProgressiveCompressor TransformerBlock happy:
# nhead = min(8, final_dim // 64) must be >= 1, so final_dim >= 64.
# For D1 that means half_dim >= 32 (final_dim = 2 * half_dim).
# Use input_dim=128 so the inner encoder dimensions stay manageable.

_INPUT_DIM = 128
_B = 4
_SEQ = 10


def _make_d1(half_dim: int = 8) -> SharedTrunkSplitOutputCompressor:
    torch.manual_seed(0)
    # Use input_dim=128, half_dim=32 so final_dim=64 and nhead=1 is valid.
    # For shape tests we use the caller's half_dim but cap at 32 for correctness.
    return SharedTrunkSplitOutputCompressor(
        input_dim=_INPUT_DIM,
        half_dim=half_dim,
    )


def _make_d2(trunk_dim: int = 16, head_dim: int = 8) -> SharedTrunkProjectionHeadCompressor:
    torch.manual_seed(0)
    return SharedTrunkProjectionHeadCompressor(
        input_dim=_INPUT_DIM,
        trunk_dim=trunk_dim,
        head_dim=head_dim,
        head_hidden_dim=head_dim,
    )


def _random_x(b: int = _B, seq: int = _SEQ, d: int = _INPUT_DIM) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(b, seq, d)


# ---------------------------------------------------------------------------
# D1 — SharedTrunkSplitOutputCompressor
# ---------------------------------------------------------------------------


def test_split_output_forward_shape():
    """forward() on (B, L, 128) returns (B, 2*half_dim)."""
    # Use half_dim=32 so final_dim=64 satisfies nhead >= 1 constraint.
    half_dim = 32
    model = _make_d1(half_dim=half_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (_B, 2 * half_dim), (
        f"Expected ({_B}, {2 * half_dim}), got {out.shape}"
    )


def test_split_output_slice_helper():
    """forward_slices() returns (z, z_A, z_B, logprob_pred) with correct slicing."""
    half_dim = 32
    model = _make_d1(half_dim=half_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        z, z_A, z_B, logprob_pred = model.forward_slices(x)

    assert z.shape == (_B, 2 * half_dim), f"z shape mismatch: {z.shape}"
    assert z_A.shape == (_B, half_dim), f"z_A shape mismatch: {z_A.shape}"
    assert z_B.shape == (_B, half_dim), f"z_B shape mismatch: {z_B.shape}"

    # z_A must equal the first half of z
    assert torch.allclose(z_A, z[:, :half_dim]), "z_A != z[:, :half_dim]"
    # z_B must equal the second half of z
    assert torch.allclose(z_B, z[:, half_dim:]), "z_B != z[:, half_dim:]"


def test_split_output_forward_matches_slices_z():
    """forward(x) and the z returned by forward_slices(x) must be identical."""
    half_dim = 32
    model = _make_d1(half_dim=half_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        z_forward = model(x)
        z_slices, _, _, _ = model.forward_slices(x)
    assert torch.allclose(z_forward, z_slices), "forward(x) != forward_slices(x)[0]"


# ---------------------------------------------------------------------------
# D2 — SharedTrunkProjectionHeadCompressor
# ---------------------------------------------------------------------------


def test_projection_head_forward_returns_trunk():
    """forward() returns (B, trunk_dim), NOT (B, head_dim). Heads must not be called."""
    trunk_dim = 64  # must be >= 64 for nhead constraint
    head_dim = 8
    model = _make_d2(trunk_dim=trunk_dim, head_dim=head_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (_B, trunk_dim), (
        f"Expected ({_B}, {trunk_dim}), got {out.shape}"
    )


def test_projection_head_with_heads_helper():
    """forward_with_heads() returns (z, zA, zB) with correct shapes."""
    trunk_dim = 64
    head_dim = 8
    model = _make_d2(trunk_dim=trunk_dim, head_dim=head_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        z, zA, zB, logprob_pred = model.forward_with_heads(x)

    assert z.shape == (_B, trunk_dim), f"z shape mismatch: {z.shape}"
    assert zA.shape == (_B, head_dim), f"zA shape mismatch: {zA.shape}"
    assert zB.shape == (_B, head_dim), f"zB shape mismatch: {zB.shape}"


def test_projection_head_forward_matches_trunk_in_with_heads():
    """forward(x) must equal the trunk z returned by forward_with_heads(x)."""
    trunk_dim = 64
    head_dim = 8
    model = _make_d2(trunk_dim=trunk_dim, head_dim=head_dim)
    model.eval()
    x = _random_x()
    with torch.no_grad():
        z_forward = model(x)
        z_trunk, _, _, _ = model.forward_with_heads(x)
    assert torch.allclose(z_forward, z_trunk), "forward(x) != forward_with_heads(x)[0]"


# ---------------------------------------------------------------------------
# Dual-loss trainer smoke test
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 32, seq: int = _SEQ, d: int = _INPUT_DIM):
    """Return a lightweight in-memory dataset compatible with the contrastive collate fn."""
    torch.manual_seed(99)

    class _SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, n, seq, d):
            self.n = n
            self.data = torch.randn(n, 2, seq, d)  # 2 views
            self.labels = torch.randint(0, 2, (n,))
            self.logprobs = torch.randn(n, seq)

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "views_activations": self.data[idx],       # (2, seq, d)
                "halu": self.labels[idx],                   # scalar int tensor
                "logprob": self.logprobs[idx],              # (seq,)
                "hashkey": f"sample_{idx}",
            }

    return _SyntheticDataset(n, seq, d)


def test_dualloss_step_runs_d1():
    """One training step via train_contrastive_logprob_recon_dualloss on D1 (CPU).

    Asserts:
    - loss is finite
    - gradients flow to trunk parameters AND decoder parameters
    """
    from activation_research.training import train_contrastive_logprob_recon_dualloss

    model = SharedTrunkSplitOutputCompressor(
        input_dim=_INPUT_DIM,
        half_dim=32,  # final_dim=64, nhead=1 valid
    )
    ds = _make_synthetic_dataset(n=8)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        train_contrastive_logprob_recon_dualloss(
            model=model,
            train_dataset=ds,
            test_dataset=None,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            temperature=0.1,
            device="cpu",
            num_workers=0,
            sub_batch_size=8,
            checkpoint_dir=ckpt_dir,
            save_every=1,
            persistent_workers=False,
            use_infinite_index_stream=False,
            recon_lambda=1.0,
        )

    # Check gradients flowed — re-run one manual step to inspect grads
    model2 = SharedTrunkSplitOutputCompressor(input_dim=_INPUT_DIM, half_dim=32)
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

    x = torch.randn(_B, _SEQ, _INPUT_DIM)
    x_2views = x.unsqueeze(1).expand(-1, 2, -1, -1)  # (B, 2, seq, d)
    bsz, num_views, seq, hidden = x_2views.shape
    x_flat = x_2views.reshape(bsz * num_views, seq, hidden)

    z_flat, zA_flat, zB_flat, logprob_pred = model2.forward_slices(x_flat)
    labels = torch.randint(0, 2, (bsz,))

    from activation_research.training import SupConLoss
    loss_fn_A = SupConLoss(temperature=0.1, ignore_label=1)
    loss_fn_B = SupConLoss(temperature=0.1, ignore_label=0)

    zA_views = zA_flat.reshape(bsz, num_views, -1)
    zB_views = zB_flat.reshape(bsz, num_views, -1)

    target = torch.randn(bsz * num_views, model2._inner.recon_seq_len)
    recon, _ = model2.recon_loss(logprob_pred, target)

    sample_ids = torch.arange(bsz, dtype=torch.long)
    loss = loss_fn_A(zA_views, labels=labels, sample_ids=sample_ids) + \
           loss_fn_B(zB_views, labels=labels, sample_ids=sample_ids) + recon

    optimizer.zero_grad()
    loss.backward()

    assert math.isfinite(loss.item()), f"loss is not finite: {loss.item()}"

    for name, param in model2.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for param: {name}"


def test_dualloss_step_runs_d2():
    """One training step via train_contrastive_logprob_recon_dualloss on D2 (CPU).

    Asserts:
    - loss is finite
    - gradients flow to trunk parameters AND both projection head parameters
    """
    from activation_research.training import train_contrastive_logprob_recon_dualloss

    model = SharedTrunkProjectionHeadCompressor(
        input_dim=_INPUT_DIM,
        trunk_dim=64,
        head_dim=8,
        head_hidden_dim=8,
    )
    ds = _make_synthetic_dataset(n=8)

    with tempfile.TemporaryDirectory() as ckpt_dir:
        train_contrastive_logprob_recon_dualloss(
            model=model,
            train_dataset=ds,
            test_dataset=None,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            temperature=0.1,
            device="cpu",
            num_workers=0,
            sub_batch_size=8,
            checkpoint_dir=ckpt_dir,
            save_every=1,
            persistent_workers=False,
            use_infinite_index_stream=False,
            recon_lambda=1.0,
        )

    # Manual step to verify gradient flow to trunk + both heads
    model2 = SharedTrunkProjectionHeadCompressor(
        input_dim=_INPUT_DIM, trunk_dim=64, head_dim=8, head_hidden_dim=8
    )
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)

    x = torch.randn(_B, _SEQ, _INPUT_DIM)
    x_2views = x.unsqueeze(1).expand(-1, 2, -1, -1)
    bsz, num_views, seq, hidden = x_2views.shape
    x_flat = x_2views.reshape(bsz * num_views, seq, hidden)

    z_flat, zA_flat, zB_flat, logprob_pred = model2.forward_with_heads(x_flat)
    labels = torch.randint(0, 2, (bsz,))

    from activation_research.training import SupConLoss
    loss_fn_A = SupConLoss(temperature=0.1, ignore_label=1)
    loss_fn_B = SupConLoss(temperature=0.1, ignore_label=0)

    zA_views = zA_flat.reshape(bsz, num_views, -1)
    zB_views = zB_flat.reshape(bsz, num_views, -1)

    target = torch.randn(bsz * num_views, model2._inner.recon_seq_len)
    recon, _ = model2.recon_loss(logprob_pred, target)

    sample_ids = torch.arange(bsz, dtype=torch.long)
    loss = loss_fn_A(zA_views, labels=labels, sample_ids=sample_ids) + \
           loss_fn_B(zB_views, labels=labels, sample_ids=sample_ids) + recon

    optimizer.zero_grad()
    loss.backward()

    assert math.isfinite(loss.item()), f"loss is not finite: {loss.item()}"

    # Trunk parameters
    for name, param in model2._inner.named_parameters():
        assert param.grad is not None, f"No gradient for trunk param: {name}"
    # Head A parameters
    for name, param in model2.head_A.named_parameters():
        assert param.grad is not None, f"No gradient for head_A param: {name}"
    # Head B parameters
    for name, param in model2.head_B.named_parameters():
        assert param.grad is not None, f"No gradient for head_B param: {name}"


# ---------------------------------------------------------------------------
# Eval surface test for D2: heads must not affect forward()
# ---------------------------------------------------------------------------


def test_eval_surface_d2_is_trunk():
    """After zeroing out head weights, model.eval(); model(x) is unchanged.

    This confirms forward() returns purely the trunk embedding and does NOT
    depend on head_A or head_B weights.
    """
    trunk_dim = 64
    head_dim = 8
    model = _make_d2(trunk_dim=trunk_dim, head_dim=head_dim)
    model.eval()
    x = _random_x()

    with torch.no_grad():
        z_before = model(x).clone()

        # Zero out all head weights
        for param in model.head_A.parameters():
            param.data.zero_()
        for param in model.head_B.parameters():
            param.data.zero_()

        z_after = model(x)

    assert torch.allclose(z_before, z_after), (
        "forward() changed after zeroing head weights — heads must not be called in forward()"
    )
