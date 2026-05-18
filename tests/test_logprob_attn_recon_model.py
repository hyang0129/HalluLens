"""Tests for LogprobAttnReconProgressiveCompressor (Mechanism F + K combined).

CPU-only; no real data or GPU.

Run with:
    pytest tests/test_logprob_attn_recon_model.py -v
"""

from __future__ import annotations

import pytest
import torch

from activation_research.model import (
    LogprobAttnReconProgressiveCompressor,
    LogprobReconProgressiveCompressor,
    ProgressiveCompressor,
)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="attn_direction"):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_direction="sideways",
        )


def test_invalid_target_raises():
    with pytest.raises(ValueError, match="attn_target"):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_target="hexgrid",
        )


def test_coarse_and_full_targets_not_implemented():
    with pytest.raises(NotImplementedError):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_target="coarse",
        )
    with pytest.raises(NotImplementedError):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_target="full",
        )


def test_active_directions_dispatch():
    # Internal ModuleDict uses "fwd" / "bwd" keys (since "forward" collides
    # with nn.Module.forward); the public direction names stay "forward" /
    # "backward".
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="forward",
    )
    assert m._active_attn_dirs == ("forward",)
    assert set(m.attn_decoders.keys()) == {"fwd"}

    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
    )
    assert m._active_attn_dirs == ("backward",)
    assert set(m.attn_decoders.keys()) == {"bwd"}

    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="both",
    )
    assert m._active_attn_dirs == ("forward", "backward")
    assert set(m.attn_decoders.keys()) == {"fwd", "bwd"}

    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="none",
    )
    assert m._active_attn_dirs == ()
    assert len(m.attn_decoders) == 0


# ---------------------------------------------------------------------------
# Inference forward equivalence
# ---------------------------------------------------------------------------

def test_forward_returns_z_only():
    m = LogprobAttnReconProgressiveCompressor(input_dim=128, final_dim=64)
    x = torch.randn(2, 6, 128)
    out = m(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 64)


def test_forward_matches_progressive_compressor_with_same_encoder_weights():
    """The inference path must be identical to ProgressiveCompressor."""
    torch.manual_seed(0)
    plain = ProgressiveCompressor(input_dim=128, final_dim=64)
    torch.manual_seed(0)
    combined = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
    )

    # The inner encoder layers should be initialised identically.
    combined.encoder.load_state_dict(plain.state_dict())
    plain.eval()
    combined.eval()
    x = torch.randn(3, 5, 128)
    out_plain = plain(x)
    out_combined = combined(x)
    torch.testing.assert_close(out_plain, out_combined)


# ---------------------------------------------------------------------------
# forward_with_recon shapes
# ---------------------------------------------------------------------------

def test_forward_with_recon_backward_only():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
        recon_seq_len=10, attn_num_stat_features=3,
    )
    x = torch.randn(2, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert z.shape == (2, 64)
    assert lp_pred.shape == (2, 10)
    assert set(attn_pred.keys()) == {"backward"}
    assert attn_pred["backward"].shape == (2, 3)


def test_forward_with_recon_both_directions():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="both",
        recon_seq_len=10, attn_num_stat_features=3,
    )
    x = torch.randn(2, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert z.shape == (2, 64)
    assert lp_pred.shape == (2, 10)
    assert set(attn_pred.keys()) == {"forward", "backward"}
    assert attn_pred["forward"].shape == (2, 3)
    assert attn_pred["backward"].shape == (2, 3)


def test_forward_with_recon_none_returns_empty_attn():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="none",
    )
    x = torch.randn(2, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert z.shape == (2, 64)
    assert lp_pred.shape == (2, m.recon_seq_len)
    assert attn_pred == {}


# ---------------------------------------------------------------------------
# Logprob recon equivalence with LogprobReconProgressiveCompressor
# ---------------------------------------------------------------------------

def test_lp_recon_matches_logprob_recon_progressive_compressor():
    """recon_loss_lp on combined model == recon_loss on F-only model."""
    torch.manual_seed(0)
    f_only = LogprobReconProgressiveCompressor(
        input_dim=128, final_dim=64, recon_seq_len=10,
        logprob_var_threshold=1e-8,
    )
    torch.manual_seed(0)
    combined = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, recon_seq_len=10,
        logprob_var_threshold=1e-8,
    )

    lp_pred = torch.randn(3, 10, requires_grad=False)
    lp_target = torch.randn(3, 10) * 0.5  # high enough variance

    loss_f, diag_f = f_only.recon_loss(lp_pred, lp_target)
    loss_c, diag_c = combined.recon_loss_lp(lp_pred, lp_target)

    torch.testing.assert_close(loss_f, loss_c)
    assert diag_f["suppressed"] == diag_c["suppressed"]


def test_lp_recon_suppressed_below_variance_threshold():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, logprob_var_threshold=1.0,
    )
    lp_pred = torch.randn(3, m.recon_seq_len)
    lp_target = torch.zeros(3, m.recon_seq_len)  # zero variance
    loss, diag = m.recon_loss_lp(lp_pred, lp_target)
    assert diag["suppressed"] is True
    assert float(loss) == 0.0


# ---------------------------------------------------------------------------
# Attention recon loss
# ---------------------------------------------------------------------------

def test_attn_recon_basic():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
        attn_var_threshold=1e-8,
    )
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3) * 0.5
    loss, diag = m.recon_loss_attn(pred, target)
    assert torch.isfinite(loss)
    assert diag["suppressed"] is False


def test_attn_recon_suppressed_below_variance_threshold():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
        attn_var_threshold=1.0,
    )
    pred = torch.randn(4, 3)
    target = torch.zeros(4, 3)  # zero variance
    loss, diag = m.recon_loss_attn(pred, target)
    assert diag["suppressed"] is True
    assert float(loss) == 0.0


def test_attn_recon_all_nan_target_suppressed():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
    )
    pred = torch.randn(4, 3)
    target = torch.full((4, 3), float("nan"))
    loss, diag = m.recon_loss_attn(pred, target)
    assert diag["suppressed"] is True
    assert float(loss) == 0.0


def test_attn_recon_partial_nan_excluded_from_loss():
    """Loss must equal MSE over non-NaN entries only."""
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
        attn_var_threshold=1e-8,
    )
    pred = torch.tensor(
        [[1.0, 2.0, 3.0],
         [0.0, 0.0, 0.0]],
        requires_grad=False,
    )
    target = torch.tensor(
        [[2.0, 3.0, 4.0],  # 3 valid entries, sq err = 1+1+1 = 3
         [float("nan"), float("nan"), float("nan")]],  # all NaN, excluded
    )
    loss, diag = m.recon_loss_attn(pred, target)
    # 3 valid entries, sum sq err = 3, mean = 1.0
    assert abs(float(loss) - 1.0) < 1e-5
    assert diag["valid_count"] == 3
