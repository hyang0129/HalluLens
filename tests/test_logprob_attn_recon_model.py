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


def test_coarse_target_constructs():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_target="coarse",
        attn_num_layers=4, attn_direction="backward",
    )
    assert m.attn_target == "coarse"
    assert m._attn_out_dim == 4 * 8 * 8  # num_layers × 8 × 8


def test_full_target_constructs():
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_target="full",
        attn_num_layers=4, attn_r_max=6, attn_direction="backward",
    )
    assert m.attn_target == "full"
    assert m._attn_out_dim == 4 * 6 * 6  # num_layers × r_max × r_max


def test_invalid_attn_loss_raises():
    with pytest.raises(ValueError, match="attn_loss"):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_loss="l1",
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


# ---------------------------------------------------------------------------
# Coarse target — decoder shape + loss
# ---------------------------------------------------------------------------

_NUM_LAYERS = 4
_R_MAX = 6


def _make_coarse_model(**kwargs):
    return LogprobAttnReconProgressiveCompressor(
        input_dim=128,
        final_dim=64,
        attn_target="coarse",
        attn_num_layers=_NUM_LAYERS,
        attn_var_threshold=1e-8,
        **kwargs,
    )


def _make_full_model(**kwargs):
    return LogprobAttnReconProgressiveCompressor(
        input_dim=128,
        final_dim=64,
        attn_target="full",
        attn_num_layers=_NUM_LAYERS,
        attn_r_max=_R_MAX,
        attn_var_threshold=1e-8,
        **kwargs,
    )


def test_coarse_decoder_output_shape_backward():
    m = _make_coarse_model(attn_direction="backward")
    x = torch.randn(3, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert z.shape == (3, 64)
    assert set(attn_pred.keys()) == {"backward"}
    # Flat output: num_layers × 8 × 8
    assert attn_pred["backward"].shape == (3, _NUM_LAYERS * 8 * 8)


def test_coarse_decoder_output_shape_both_directions():
    m = _make_coarse_model(attn_direction="both")
    x = torch.randn(2, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert set(attn_pred.keys()) == {"forward", "backward"}
    assert attn_pred["forward"].shape == (2, _NUM_LAYERS * 8 * 8)
    assert attn_pred["backward"].shape == (2, _NUM_LAYERS * 8 * 8)


def test_full_decoder_output_shape_backward():
    m = _make_full_model(attn_direction="backward")
    x = torch.randn(3, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert set(attn_pred.keys()) == {"backward"}
    # Flat output: num_layers × r_max × r_max
    assert attn_pred["backward"].shape == (3, _NUM_LAYERS * _R_MAX * _R_MAX)


def test_full_decoder_output_shape_both_directions():
    m = _make_full_model(attn_direction="both")
    x = torch.randn(2, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert set(attn_pred.keys()) == {"forward", "backward"}
    assert attn_pred["forward"].shape == (2, _NUM_LAYERS * _R_MAX * _R_MAX)
    assert attn_pred["backward"].shape == (2, _NUM_LAYERS * _R_MAX * _R_MAX)


def test_coarse_decoder_output_dtype():
    m = _make_coarse_model(attn_direction="backward")
    x = torch.randn(2, 6, 128)
    _, _, attn_pred = m.forward_with_recon(x)
    assert attn_pred["backward"].dtype == torch.float32


def test_full_decoder_output_dtype():
    m = _make_full_model(attn_direction="backward")
    x = torch.randn(2, 6, 128)
    _, _, attn_pred = m.forward_with_recon(x)
    assert attn_pred["backward"].dtype == torch.float32


# ---------------------------------------------------------------------------
# NaN masking for coarse/full targets
# ---------------------------------------------------------------------------

def test_coarse_mse_out_of_range_layer_contributes_zero():
    """A layer slice that is entirely NaN must not contribute to the loss."""
    m = _make_coarse_model(attn_direction="backward")
    D = _NUM_LAYERS * 8 * 8
    pred = torch.randn(2, D)
    # Layer 0 valid (non-NaN), layer 1..3 entirely NaN.
    target = torch.full((2, D), float("nan"))
    target[:, : 8 * 8] = torch.randn(2, 8 * 8)  # fill layer-0 cells

    loss, diag = m.recon_loss_attn(pred, target)
    assert torch.isfinite(loss)
    assert diag["suppressed"] is False
    # Only 2 × 64 = 128 valid cells (layer 0 only).
    assert diag["valid_count"] == 2 * 8 * 8


def test_full_mse_out_of_response_cells_contribute_zero():
    """Cells beyond r_eff (NaN-filled by dataset) must not contribute."""
    m = _make_full_model(attn_direction="backward")
    D = _NUM_LAYERS * _R_MAX * _R_MAX
    pred = torch.randn(4, D)
    # All valid.
    target = torch.randn(4, D)
    loss_all_valid, _ = m.recon_loss_attn(pred, target)

    # NaN half the cells.
    target_partial = target.clone()
    target_partial[:, D // 2:] = float("nan")
    loss_partial, diag_partial = m.recon_loss_attn(pred, target_partial)

    # With half the cells NaN, the valid_count drops.
    assert diag_partial["valid_count"] == 4 * (D // 2)
    # Both losses are finite.
    assert torch.isfinite(loss_all_valid)
    assert torch.isfinite(loss_partial)


# ---------------------------------------------------------------------------
# MSE + KL loss paths — finite outputs and finite gradients
# ---------------------------------------------------------------------------

def test_coarse_mse_loss_no_nan_gradients():
    m = _make_coarse_model(attn_direction="backward", attn_loss="mse")
    x = torch.randn(4, 6, 128, requires_grad=False)
    m.train()
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * 8 * 8
    # Target with some NaN cells (simulates short responses).
    target = torch.randn(4, D)
    target[:, D // 2:] = float("nan")

    loss, _ = m.recon_loss_attn(attn_pred["backward"], target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


def test_full_mse_loss_no_nan_gradients():
    m = _make_full_model(attn_direction="backward", attn_loss="mse")
    x = torch.randn(4, 6, 128, requires_grad=False)
    m.train()
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * _R_MAX * _R_MAX
    target = torch.randn(4, D)
    target[:, D // 2:] = float("nan")

    loss, _ = m.recon_loss_attn(attn_pred["backward"], target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


def test_coarse_kl_loss_no_nan_gradients():
    m = _make_coarse_model(attn_direction="backward", attn_loss="kl")
    x = torch.randn(4, 6, 128, requires_grad=False)
    m.train()
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * 8 * 8
    # Row-normalised target (each 8×8 = 64-cell row sums to 1).
    target_raw = torch.rand(4, D).abs() + 0.01
    # Normalise each 64-cell row.
    target_3d = target_raw.reshape(4, _NUM_LAYERS, 64)
    target_3d = target_3d / target_3d.sum(dim=-1, keepdim=True)
    target = target_3d.reshape(4, D)
    # NaN one full layer row per sample to test skip logic.
    target[:, : 64] = float("nan")

    loss, _ = m.recon_loss_attn(attn_pred["backward"], target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


def test_full_kl_loss_no_nan_gradients():
    m = _make_full_model(attn_direction="backward", attn_loss="kl")
    x = torch.randn(4, 6, 128, requires_grad=False)
    m.train()
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * _R_MAX * _R_MAX
    map_size = _R_MAX * _R_MAX
    target_raw = torch.rand(4, D).abs() + 0.01
    target_3d = target_raw.reshape(4, _NUM_LAYERS, map_size)
    target_3d = target_3d / target_3d.sum(dim=-1, keepdim=True)
    target = target_3d.reshape(4, D)
    target[:, :map_size] = float("nan")

    loss, _ = m.recon_loss_attn(attn_pred["backward"], target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


# ---------------------------------------------------------------------------
# Round-trip smoke: synthetic capture → forward+backward → finite grads
# ---------------------------------------------------------------------------

def test_coarse_round_trip_forward_backward():
    """End-to-end: model forward + loss backward produces finite grads."""
    torch.manual_seed(42)
    m = _make_coarse_model(attn_direction="both")
    m.train()

    B, L, D_in = 6, 8, 128
    x = torch.randn(B, L, D_in)
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * 8 * 8
    target = torch.randn(B, D)
    target[:, D // 4: D // 2] = float("nan")

    loss_fwd, _ = m.recon_loss_attn(attn_pred["forward"], target)
    loss_bwd, _ = m.recon_loss_attn(attn_pred["backward"], target)
    (loss_fwd + loss_bwd).backward()

    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"


def test_full_round_trip_forward_backward():
    """End-to-end: model forward + loss backward produces finite grads."""
    torch.manual_seed(42)
    m = _make_full_model(attn_direction="both")
    m.train()

    B, L, D_in = 4, 8, 128
    x = torch.randn(B, L, D_in)
    z, lp_pred, attn_pred = m.forward_with_recon(x)

    D = _NUM_LAYERS * _R_MAX * _R_MAX
    target = torch.randn(B, D)
    target[:, D // 3:] = float("nan")

    loss_fwd, _ = m.recon_loss_attn(attn_pred["forward"], target)
    loss_bwd, _ = m.recon_loss_attn(attn_pred["backward"], target)
    (loss_fwd + loss_bwd).backward()

    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"
