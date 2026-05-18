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


def test_coarse_target_not_implemented():
    # 'coarse' is permanently out of scope (issue #82).
    with pytest.raises(NotImplementedError):
        LogprobAttnReconProgressiveCompressor(
            input_dim=128, final_dim=64, attn_target="coarse",
        )


def test_full_target_succeeds():
    # 'full' is implemented in this PR — must not raise.
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_target="full", attn_r_max=6,
        attn_direction="both",
    )
    assert m.attn_target == "full"
    assert m._attn_out_dim == 6 * 7  # r_max * (r_max + 1)


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
# Full attention target (issue #82)
# ---------------------------------------------------------------------------

def _make_full_model(r_max=4, direction="both"):
    return LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64,
        attn_target="full",
        attn_r_max=r_max,
        attn_direction=direction,
        attn_var_threshold=1e-12,
    )


def _make_full_target(batch=2, r_max=4, r_eff=3):
    """Build a synthetic (batch, r_max, r_max+1) target with prompt-sink augmentation."""
    import math
    target = torch.full((batch, r_max, r_max + 1), float("nan"))
    for b in range(batch):
        for q in range(r_eff):
            # Uniform distribution over r_eff response keys.
            val = 1.0 / r_eff
            target[b, q, :r_eff] = val
            # Prompt-sink: leftover mass (here zero — rows already sum to 1).
            # Add a small sink mass so the test is more realistic.
            resp_sum = r_eff * val
            target[b, q, :r_eff] = target[b, q, :r_eff] * 0.8
            target[b, q, r_max] = 1.0 - float(target[b, q, :r_eff].sum())
    return target.reshape(batch, r_max * (r_max + 1))


def test_full_target_constructor_succeeds():
    m = _make_full_model(r_max=6)
    assert m.attn_target == "full"
    assert m._attn_out_dim == 6 * 7


def test_full_decoder_output_shape_both_directions():
    r_max = 4
    m = _make_full_model(r_max=r_max, direction="both")
    x = torch.randn(3, 6, 128)
    z, lp_pred, attn_pred = m.forward_with_recon(x)
    assert z.shape == (3, 64)
    assert set(attn_pred.keys()) == {"forward", "backward"}
    assert attn_pred["forward"].shape == (3, r_max * (r_max + 1))
    assert attn_pred["backward"].shape == (3, r_max * (r_max + 1))


def test_full_decoder_output_shape_forward_only():
    r_max = 4
    m = _make_full_model(r_max=r_max, direction="forward")
    x = torch.randn(2, 5, 128)
    _, _, attn_pred = m.forward_with_recon(x)
    assert set(attn_pred.keys()) == {"forward"}
    assert attn_pred["forward"].shape == (2, r_max * (r_max + 1))


def test_full_kl_loss_finite_no_nan_grads():
    """KL loss must be finite and produce non-NaN gradients."""
    r_max, r_eff = 4, 3
    m = _make_full_model(r_max=r_max)
    x = torch.randn(2, 6, 128, requires_grad=False)
    _, _, attn_pred = m.forward_with_recon(x)

    target = _make_full_target(batch=2, r_max=r_max, r_eff=r_eff)

    loss, diag = m.recon_loss_attn(attn_pred["forward"], target)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert not diag["suppressed"]

    loss.backward()
    for name, p in m.attn_decoders["fwd"].named_parameters():
        assert p.grad is not None, f"param {name} has no grad"
        assert torch.isfinite(p.grad).all(), f"param {name} has NaN grad"


def test_full_kl_loss_out_of_range_layer_zero_loss():
    """Full-NaN target (out-of-range layer) must contribute 0 loss, no grad."""
    r_max = 4
    m = _make_full_model(r_max=r_max)
    x = torch.randn(2, 6, 128)
    _, _, attn_pred = m.forward_with_recon(x)

    # Full-NaN target simulates an out-of-range layer.
    target = torch.full((2, r_max * (r_max + 1)), float("nan"))
    loss, diag = m.recon_loss_attn(attn_pred["forward"], target)
    assert float(loss) == 0.0
    assert diag["suppressed"]


def test_full_kl_loss_backward_direction():
    """KL loss for the backward decoder must also run cleanly."""
    r_max, r_eff = 4, 3
    m = _make_full_model(r_max=r_max)
    x = torch.randn(2, 6, 128)
    _, _, attn_pred = m.forward_with_recon(x)

    target = _make_full_target(batch=2, r_max=r_max, r_eff=r_eff)
    loss, diag = m.recon_loss_attn(attn_pred["backward"], target)
    assert torch.isfinite(loss)
    assert not diag["suppressed"]


def test_stats_target_loss_still_mse():
    """Stats path must still use MSE after this PR (regression check)."""
    m = LogprobAttnReconProgressiveCompressor(
        input_dim=128, final_dim=64, attn_direction="backward",
        attn_var_threshold=1e-8, attn_target="stats",
    )
    pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor(
        [[2.0, 3.0, 4.0],
         [float("nan"), float("nan"), float("nan")]],
    )
    loss, diag = m.recon_loss_attn(pred, target)
    assert abs(float(loss) - 1.0) < 1e-5
    assert diag["valid_count"] == 3


def test_full_round_trip_forward_backward_step(tmp_path):
    """Smoke: tiny synthetic capture → dataset → full model → fwd + bwd step."""
    import sys, os
    sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
    from tests.test_memmap_contrastive_dataset import _make_full_capture_dir, _SMALL_CFG
    from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset

    r_max = _SMALL_CFG["r_max"]          # 6
    capture = _make_full_capture_dir(tmp_path, n_samples=5)

    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3],
        include_response_attention=True,
        attention_summary="full",
        attention_target_layer_offset_forward=1,
        attention_target_layer_offset_backward=1,
    )

    m = LogprobAttnReconProgressiveCompressor(
        input_dim=_SMALL_CFG["hidden_dim"],
        final_dim=64,
        attn_target="full",
        attn_r_max=r_max,
        attn_direction="both",
        attn_var_threshold=1e-12,
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    # Build a small batch manually (no DataLoader needed).
    samples = [ds[i] for i in range(3)]
    # Stack views_activations: (B, K, max_resp, hidden_dim)
    acts = torch.stack([s["views_activations"] for s in samples])   # (3, 2, mr, hd)
    attn_fwd = torch.stack([s["attention_forward"] for s in samples])   # (3, 2, r_max, r_max+1)
    attn_bwd = torch.stack([s["attention_backward"] for s in samples])

    B, K = acts.shape[:2]
    # Flatten (B, K) → (B*K) for encoder.
    acts_flat = acts.reshape(B * K, *acts.shape[2:])    # (6, mr, hd)
    _, _, attn_pred = m.forward_with_recon(acts_flat)

    # Flatten targets to (N, r_max * (r_max+1)).
    attn_fwd_flat = attn_fwd.reshape(B * K, r_max * (r_max + 1))
    attn_bwd_flat = attn_bwd.reshape(B * K, r_max * (r_max + 1))

    loss_fwd, _ = m.recon_loss_attn(attn_pred["forward"], attn_fwd_flat)
    loss_bwd, _ = m.recon_loss_attn(attn_pred["backward"], attn_bwd_flat)
    loss = loss_fwd + loss_bwd

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Every decoder parameter must have a finite gradient.
    for direction_key in ("fwd", "bwd"):
        for name, p in m.attn_decoders[direction_key].named_parameters():
            assert p.grad is not None, f"attn_decoders[{direction_key}].{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"NaN grad in attn_decoders[{direction_key}].{name}"
