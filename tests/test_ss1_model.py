"""Unit tests for the SS-1 ``SharedSpineDualHeadCompressor`` (issue #129).

Covers the two behaviours that distinguish SS-1 from the D2 shared-trunk model:
  * dual deep heads produce correctly-shaped embeddings, and
  * ``spine_supcon_grad_scale`` routes only a scaled fraction of the head loss
    gradient into the shared spine (0.0 = full stop-grad), while leaving the
    forward pass — and therefore the head embeddings — unchanged.

Run: ``pytest tests/test_ss1_model.py`` (needs torch).
"""
import torch

from activation_research.model import SharedSpineDualHeadCompressor

# Block OUTPUT dims must stay >= 64 (encoder nhead = min(8, dim//64)); the
# default schedule halves INPUT_DIM down to TRUNK, so keep both >= 64.
INPUT_DIM, TRUNK, HEAD, L, RSEQ, B = 256, 64, 32, 5, 4, 4


def _make(scale: float) -> SharedSpineDualHeadCompressor:
    torch.manual_seed(0)
    return SharedSpineDualHeadCompressor(
        input_dim=INPUT_DIM, trunk_dim=TRUNK, head_dim=HEAD,
        head_hidden_dim=HEAD, head_depth=2, spine_supcon_grad_scale=scale,
        recon_seq_len=RSEQ, recon_hidden_dim=8,
    ).eval()  # eval() disables dropout → deterministic forward for the grad test


def _spine_grad_norm(m: SharedSpineDualHeadCompressor) -> float:
    return float(sum(
        (p.grad.norm() ** 2) for p in m._inner.parameters() if p.grad is not None
    ) ** 0.5)


def _head_grad_norm(m: SharedSpineDualHeadCompressor) -> float:
    ps = list(m.head_A.parameters()) + list(m.head_B.parameters())
    return float(sum((p.grad.norm() ** 2) for p in ps if p.grad is not None) ** 0.5)


def test_shapes():
    m = _make(0.0)
    x = torch.randn(B, L, INPUT_DIM)
    assert m(x).shape == (B, TRUNK)
    zA, zB = m.head_embeddings(x)
    assert zA.shape == (B, HEAD) and zB.shape == (B, HEAD)
    z, zA2, zB2, lp = m.forward_with_heads(x)
    assert z.shape == (B, TRUNK)
    assert zA2.shape == (B, HEAD) and zB2.shape == (B, HEAD)
    assert lp.shape == (B, RSEQ)


def test_recon_loss_callable():
    m = _make(0.0)
    x = torch.randn(B, L, INPUT_DIM)
    _, _, _, lp = m.forward_with_heads(x)
    loss, diag = m.recon_loss(lp, torch.randn(B, L))  # (scalar Tensor, diag dict)
    assert torch.isfinite(loss) and isinstance(diag, dict)


def _head_loss_backward(scale: float):
    m = _make(scale)
    # _make sets manual_seed(0) and constructs deterministically (scale is a stored
    # float, not a param), so this x is identical across scale values.
    x = torch.randn(B, L, INPUT_DIM)
    m.zero_grad(set_to_none=True)
    _, zA, zB, _ = m.forward_with_heads(x)
    (zA.sum() + zB.sum()).backward()
    return m, zA.detach().clone(), _spine_grad_norm(m), _head_grad_norm(m)


def test_grad_scale_zero_is_stop_grad():
    """scale=0.0 → spine receives NO gradient from the head losses."""
    _, _, spine_g, head_g = _head_loss_backward(0.0)
    assert spine_g == 0.0, f"spine grad should be 0 at scale=0, got {spine_g}"
    assert head_g > 0.0, "heads should still receive gradient"


def test_grad_scale_one_flows():
    _, _, spine_g, _ = _head_loss_backward(1.0)
    assert spine_g > 0.0, "spine should receive gradient at scale=1.0"


def test_grad_scale_is_proportional_and_forward_unchanged():
    """Forward (head embeddings) identical across scales; spine grad ∝ scale."""
    # Build with identical seed so weights match; vary only the scale.
    x = torch.randn(B, L, INPUT_DIM)

    def run(scale):
        m = _make(scale)
        m.zero_grad(set_to_none=True)
        _, zA, zB, _ = m.forward_with_heads(x)
        out = zA.detach().clone()
        (zA.sum() + zB.sum()).backward()
        return out, _spine_grad_norm(m)

    zA_half, g_half = run(0.5)
    zA_one, g_one = run(1.0)
    # Forward is unchanged by the gradient scale (zc == z numerically).
    assert torch.allclose(zA_half, zA_one, atol=1e-6), "head forward must not depend on grad scale"
    # Spine grad at 0.5 should be ~half of at 1.0.
    assert abs(g_half - 0.5 * g_one) < 1e-4 * max(1.0, g_one), f"{g_half} vs 0.5*{g_one}"


if __name__ == "__main__":
    test_shapes(); test_recon_loss_callable()
    test_grad_scale_zero_is_stop_grad(); test_grad_scale_one_flows()
    test_grad_scale_is_proportional_and_forward_unchanged()
    print("all SS-1 model tests passed")
