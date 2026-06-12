"""Unit tests for LogprobReconMLPIntake intake variants (#132).

Covers the base MLP, SwiGLU, ResNet-MLP intakes (+ normalize_input) — shapes,
recon head, and that gradient reaches the intake. No attention, so any dims work.
"""
import torch

from activation_research.model import LogprobReconMLPIntake

IN, FINAL, HID, L, RSEQ, B = 256, 64, 128, 5, 4, 4


def _make(intake_type, normalize_input=False):
    torch.manual_seed(0)
    return LogprobReconMLPIntake(
        input_dim=IN, final_dim=FINAL, intake_type=intake_type, hidden_dim=HID,
        depth=2, normalize_input=normalize_input, recon_seq_len=RSEQ, recon_hidden_dim=16,
    )


def _gnorm(params):
    return float(sum((p.grad.norm() ** 2) for p in params if p.grad is not None) ** 0.5)


def test_all_intakes_shapes_and_grad():
    for itype in ("mlp", "swiglu", "resnet"):
        m = _make(itype)
        x = torch.randn(B, L, IN)
        assert m(x).shape == (B, FINAL), itype
        z, lp = m.forward_with_recon(x)
        assert z.shape == (B, FINAL) and lp.shape == (B, RSEQ), itype
        loss, diag = m.recon_loss(lp, torch.randn(B, L))
        assert torch.isfinite(loss) and isinstance(diag, dict), itype
        # gradient reaches the intake block
        m.zero_grad(set_to_none=True)
        (m(x).sum()).backward()
        assert _gnorm(m.encoder.intake.parameters()) > 0, f"{itype}: intake got no gradient"


def test_normalize_input_adds_norm():
    m = _make("mlp", normalize_input=True)
    assert hasattr(m.encoder, "input_norm") and isinstance(m.encoder.input_norm, torch.nn.LayerNorm)
    assert m(torch.randn(B, L, IN)).shape == (B, FINAL)


def test_swiglu_is_gated():
    import activation_research.model as M
    m = _make("swiglu")
    assert isinstance(m.encoder.intake, M._SwiGLUIntake)


if __name__ == "__main__":
    test_all_intakes_shapes_and_grad()
    test_normalize_input_adds_norm()
    test_swiglu_is_gated()
    print("all intake model tests passed")
