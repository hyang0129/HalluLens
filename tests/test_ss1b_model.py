"""Unit tests for SS-1b ``SharedStemDualBranchCompressor`` (issue #129).

Verifies the early-split design: a shared input stem trained by BOTH conventions
(full, cooperative gradient) feeding two INDEPENDENT branch sub-encoders.

Run: ``pytest tests/test_ss1b_model.py`` (needs torch).
"""
import torch

from activation_research.model import SharedStemDualBranchCompressor

# Block output dims must be >= 64 (encoder nhead = min(8, dim//64)).
INPUT_DIM, STEM, FINAL, L, RSEQ, B = 256, 128, 64, 5, 4, 4


def _make():
    torch.manual_seed(0)
    return SharedStemDualBranchCompressor(
        input_dim=INPUT_DIM, stem_dim=STEM, branch_block_dims=[128, 64],
        final_dim=FINAL, recon_seq_len=RSEQ, recon_hidden_dim=16,
    ).eval()


def _gnorm(params):
    return float(sum((p.grad.norm() ** 2) for p in params if p.grad is not None) ** 0.5)


def test_shapes():
    m = _make()
    x = torch.randn(B, L, INPUT_DIM)
    assert m(x).shape == (B, FINAL)
    zA, zB = m.head_embeddings(x)
    assert zA.shape == (B, FINAL) and zB.shape == (B, FINAL)
    assert m.embed_head(x, "A").shape == (B, FINAL)
    stem_pooled, zA2, zB2, lp = m.forward_with_heads(x)
    assert stem_pooled.shape == (B, STEM)
    assert zA2.shape == (B, FINAL) and zB2.shape == (B, FINAL)
    assert lp.shape == (B, RSEQ)


def test_recon_loss_callable():
    m = _make()
    _, _, _, lp = m.forward_with_heads(torch.randn(B, L, INPUT_DIM))
    loss, diag = m.recon_loss(lp, torch.randn(B, L))
    assert torch.isfinite(loss) and isinstance(diag, dict)


def test_shared_stem_trained_by_both_branches():
    """Stem gets gradient from both heads; each branch from its own head."""
    m = _make()
    m.zero_grad(set_to_none=True)
    _, zA, zB, _ = m.forward_with_heads(torch.randn(B, L, INPUT_DIM))
    (zA.sum() + zB.sum()).backward()
    assert _gnorm(m.stem.parameters()) > 0, "shared stem must receive gradient"
    assert _gnorm(m.branch_A.parameters()) > 0
    assert _gnorm(m.branch_B.parameters()) > 0


def test_branches_are_independent():
    """A loss on head A alone leaves branch B untouched but still trains the stem."""
    m = _make()
    m.zero_grad(set_to_none=True)
    _, zA, zB, _ = m.forward_with_heads(torch.randn(B, L, INPUT_DIM))
    zA.sum().backward()
    assert _gnorm(m.branch_A.parameters()) > 0, "branch A should get gradient"
    assert _gnorm(m.branch_B.parameters()) == 0.0, "branch B must be independent of head A"
    assert _gnorm(m.stem.parameters()) > 0, "shared stem still gets gradient from head A"


def test_ss1c_linear_stem():
    """SS-1c: a lean LINEAR shared stem (no attention) + deep independent branches."""
    import torch.nn as nn
    torch.manual_seed(0)
    m = SharedStemDualBranchCompressor(
        input_dim=INPUT_DIM, stem_dim=STEM, stem_type="linear",
        branch_block_dims=[STEM, STEM, FINAL], final_dim=FINAL,
        recon_seq_len=RSEQ, recon_hidden_dim=16,
    ).eval()
    assert isinstance(m.stem, nn.Linear), "linear stem must be nn.Linear (no attention)"
    x = torch.randn(B, L, INPUT_DIM)
    stem_pooled, zA, zB, lp = m.forward_with_heads(x)
    assert stem_pooled.shape == (B, STEM)
    assert zA.shape == (B, FINAL) and zB.shape == (B, FINAL)
    # shared (lean) stem trained by both; branches independent
    m.zero_grad(set_to_none=True)
    zA.sum().backward()
    assert _gnorm(m.stem.parameters()) > 0
    assert _gnorm(m.branch_A.parameters()) > 0
    assert _gnorm(m.branch_B.parameters()) == 0.0


if __name__ == "__main__":
    test_shapes(); test_recon_loss_callable()
    test_shared_stem_trained_by_both_branches(); test_branches_are_independent()
    test_ss1c_linear_stem()
    print("all SS-1b/SS-1c model tests passed")
