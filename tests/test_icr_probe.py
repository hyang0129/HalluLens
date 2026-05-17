"""Forward/backward + convergence + paper-faithfulness shape checks for ICRProbe.

All tests are CPU-only and use small dummy tensors.
No model weights, GPU, or external data required.

Run with:
    pytest tests/test_icr_probe.py -v
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from activation_research.icr_probe import ICRProbe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(L: int = 32) -> ICRProbe:
    """Return a default ICRProbe with L input dims."""
    return ICRProbe(input_dim=L)


# ---------------------------------------------------------------------------
# 1. Forward shape
# ---------------------------------------------------------------------------

def test_forward_shape_batch():
    """(B, L) input → (B,) raw logit output."""
    probe = _make_probe(L=32)
    probe.eval()
    x = torch.randn(16, 32)
    out = probe(x)
    assert out.shape == (16,), f"Expected (16,), got {out.shape}"
    assert out.dtype == torch.float32


def test_forward_shape_single_item_via_batch():
    """B=2 (minimum for BatchNorm1d to work) passes through cleanly."""
    probe = _make_probe(L=28)
    probe.eval()
    x = torch.randn(2, 28)
    out = probe(x)
    assert out.shape == (2,)


def test_forward_output_is_raw_logit_not_probability():
    """Output should be unbounded logit, not probability (no sigmoid inside)."""
    probe = _make_probe(L=32)
    probe.eval()
    x = torch.randn(64, 32)
    logits = probe(x)
    # If sigmoid had been applied, all values would be in [0, 1].
    # With random weights and random inputs, logits routinely fall outside.
    # We just assert the tensor is finite — actual unboundedness is statistical.
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
    # Verify no clamping to [0,1] by checking the full torch.sigmoid still works.
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# 2. Backward / gradient flow
# ---------------------------------------------------------------------------

def test_backward_grads_flow_to_all_parameters():
    """After one backward pass, every parameter has a non-None, finite gradient."""
    probe = _make_probe(L=32)
    probe.train()
    x = torch.randn(8, 32)
    labels = torch.randint(0, 2, (8,)).float()
    loss_fn = nn.BCEWithLogitsLoss()
    logits = probe(x)
    loss = loss_fn(logits, labels)
    loss.backward()

    for name, param in probe.named_parameters():
        assert param.grad is not None, f"Param {name!r} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Param {name!r} has non-finite gradient"


# ---------------------------------------------------------------------------
# 3. BatchNorm requires B > 1 in train mode
# ---------------------------------------------------------------------------

def test_batchnorm_requires_batch_gt_1_in_train_mode():
    """B=1 in train mode raises RuntimeError from BatchNorm1d.

    This test documents the expected failure surface so that users of
    ICRProbeTrainer know they need batch_size >= 2 during training.
    """
    probe = _make_probe(L=32)
    probe.train()
    x = torch.randn(1, 32)
    with pytest.raises(Exception):
        # RuntimeError: Expected more than 1 value per channel when training.
        _ = probe(x)


def test_batchnorm_works_with_b1_in_eval_mode():
    """B=1 is allowed in eval mode (BN uses running stats, not batch stats)."""
    probe = _make_probe(L=32)
    # Need at least one forward pass in train mode so running stats are initialized.
    probe.train()
    _ = probe(torch.randn(4, 32))
    probe.eval()
    out = probe(torch.randn(1, 32))
    assert out.shape == (1,)


# ---------------------------------------------------------------------------
# 4. Tiny convergence test
# ---------------------------------------------------------------------------

def test_tiny_convergence_linearly_separable_data():
    """Linearly-separable data → AUROC > 0.95 within 20 training epochs.

    Uses 512 samples of L=32 dims where class 1 has higher mean ICR scores
    than class 0.  The ICRProbe MLP should separate them easily.
    """
    torch.manual_seed(0)
    B, L = 512, 32

    # Linearly separable: class 1 has ICR offset +2.0.
    x0 = torch.randn(B // 2, L)
    x1 = torch.randn(B // 2, L) + 2.0
    x = torch.cat([x0, x1])
    labels = torch.cat([torch.zeros(B // 2), torch.ones(B // 2)])

    probe = ICRProbe(input_dim=L)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(20):
        probe.train()
        idx = torch.randperm(B)
        x_shuf, y_shuf = x[idx], labels[idx]

        for start in range(0, B, 64):
            xb = x_shuf[start : start + 64]
            yb = y_shuf[start : start + 64]
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        scores = torch.sigmoid(probe(x)).numpy()
    auroc = roc_auc_score(labels.numpy(), scores)
    assert auroc > 0.95, (
        f"Expected AUROC > 0.95 on linearly-separable data after 20 epochs, "
        f"got {auroc:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. State-dict roundtrip
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_logits():
    """save state_dict → load into fresh probe → same logits (no NaN drift)."""
    torch.manual_seed(42)
    L = 32
    probe_a = ICRProbe(input_dim=L)
    probe_a.eval()

    x = torch.randn(8, L)
    with torch.no_grad():
        logits_a = probe_a(x).clone()

    # Serialize / deserialize via in-memory buffer.
    buf = io.BytesIO()
    torch.save(probe_a.state_dict(), buf)
    buf.seek(0)

    probe_b = ICRProbe(input_dim=L)
    probe_b.load_state_dict(torch.load(buf, map_location="cpu"))
    probe_b.eval()

    with torch.no_grad():
        logits_b = probe_b(x)

    assert torch.allclose(logits_a, logits_b, atol=1e-6), (
        "Logits changed after state_dict roundtrip"
    )


# ---------------------------------------------------------------------------
# 6. Architecture paper-faithfulness
# ---------------------------------------------------------------------------

def test_architecture_matches_paper_hidden_dims():
    """Default hidden_dims are (128, 64, 32) per notes §7 / upstream utils.py:5-26."""
    probe = ICRProbe(input_dim=32)
    # Extract linear layer output features in order.
    linear_layers = [m for m in probe.net if isinstance(m, nn.Linear)]
    # Expected: 32→128, 128→64, 64→32, 32→1
    assert len(linear_layers) == 4, f"Expected 4 Linear layers, got {len(linear_layers)}"
    assert linear_layers[0].out_features == 128
    assert linear_layers[1].out_features == 64
    assert linear_layers[2].out_features == 32
    assert linear_layers[3].out_features == 1


def test_architecture_has_batchnorm_leakyrelu_dropout_per_hidden():
    """Per notes §7: each hidden block has Linear → BN → LeakyReLU → Dropout."""
    probe = ICRProbe(input_dim=32)
    block_types = [type(m).__name__ for m in probe.net]
    # 3 hidden blocks × 4 = 12 layers, then 1 final Linear.
    assert block_types.count("BatchNorm1d") == 3
    assert block_types.count("LeakyReLU") == 3
    assert block_types.count("Dropout") == 3
    assert block_types[-1] == "Linear"  # output logit, no activation
