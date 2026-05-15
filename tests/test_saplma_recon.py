"""Unit tests for ``SaplmaWithReconHead`` (issue #67).

Covers:
- ``forward`` returns ``(B, 1)`` sigmoid probabilities in ``[0, 1]``.
- ``forward_with_recon`` returns the expected 3-tuple with a finite
  reconstruction loss.
- ``recon_loss`` suppresses when the logprob variance falls below the
  threshold.
- Inference path is identical to a plain ``SimpleHaluClassifier`` when
  body and head weights are copied across.
"""

import torch

from activation_research.model import SaplmaWithReconHead, SimpleHaluClassifier


def test_forward_returns_sigmoid_probabilities():
    torch.manual_seed(0)
    model = SaplmaWithReconHead(input_dim=64, hidden_dims=(32, 16, 8))
    x = torch.randn(4, 7, 64)
    preds = model(x)
    assert preds.shape == (4, 1)
    assert torch.all(preds >= 0.0)
    assert torch.all(preds <= 1.0)


def test_forward_with_recon_shapes_and_recon_loss():
    torch.manual_seed(0)
    model = SaplmaWithReconHead(
        input_dim=64,
        hidden_dims=(32, 16, 8),
        recon_seq_len=12,
        recon_hidden_dim=16,
    )
    x = torch.randn(4, 7, 64)
    sigmoid_logit, z, logprob_pred = model.forward_with_recon(x)
    assert sigmoid_logit.shape == (4, 1)
    assert z.shape == (4, 8)
    assert logprob_pred.shape == (4, 12)

    logprob_target = torch.randn(4, 20)
    loss, diag = model.recon_loss(logprob_pred, logprob_target)
    assert torch.isfinite(loss)
    assert not diag["suppressed"]
    assert loss.ndim == 0


def test_recon_loss_suppressed_when_target_is_near_constant():
    model = SaplmaWithReconHead(input_dim=64, hidden_dims=(32, 16, 8), recon_seq_len=12)
    logprob_pred = torch.zeros(4, 12)
    logprob_target = torch.full((4, 20), -0.5)
    loss, diag = model.recon_loss(logprob_pred, logprob_target)
    assert diag["suppressed"]
    assert float(loss) == 0.0


def test_inference_path_matches_simple_halu_classifier():
    """Inference (``forward``) must be bit-identical to ``SimpleHaluClassifier``
    with the same body+head weights, so cached AUROC comparisons stay valid.
    """
    torch.manual_seed(0)
    model = SaplmaWithReconHead(input_dim=64, hidden_dims=(32, 16, 8), dropout=0.0)
    model.eval()

    baseline = SimpleHaluClassifier(input_dim=64, hidden_dims=[32, 16, 8], dropout=0.0)
    baseline.eval()

    # Copy weights: SaplmaWithReconHead.body is [Linear, ReLU, Dropout]x3 and
    # head is Linear; SimpleHaluClassifier.classifier is the same sequence
    # plus a final Linear all in one Sequential.
    body_layers = list(model.body.children())
    head_layer = model.head
    baseline_layers = list(baseline.classifier.children())

    # Indices 0, 3, 6 are the three hidden Linears (separated by ReLU+Dropout).
    for src, dst in zip(
        [body_layers[0], body_layers[3], body_layers[6], head_layer],
        [baseline_layers[0], baseline_layers[3], baseline_layers[6], baseline_layers[9]],
    ):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)

    x = torch.randn(8, 5, 64)
    with torch.no_grad():
        a = model(x)
        b = baseline(x)
    assert torch.allclose(a, b, atol=1e-6)
