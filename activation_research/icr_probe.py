"""ICR Probe MLP per notes §7 / upstream utils.py:5-26.

Zhang et al. ACL 2025. ICR Probe: Tracking Hidden State Dynamics for
Reliable Hallucination Detection in LLMs. arXiv:2507.16488.

Architecture: L → 128 → 64 → 32 → 1
Per hidden layer: Linear → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.3)
Output: raw logit (no sigmoid). Train with BCEWithLogitsLoss.

Why no built-in sigmoid: BCEWithLogitsLoss is numerically more stable than
BCELoss + Sigmoid (equivalent math, better behavior near saturation).
Eval threshold is logit > 0  ≡  p > 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ICRProbe(nn.Module):
    """ICR Probe per Zhang et al. ACL 2025 (upstream utils.py:5-26).

    Parameters
    ----------
    input_dim : int
        Number of transformer layers L.  Each sample is a 1-D vector of L
        ICR scores (one per layer, averaged over response tokens).
    hidden_dims : tuple[int, ...]
        Hidden layer widths.  Default (128, 64, 32) matches upstream.
    dropout : float
        Dropout rate applied after each BatchNorm+LeakyReLU block.
        Per notes §7: 0.3.
    leaky_slope : float
        Negative slope for LeakyReLU.  Per notes §7: 0.01.

    Notes
    -----
    Per notes §7 / upstream utils.py:5-26:
        BN + LeakyReLU(0.01) + Dropout(0.3) after each Linear.
    The output is a raw logit; callers apply sigmoid for probabilities or
    use BCEWithLogitsLoss directly during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                # Per notes §7 / upstream utils.py:5-26: BN + LeakyReLU(0.01) + Dropout(0.3).
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=leaky_slope),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, L) ICR scores to (B,) raw logits.

        Parameters
        ----------
        x : Tensor, shape (B, L)
            Per-layer ICR scores, one row per sample.

        Returns
        -------
        Tensor, shape (B,)
            Raw logits (no sigmoid applied).
        """
        return self.net(x).squeeze(-1)
