"""Activation-space augmentations for contrastive training.

All augmentation functions receive ``views`` of shape ``(B, K, T, H)`` where:
  B = batch size
  K = num_views (always 2)
  T = sequence length (pad_length + 1 = 64)
  H = hidden dim

They also receive ``labels`` of shape ``(B,)`` (needed by mixup).

None of the functions modify the input tensor in-place.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def whole_token_dropout(
    views: Tensor,
    labels: Tensor,
    p: float = 0.15,
    asymmetric: bool = False,
) -> Tensor:
    """Zero out entire token positions with probability ``p``.

    A binary mask of shape ``(B, T)`` is sampled; zeroed positions have all H
    features set to zero.  When ``asymmetric=True`` only view index 0 is
    masked; view index 1 is returned unchanged.

    Parameters
    ----------
    views:
        Shape ``(B, K, T, H)``.
    labels:
        Shape ``(B,)`` — not used here but kept for a uniform signature.
    p:
        Dropout probability per token position.
    asymmetric:
        When True apply the mask only to ``views[:, 0]``.
    """
    B, K, T, H = views.shape
    # Sample mask: (B, T) -> broadcast to (B, 1, T, 1)
    keep = torch.bernoulli(torch.full((B, T), 1.0 - p, device=views.device, dtype=views.dtype))
    mask = keep.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)

    out = views.clone()
    if asymmetric:
        out[:, 0] = out[:, 0] * mask[:, 0]  # (B, T, H) * (B, T, 1)
    else:
        out = out * mask  # broadcasts over K and H
    return out


def channel_dropout(
    views: Tensor,
    labels: Tensor,
    p: float = 0.10,
    asymmetric: bool = False,
) -> Tensor:
    """Zero out feature channels (dim=3) with probability ``p``.

    A binary mask of shape ``(B, H)`` is sampled; zeroed channels are set to
    zero across all T positions.  When ``asymmetric=True`` only view 0 is
    masked.

    Parameters
    ----------
    views:
        Shape ``(B, K, T, H)``.
    labels:
        Shape ``(B,)`` — not used.
    p:
        Dropout probability per channel.
    asymmetric:
        When True apply the mask only to ``views[:, 0]``.
    """
    B, K, T, H = views.shape
    # Sample mask: (B, H) -> broadcast to (B, 1, 1, H)
    keep = torch.bernoulli(torch.full((B, H), 1.0 - p, device=views.device, dtype=views.dtype))
    mask = keep.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H)

    out = views.clone()
    if asymmetric:
        out[:, 0] = out[:, 0] * mask[:, 0]  # (B, T, H) * (B, 1, H)
    else:
        out = out * mask  # broadcasts over K and T
    return out


def mixup_intra_label(
    views: Tensor,
    labels: Tensor,
    alpha: float = 0.4,
    asymmetric: bool = False,
) -> Tensor:
    """Convex combination of same-label activation pairs.

    For each sample i, find another sample j in the batch with the same label.
    Mix: ``lam * views[i] + (1-lam) * views[j]`` where
    ``lam ~ Beta(alpha, alpha)``.

    Samples in singleton label groups (only one sample with that label) are
    mixed with themselves, leaving them unchanged.

    When ``asymmetric=True`` only view 0 is mixed; view 1 stays clean.

    Known limitation (documented in issue #81): under ``ignore_label=1``,
    hallu examples (label=1) never form positive pairs in the SupCon loss.
    Mixing two hallu activations still produces a sample with label=1 that has
    no cross-sample positive pairs.

    Parameters
    ----------
    views:
        Shape ``(B, K, T, H)``.
    labels:
        Shape ``(B,)``.
    alpha:
        Beta distribution concentration parameter.
    asymmetric:
        When True mix only view 0.
    """
    B, K, T, H = views.shape
    device = views.device
    dtype = views.dtype

    # Draw lam per sample: (B, 1, 1, 1)
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((B,)).to(device=device, dtype=dtype).view(B, 1, 1, 1)

    # Build partner indices for each sample (same label, random permutation)
    partner_idx = torch.arange(B, device=device)
    unique_labels = labels.unique()
    for lbl in unique_labels:
        idx = (labels == lbl).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            # singleton — mix with itself (no change)
            continue
        perm = idx[torch.randperm(idx.numel(), device=device)]
        partner_idx[idx] = perm

    out = views.clone()
    partners = views[partner_idx]  # (B, K, T, H)

    if asymmetric:
        # Mix only view 0; lam broadcasts over (T, H)
        out[:, 0] = lam[:, 0] * out[:, 0] + (1 - lam[:, 0]) * partners[:, 0]
    else:
        out = lam * out + (1 - lam) * partners
    return out


def contiguous_block_dropout(
    views: Tensor,
    labels: Tensor,
    p_min: float = 0.20,
    p_max: float = 0.40,
    max_block_size: int = 8,
    asymmetric: bool = False,
) -> Tensor:
    """Zero out contiguous blocks of token positions.

    For each sample, drops p_min–p_max of the T token positions as contiguous
    blocks of size ≤ max_block_size.  Blocks are placed greedily without
    overlap until the per-sample target drop count is reached.

    Parameters
    ----------
    views:
        Shape ``(B, K, T, H)``.
    labels:
        Shape ``(B,)`` — not used, kept for uniform signature.
    p_min, p_max:
        Range for per-sample drop fraction (uniform sample).
    max_block_size:
        Maximum number of contiguous tokens per block.
    asymmetric:
        When True apply masking only to view 0; view 1 stays clean.
    """
    B, K, T, H = views.shape
    device = views.device

    p_vals = torch.rand(B) * (p_max - p_min) + p_min  # (B,) on CPU
    n_drop = (p_vals * T).round().long().clamp(min=1)  # (B,) tokens to drop

    # Build binary keep-masks on CPU then move once.
    masks = torch.ones(B, T, dtype=torch.float32)
    for b in range(B):
        remaining = int(n_drop[b])
        attempts = 0
        while remaining > 0 and attempts < T * 4:
            bsz = min(max_block_size, remaining)
            bsz = int(torch.randint(1, bsz + 1, (1,)))
            max_start = T - bsz
            if max_start < 0:
                break
            start = int(torch.randint(0, max_start + 1, (1,)))
            # Only place if all positions are still unmasked (no overlap).
            if masks[b, start : start + bsz].all():
                masks[b, start : start + bsz] = 0.0
                remaining -= bsz
            attempts += 1

    # (B, T) → (B, 1, T, 1) for broadcasting over K and H.
    float_mask = masks.to(device=device).unsqueeze(1).unsqueeze(-1)

    out = views.clone()
    if asymmetric:
        out[:, 0] = out[:, 0] * float_mask[:, 0]
    else:
        out = out * float_mask
    return out


_AUG_REGISTRY = {
    "whole_token_dropout": whole_token_dropout,
    "channel_dropout": channel_dropout,
    "mixup_intra_label": mixup_intra_label,
    "contiguous_block_dropout": contiguous_block_dropout,
}


class AugmentationComposer:
    """Sequential composition of augmentation functions.

    Parameters
    ----------
    augmentations:
        List of dicts, each with keys ``"type"`` and ``"params"``.
        Supported types: ``whole_token_dropout``, ``channel_dropout``,
        ``mixup_intra_label``.
    asymmetric:
        When True the asymmetric flag is passed to every augmentation,
        overriding any per-augmentation setting.
    """

    def __init__(
        self,
        augmentations: list[dict] | None,
        asymmetric: bool = False,
    ) -> None:
        self.augmentations = augmentations or []
        self.asymmetric = asymmetric

    def __call__(self, views: Tensor, labels: Tensor) -> Tensor:
        if not self.augmentations:
            return views
        out = views
        for aug in self.augmentations:
            aug_type = aug["type"]
            if aug_type not in _AUG_REGISTRY:
                raise ValueError(
                    f"Unknown augmentation type '{aug_type}'. "
                    f"Supported: {list(_AUG_REGISTRY)}"
                )
            fn = _AUG_REGISTRY[aug_type]
            params = dict(aug.get("params", {}))
            params["asymmetric"] = self.asymmetric
            out = fn(out, labels, **params)
        return out
