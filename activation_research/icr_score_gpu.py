"""
Batched GPU equivalent of activation_research.icr_score.compute_icr_score.

Implements the same z-score → softmax → JSD formula as icr_score.py, vectorized
across (B, L) using torch tensor ops. Falls back to CPU when CUDA is unavailable;
correctness must hold either way.

Memory formula (for capacity planning):
  Intermediate tensor peak ≈ B × L × r_max × k × hidden_dim × 4 bytes (fp32).
  At B=4, L=32, r_max=64, k≈6 (top_p=0.1×64), hidden_dim=4096:
    4 × 32 × 64 × 6 × 4096 × 4 = ~770 MB — fits on H100 80GB.
  Scale k or B up before hidden_dim dominates.
"""

from __future__ import annotations

import torch


def compute_icr_per_layer_batched_gpu(
    response_attn: torch.Tensor,   # (B, L, r_max, r_max) fp16 or fp32
    h_block_input: torch.Tensor,   # (B, L, r_max, hidden_dim) fp32
    delta_h: torch.Tensor,         # (B, L, r_max, hidden_dim) fp32
    response_lens: torch.Tensor,   # (B,) int64
    top_p: float = 0.1,
) -> torch.Tensor:                  # (B, L) float32
    """Batched GPU equivalent of icr_score.compute_icr_score looped over B×L.

    Implements the same formula as upstream's icr_score.py (z-score → softmax → JSD
    on top-p subsets), vectorized across (B, L). Falls back to CPU when CUDA is
    unavailable; correctness must hold either way.
    """
    B, L, r_max, _ = response_attn.shape
    device = h_block_input.device

    # Why: fp16 attention input matches the memmap storage format; all math is fp32.
    attn = response_attn.to(dtype=torch.float32, device=device)
    h_in = h_block_input.to(dtype=torch.float32, device=device)
    dh = delta_h.to(dtype=torch.float32, device=device)

    rlens = response_lens.to(dtype=torch.int64, device=device)   # (B,)

    # Why: padding-zero rows in the attention matrix must not be selected by top-k;
    # without this mask, a response_len=2 sample still contributes r_max-2 zero rows
    # to top-k, and those rows receive non-zero JSD from the zero-vector projections.
    pos = torch.arange(r_max, device=device)  # (r_max,)
    row_mask = pos.unsqueeze(0) < rlens.unsqueeze(1)  # (B, r_max)
    attn_masked = attn * row_mask[:, None, :, None] * row_mask[:, None, None, :]

    # Per-sample k = max(1, int(top_p * response_len)) — matches the numpy reference
    # exactly. k varies per sample. Use k_max (largest k in the batch) for a unified
    # top-k gather, then zero out positions beyond each sample's true k.
    k_per_sample = (top_p * rlens.float()).long().clamp(min=1)   # (B,)
    k_max = int(k_per_sample.max().item())

    # Top-k key positions per (B, L, query_position) — unified gather at k_max.
    # topk_indices: (B, L, r_max, k_max)
    _, topk_indices = torch.topk(attn_masked, k=k_max, dim=-1, sorted=False)

    a_topk = attn_masked.gather(-1, topk_indices)  # (B, L, r_max, k_max)

    # Gather h_block_input at top-k positions for each query.
    D = h_in.shape[-1]
    idx_exp = topk_indices.unsqueeze(-1).expand(B, L, r_max, k_max, D)
    # Why: insert the QUERY axis (dim 2) as the broadcast singleton so the original
    # r_max axis of h_in remains the key axis for gather. With .unsqueeze(3) (the
    # wrong axis), every k position collapses to the same h_in[q], yielding identical
    # projections for all k keys and a degenerate softmax(zscore(...)).
    h_topk = h_in.unsqueeze(2).expand(B, L, r_max, r_max, D).gather(3, idx_exp)  # (B, L, r_max, k_max, D)

    # Projection: w_i = (delta_h[q] · h_topk[q,j]) / (||h_topk[q,j]|| + 1e-8)
    dh_exp = dh.unsqueeze(3)                                  # (B, L, r_max, 1, D)
    numerator = (dh_exp * h_topk).sum(dim=-1)                 # (B, L, r_max, k_max)
    denom = h_topk.norm(dim=-1).clamp(min=1e-8)               # (B, L, r_max, k_max)
    w_topk = numerator / denom                                 # (B, L, r_max, k_max)

    # Zero out the tail positions beyond each sample's per-sample k so that the
    # JSD for samples with k < k_max is computed only over their true k positions.
    # Why: the zscore+softmax normalization in _jsd treats all k_max positions equally,
    # so truncation via masking is not equivalent to a smaller gather — we must zero-
    # score the tail entries rather than exclude them. The correct approach is to split
    # per distinct k value (rare: k_per_sample varies by ≤ r_max steps).
    #
    # Practical solution: because k differences are small (max spread = r_max steps),
    # run a per-unique-k loop. For the common case where all k_per_sample are equal
    # (any batch where floor(top_p * rlen) is constant across samples), this is one
    # iteration — same cost as the fully-vectorized path.
    icr = torch.zeros(B, L, device=device)

    unique_ks = k_per_sample.unique()
    for k_val in unique_ks:
        k = int(k_val.item())
        mask_b = (k_per_sample == k_val)   # (B,) bool

        if not mask_b.any():
            continue

        a_sub = a_topk[mask_b, :, :, :k]   # (B_sub, L, r_max, k)
        w_sub = w_topk[mask_b, :, :, :k]   # (B_sub, L, r_max, k)

        a_norm = _softmax_zscore(a_sub)     # (B_sub, L, r_max, k)
        w_norm = _softmax_zscore(w_sub)     # (B_sub, L, r_max, k)
        m = 0.5 * (a_norm + w_norm)
        jsd = 0.5 * (_kl(a_norm, m) + _kl(w_norm, m))   # (B_sub, L, r_max)

        # Mask positions past each sample's response_len; divide by response_len.
        rlens_sub = rlens[mask_b]                                    # (B_sub,)
        pos_mask = (pos.unsqueeze(0) < rlens_sub.unsqueeze(1)).float()  # (B_sub, r_max)
        jsd_masked = jsd * pos_mask[:, None, :]                          # (B_sub, L, r_max)
        rlens_f = rlens_sub.float().clamp(min=1).unsqueeze(1)            # (B_sub, 1)
        icr[mask_b] = jsd_masked.sum(dim=-1) / rlens_f                  # (B_sub, L)

    return icr.float()


def _zscore_last(x: torch.Tensor) -> torch.Tensor:
    """Z-score along last dim; clamp std to 1e-8 (matches numpy reference)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-8)
    return (x - mean) / std


def _softmax_zscore(x: torch.Tensor) -> torch.Tensor:
    """Z-score then softmax along last dim."""
    z = _zscore_last(x)
    # Subtract max for numerical stability (matches numpy _softmax).
    z = z - z.max(dim=-1, keepdim=True).values
    e = z.exp()
    return e / e.sum(dim=-1, keepdim=True)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence along last dim; returns tensor with last dim collapsed."""
    # Why 1e-12 inside log: matches upstream's _kl_divergence; prevents log(0) when
    # distributions have near-zero mass after softmax.
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1)
