"""
attention_recompute.py

Recomputes per-block head-averaged attention weights from cached hidden states,
yielding the response-to-response attention sub-block required by ICR Probe.

Cross-region attention (response-to-prompt and prompt-to-response) is
intentionally discarded. Per icr_score.py:104-127, the ICR Score zeros
prompt-side attention scores before top-k, so only the response-to-response
sub-block is needed for the probe path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _build_causal_mask(T: int, device: str) -> torch.Tensor:
    """Return a (1, 1, T, T) additive float causal attention mask.

    Values are 0.0 where a token may attend and -inf where it is masked.
    Upper-triangular positions (future tokens) are masked out.
    """
    mask = torch.zeros(1, 1, T, T, device=device)
    upper = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(upper, float("-inf"))
    return mask


def _head_average_resp_to_resp(
    attn: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """Slice the response-to-response quadrant and average across heads.

    Args:
        attn: (n_heads, T, T) full attention after softmax.
        prompt_len: Number of prompt tokens.
        response_len: Number of response tokens.

    Returns:
        (response_len, response_len) float32 — rows are query response tokens,
        columns are key response tokens.
    """
    start = prompt_len
    end = prompt_len + response_len
    resp_block = attn[:, start:end, start:end]  # (n_heads, R, R)
    return resp_block.float().mean(dim=0)  # (R, R)


def _call_self_attn(
    block: nn.Module,
    normed: torch.Tensor,
    causal_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Call block.self_attn with output_attentions=True, handling API variants.

    Tries the standard position_ids API first; falls back to the newer
    position_embeddings API (transformers >= ~4.46) on TypeError.

    Returns:
        (n_heads, T, T) float32 attention weight tensor.

    Raises:
        RuntimeError: If the module returns None for attention weights, which
            happens when the model was loaded without attn_implementation='eager'.
    """
    def _extract(out: Tuple) -> torch.Tensor:
        weights = out[1]
        if weights is None:
            raise RuntimeError(
                "block.self_attn returned None for attention weights. "
                "Load model with attn_implementation='eager' to enable weight output."
            )
        return weights.squeeze(0).float()  # (n_heads, T, T)

    # Standard API: self_attn computes RoPE internally from position_ids.
    try:
        return _extract(
            block.self_attn(
                normed,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=True,
            )
        )
    except TypeError:
        pass

    # Newer API (transformers >= ~4.46): position_embeddings pre-computed outside.
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if hasattr(block, "rotary_emb"):
        cos, sin = block.rotary_emb(normed, position_ids)
        position_embeddings = (cos, sin)
    elif hasattr(block.self_attn, "rotary_emb"):
        cos, sin = block.self_attn.rotary_emb(normed, position_ids)
        position_embeddings = (cos, sin)

    return _extract(
        block.self_attn(
            normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            output_attentions=True,
        )
    )


def recompute_block_attention(
    h_prev: torch.Tensor,
    block: nn.Module,
    prompt_len: int,
    response_len: int,
    position_ids: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Recompute head-averaged response-to-response attention for one transformer block.

    Feeds the cached full-sequence hidden state through the block's attention
    sublayer (with RMSNorm applied internally) and returns the response-side
    attention sub-block averaged across all heads.

    The block must be loaded with ``attn_implementation='eager'`` so that
    ``output_attentions=True`` returns the attention weight tensor. The
    ``LlamaSdpaAttention`` / ``Qwen3SdpaAttention`` variants do NOT return
    weights and will raise RuntimeError.

    Args:
        h_prev: (T, H) float32 full-sequence hidden state entering this block,
                where T = prompt_len + response_len.  Stored zarr activations are
                float16; cast to float32 before passing.  No RMSNorm is applied
                here — h_prev is the pre-norm block input and the block's internal
                forward handles normalization.
        block: A single HF transformer block (e.g. ``model.model.layers[ℓ]``).
               The caller is responsible for loading the model and passing the block.
        prompt_len: Number of prompt tokens.  Prompt positions are 0..prompt_len-1.
        response_len: Number of response tokens.  Response positions are
                      prompt_len..T-1.
        position_ids: (T,) int64 or None.  If None, inferred as ``arange(T)``,
                      matching original inference positions.
        device: Device string for computation (default "cpu").

    Returns:
        (response_len, response_len) float32 tensor of head-averaged
        response-to-response attention probabilities.  Row i is query response
        token i; column j is key response token j.  Upper triangle is 0 due to
        causal masking.

    Raises:
        ValueError: If h_prev token count != prompt_len + response_len, or if
                    response_len < 1, or prompt_len < 0.
        RuntimeError: If block.self_attn returns None attention weights (use
                      attn_implementation='eager').
    """
    T = prompt_len + response_len
    if h_prev.shape[0] != T:
        raise ValueError(
            f"h_prev has {h_prev.shape[0]} tokens but "
            f"prompt_len={prompt_len} + response_len={response_len} = {T}"
        )
    if response_len < 1:
        raise ValueError(f"response_len must be >= 1, got {response_len}")
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be >= 0, got {prompt_len}")

    h_prev = h_prev.to(device=device, dtype=torch.float32)

    if position_ids is None:
        position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
    else:
        position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)  # (1, T)

    causal_mask = _build_causal_mask(T, device)
    block = block.to(device).eval()

    with torch.no_grad():
        hidden_states = h_prev.unsqueeze(0)  # (1, T, H)

        # Apply pre-attention layer norm; handles Llama (input_layernorm) and Qwen3.
        if hasattr(block, "input_layernorm"):
            normed = block.input_layernorm(hidden_states)
        elif hasattr(block, "ln_1"):
            normed = block.ln_1(hidden_states)
        else:
            normed = hidden_states

        attn_weights = _call_self_attn(block, normed, causal_mask, position_ids)

    return _head_average_resp_to_resp(attn_weights, prompt_len, response_len)
