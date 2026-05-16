"""
generate_capture.py — stitching + logprob primitives for the single-process
inference-capture pipeline (Issue #72).

This module is the load-bearing contract for the rewrite: it turns the
heterogeneous outputs of `model.generate(..., output_attentions=True,
output_hidden_states=True, return_dict_in_generate=True)` into the
fixed-shape tensors that the memmap writer persists.

Contracts (must hold to match upstream XavierZhang2002/ICR_Probe byte-for-byte
on the response-to-response sub-block; see specs/issue_72_inference_capture_rewrite.md
"Upstream stitching contract" and notes/icr_probe_paper_notes.md §1.5):

    out.attentions: tuple of length 1 + response_len.
      [0]             — prefill: per-layer (batch, num_heads, prompt_len, prompt_len)
      [t>=1]          — decode step that emitted response token t-1:
                        per-layer (batch, num_heads, 1, prompt_len + t)

    out.hidden_states: tuple of length 1 + response_len.
      [0]             — prefill: per-layer (batch, prompt_len, hidden_dim)
                        for layers 0..L (where 0 is the embedding output)
      [t>=1]          — decode step: per-layer (batch, 1, hidden_dim) for
                        layers 0..L for the newly-emitted token

    out.scores: tuple of length response_len; each is (batch, vocab_size) logits.

Functions exposed:

    stitch_response_to_response(attentions, prompt_len, r_max,
                                response_len=None) -> np.ndarray
        Returns head-averaged response-to-response attention of shape
        (num_layers, r_max, r_max) float16, zero-padded past response_len.
        Skips attentions[0] entirely. For each decode step t>=1, slices the
        key dimension at [prompt_len : prompt_len + r_max] and head-averages
        per layer. Row t-1 of the output is for response token t-1.

    stitch_response_hidden_states(hidden_states, prompt_len, max_response_len)
        -> np.ndarray
        Returns (num_layers + 1, max_response_len, hidden_dim) float16,
        zero-padded past response_len. Layer 0 is the embedding output.

    stitch_prompt_hidden_states(hidden_states, prompt_len, max_prompt_len)
        -> np.ndarray
        Returns (num_layers + 1, max_prompt_len, hidden_dim) float16,
        zero-padded past prompt_len. Pulled from attentions[0] / prefill.

    extract_logprobs(scores, generated_token_ids, top_k=20)
        -> (token_logprobs, topk_ids, topk_logprobs)
        token_logprobs: (response_len,) float32 — log P(generated_id | context)
        topk_ids: (response_len, top_k) int32
        topk_logprobs: (response_len, top_k) float32
        All computed on CPU after a single GPU->CPU move per decode step.

Implementation requirements:
- All returned arrays must be numpy (not torch). Convert via .detach().cpu().numpy()
  and cast to the target dtype before returning.
- For samples with response_len < r_max, pad with zeros; never crash on short
  responses.
- For samples with response_len > r_max, truncate silently — the spec accepts
  this (Qwen3 HotpotQA truncation policy, PR #71 known follow-up #2).
- batch dimension is always 1 in v1 capture; assert or squeeze accordingly.

Unit-test target: tests/test_generate_capture.py
- Hand-stitched reference on a tiny model (e.g. sshleifer/tiny-gpt2) — assert
  shape, dtype, and numerical equivalence vs. a naive scalar loop over the
  upstream tuple format.
- Cover edge cases: response_len == 0, response_len == 1, response_len > r_max,
  response_len < r_max (padding correctness).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def stitch_response_to_response(
    attentions: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    r_max: int,
    response_len: int | None = None,
) -> np.ndarray:
    """Head-averaged response-to-response attention from heterogeneous generate() output.

    Args:
        attentions: out.attentions from model.generate(...).
        prompt_len: number of prompt tokens (used to offset the key slice).
        r_max: max stored response length (output dim).
        response_len: actual response length; defaults to len(attentions) - 1.

    Returns:
        (num_layers, r_max, r_max) float16 numpy array, zero-padded past
        min(response_len, r_max).
    """
    if response_len is None:
        # Why: attentions[0] is the prefill step, not a response token.
        response_len = len(attentions) - 1

    # Why: attentions[0] is prompt-to-prompt (prefill); ICR zeroes cross-region
    # attention before top-k anyway (notes §9), so we never need it.
    decode_steps = attentions[1:]

    # Infer num_layers from the first available decode step, or from prefill if
    # response_len == 0.
    if len(decode_steps) > 0:
        num_layers = len(decode_steps[0])
    else:
        num_layers = len(attentions[0])

    out = np.zeros((num_layers, r_max, r_max), dtype=np.float16)

    n_steps = min(response_len, r_max, len(decode_steps))
    for t in range(n_steps):
        # decode_steps[t] is a tuple of length num_layers; each element is
        # (batch=1, num_heads, 1, prompt_len + t + 1).
        for layer_idx, layer_attn in enumerate(decode_steps[t]):
            # Why: batch dim is always 1 in v1; squeeze it to (num_heads, 1, key_len).
            assert layer_attn.shape[0] == 1, (
                f"Expected batch=1, got {layer_attn.shape[0]}"
            )
            layer_attn = layer_attn.squeeze(0)  # (num_heads, 1, key_len)

            # Slice the response-to-response key range. Key positions
            # [prompt_len : prompt_len + r_max] are the response tokens seen so far.
            key_len = layer_attn.shape[-1]
            slice_end = min(prompt_len + r_max, key_len)
            row = layer_attn[:, 0, prompt_len:slice_end]  # (num_heads, n_keys)

            # Pad to r_max along the key dimension so the result is always r_max wide.
            n_keys = row.shape[-1]
            if n_keys < r_max:
                pad = r_max - n_keys
                row = F.pad(row, (0, pad))  # (num_heads, r_max)

            # Head-average per notes §4: simple mean over all heads.
            row_avg = row.float().mean(dim=0)  # (r_max,)
            out[layer_idx, t] = row_avg.detach().cpu().numpy().astype(np.float16)

    return out


def stitch_response_hidden_states(
    hidden_states: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    max_response_len: int,
) -> np.ndarray:
    """Stack per-token decode hidden states into a fixed-shape array.

    Args:
        hidden_states: out.hidden_states from model.generate(...).
        prompt_len: number of prompt tokens (unused here; kept for API symmetry).
        max_response_len: output time dimension.

    Returns:
        (num_layers + 1, max_response_len, hidden_dim) float16 numpy array,
        zero-padded past min(response_len, max_response_len). Layer 0 is the
        embedding output (HF convention).
    """
    # Why: hidden_states[0] is the prefill pass; hidden_states[t>=1] are decode
    # steps. Each element of hidden_states[0] is (batch, prompt_len, hidden_dim)
    # for one layer (0 = embedding output, 1..L = block outputs).
    prefill_layers = hidden_states[0]

    # Why: batch dim is always 1 in v1; assert and squeeze.
    assert prefill_layers[0].shape[0] == 1, (
        f"Expected batch=1, got {prefill_layers[0].shape[0]}"
    )

    num_layers_plus1 = len(prefill_layers)  # L+1 (embedding + L blocks)
    hidden_dim = prefill_layers[0].shape[-1]

    decode_steps = hidden_states[1:]  # length == response_len
    response_len = len(decode_steps)
    n_steps = min(response_len, max_response_len)

    out = np.zeros((num_layers_plus1, max_response_len, hidden_dim), dtype=np.float16)

    for t in range(n_steps):
        # decode_steps[t] is a tuple of length num_layers_plus1; each element
        # is (batch=1, 1, hidden_dim) for the newly-emitted token.
        for layer_idx, layer_hs in enumerate(decode_steps[t]):
            assert layer_hs.shape[0] == 1, (
                f"Expected batch=1, got {layer_hs.shape[0]}"
            )
            # (1, 1, hidden_dim) -> (hidden_dim,)
            token_hs = layer_hs[0, 0].detach().cpu().float().numpy()
            out[layer_idx, t] = token_hs.astype(np.float16)

    return out


def stitch_prompt_hidden_states(
    hidden_states: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    max_prompt_len: int,
) -> np.ndarray:
    """Extract prompt hidden states from the prefill pass.

    Args:
        hidden_states: out.hidden_states from model.generate(...).
        prompt_len: actual number of prompt tokens.
        max_prompt_len: output time dimension (truncate or zero-pad to this).

    Returns:
        (num_layers + 1, max_prompt_len, hidden_dim) float16 numpy array,
        zero-padded past min(prompt_len, max_prompt_len). Pulled from
        hidden_states[0] (the prefill forward pass).
    """
    # Why: hidden_states[0] is the full prefill pass; all subsequent elements
    # are single-token decode steps with no prompt information.
    prefill_layers = hidden_states[0]

    assert prefill_layers[0].shape[0] == 1, (
        f"Expected batch=1, got {prefill_layers[0].shape[0]}"
    )

    num_layers_plus1 = len(prefill_layers)
    hidden_dim = prefill_layers[0].shape[-1]

    out = np.zeros((num_layers_plus1, max_prompt_len, hidden_dim), dtype=np.float16)

    n_tokens = min(prompt_len, max_prompt_len)
    for layer_idx, layer_hs in enumerate(prefill_layers):
        # layer_hs: (batch=1, prompt_len, hidden_dim)
        tokens = layer_hs[0, :n_tokens].detach().cpu().float().numpy()
        out[layer_idx, :n_tokens] = tokens.astype(np.float16)

    return out


def extract_logprobs(
    scores: Tuple[Any, ...],
    generated_token_ids: Any,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-step log probabilities and top-k alternatives from generate() scores.

    Args:
        scores: out.scores from model.generate(...) — tuple of length
                response_len, each (batch=1, vocab_size) logits tensor.
        generated_token_ids: 1-D tensor of length response_len with the
                             actually-sampled token IDs (out.sequences[0][prompt_len:]).
        top_k: number of alternative tokens to keep per position.

    Returns:
        (token_logprobs, topk_ids, topk_logprobs):
            token_logprobs: (response_len,) float32 — log P(generated | context).
            topk_ids:       (response_len, top_k) int32.
            topk_logprobs:  (response_len, top_k) float32.
    """
    response_len = len(scores)

    # Convert to 1-D CPU int tensor for indexing, regardless of input type.
    if isinstance(generated_token_ids, np.ndarray):
        gen_ids = torch.from_numpy(generated_token_ids.astype(np.int64))
    else:
        gen_ids = generated_token_ids.long().cpu()

    token_logprobs = np.empty(response_len, dtype=np.float32)
    topk_ids_arr = np.empty((response_len, top_k), dtype=np.int32)
    topk_logprobs_arr = np.empty((response_len, top_k), dtype=np.float32)

    for t, step_logits in enumerate(scores):
        # Why: move off GPU before any numpy work; one transfer per step avoids
        # accumulating a large device tensor before log_softmax.
        logits_cpu = step_logits.detach().cpu()

        # Why: batch dim is always 1 in v1; squeeze to (vocab_size,).
        assert logits_cpu.shape[0] == 1, (
            f"Expected batch=1, got {logits_cpu.shape[0]}"
        )
        logits_1d = logits_cpu[0].float()  # (vocab_size,)

        log_probs = F.log_softmax(logits_1d, dim=-1)  # (vocab_size,)

        token_id = int(gen_ids[t])
        token_logprobs[t] = float(log_probs[token_id])

        # top_k capped at vocab size to avoid errors on tiny models.
        actual_k = min(top_k, logits_1d.shape[0])
        tk_vals, tk_ids = torch.topk(log_probs, k=actual_k)

        topk_logprobs_arr[t, :actual_k] = tk_vals.numpy().astype(np.float32)
        topk_ids_arr[t, :actual_k] = tk_ids.numpy().astype(np.int32)

        # Pad remaining slots if vocab < top_k (rare, only on tiny test models).
        if actual_k < top_k:
            topk_logprobs_arr[t, actual_k:] = 0.0
            topk_ids_arr[t, actual_k:] = 0

    return token_logprobs, topk_ids_arr, topk_logprobs_arr
