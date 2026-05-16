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


def stitch_response_to_response(
    attentions: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    r_max: int,
    response_len: int | None = None,
) -> np.ndarray:
    """STUB — to be implemented by sonnet A. See module docstring for contract.

    Args:
        attentions: out.attentions from model.generate(...).
        prompt_len: number of prompt tokens (used to offset the key slice).
        r_max: max stored response length (output dim).
        response_len: actual response length; defaults to len(attentions) - 1.

    Returns:
        (num_layers, r_max, r_max) float16 numpy array, zero-padded past
        min(response_len, r_max).
    """
    raise NotImplementedError("Implement per generate_capture.py module docstring.")


def stitch_response_hidden_states(
    hidden_states: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    max_response_len: int,
) -> np.ndarray:
    """STUB — to be implemented by sonnet A. See module docstring for contract.

    Returns:
        (num_layers + 1, max_response_len, hidden_dim) float16 numpy array,
        zero-padded past min(response_len, max_response_len). Layer 0 is the
        embedding output (HF convention).
    """
    raise NotImplementedError("Implement per generate_capture.py module docstring.")


def stitch_prompt_hidden_states(
    hidden_states: Tuple[Tuple[Any, ...], ...],
    prompt_len: int,
    max_prompt_len: int,
) -> np.ndarray:
    """STUB — to be implemented by sonnet A. See module docstring for contract.

    Returns:
        (num_layers + 1, max_prompt_len, hidden_dim) float16 numpy array,
        zero-padded past min(prompt_len, max_prompt_len). Pulled from
        hidden_states[0] (the prefill forward pass).
    """
    raise NotImplementedError("Implement per generate_capture.py module docstring.")


def extract_logprobs(
    scores: Tuple[Any, ...],
    generated_token_ids: Any,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STUB — to be implemented by sonnet A. See module docstring for contract.

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
    raise NotImplementedError("Implement per generate_capture.py module docstring.")
