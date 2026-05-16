"""
Tests for activation_logging/generate_capture.py.

All tests run on CPU. The model fixture uses sshleifer/tiny-gpt2 (2-layer
causal LM) with attn_implementation='eager' so that output_attentions=True
returns real attention weight tensors rather than None.

The "naive" reference paths deliberately replicate the upstream
XavierZhang2002/ICR_Probe stitching logic described in
notes/icr_probe_paper_notes.md §1.5 so numerical-equivalence failures
identify actual stitching bugs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_logging.generate_capture import (
    extract_logprobs,
    stitch_prompt_hidden_states,
    stitch_response_hidden_states,
    stitch_response_to_response,
)

MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def generate_output(model_and_tokenizer):
    """Run model.generate on a short prompt and return (out, prompt_len)."""
    model, tokenizer = model_and_tokenizer
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=6,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
    return out, prompt_len


# ---------------------------------------------------------------------------
# stitch_response_to_response
# ---------------------------------------------------------------------------

def test_stitch_response_to_response_shape_and_dtype(generate_output):
    out, prompt_len = generate_output
    num_layers = len(out.attentions[0])
    r_max = 4

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    assert result.shape == (num_layers, r_max, r_max)
    assert result.dtype == np.float16


def test_stitch_response_to_response_equivalence(generate_output):
    """Stream-stitched result must match the naive slice-then-average reference."""
    out, prompt_len = generate_output
    r_max = 8

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    # Naive reference: for each decode step t>=1, directly slice the key dim
    # and mean over heads.
    decode_steps = out.attentions[1:]
    num_layers = len(decode_steps[0])
    response_len = len(decode_steps)
    n_steps = min(response_len, r_max)

    ref = np.zeros((num_layers, r_max, r_max), dtype=np.float32)
    for t in range(n_steps):
        for layer_idx, layer_attn in enumerate(decode_steps[t]):
            arr = layer_attn[0, :, 0, prompt_len:].float()  # (heads, keys_so_far)
            n_keys = arr.shape[-1]
            if n_keys < r_max:
                arr = F.pad(arr, (0, r_max - n_keys))
            else:
                arr = arr[:, :r_max]
            ref[layer_idx, t] = arr.mean(dim=0).detach().numpy()

    assert np.max(np.abs(result.astype(np.float32) - ref)) < 1e-3


def test_stitch_response_to_response_truncation(generate_output):
    """When response_len > r_max the output is still exactly r_max wide."""
    out, prompt_len = generate_output
    response_len = len(out.attentions) - 1
    # Choose r_max smaller than actual response so truncation is exercised.
    r_max = max(1, response_len - 1)

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    num_layers = len(out.attentions[0])
    assert result.shape == (num_layers, r_max, r_max)
    assert result.dtype == np.float16


def test_stitch_response_to_response_padding(generate_output):
    """When r_max > response_len, rows past response_len are zero."""
    out, prompt_len = generate_output
    response_len = len(out.attentions) - 1
    r_max = response_len + 4

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    # Rows [response_len:] in the query dimension must be all-zero.
    assert np.all(result[:, response_len:, :] == 0)


def test_stitch_response_to_response_zero_response_len(generate_output):
    """Passing response_len=0 returns an all-zero array without crashing."""
    out, prompt_len = generate_output
    r_max = 4
    num_layers = len(out.attentions[0])

    result = stitch_response_to_response(
        out.attentions, prompt_len, r_max, response_len=0
    )

    assert result.shape == (num_layers, r_max, r_max)
    assert result.dtype == np.float16
    assert np.all(result == 0)


# ---------------------------------------------------------------------------
# stitch_response_hidden_states
# ---------------------------------------------------------------------------

def test_stitch_response_hidden_states_layer_zero_is_embedding(
    model_and_tokenizer, generate_output
):
    """Layer-0 entries match HF hidden_states[0] (token emb + positional emb)."""
    model, tokenizer = model_and_tokenizer
    out, prompt_len = generate_output

    response_len = len(out.hidden_states) - 1  # subtract prefill step
    max_response_len = response_len + 2
    hidden_dim = out.hidden_states[0][0].shape[-1]

    result = stitch_response_hidden_states(
        out.hidden_states, prompt_len, max_response_len
    )

    # HF hidden_states layer 0 is the embedding output: wte(id) + wpe(position).
    # For each decode step t (1-indexed in hidden_states), compute expected
    # directly from the raw tensors that model.generate stores.
    expected = np.zeros((response_len, hidden_dim), dtype=np.float32)
    for t in range(response_len):
        # hidden_states[t+1] is decode step t (0-indexed), layer-0 entry.
        hs_t_layer0 = out.hidden_states[t + 1][0]  # (batch=1, 1, hidden_dim)
        expected[t] = hs_t_layer0[0, 0].detach().cpu().float().numpy()

    # result[0] is layer-0 (embedding output). Compare valid rows only.
    actual = result[0, :response_len].astype(np.float32)
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-2


# ---------------------------------------------------------------------------
# stitch_prompt_hidden_states
# ---------------------------------------------------------------------------

def test_stitch_prompt_hidden_states_shape_and_truncation(generate_output):
    """Shape is (L+1, max_prompt_len, H) and truncation works correctly."""
    out, prompt_len = generate_output
    num_layers_plus1 = len(out.hidden_states[0])
    hidden_dim = out.hidden_states[0][0].shape[-1]

    # Case 1: max_prompt_len == prompt_len — exact fit.
    result_exact = stitch_prompt_hidden_states(
        out.hidden_states, prompt_len, prompt_len
    )
    assert result_exact.shape == (num_layers_plus1, prompt_len, hidden_dim)
    assert result_exact.dtype == np.float16

    # Case 2: max_prompt_len > prompt_len — zero-padded past prompt_len.
    big_max = prompt_len + 3
    result_pad = stitch_prompt_hidden_states(out.hidden_states, prompt_len, big_max)
    assert result_pad.shape == (num_layers_plus1, big_max, hidden_dim)
    assert np.all(result_pad[:, prompt_len:, :] == 0)

    # Case 3: max_prompt_len < prompt_len — truncated, no crash.
    small_max = max(1, prompt_len - 1)
    result_trunc = stitch_prompt_hidden_states(
        out.hidden_states, prompt_len, small_max
    )
    assert result_trunc.shape == (num_layers_plus1, small_max, hidden_dim)


# ---------------------------------------------------------------------------
# extract_logprobs
# ---------------------------------------------------------------------------

def test_extract_logprobs_basic(generate_output):
    """token_logprobs[t] is the actual log P(generated_id | context)."""
    out, prompt_len = generate_output
    response_ids = out.sequences[0][prompt_len:]
    response_len = len(response_ids)

    token_logprobs, topk_ids, topk_logprobs = extract_logprobs(
        out.scores, response_ids, top_k=20
    )

    # Reference: compute log P(token_id) from scratch for each step.
    for t, step_logits in enumerate(out.scores):
        lp = F.log_softmax(step_logits[0].float(), dim=-1)
        expected = float(lp[int(response_ids[t])])
        assert abs(float(token_logprobs[t]) - expected) < 1e-5, (
            f"Step {t}: expected {expected}, got {token_logprobs[t]}"
        )

    # For greedy decoding, the generated token is by definition the argmax, so
    # it must appear in the top-k results for every step.
    for t in range(response_len):
        tid = int(response_ids[t])
        assert tid in topk_ids[t], (
            f"Step {t}: generated token {tid} not in top-k ids {topk_ids[t]}"
        )


def test_extract_logprobs_shapes(generate_output):
    """Output shapes and dtypes must be exactly as documented."""
    out, prompt_len = generate_output
    response_ids = out.sequences[0][prompt_len:]
    response_len = len(response_ids)
    k = 10

    token_logprobs, topk_ids, topk_logprobs = extract_logprobs(
        out.scores, response_ids, top_k=k
    )

    assert token_logprobs.shape == (response_len,)
    assert token_logprobs.dtype == np.float32

    assert topk_ids.shape == (response_len, k)
    assert topk_ids.dtype == np.int32

    assert topk_logprobs.shape == (response_len, k)
    assert topk_logprobs.dtype == np.float32
