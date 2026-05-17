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
    """Stream-stitched result must match the naive slice-then-average reference.

    Reference semantic: response position q comes from attentions[q]'s LAST
    query row. q=0 reads from the prefill's last prompt query (response slice
    empty → naturally zero); q>=1 reads from the sole decode-pass row.
    """
    out, prompt_len = generate_output
    r_max = 8

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    num_layers = len(out.attentions[0])
    response_len = len(out.attentions)
    n_steps = min(response_len, r_max)

    ref = np.zeros((num_layers, r_max, r_max), dtype=np.float32)
    for q in range(n_steps):
        for layer_idx, layer_attn in enumerate(out.attentions[q]):
            arr = layer_attn[0, :, -1, prompt_len:].float()  # (heads, keys_after_prompt)
            n_keys = arr.shape[-1]
            if n_keys < r_max:
                arr = F.pad(arr, (0, r_max - n_keys))
            else:
                arr = arr[:, :r_max]
            ref[layer_idx, q] = arr.mean(dim=0).detach().numpy()

    assert np.max(np.abs(result.astype(np.float32) - ref)) < 1e-3


def test_stitch_response_to_response_truncation(generate_output):
    """When response_len > r_max the output is still exactly r_max wide."""
    out, prompt_len = generate_output
    response_len = len(out.attentions)
    r_max = max(1, response_len - 1)

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    num_layers = len(out.attentions[0])
    assert result.shape == (num_layers, r_max, r_max)
    assert result.dtype == np.float16


def test_stitch_response_to_response_padding(generate_output):
    """When r_max > response_len, rows past response_len are zero."""
    out, prompt_len = generate_output
    response_len = len(out.attentions)
    r_max = response_len + 4

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

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


def test_stitch_response_to_response_q0_is_zero(generate_output):
    """q=0 row is zero because the prefill's response sub-block is empty.

    The prefill's last query (prompt_len-1) attends only over prompt keys;
    the slice [prompt_len:] is empty, so the row pads to all-zero.
    """
    out, prompt_len = generate_output
    r_max = 8

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    assert np.all(result[:, 0, :] == 0)


def test_stitch_response_to_response_last_position_filled(generate_output):
    """Regression: q=response_len-1 must be FILLED (not the old zero row).

    The pre-fix stitcher used attentions[1:] and so left q=R-1 untouched —
    this test guards against that off-by-one returning.
    """
    out, prompt_len = generate_output
    response_len = len(out.attentions)
    r_max = response_len

    result = stitch_response_to_response(out.attentions, prompt_len, r_max)

    # response_len-1 row has key range [prompt_len : prompt_len+(response_len-1)],
    # which is response_len-1 nonzero entries (for response_len >= 2).
    if response_len >= 2:
        last_q_row = result[:, response_len - 1, :]
        assert np.any(last_q_row != 0), (
            f"q={response_len - 1} row is all zeros — off-by-one regression."
        )


# ---------------------------------------------------------------------------
# stitch_response_hidden_states
# ---------------------------------------------------------------------------

def test_stitch_response_hidden_states_layer_zero_is_embedding(
    model_and_tokenizer, generate_output
):
    """Layer-0 entries match HF hidden_states[q] (token emb + positional emb).

    Semantic: response position q comes from hidden_states[q]'s LAST position
    — for q=0 (prefill) that's the last prompt position; for q>=1 (decode)
    that's the sole new position.
    """
    out, prompt_len = generate_output

    response_len = len(out.hidden_states)
    max_response_len = response_len + 2
    hidden_dim = out.hidden_states[0][0].shape[-1]

    result = stitch_response_hidden_states(
        out.hidden_states, prompt_len, max_response_len
    )

    expected = np.zeros((response_len, hidden_dim), dtype=np.float32)
    for q in range(response_len):
        hs_q_layer0 = out.hidden_states[q][0]  # (batch=1, query_len, hidden_dim)
        expected[q] = hs_q_layer0[0, -1].detach().cpu().float().numpy()

    actual = result[0, :response_len].astype(np.float32)
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-2


def test_stitch_response_hidden_states_last_position_filled(generate_output):
    """Regression: q=response_len-1 hidden state must be filled (not zero)."""
    out, prompt_len = generate_output
    response_len = len(out.hidden_states)
    max_response_len = response_len

    result = stitch_response_hidden_states(
        out.hidden_states, prompt_len, max_response_len
    )

    # Compare the last response position against hidden_states[response_len-1]'s
    # last position, layer 0.
    expected_last = out.hidden_states[response_len - 1][0][0, -1].detach().cpu().float().numpy()
    actual_last = result[0, response_len - 1].astype(np.float32)
    assert np.max(np.abs(actual_last - expected_last)) < 1e-2


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
