"""
Tests for the batched stitching primitives in activation_logging/generate_capture.py.

Strategy: run each *_batched primitive on a B=3 batch of prompts with different
lengths (simulating left-padding). For each sample b, assert that
    batched_output[b] == unbatched_call_on_prompt_b
within fp16 tolerance (max abs diff < 1e-3).

Also verifies pad-step skipping: a sample with response_len=2 must not be
contaminated by attention rows from decode steps t>2.

All tests run on CPU using sshleifer/tiny-gpt2 (2-layer causal LM) with
attn_implementation='eager'.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

MODEL_ID = "sshleifer/tiny-gpt2"
PROMPTS = [
    "The capital of France is",
    "A",
    "What is the largest planet in the solar system?",
]
MAX_NEW_TOKENS = 8
R_MAX = 6
TOP_K = 10


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def batched_output(model_and_tokenizer):
    """Single batched model.generate() call over all 3 prompts with left-padding."""
    model, tokenizer = model_and_tokenizer
    tokenizer.padding_side = "left"
    batch = tokenizer(
        PROMPTS,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()
    padded_prompt_len = batch.input_ids.shape[1]

    response_lens = np.empty(len(PROMPTS), dtype=np.int32)
    for b in range(len(PROMPTS)):
        resp_b = out.sequences[b, padded_prompt_len:]
        eos_pos = (resp_b == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos_pos) > 0:
            response_lens[b] = int(eos_pos[0].item()) + 1
        else:
            response_lens[b] = int(resp_b.shape[0])

    return {
        "out": out,
        "prompt_lens": prompt_lens,
        "response_lens": response_lens,
        "padded_prompt_len": padded_prompt_len,
    }


@pytest.fixture(scope="module")
def unbatched_outputs(model_and_tokenizer):
    """One model.generate() call per prompt (B=1, no padding)."""
    model, tokenizer = model_and_tokenizer
    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_attentions=True,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_len = out.sequences.shape[1] - prompt_len
        results.append({
            "out": out,
            "prompt_len": prompt_len,
            "response_len": response_len,
        })
    return results


# ---------------------------------------------------------------------------
# stitch_response_to_response_batched
# ---------------------------------------------------------------------------

def test_stitch_response_to_response_batched_shape(batched_output):
    from activation_logging.generate_capture import stitch_response_to_response_batched

    d = batched_output
    result = stitch_response_to_response_batched(
        d["out"].attentions, d["prompt_lens"], d["response_lens"], R_MAX
    )
    B = len(PROMPTS)
    num_layers = len(d["out"].attentions[1]) if len(d["out"].attentions) > 1 else len(d["out"].attentions[0])
    assert result.shape == (B, num_layers, R_MAX, R_MAX)
    assert result.dtype == np.float16


def test_stitch_response_to_response_batched_vs_unbatched(batched_output, unbatched_outputs):
    from activation_logging.generate_capture import (
        stitch_response_to_response,
        stitch_response_to_response_batched,
    )

    d = batched_output
    batched = stitch_response_to_response_batched(
        d["out"].attentions, d["prompt_lens"], d["response_lens"], R_MAX
    )

    for b, u in enumerate(unbatched_outputs):
        ref = stitch_response_to_response(
            u["out"].attentions, u["prompt_len"], R_MAX, u["response_len"]
        )
        diff = float(np.max(np.abs(batched[b].astype(np.float32) - ref.astype(np.float32))))
        assert diff < 1e-3, (
            f"sample b={b}: max|batched - unbatched| = {diff:.4e} >= 1e-3"
        )


def test_stitch_response_to_response_batched_pad_step_skip(model_and_tokenizer):
    """Samples with early EOS must not contaminate rows past response_len."""
    from activation_logging.generate_capture import stitch_response_to_response_batched

    model, tokenizer = model_and_tokenizer
    tokenizer.padding_side = "left"
    batch = tokenizer(PROMPTS, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()
    padded_prompt_len = batch.input_ids.shape[1]

    response_lens = np.empty(len(PROMPTS), dtype=np.int32)
    for b in range(len(PROMPTS)):
        resp_b = out.sequences[b, padded_prompt_len:]
        eos_pos = (resp_b == tokenizer.eos_token_id).nonzero(as_tuple=False)
        response_lens[b] = int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else int(resp_b.shape[0])

    # Force a short response_len for sample 0 to test the skip logic.
    truncated_lens = response_lens.copy()
    truncated_lens[0] = min(2, response_lens[0])

    result = stitch_response_to_response_batched(
        out.attentions, prompt_lens, truncated_lens, R_MAX
    )

    short_len = int(truncated_lens[0])
    if short_len < R_MAX:
        tail = result[0, :, short_len:, :]
        assert np.all(tail == 0), "rows past response_lens[0] are non-zero (pad-step leaked)"


# ---------------------------------------------------------------------------
# stitch_response_hidden_states_batched
# ---------------------------------------------------------------------------

def test_stitch_response_hidden_states_batched_shape(batched_output):
    from activation_logging.generate_capture import stitch_response_hidden_states_batched

    d = batched_output
    num_layers_plus1 = len(d["out"].hidden_states[0])
    hidden_dim = d["out"].hidden_states[0][0].shape[-1]
    result = stitch_response_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["response_lens"], MAX_NEW_TOKENS
    )
    B = len(PROMPTS)
    assert result.shape == (B, num_layers_plus1, MAX_NEW_TOKENS, hidden_dim)
    assert result.dtype == np.float16


def test_stitch_response_hidden_states_batched_vs_unbatched(batched_output, unbatched_outputs):
    from activation_logging.generate_capture import (
        stitch_response_hidden_states,
        stitch_response_hidden_states_batched,
    )

    d = batched_output
    batched = stitch_response_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["response_lens"], MAX_NEW_TOKENS
    )

    for b, u in enumerate(unbatched_outputs):
        ref = stitch_response_hidden_states(
            u["out"].hidden_states, u["prompt_len"], MAX_NEW_TOKENS
        )
        # Compare only the rows actually populated (min of the two response_lens).
        rlen = min(int(d["response_lens"][b]), u["response_len"], MAX_NEW_TOKENS)
        diff = float(np.max(np.abs(
            batched[b, :, :rlen].astype(np.float32) - ref[:, :rlen].astype(np.float32)
        )))
        assert diff < 1e-3, (
            f"sample b={b}: max|batched - unbatched| hidden states = {diff:.4e} >= 1e-3"
        )


# ---------------------------------------------------------------------------
# stitch_prompt_hidden_states_batched
# ---------------------------------------------------------------------------

def test_stitch_prompt_hidden_states_batched_shape(batched_output):
    from activation_logging.generate_capture import stitch_prompt_hidden_states_batched

    d = batched_output
    num_layers_plus1 = len(d["out"].hidden_states[0])
    hidden_dim = d["out"].hidden_states[0][0].shape[-1]
    max_prompt_len = int(d["prompt_lens"].max())
    result = stitch_prompt_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], max_prompt_len
    )
    B = len(PROMPTS)
    assert result.shape == (B, num_layers_plus1, max_prompt_len, hidden_dim)
    assert result.dtype == np.float16


def test_stitch_prompt_hidden_states_batched_vs_unbatched(batched_output, unbatched_outputs):
    from activation_logging.generate_capture import (
        stitch_prompt_hidden_states,
        stitch_prompt_hidden_states_batched,
    )

    d = batched_output
    max_prompt_len = int(d["prompt_lens"].max())
    batched = stitch_prompt_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], max_prompt_len
    )

    for b, u in enumerate(unbatched_outputs):
        ref = stitch_prompt_hidden_states(
            u["out"].hidden_states, u["prompt_len"], max_prompt_len
        )
        plen = min(int(d["prompt_lens"][b]), u["prompt_len"], max_prompt_len)
        diff = float(np.max(np.abs(
            batched[b, :, :plen].astype(np.float32) - ref[:, :plen].astype(np.float32)
        )))
        assert diff < 1e-3, (
            f"sample b={b}: max|batched - unbatched| prompt hs = {diff:.4e} >= 1e-3"
        )


# ---------------------------------------------------------------------------
# extract_logprobs_batched
# ---------------------------------------------------------------------------

def test_extract_logprobs_batched_shape(batched_output):
    from activation_logging.generate_capture import extract_logprobs_batched

    d = batched_output
    token_lp, topk_ids, topk_lp = extract_logprobs_batched(
        d["out"].scores, d["out"].sequences, d["prompt_lens"], d["response_lens"], TOP_K
    )
    B = len(PROMPTS)
    R = int(d["response_lens"].max())
    assert token_lp.shape == (B, R)
    assert token_lp.dtype == np.float32
    assert topk_ids.shape == (B, R, TOP_K)
    assert topk_ids.dtype == np.int32
    assert topk_lp.shape == (B, R, TOP_K)
    assert topk_lp.dtype == np.float32


def test_extract_logprobs_batched_vs_unbatched(batched_output, unbatched_outputs):
    from activation_logging.generate_capture import (
        extract_logprobs,
        extract_logprobs_batched,
    )

    d = batched_output
    token_lp_b, topk_ids_b, topk_lp_b = extract_logprobs_batched(
        d["out"].scores, d["out"].sequences, d["prompt_lens"], d["response_lens"], TOP_K
    )

    padded_prompt_len = d["padded_prompt_len"]
    for b, u in enumerate(unbatched_outputs):
        rlen = min(int(d["response_lens"][b]), u["response_len"])
        resp_ids = u["out"].sequences[0][u["prompt_len"]:]
        ref_lp, _, _ = extract_logprobs(u["out"].scores, resp_ids, top_k=TOP_K)
        diff = float(np.max(np.abs(token_lp_b[b, :rlen] - ref_lp[:rlen])))
        assert diff < 1e-3, (
            f"sample b={b}: max|batched - unbatched| logprobs = {diff:.4e} >= 1e-3"
        )
