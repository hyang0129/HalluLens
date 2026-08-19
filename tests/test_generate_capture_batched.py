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


# ---------------------------------------------------------------------------
# Capture-convention invariants: what IS response position q=0?
#
# The entire first-state / pre-generation line of work (issue #151, #159, #160)
# rests on one claim: ``response_activations[:, :, 0, :]`` is the hidden state at
# the FINAL PROMPT TOKEN — the prefill state whose logits produce response token
# zero — and therefore conditions on no sampled response token.
#
# That claim is currently protected only by a comment in generate_capture.py plus
# a ``tokenizer.padding_side = "left"`` assignment in a DIFFERENT file
# (scripts/capture_inference.py). stitch_response_hidden_states_batched takes
# ``layer_cpu[b, -1]`` at q=0 and never consults prompt_lens, so a right-padding
# regression would silently make every short prompt's q=0 a PAD-position state —
# producing plausible-looking but meaningless results with no test failure.
#
# These tests pin the convention down against an independent forward pass.
# ---------------------------------------------------------------------------

def _bare_prompt_hidden_states(model, tokenizer, prompt):
    """Forward pass on the prompt alone: no generation, no padding, no batch.

    Returns a tuple of (num_layers + 1) tensors, each (seq_len, hidden_dim).
    This is ground truth that provably conditions on no response token.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return tuple(h[0] for h in out.hidden_states)


def test_response_position_zero_is_final_prompt_token(
    model_and_tokenizer, batched_output
):
    """q=0 == the last prompt token's state from an independent prompt-only pass.

    This is the load-bearing assertion. It is independent of the stitching code:
    it compares against a separate forward pass that contains no response tokens
    at all, so passing it proves q=0 cannot encode a sampled response token.
    """
    from activation_logging.generate_capture import (
        stitch_response_hidden_states_batched,
    )

    model, tokenizer = model_and_tokenizer
    d = batched_output
    resp = stitch_response_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["response_lens"], R_MAX
    )

    for b, prompt in enumerate(PROMPTS):
        ref_layers = _bare_prompt_hidden_states(model, tokenizer, prompt)
        for layer_idx, ref in enumerate(ref_layers):
            got = resp[b, layer_idx, 0, :].astype(np.float32)
            want = ref[-1].detach().cpu().numpy().astype(np.float32)
            assert np.max(np.abs(got - want)) < 1e-2, (
                f"prompt {b!r} layer {layer_idx}: response position 0 is not the "
                f"final prompt token state. If this fails, check that "
                f"tokenizer.padding_side == 'left' wherever capture runs."
            )


def test_response_position_zero_matches_prompt_activations_tail(batched_output):
    """Internal consistency: response[:, :, 0] == prompt[:, :, prompt_len - 1].

    Note the index: prompt_activations is zero-padded PAST prompt_len, so its
    index -1 is a zero pad, not the last real token. Comparing at -1 would fail
    for reasons unrelated to the convention under test.
    """
    from activation_logging.generate_capture import (
        stitch_prompt_hidden_states_batched,
        stitch_response_hidden_states_batched,
    )

    d = batched_output
    resp = stitch_response_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["response_lens"], R_MAX
    )
    prompt = stitch_prompt_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["padded_prompt_len"]
    )

    for b in range(len(PROMPTS)):
        tail = int(d["prompt_lens"][b]) - 1
        got = resp[b, :, 0, :].astype(np.float32)
        want = prompt[b, :, tail, :].astype(np.float32)
        assert np.max(np.abs(got - want)) < 1e-2, (
            f"prompt {b}: response position 0 disagrees with prompt position "
            f"{tail} (prompt_len={d['prompt_lens'][b]})"
        )


def test_response_position_one_conditions_on_first_generated_token(
    model_and_tokenizer, batched_output
):
    """q=1 DOES condition on response token 0 — confirms the off-by-one.

    Documents the other half of the convention: position q holds the state that
    PRODUCED token q, so q>=1 has read tokens <q. Without this, a reader could
    reasonably assume q indexes the state that READ token q.
    """
    from activation_logging.generate_capture import (
        stitch_response_hidden_states_batched,
    )

    model, tokenizer = model_and_tokenizer
    d = batched_output
    resp = stitch_response_hidden_states_batched(
        d["out"].hidden_states, d["prompt_lens"], d["response_lens"], R_MAX
    )
    padded = d["padded_prompt_len"]

    for b, prompt in enumerate(PROMPTS):
        if d["response_lens"][b] < 2:
            continue
        first_gen = d["out"].sequences[b, padded].item()
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[first_gen]])], dim=1)
        with torch.no_grad():
            ref = model(ids, output_hidden_states=True).hidden_states

        for layer_idx, layer_hs in enumerate(ref):
            got = resp[b, layer_idx, 1, :].astype(np.float32)
            want = layer_hs[0, -1].detach().cpu().numpy().astype(np.float32)
            assert np.max(np.abs(got - want)) < 1e-2, (
                f"prompt {b!r} layer {layer_idx}: response position 1 does not "
                f"match a forward pass over prompt + first generated token."
            )


def test_right_padding_would_break_position_zero(model_and_tokenizer):
    """Negative control: the q=0 convention DEPENDS on left padding.

    stitch_response_hidden_states_batched indexes [b, -1] at q=0 without
    consulting prompt_lens. Under right padding that is a pad-position state for
    every prompt shorter than the batch max. This test fails loudly if someone
    changes padding_side, which no other test would catch.
    """
    from activation_logging.generate_capture import (
        stitch_response_hidden_states_batched,
    )

    model, tokenizer = model_and_tokenizer
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        batch = tokenizer(PROMPTS, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()
        response_lens = np.full(len(PROMPTS), MAX_NEW_TOKENS, dtype=np.int32)
        resp = stitch_response_hidden_states_batched(
            out.hidden_states, prompt_lens, response_lens, R_MAX
        )

        # The shortest prompt carries the most padding; its q=0 must NOT equal
        # the true final-prompt-token state under right padding.
        b_short = int(np.argmin(prompt_lens))
        ref_layers = _bare_prompt_hidden_states(model, tokenizer, PROMPTS[b_short])
        last_layer = len(ref_layers) - 1
        got = resp[b_short, last_layer, 0, :].astype(np.float32)
        want = ref_layers[last_layer][-1].detach().cpu().numpy().astype(np.float32)
        assert np.max(np.abs(got - want)) > 1e-2, (
            "Right padding did not break the q=0 invariant. Either the stitcher "
            "now consults prompt_lens (good — update this test), or the fixture "
            "no longer has variable-length prompts (bad — the guarantee is void)."
        )
    finally:
        tokenizer.padding_side = original_side
