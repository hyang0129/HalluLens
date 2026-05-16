"""
End-to-end equivalence gate for the batched capture path (Issue #72 spec §4).

Asserts that running the stitching pipeline with B=3 over three prompts
produces per-sample output byte-identical (within fp16 tolerance) to running
B=1 over each prompt individually.

This is the gating contract: if this test passes, the batched path is correct.

All tests run on CPU using sshleifer/tiny-gpt2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

MODEL_ID = "sshleifer/tiny-gpt2"
PROMPTS = [
    "The capital of France is",
    "Hello",
    "What is the largest planet?",
]
MAX_NEW_TOKENS = 8
MAX_PROMPT_LEN = 32
R_MAX = 6
TOP_K = 10
TOLERANCE = 1e-3


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


def _run_batched(model, tokenizer):
    """Run the full batched stitch pipeline over all PROMPTS at once."""
    from activation_logging.generate_capture import (
        extract_logprobs_batched,
        stitch_prompt_hidden_states_batched,
        stitch_response_hidden_states_batched,
        stitch_response_to_response_batched,
    )

    tokenizer.padding_side = "left"
    batch = tokenizer(
        PROMPTS,
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_LEN,
        return_tensors="pt",
    )
    prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()
    padded_prompt_len = batch.input_ids.shape[1]
    B = len(PROMPTS)

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

    response_lens = np.empty(B, dtype=np.int32)
    for b in range(B):
        resp_b = out.sequences[b, padded_prompt_len:]
        eos_pos = (resp_b == tokenizer.eos_token_id).nonzero(as_tuple=False)
        response_lens[b] = int(eos_pos[0].item()) + 1 if len(eos_pos) > 0 else int(resp_b.shape[0])

    resp_attn = stitch_response_to_response_batched(
        out.attentions, prompt_lens, response_lens, R_MAX
    )
    resp_hs = stitch_response_hidden_states_batched(
        out.hidden_states, prompt_lens, response_lens, MAX_NEW_TOKENS
    )
    prompt_hs = stitch_prompt_hidden_states_batched(
        out.hidden_states, prompt_lens, MAX_PROMPT_LEN
    )
    token_lp, topk_ids, topk_lp = extract_logprobs_batched(
        out.scores, out.sequences, prompt_lens, response_lens, TOP_K
    )

    return {
        "resp_attn": resp_attn,
        "resp_hs": resp_hs,
        "prompt_hs": prompt_hs,
        "token_lp": token_lp,
        "topk_ids": topk_ids,
        "topk_lp": topk_lp,
        "prompt_lens": prompt_lens,
        "response_lens": response_lens,
        "padded_prompt_len": padded_prompt_len,
    }


def _run_unbatched(model, tokenizer):
    """Run the B=1 stitch pipeline once per prompt."""
    from activation_logging.generate_capture import (
        extract_logprobs,
        stitch_prompt_hidden_states,
        stitch_response_hidden_states,
        stitch_response_to_response,
    )

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
        resp_ids = out.sequences[0][prompt_len:]

        resp_attn = stitch_response_to_response(out.attentions, prompt_len, R_MAX, response_len)
        resp_hs = stitch_response_hidden_states(out.hidden_states, prompt_len, MAX_NEW_TOKENS)
        prompt_hs = stitch_prompt_hidden_states(out.hidden_states, prompt_len, MAX_PROMPT_LEN)
        token_lp, topk_ids, topk_lp = extract_logprobs(out.scores, resp_ids, top_k=TOP_K)

        results.append({
            "resp_attn": resp_attn,
            "resp_hs": resp_hs,
            "prompt_hs": prompt_hs,
            "token_lp": token_lp,
            "topk_ids": topk_ids,
            "topk_lp": topk_lp,
            "prompt_len": prompt_len,
            "response_len": response_len,
        })
    return results


@pytest.fixture(scope="module")
def batched_results(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return _run_batched(model, tokenizer)


@pytest.fixture(scope="module")
def unbatched_results(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return _run_unbatched(model, tokenizer)


def test_response_attention_equivalence(batched_results, unbatched_results):
    for b, u in enumerate(unbatched_results):
        rlen = min(int(batched_results["response_lens"][b]), u["response_len"], R_MAX)
        bat = batched_results["resp_attn"][b, :, :rlen, :rlen].astype(np.float32)
        ref = u["resp_attn"][:, :rlen, :rlen].astype(np.float32)
        diff = float(np.max(np.abs(bat - ref)))
        assert diff < TOLERANCE, (
            f"resp_attn b={b}: max diff {diff:.4e} >= {TOLERANCE}"
        )


def test_response_hidden_states_equivalence(batched_results, unbatched_results):
    for b, u in enumerate(unbatched_results):
        rlen = min(int(batched_results["response_lens"][b]), u["response_len"], MAX_NEW_TOKENS)
        bat = batched_results["resp_hs"][b, :, :rlen].astype(np.float32)
        ref = u["resp_hs"][:, :rlen].astype(np.float32)
        diff = float(np.max(np.abs(bat - ref)))
        assert diff < TOLERANCE, (
            f"resp_hs b={b}: max diff {diff:.4e} >= {TOLERANCE}"
        )


def test_prompt_hidden_states_equivalence(batched_results, unbatched_results):
    for b, u in enumerate(unbatched_results):
        plen = min(int(batched_results["prompt_lens"][b]), u["prompt_len"], MAX_PROMPT_LEN)
        bat = batched_results["prompt_hs"][b, :, :plen].astype(np.float32)
        ref = u["prompt_hs"][:, :plen].astype(np.float32)
        diff = float(np.max(np.abs(bat - ref)))
        assert diff < TOLERANCE, (
            f"prompt_hs b={b}: max diff {diff:.4e} >= {TOLERANCE}"
        )


def test_logprobs_equivalence(batched_results, unbatched_results):
    for b, u in enumerate(unbatched_results):
        rlen = min(int(batched_results["response_lens"][b]), u["response_len"])
        bat = batched_results["token_lp"][b, :rlen]
        ref = u["token_lp"][:rlen]
        diff = float(np.max(np.abs(bat - ref)))
        assert diff < TOLERANCE, (
            f"token_lp b={b}: max diff {diff:.4e} >= {TOLERANCE}"
        )
