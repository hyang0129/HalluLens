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


def test_icr_scores_gpu_matches_numpy_reference(batched_results, unbatched_results):
    """GPU ICR path (B>1) must match per-sample numpy reference within 1e-4.

    This is the gate that says "GPU ICR is safe for Phase 1": the batched GPU
    kernel must reproduce the B=1 numpy-reference ICR scores within the
    allowed fp16/fp32 drift budget.
    """
    from activation_research.icr_score import compute_icr_score
    from activation_research.icr_score_gpu import compute_icr_per_layer_batched_gpu
    from scripts.capture_inference import compute_icr_per_layer

    ICR_TOLERANCE = 1e-4

    resp_attn = batched_results["resp_attn"]   # (B, L, r_max, r_max) float16
    resp_hs = batched_results["resp_hs"]       # (B, L+1, max_resp, hidden_dim) float16
    response_lens = batched_results["response_lens"]  # (B,) int32

    B = resp_attn.shape[0]
    L = resp_attn.shape[1]

    # Numpy reference: per-sample B=1 loop (same as the production B=1 path).
    numpy_icr = np.stack([
        compute_icr_per_layer(resp_attn[b], resp_hs[b], int(response_lens[b]))
        for b in range(B)
    ])  # (B, L)

    # GPU path: batched kernel with the same slicing logic as _run_batch.
    r_max = resp_attn.shape[2]
    h_in_np = resp_hs[:, :-1, :r_max].astype(np.float32)   # (B, L, r_max, D)
    h_out_np = resp_hs[:, 1:, :r_max].astype(np.float32)   # (B, L, r_max, D)
    gpu_icr = compute_icr_per_layer_batched_gpu(
        torch.from_numpy(resp_attn.astype(np.float32)),
        torch.from_numpy(h_in_np),
        torch.from_numpy(h_out_np - h_in_np),
        torch.from_numpy(response_lens.astype(np.int64)),
        top_p=0.1,
    ).numpy()   # (B, L)

    for b in range(B):
        max_diff = float(np.max(np.abs(gpu_icr[b] - numpy_icr[b])))
        assert max_diff < ICR_TOLERANCE, (
            f"ICR gate b={b} (response_len={response_lens[b]}): "
            f"max|gpu - numpy|={max_diff:.2e} >= {ICR_TOLERANCE}"
        )

    assert np.all(np.isfinite(gpu_icr)), "GPU ICR output contains NaN/Inf"
