"""
Numerical-equivalence gate for the Issue #72 capture rewrite.

Asserts that our stream-stitched response-to-response attention
(activation_logging.generate_capture.stitch_response_to_response) matches
the upstream XavierZhang2002/ICR_Probe reference (_pre_process_attn) on the
response-to-response sub-block to fp16 tolerance.

If this test fails, the stitching contract is broken and ICR scores from
the new capture path are NOT comparable to the published baseline.

The vendored upstream code lives in external/ICR_Probe/icr_score_reference.py
(Apache 2.0, see external/ICR_Probe/LICENSE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Add external/ICR_Probe to sys.path so we can import the vendored reference.
_VENDORED_DIR = Path(__file__).resolve().parent.parent / "external" / "ICR_Probe"
if str(_VENDORED_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDORED_DIR))


@pytest.fixture(scope="module")
def tiny_generate_output():
    """Run model.generate(...) once with output_attentions=True on a tiny model.

    Returned dict contains:
        - prompt_len: int
        - r_max: int (capped by max_new_tokens; chosen so response_len < r_max)
        - response_len: int (actual)
        - hidden_states: out.hidden_states
        - attentions: out.attentions
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=torch.float32
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=8,
            do_sample=False,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_len = out.sequences.shape[1] - prompt_len
    return {
        "prompt_len": prompt_len,
        "r_max": 16,
        "response_len": response_len,
        "hidden_states": out.hidden_states,
        "attentions": out.attentions,
    }


def test_response_to_response_equivalence(tiny_generate_output):
    """Our stream-stitched attention must match upstream _pre_process_attn to fp16 tol."""
    from activation_logging.generate_capture import stitch_response_to_response

    # Import the vendored upstream class.
    from icr_score_reference import ICRScore

    prompt_len = tiny_generate_output["prompt_len"]
    r_max = tiny_generate_output["r_max"]
    response_len = tiny_generate_output["response_len"]
    attentions = tiny_generate_output["attentions"]
    hidden_states = tiny_generate_output["hidden_states"]

    ours = stitch_response_to_response(attentions, prompt_len, r_max)
    assert ours.shape == (len(attentions[1][0]) if response_len > 0 else len(attentions[0]),
                          r_max, r_max)
    assert ours.dtype == np.float16

    # Upstream path: instantiate ICRScore. core_positions controls cross-region
    # masking — keep both prompt-to-prompt AND response-to-response. We then
    # slice out only the response-to-response sub-block.
    core_positions = {
        "user_prompt_start": 0,
        "user_prompt_end": prompt_len,
        "response_start": prompt_len,
    }
    upstream = ICRScore(
        hidden_states=hidden_states,
        attentions=attentions,
        skew_threshold=0,
        entropy_threshold=1e5,
        core_positions=core_positions,
        icr_device=None,
    )

    # upstream.origin_attentions: (L, H, T, T) where T = prompt_len + response_len.
    # The response-to-response sub-block is at [prompt_len:, prompt_len:].
    L, H, T, _ = upstream.origin_attentions.shape
    r_actual = min(response_len, r_max)

    upstream_sub = upstream.origin_attentions[
        :, :, prompt_len : prompt_len + r_actual, prompt_len : prompt_len + r_actual
    ]  # (L, H, r_actual, r_actual)

    # Head-average to match our (L, r, r) layout.
    upstream_head_avg = upstream_sub.mean(dim=1).detach().cpu().numpy()  # (L, r_actual, r_actual)

    # Compare against the populated region only.
    ours_populated = ours[:L, :r_actual, :r_actual]

    max_diff = float(np.max(np.abs(ours_populated.astype(np.float32) - upstream_head_avg.astype(np.float32))))
    print(f"\n  response_len={response_len}, r_actual={r_actual}, L={L}, H={H}, T={T}")
    print(f"  max|ours - upstream| = {max_diff:.6e} (tol: 1e-3)")

    # fp16 tolerance per spec.
    assert max_diff < 1e-3, (
        f"Stitching contract broken: max diff {max_diff:.6e} >= 1e-3. "
        "The capture path's response_attention will not be apples-to-apples "
        "with the published ICR Probe baseline."
    )


def test_response_to_response_zeropad_past_response_len(tiny_generate_output):
    """Rows past response_len must be zero (no leaked state from earlier tokens)."""
    from activation_logging.generate_capture import stitch_response_to_response

    prompt_len = tiny_generate_output["prompt_len"]
    r_max = tiny_generate_output["r_max"]
    response_len = tiny_generate_output["response_len"]
    attentions = tiny_generate_output["attentions"]

    ours = stitch_response_to_response(attentions, prompt_len, r_max)

    if response_len < r_max:
        # Rows response_len..r_max-1 should be all zero.
        tail = ours[:, response_len:, :]
        assert np.all(tail == 0.0), "rows past response_len contain non-zero data"
