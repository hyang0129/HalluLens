"""
capture_inference.py — single-process inference capture for the ICR Probe
data pipeline (Issue #72).

Orchestrator only — primitives live in activation_logging/generate_capture.py;
storage lives in activation_logging/inference_capture_writer.py.

Usage:
    python scripts/capture_inference.py \\
        --task hotpotqa \\
        --split validation \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --out-dir shared/icr_capture/hotpotqa_Llama-3.1-8B-Instruct \\
        --max-prompt-len 512 \\
        --max-response-len 256 \\
        --r-max 64 \\
        --top-k 20 \\
        --n-samples 100  # for smoketest; omit for full split

End-to-end loop (see spec "Architecture" section):

  1. Load HF model with attn_implementation='eager' (required for output_attentions).
  2. Load tokenizer + task-module dataset iterator + task-module is_correct().
  3. For each sample:
     a. Tokenize prompt.
     b. model.generate(..., output_attentions=True, output_hidden_states=True,
                       return_dict_in_generate=True, do_sample=False).
     c. stitch_response_to_response(...) -> response_attention (num_layers, r_max, r_max)
     d. stitch_response_hidden_states(...) -> response_activations
     e. stitch_prompt_hidden_states(...)   -> prompt_activations
     f. extract_logprobs(...)              -> token_logprobs, topk_ids, topk_logprobs
     g. compute_icr_score per layer (uses activation_research.icr_score.compute_icr_score)
        — returns (num_layers,) fp32 score vector.
     h. tokenizer.decode response, compute hallucinated = not is_correct(...).
     i. writer.append(...) — passes prompt_hash = sha256(prompt) for resume keying.
  4. writer.finalize() synthesizes eval_results.json from meta.jsonl.

Resume semantics:
  - On startup, open writer in mode='a' if out_dir/config.json exists, else 'w'.
  - Skip samples whose sha256(prompt) is in writer.written_prompt_hashes.
  - Run starts from next_index().

Task module contract (existing in tasks/llmsknow/<task>.py):
  - Each task module exposes is_correct(generated_text, answer_or_answers) -> bool.
  - Dataset rows yield (key, prompt, question, answer, passthrough_dict).
  - Closed-book LLMsKnow tasks: hotpotqa, mmlu, natural_questions, popqa, sciq,
    searchqa. (movies excluded — no train split, can't be used for training data.)

Label contract (spec "Label contract"):
  - hallucinated = not is_correct(decoded_response, sample.answer).
  - Computed inline immediately after model.generate().
  - meta.jsonl includes the bool; finalize() synthesizes eval_results.json
    from meta.jsonl with array format {"halu_test_res": [...], "abstantion": [...]}.
  - --step eval-only re-runs is_correct() over an existing generation.jsonl
    without touching GPU.

Not orchestrated here (downstream consumers):
  - Probe training (Issue #70).
  - SE / P(true) / SelfCheckGPT — those continue to read the synthesized
    eval_results.json + generation.jsonl, unchanged from the existing baseline
    consumer contracts.
"""

from __future__ import annotations

import argparse


def main() -> int:
    """STUB — to be implemented by sonnet C. See module docstring for contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-response-len", type=int, default=256)
    parser.add_argument("--r-max", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--step", choices=("capture", "eval-only"), default="capture")
    args = parser.parse_args()

    raise NotImplementedError(
        "Implement per capture_inference.py module docstring (Issue #72 orchestrator)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
