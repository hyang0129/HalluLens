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
        --max-response-len 64 \\
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
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Why: `python scripts/capture_inference.py` adds only scripts/ to sys.path,
# not the repo root — so first-party packages like activation_logging fail to
# import. Mirror the convention used by every other script in this dir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
# Maps CLI --task name to the task module's load function, is_correct function,
# and a prompt-builder function. Extend this dict to add tasks.
#
# Each entry: (load_fn_name, is_correct_adapter, prompt_fn)
#   - load_fn_name: name of the loader in the module (str)
#   - is_correct_adapter: callable(task_module, generation, sample) -> bool
#     (indirects over is_correct signature differences between tasks)
#   - prompt_fn: callable(task_module, sample) -> str
#
# Why a registry instead of duck-typing alone: popqa's is_correct takes a list
# while all other tasks take a string, so a thin adapter is needed. NQ's
# is_correct is inside a class rather than module-level, requiring a local shim.

def _correct_str(task_module: Any, generation: str, sample: dict) -> bool:
    """Adapter for tasks where is_correct(generation, answer: str)."""
    return task_module.is_correct(generation, sample["answer"])


def _correct_list(task_module: Any, generation: str, sample: dict) -> bool:
    """Adapter for popqa where is_correct(generation, possible_answers: list)."""
    return task_module.is_correct(generation, sample["possible_answers"])


def _correct_nq(task_module: Any, generation: str, sample: dict) -> bool:
    """Adapter for NQ which has no module-level is_correct."""
    answer = sample["answer"]
    return answer.lower() in generation.lower()


def _prompt_default(task_module: Any, sample: dict) -> str:
    """Use task_module.format_prompt if available, else fall back to sample['question']."""
    if hasattr(task_module, "format_prompt"):
        return task_module.format_prompt(sample["question"])
    return sample["question"]


# (load_fn_name, is_correct_adapter, default_split)
_TASK_REGISTRY: dict[str, tuple[str, Any, str]] = {
    "hotpotqa": ("load_hotpotqa", _correct_str, "validation"),
    "mmlu": ("load_mmlu_data", _correct_str, "test"),
    "popqa": ("load_popqa_data", _correct_list, "test"),
    "natural_questions": ("load_nq_data", _correct_nq, "test"),
    "sciq": ("load_sciq_data", _correct_str, "test"),
    "searchqa": ("load_searchqa_data", _correct_str, "train"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pad_to(arr: np.ndarray, target_len: int, fill_value: Any) -> np.ndarray:
    """Pad 1-D array to target_len with fill_value, or truncate if longer."""
    arr = np.asarray(arr).ravel()
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    pad_width = target_len - arr.shape[0]
    return np.concatenate([arr, np.full(pad_width, fill_value, dtype=arr.dtype)])


def pad_2d(arr: np.ndarray, target_rows: int, target_cols: int, fill_value: Any) -> np.ndarray:
    """Pad 2-D array to (target_rows, target_cols), truncating dims that are too large."""
    arr = np.asarray(arr)
    rows, cols = arr.shape
    # Truncate
    arr = arr[:target_rows, :target_cols]
    rows, cols = arr.shape
    # Pad columns
    if cols < target_cols:
        arr = np.pad(arr, ((0, 0), (0, target_cols - cols)),
                     constant_values=fill_value)
    # Pad rows
    if rows < target_rows:
        arr = np.pad(arr, ((0, target_rows - rows), (0, 0)),
                     constant_values=fill_value)
    return arr


def compute_icr_per_layer(
    resp_attn: np.ndarray,
    resp_hs: np.ndarray,
    response_len: int,
    top_p: float = 0.1,
) -> np.ndarray:
    """Compute ICR score for each layer, returning (num_layers,) float32 array.

    Args:
        resp_attn: (num_layers, r_max, r_max) float16 — response-to-response attention.
        resp_hs: (num_layers+1, max_response_len, hidden_dim) float16 — response
                 hidden states. Layer 0 is the embedding output.
        response_len: actual response length.
        top_p: top-p fraction for ICR attention masking.

    Returns:
        (num_layers,) float32 array of per-layer ICR scores.
    """
    from activation_research.icr_score import compute_icr_score

    num_layers = resp_attn.shape[0]
    r_max = resp_attn.shape[1]
    scores = np.zeros(num_layers, dtype=np.float32)
    for l in range(num_layers):
        # Why: compute_icr_score expects h_block_input/delta_h aligned to the
        # attention sub-block (R=r_max), not the full max_response_len.
        # resp_hs[l] is block input h^{l-1}; resp_hs[l+1] is block output h^l.
        h_block_input = resp_hs[l][:r_max].astype(np.float32)
        h_block_output = resp_hs[l + 1][:r_max].astype(np.float32)
        delta_h = h_block_output - h_block_input
        scores[l] = compute_icr_score(
            response_attn=resp_attn[l].astype(np.float32),
            h_block_input=h_block_input,
            delta_h=delta_h,
            response_len=min(response_len, r_max),
            top_p=top_p,
        )
    return scores


def load_model_eager(model_name: str):
    """Load tokenizer and model in eager attention mode (required for output_attentions)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model (eager, fp16): %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_prompt(sample: dict, task_module: Any) -> str:
    """Build the prompt string for a sample using task_module.format_prompt if available."""
    return _prompt_default(task_module, sample)


def build_writer_config(model: Any, args: argparse.Namespace) -> dict:
    """Assemble config.json dict from loaded model and CLI args."""
    import torch

    cfg = model.config
    # num_layers: number of transformer blocks (not counting the embedding)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        # Fallback for model configs that use a different attribute name
        num_layers = getattr(cfg, "n_layer", None)
    hidden_dim = getattr(cfg, "hidden_size", None)

    return {
        "model_name": args.model,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "r_max": args.r_max,
        "dtype": "float16",
        "response_logprobs_top_k": args.top_k,
        "max_prompt_len": args.max_prompt_len,
        "max_response_len": args.max_response_len,
    }


# ---------------------------------------------------------------------------
# Step: capture (batched helper)
# ---------------------------------------------------------------------------

def _run_batch(
    pending: list,
    tokenizer: Any,
    model: Any,
    task_module: Any,
    is_correct_adapter: Any,
    writer: Any,
    args: argparse.Namespace,
) -> None:
    """Tokenize, generate, stitch, and write one batch of pending samples."""
    import torch
    from activation_logging.generate_capture import (
        extract_logprobs_batched,
        stitch_prompt_hidden_states_batched,
        stitch_response_hidden_states_batched,
        stitch_response_to_response_batched,
    )

    assert tokenizer.pad_token is not None, (
        "tokenizer.pad_token must be set before batched generate"
    )
    tokenizer.padding_side = "left"

    prompts = [s["prompt"] for s in pending]
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_len,
        return_tensors="pt",
    ).to(model.device)

    prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()  # (B,)
    padded_prompt_len = batch.input_ids.shape[1]
    B = len(pending)

    with torch.no_grad():
        out = model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=args.max_response_len,
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
        eos_positions = (resp_b == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            response_lens[b] = int(eos_positions[0].item()) + 1
        else:
            response_lens[b] = int(resp_b.shape[0])

    resp_attn = stitch_response_to_response_batched(
        out.attentions, prompt_lens, response_lens, args.r_max
    )
    resp_hs = stitch_response_hidden_states_batched(
        out.hidden_states, prompt_lens, response_lens, args.max_response_len
    )
    prompt_hs = stitch_prompt_hidden_states_batched(
        out.hidden_states, prompt_lens, args.max_prompt_len
    )
    token_lp, topk_ids, topk_lp = extract_logprobs_batched(
        out.scores, out.sequences, prompt_lens, response_lens, args.top_k
    )

    for b in range(B):
        s = pending[b]
        sample = s["sample"]
        resp_ids = out.sequences[b, padded_prompt_len:padded_prompt_len + int(response_lens[b])]
        decoded = tokenizer.decode(resp_ids, skip_special_tokens=True)
        hallucinated = not is_correct_adapter(task_module, decoded, sample)

        icr_per_layer = compute_icr_per_layer(
            resp_attn[b], resp_hs[b], int(response_lens[b]), top_p=0.1
        )

        p_len = int(prompt_lens[b])
        real_token_start = padded_prompt_len - p_len
        prompt_ids_np = pad_to(
            batch.input_ids[b, real_token_start:].cpu().numpy(),
            args.max_prompt_len, -1
        ).astype(np.int32)
        response_ids_np = pad_to(
            resp_ids.cpu().numpy(), args.max_response_len, -1
        ).astype(np.int32)
        token_lp_np = pad_to(token_lp[b], args.max_response_len, np.nan).astype(np.float32)
        topk_ids_np = pad_2d(topk_ids[b], args.max_response_len, args.top_k, -1).astype(np.int32)
        topk_lp_np = pad_2d(topk_lp[b], args.max_response_len, args.top_k, np.nan).astype(np.float32)

        passthrough = {
            k: v for k, v in sample.items()
            if k not in ("question", "answer", "possible_answers")
        }
        answer_raw = sample.get("answer", sample.get("possible_answers", ""))
        answer_str = json.dumps(answer_raw) if isinstance(answer_raw, list) else str(answer_raw)

        gen_record = {
            "prompt": s["prompt"],
            "generation": decoded,
            "answer": answer_str,
            "question": sample.get("question", ""),
            "hallucinated": hallucinated,
            "prompt_hash": s["prompt_hash"],
            "sample_index": s["sample_index"],
            **passthrough,
        }

        writer.append(
            sample_index=s["sample_index"],
            prompt_hash=s["prompt_hash"],
            key=str(sample.get("id", s["dataset_index"])),
            prompt_len=p_len,
            response_len=int(response_lens[b]),
            prompt_activations=prompt_hs[b],
            response_activations=resp_hs[b],
            response_attention=resp_attn[b],
            prompt_token_ids=prompt_ids_np,
            response_token_ids=response_ids_np,
            response_token_logprobs=token_lp_np,
            response_topk_token_ids=topk_ids_np,
            response_topk_logprobs=topk_lp_np,
            icr_score_per_layer=icr_per_layer,
            hallucinated=hallucinated,
            generation_record=gen_record,
        )

        logger.info(
            "  [batch] sample_index=%d  response_len=%d  hallucinated=%s  icr_mean=%.4f",
            s["sample_index"], int(response_lens[b]), hallucinated, float(icr_per_layer.mean()),
        )


# ---------------------------------------------------------------------------
# Step: capture
# ---------------------------------------------------------------------------

def _run_capture(args: argparse.Namespace) -> int:
    import torch
    from activation_logging.generate_capture import (
        extract_logprobs,
        stitch_prompt_hidden_states,
        stitch_response_hidden_states,
        stitch_response_to_response,
    )
    from activation_logging.inference_capture_writer import InferenceCaptureWriter

    # Resolve task module and loader
    if args.task not in _TASK_REGISTRY:
        logger.error("Unknown task '%s'. Known tasks: %s", args.task, list(_TASK_REGISTRY))
        return 1

    load_fn_name, is_correct_adapter, default_split = _TASK_REGISTRY[args.task]
    split = args.split if args.split else default_split

    module_path = f"tasks.llmsknow.{args.task}"
    logger.info("Importing task module: %s", module_path)
    task_module = importlib.import_module(module_path)
    load_fn = getattr(task_module, load_fn_name)

    logger.info("Loading dataset: task=%s split=%s n_samples=%s", args.task, split, args.n_samples)
    load_kwargs: dict[str, Any] = {"split": split}
    if args.n_samples is not None:
        load_kwargs["n_samples"] = args.n_samples
    dataset = load_fn(**load_kwargs)
    logger.info("Dataset size: %d", len(dataset))

    # Load model
    tokenizer, model = load_model_eager(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine resume mode
    out_path = Path(args.out_dir)
    config_path = out_path / "config.json"
    writer_mode = "a" if config_path.exists() else "w"

    config_dict = build_writer_config(model, args)
    logger.info("Writer mode: %s  out_dir: %s  batch_size: %d", writer_mode, args.out_dir, args.batch_size)

    batch_size = args.batch_size

    with InferenceCaptureWriter(
        args.out_dir,
        mode=writer_mode,
        n_samples=len(dataset),
        config_dict=config_dict,
    ) as writer:
        if batch_size <= 1:
            for i, sample in enumerate(dataset):
                prompt = build_prompt(sample, task_module)
                p_hash = sha256(prompt)

                if writer.is_written(p_hash):
                    logger.debug("Skipping already-written sample %d (hash=%s)", i, p_hash[:8])
                    continue

                sample_index = writer.next_index()
                logger.info(
                    "Sample %d/%d  index=%d  task=%s",
                    i + 1, len(dataset), sample_index, args.task,
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                prompt_len = inputs.input_ids.shape[1]

                with torch.no_grad():
                    out = model.generate(
                        inputs.input_ids,
                        output_attentions=True,
                        output_hidden_states=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=args.max_response_len,
                        do_sample=False,
                    )

                response_ids = out.sequences[0][prompt_len:]
                response_len = response_ids.shape[0]
                decoded = tokenizer.decode(response_ids, skip_special_tokens=True)

                hallucinated = not is_correct_adapter(task_module, decoded, sample)

                resp_attn = stitch_response_to_response(
                    out.attentions, prompt_len, args.r_max, response_len
                )
                resp_hs = stitch_response_hidden_states(
                    out.hidden_states, prompt_len, args.max_response_len
                )
                prompt_hs = stitch_prompt_hidden_states(
                    out.hidden_states, prompt_len, args.max_prompt_len
                )

                token_lp, topk_ids, topk_lp = extract_logprobs(
                    out.scores, response_ids, top_k=args.top_k
                )

                icr_per_layer = compute_icr_per_layer(
                    resp_attn, resp_hs, int(response_len), top_p=0.1
                )

                prompt_ids_np = pad_to(
                    inputs.input_ids[0].cpu().numpy(), args.max_prompt_len, -1
                ).astype(np.int32)
                response_ids_np = pad_to(
                    response_ids.cpu().numpy(), args.max_response_len, -1
                ).astype(np.int32)
                token_lp_np = pad_to(token_lp, args.max_response_len, np.nan).astype(np.float32)
                topk_ids_np = pad_2d(topk_ids, args.max_response_len, args.top_k, -1).astype(np.int32)
                topk_lp_np = pad_2d(topk_lp, args.max_response_len, args.top_k, np.nan).astype(np.float32)

                passthrough = {
                    k: v for k, v in sample.items()
                    if k not in ("question", "answer", "possible_answers")
                }

                answer_raw = sample.get("answer", sample.get("possible_answers", ""))
                answer_str = json.dumps(answer_raw) if isinstance(answer_raw, list) else str(answer_raw)

                gen_record = {
                    "prompt": prompt,
                    "generation": decoded,
                    "answer": answer_str,
                    "question": sample.get("question", ""),
                    "hallucinated": hallucinated,
                    "prompt_hash": p_hash,
                    "sample_index": sample_index,
                    **passthrough,
                }

                writer.append(
                    sample_index=sample_index,
                    prompt_hash=p_hash,
                    key=str(sample.get("id", i)),
                    prompt_len=prompt_len,
                    response_len=int(response_len),
                    prompt_activations=prompt_hs,
                    response_activations=resp_hs,
                    response_attention=resp_attn,
                    prompt_token_ids=prompt_ids_np,
                    response_token_ids=response_ids_np,
                    response_token_logprobs=token_lp_np,
                    response_topk_token_ids=topk_ids_np,
                    response_topk_logprobs=topk_lp_np,
                    icr_score_per_layer=icr_per_layer,
                    hallucinated=hallucinated,
                    generation_record=gen_record,
                )

                logger.info(
                    "  response_len=%d  hallucinated=%s  icr_mean=%.4f",
                    response_len, hallucinated, float(icr_per_layer.mean()),
                )
        else:
            pending_samples: list[dict] = []
            for i, sample in enumerate(dataset):
                prompt = build_prompt(sample, task_module)
                p_hash = sha256(prompt)

                if writer.is_written(p_hash):
                    logger.debug("Skipping already-written sample %d (hash=%s)", i, p_hash[:8])
                    continue

                sample_index = writer.next_index()
                logger.info(
                    "Sample %d/%d  index=%d  task=%s (pending batch)",
                    i + 1, len(dataset), sample_index, args.task,
                )
                pending_samples.append({
                    "dataset_index": i,
                    "sample": sample,
                    "prompt": prompt,
                    "prompt_hash": p_hash,
                    "sample_index": sample_index,
                })

                if len(pending_samples) == batch_size:
                    _run_batch(pending_samples, tokenizer, model, task_module,
                               is_correct_adapter, writer, args)
                    pending_samples = []

            if pending_samples:
                _run_batch(pending_samples, tokenizer, model, task_module,
                           is_correct_adapter, writer, args)

    logger.info("Capture complete. Output dir: %s", args.out_dir)
    return 0


# ---------------------------------------------------------------------------
# Step: eval-only
# ---------------------------------------------------------------------------

def _run_eval_only(args: argparse.Namespace) -> int:
    """Re-run is_correct over an existing generation.jsonl without GPU.

    Rewrites eval_results.json and patches the hallucinated field in meta.jsonl
    for any rows where the evaluator result has changed. Useful when the
    evaluator logic changes and data needs re-labelling without re-inference.
    """
    if args.task not in _TASK_REGISTRY:
        logger.error("Unknown task '%s'. Known tasks: %s", args.task, list(_TASK_REGISTRY))
        return 1

    _, is_correct_adapter, _ = _TASK_REGISTRY[args.task]
    module_path = f"tasks.llmsknow.{args.task}"
    task_module = importlib.import_module(module_path)

    out_path = Path(args.out_dir)
    gen_path = out_path / "generation.jsonl"
    eval_path = out_path / "eval_results.json"

    if not gen_path.exists():
        logger.error("generation.jsonl not found at %s", gen_path)
        return 1

    logger.info("Reading generation.jsonl: %s", gen_path)
    records = []
    with gen_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        logger.error("generation.jsonl is empty")
        return 1

    # Re-evaluate — build a synthetic sample dict from the stored generation record
    # so the is_correct adapter can extract answer / possible_answers uniformly.
    halu_test_res = []
    for rec in records:
        generation = rec.get("generation", "")
        # Reconstruct the answer field the adapter expects
        answer_raw = rec.get("answer", "")
        # popqa answers are stored as JSON-encoded lists
        if args.task == "popqa":
            try:
                answer_raw = json.loads(answer_raw)
            except (json.JSONDecodeError, TypeError):
                pass
            sample_proxy = {"possible_answers": answer_raw}
        else:
            sample_proxy = {"answer": answer_raw}

        correct = is_correct_adapter(task_module, generation, sample_proxy)
        halu_test_res.append(not correct)

    # Build eval_results.json in the array format that ActivationParser expects
    result = {
        "halu_test_res": halu_test_res,
        "abstantion": [False] * len(halu_test_res),
    }
    eval_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "Wrote eval_results.json: %d samples, %d hallucinated",
        len(halu_test_res), sum(halu_test_res),
    )

    # Patch meta.jsonl if it exists — rewrite hallucinated field in place
    meta_path = out_path / "meta.jsonl"
    if meta_path.exists():
        meta_lines = []
        with meta_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta_obj = json.loads(line)
                idx = meta_obj.get("sample_index")
                if idx is not None and 0 <= idx < len(halu_test_res):
                    meta_obj["hallucinated"] = halu_test_res[idx]
                meta_lines.append(json.dumps(meta_obj))
        meta_path.write_text("\n".join(meta_lines) + "\n")
        logger.info("Patched meta.jsonl: %d lines", len(meta_lines))

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Single-process inference capture for ICR Probe data pipeline (Issue #72)."
    )
    parser.add_argument("--task", required=True,
                        choices=list(_TASK_REGISTRY),
                        help="LLMsKnow task name.")
    parser.add_argument("--split", default=None,
                        help="Dataset split (defaults per task: hotpotqa→validation, mmlu→test, etc.).")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory (shared/icr_capture/<dataset>_<model_slug>/).")
    parser.add_argument("--max-prompt-len", type=int, default=512,
                        help="Max prompt length in tokens (padding / truncation target).")
    parser.add_argument("--max-response-len", type=int, default=64,
                        help="Max new tokens to generate; also response storage width. "
                             "Default 64 matches r_max — ICR scoring only consumes the "
                             "first r_max response positions (cross-region masking per "
                             "icr_probe_paper_notes.md §9), so generating past that is "
                             "wasted GPU time.")
    parser.add_argument("--r-max", type=int, default=64,
                        help="Max response-to-response attention window (r_max × r_max per layer).")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top-k alternative tokens to store per position.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Cap on samples (omit for full split).")
    parser.add_argument("--step", choices=("capture", "eval-only"), default="capture",
                        help="capture (default): full GPU pipeline. "
                             "eval-only: re-run is_correct over existing generation.jsonl.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of samples per generate() call. B=1 uses the B=1 path "
                             "(no left-padding). B>1 uses batched stitching primitives. "
                             "Llama-3.1-8B at max_prompt=512, max_response=64: B=4 fits "
                             "comfortably on H100 80GB.")
    args = parser.parse_args()

    if args.step == "capture":
        return _run_capture(args)
    else:
        return _run_eval_only(args)


if __name__ == "__main__":
    raise SystemExit(main())
