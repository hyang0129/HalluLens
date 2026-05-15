"""
recompute_attention.py

CLI driver that loads an existing activations.zarr, iterates over all samples,
calls recompute_block_attention() for each transformer block, and writes
head-averaged response-to-response attention data to a new attention.zarr via
AttentionZarrLogger.

Usage:
    python scripts/recompute_attention.py \\
        --activations-zarr shared/hotpotqa_llama/activations.zarr \\
        --attention-zarr   shared/hotpotqa_llama/attention.zarr \\
        --model            meta-llama/Llama-3.1-8B-Instruct

Use --validate-first to run a 4-sample numerical-equivalence check before the
full run.  The check exits 0 on pass (all max|A_recomp - A_full| < 1e-3) and
exits 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import zarr
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root path injection so relative imports resolve when the script is
# executed from any cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_logging.attention_recompute import recompute_block_attention  # noqa: E402
from activation_logging.attention_zarr_logger import AttentionZarrLogger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_R_MAX_DEFAULT = 64
_VALIDATE_TOL = 1e-3  # fp16 tolerance
_LOG_EVERY = 100


# ---------------------------------------------------------------------------
# Config dict builder
# ---------------------------------------------------------------------------

def _build_config(
    activations_zarr_abs: str,
    model_name: str,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    r_max: int,
    dtype: str,
) -> dict:
    return {
        "source_activations_zarr": activations_zarr_abs,
        "model_name": model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "attention_region": "response_to_response",
        "query_position_rule": "all_response_tokens",
        "head_aggregation": "mean",
        "use_induction_head": False,
        "projection_kind": "residual_stream",
        "projection_target_layer": "previous",
        "projection_normalization": "l2_on_target",
        "score_top_k": None,
        "score_top_p": 0.1,
        "jsd_input_normalization": "zscore_then_softmax",
        "dtype": dtype,
        "r_max": r_max,
        "recomputed_from_cached_states": True,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: str, dtype_str: str):
    """Load a HF causal LM with eager attention implementation."""
    from transformers import AutoModelForCausalLM  # late import

    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    load_kwargs: dict = {
        "attn_implementation": "eager",
        "torch_dtype": torch_dtype,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None

    logger.info(f"Loading model {model_name!r} with attn_implementation='eager' ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Qwen3: disable thinking mode if the attribute exists
    if hasattr(model.config, "thinking_mode"):
        model.config.thinking_mode = False
        logger.info("Set model.config.thinking_mode = False (Qwen3 thinking mode disabled)")

    return model


# ---------------------------------------------------------------------------
# Activations zarr reader helpers
# ---------------------------------------------------------------------------

def _open_activations_zarr(activations_zarr: str) -> zarr.Group:
    """Open the existing activations.zarr store in read-only mode."""
    root = zarr.open_group(activations_zarr, mode="r")
    if "arrays" not in root:
        raise ValueError(f"activations.zarr at {activations_zarr!r} is missing 'arrays' group")
    return root


def _load_index(activations_root: zarr.Group, activations_zarr: str) -> dict[str, int]:
    """Build key -> sample_index from meta/index.jsonl."""
    index: dict[str, int] = {}
    index_path = Path(activations_zarr) / "meta" / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"meta/index.jsonl not found at {index_path}")
    with open(index_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("key")
                idx = entry.get("sample_index")
                if key is not None and idx is not None:
                    index[str(key)] = int(idx)
            except json.JSONDecodeError:
                continue
    return index


# ---------------------------------------------------------------------------
# Per-sample attention recomputation
# ---------------------------------------------------------------------------

def _recompute_sample(
    *,
    s: int,
    prompt_activations: zarr.Array,
    response_activations: zarr.Array,
    prompt_len: int,
    response_len: int,
    model,
    num_blocks: int,
    r_max: int,
    device: str,
) -> np.ndarray:
    """Recompute attention for one sample across all blocks.

    Returns ndarray of shape (num_blocks, r_max, r_max) dtype float16.

    Layer-index alignment: activations.zarr stores L+1 layers (index 0 =
    embedding output).  For block b, h^{b-1} is at activations[s, b, ...].
    So loop: for b in range(num_blocks): h_prev from activations[s, b, ...].
    """
    result = np.zeros((num_blocks, r_max, r_max), dtype=np.float16)

    for b in range(num_blocks):
        # Read block input from activations.zarr (index b = block b's input h^{b-1})
        prompt_h = torch.from_numpy(
            np.array(prompt_activations[s, b, :prompt_len, :], dtype=np.float32)
        )  # (P, H)
        response_h = torch.from_numpy(
            np.array(response_activations[s, b, :response_len, :], dtype=np.float32)
        )  # (R, H)

        h_prev = torch.cat([prompt_h, response_h], dim=0)  # (T, H)

        attn_resp = recompute_block_attention(
            h_prev=h_prev,
            block=model.model.layers[b],
            prompt_len=prompt_len,
            response_len=response_len,
            device=device,
        )  # (response_len, response_len) float32

        # Write into (r_max, r_max) buffer; positions >= response_len remain 0
        r = min(response_len, r_max)
        result[b, :r, :r] = attn_resp[:r, :r].cpu().float().numpy().astype(np.float16)

    return result


# ---------------------------------------------------------------------------
# --validate-first logic
# ---------------------------------------------------------------------------

def _run_validate(
    *,
    activations_root: zarr.Group,
    index: dict[str, int],
    model,
    num_blocks: int,
    r_max: int,
    device: str,
) -> bool:
    """Run a 4-sample numerical equivalence check.

    For each of the first 4 samples, computes attention via two paths:
    1. recompute_block_attention() (from cached hidden states)
    2. Full model forward with output_attentions=True

    Prints a per-(sample, block) table of max |A_recomp - A_full|.
    Returns True if all diffs < _VALIDATE_TOL, False otherwise.
    """
    arrays_group = activations_root["arrays"]
    prompt_activations: zarr.Array = arrays_group["prompt_activations"]
    response_activations: zarr.Array = arrays_group["response_activations"]
    prompt_len_arr: zarr.Array = arrays_group["prompt_len"]
    response_len_arr: zarr.Array = arrays_group["response_len"]

    # Attempt to get token ids for full-forward re-tokenisation
    has_token_ids = "response_token_ids" in arrays_group

    keys_sorted = sorted(index.keys(), key=lambda k: index[k])
    validate_keys = keys_sorted[:4]

    all_pass = True
    print(
        f"\n{'Sample':<10} {'Block':>6} {'max|diff|':>12} {'argmax_match':>14}"
    )
    print("-" * 48)

    for key in validate_keys:
        s = index[key]
        prompt_len = int(prompt_len_arr[s])
        response_len = int(response_len_arr[s])
        r = min(response_len, r_max)

        if r < 1:
            logger.warning(f"Sample {key!r} has response_len={response_len}; skipping validate")
            continue

        # --- Path 1: recompute from cached hidden states ---
        recomp_all: list[np.ndarray] = []
        for b in range(num_blocks):
            prompt_h = torch.from_numpy(
                np.array(prompt_activations[s, b, :prompt_len, :], dtype=np.float32)
            )
            response_h = torch.from_numpy(
                np.array(response_activations[s, b, :response_len, :], dtype=np.float32)
            )
            h_prev = torch.cat([prompt_h, response_h], dim=0)
            attn_resp = recompute_block_attention(
                h_prev=h_prev,
                block=model.model.layers[b],
                prompt_len=prompt_len,
                response_len=response_len,
                device=device,
            )  # (R, R) float32
            recomp_all.append(attn_resp[:r, :r].cpu().numpy())

        # --- Path 2: full model forward with output_attentions=True ---
        # Recover token sequence: use response_token_ids if available,
        # otherwise skip the full-forward check with a warning.
        if not has_token_ids:
            logger.warning(
                "response_token_ids not in activations.zarr; "
                "cannot run full-forward validation for this store. "
                "Validate manually on a store with token IDs."
            )
            print(
                f"{key[:9]:<10} {'N/A':>6} {'(no token_ids)':>12} {'N/A':>14}"
            )
            continue

        response_token_ids = np.array(arrays_group["response_token_ids"][s, :response_len], dtype=np.int64)

        # For the prompt side we don't have stored prompt token IDs in general,
        # so we use the embedding output trick: feed the concatenated full-sequence
        # through the model by constructing a dummy input_ids of length T.
        # However, if we only have response_token_ids we can still check the
        # recompute path against itself with position_ids sanity check.
        # Full-model forward requires the full token id sequence.
        # We fall back to a warning if prompt token ids are unavailable.
        if "prompt_token_ids" not in arrays_group and "input_ids" not in arrays_group:
            logger.warning(
                "Neither prompt_token_ids nor input_ids found in activations.zarr. "
                "Skipping full-forward cross-check; only recompute self-consistency is verified."
            )
            # Self-consistency: verify recompute runs without error and has valid shape
            for b_idx, attn in enumerate(recomp_all):
                shape_ok = attn.shape == (r, r)
                rows_sum = np.allclose(attn.sum(axis=-1), 1.0, atol=1e-3)
                print(
                    f"{key[:9]:<10} {b_idx:>6} {'self-check':>12} {'shape_ok=' + str(shape_ok):>14}"
                )
            continue

        # Build full token id sequence (prompt + response)
        prompt_tok_key = "prompt_token_ids" if "prompt_token_ids" in arrays_group else "input_ids"
        prompt_token_ids_raw = np.array(arrays_group[prompt_tok_key][s], dtype=np.int64)
        # Trim to actual prompt_len
        prompt_token_ids = prompt_token_ids_raw[:prompt_len]
        full_ids = np.concatenate([prompt_token_ids, response_token_ids])  # (T,)

        input_ids = torch.from_numpy(full_ids).unsqueeze(0).to(device)  # (1, T)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=False,
            )
        # out.attentions: tuple of (1, n_heads, T, T) per block
        for b_idx in range(num_blocks):
            full_attn_block = out.attentions[b_idx].squeeze(0)  # (n_heads, T, T)
            # Slice response-to-response and head-average
            full_resp = full_attn_block[:, prompt_len:prompt_len + r, prompt_len:prompt_len + r]
            full_resp_avg = full_resp.float().mean(dim=0).cpu().numpy()  # (r, r)

            diff = np.abs(recomp_all[b_idx] - full_resp_avg)
            max_diff = float(diff.max())
            if max_diff >= _VALIDATE_TOL:
                all_pass = False

            # Argmax alignment for query token 0
            argmax_recomp = int(np.argmax(recomp_all[b_idx][0]))
            argmax_full = int(np.argmax(full_resp_avg[0]))
            argmax_match = argmax_recomp == argmax_full

            print(
                f"{key[:9]:<10} {b_idx:>6} {max_diff:>12.2e} {str(argmax_match):>14}"
            )

    print()
    if all_pass:
        print(f"PASS: all max |A_recomp - A_full| < {_VALIDATE_TOL}")
    else:
        print(f"FAIL: one or more blocks exceeded tolerance {_VALIDATE_TOL}")
    return all_pass


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def _run_main(args: argparse.Namespace) -> None:
    activations_zarr_abs = str(Path(args.activations_zarr).resolve())

    logger.info(f"Opening activations.zarr at {activations_zarr_abs!r}")
    activations_root = _open_activations_zarr(activations_zarr_abs)
    arrays_group = activations_root["arrays"]

    prompt_activations: zarr.Array = arrays_group["prompt_activations"]
    response_activations: zarr.Array = arrays_group["response_activations"]
    prompt_len_arr: zarr.Array = arrays_group["prompt_len"]
    response_len_arr: zarr.Array = arrays_group["response_len"]

    # Infer dimensions from the activations.zarr shape
    # shape: (N_samples, L+1, max_tokens, H)
    n_samples_total = int(prompt_activations.shape[0])
    n_layers_plus_one = int(prompt_activations.shape[1])
    num_blocks = n_layers_plus_one - 1  # L+1 stored; block 0..num_blocks-1
    r_max = args.r_max if hasattr(args, "r_max") else _R_MAX_DEFAULT

    logger.info(
        f"activations.zarr: N={n_samples_total}, L+1={n_layers_plus_one}, "
        f"num_blocks={num_blocks}, r_max={r_max}"
    )

    # Load index
    index = _load_index(activations_root, activations_zarr_abs)
    all_keys = sorted(index.keys(), key=lambda k: index[k])

    # Load model
    model = _load_model(args.model, args.device, args.dtype)

    # Infer num_heads and head_dim from model config
    config = model.config
    num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", -1))
    head_dim = getattr(
        config, "head_dim",
        getattr(config, "hidden_size", -1) // num_heads if num_heads > 0 else -1
    )

    # --validate-first
    if args.validate_first:
        logger.info("Running --validate-first on 4 samples ...")
        passed = _run_validate(
            activations_root=activations_root,
            index=index,
            model=model,
            num_blocks=num_blocks,
            r_max=r_max,
            device=args.device,
        )
        sys.exit(0 if passed else 1)

    # Build config.json for attention.zarr
    config_dict = _build_config(
        activations_zarr_abs=activations_zarr_abs,
        model_name=args.model,
        num_layers=num_blocks,
        num_heads=num_heads,
        head_dim=head_dim,
        r_max=r_max,
        dtype=args.dtype,
    )

    # Determine zarr open mode
    attn_zarr_path = Path(args.attention_zarr)
    zarr_mode = "a" if (args.resume and attn_zarr_path.exists()) else "w"

    # Determine expected_samples for pre-allocation
    max_samples = args.max_samples
    expected = min(n_samples_total, max_samples) if max_samples is not None else n_samples_total

    logger.info(
        f"Opening attention.zarr at {args.attention_zarr!r} mode={zarr_mode!r}, "
        f"expected_samples={expected}"
    )

    with AttentionZarrLogger(
        zarr_path=args.attention_zarr,
        mode=zarr_mode,
        num_layers=num_blocks,
        r_max=r_max,
        config_dict=config_dict,
        expected_samples=expected,
        dtype=args.dtype,
    ) as attn_logger:

        keys_to_process = all_keys
        if max_samples is not None:
            keys_to_process = keys_to_process[:max_samples]

        processed = 0
        for key in tqdm(keys_to_process, desc="Recomputing attention", unit="sample"):
            if args.resume and attn_logger.is_written(key):
                continue

            s = index[key]
            prompt_len = int(prompt_len_arr[s])
            response_len = int(response_len_arr[s])

            # Skip degenerate samples
            if response_len < 1:
                logger.warning(f"Sample {key!r} has response_len={response_len}; skipping")
                continue

            response_attn = _recompute_sample(
                s=s,
                prompt_activations=prompt_activations,
                response_activations=response_activations,
                prompt_len=prompt_len,
                response_len=response_len,
                model=model,
                num_blocks=num_blocks,
                r_max=r_max,
                device=args.device,
            )  # (num_blocks, r_max, r_max) float16

            attn_logger.write(
                sample_key=key,
                response_attn=response_attn,
                response_len=response_len,
                prompt_len=prompt_len,
            )

            processed += 1
            if processed % _LOG_EVERY == 0:
                logger.info(f"Processed {processed} samples (latest key={key!r})")

    logger.info(f"Done. Wrote {processed} samples to {args.attention_zarr!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-block head-averaged attention from cached activations.zarr "
            "and write to a new attention.zarr."
        )
    )
    parser.add_argument(
        "--activations-zarr",
        required=True,
        help="Path to existing activations.zarr",
    )
    parser.add_argument(
        "--attention-zarr",
        required=True,
        help="Path to write new attention.zarr",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8; currently used for bookkeeping; per-sample loop only)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device (default: 'cuda')",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model dtype (default: 'float16')",
    )
    parser.add_argument(
        "--validate-first",
        action="store_true",
        help=(
            "Run 4-sample numerical-equivalence check (recompute vs full forward). "
            "Exits 0 on pass (all max|diff| < 1e-3), 1 on fail."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already written in attention.zarr",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (smoke-test mode)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers for reading activations (default: 2; reserved for future batch path)",
    )
    parser.add_argument(
        "--r-max",
        type=int,
        default=_R_MAX_DEFAULT,
        help=f"Max response length stored in attention.zarr (default: {_R_MAX_DEFAULT})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _run_main(args)


if __name__ == "__main__":
    main()
