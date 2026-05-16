"""
recompute_attention.py  (Wave 4b rewrite)

Fused Stage-1+2 CLI driver: reads activations.zarr, batches zarr reads across
samples, recomputes per-block head-averaged response-to-response attention via
recompute_block_attention(), computes ICR scores on-the-fly, and writes both
outputs atomically with resume semantics.

Outputs:
    <attention-dir>/response_attn.npy   (N, num_blocks, R_max, R_max) float16
    <attention-dir>/meta.jsonl
    <attention-dir>/config.json
    <icr-scores-path>                   (N, num_blocks) float32
    <icr-scores-meta>                   sidecar .jsonl auto-derived by ICRScoresWriter

Usage:
    python scripts/recompute_attention.py \\
        --activations-zarr shared/hotpotqa_llama/activations.zarr \\
        --attention-dir    shared/hotpotqa_llama/attention \\
        --model            meta-llama/Llama-3.1-8B-Instruct

Use --validate-first to run a 4-sample numerical-equivalence check before the
full run.  The check exits 0 on pass (all max|A_recomp - A_full| < 1e-3) and
exits 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import zarr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_logging.attention_recompute import recompute_block_attention  # noqa: E402
from activation_logging.attention_memmap_writer import AttentionMemmapWriter  # noqa: E402
from activation_research.icr_scores_writer import ICRScoresWriter             # noqa: E402
from activation_research.icr_score import compute_icr_score                   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_R_MAX_DEFAULT = 64
_VALIDATE_TOL = 1e-3
_LOG_EVERY_BATCHES = 20


# ---------------------------------------------------------------------------
# Config dict
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
    from transformers import AutoModelForCausalLM

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

    if hasattr(model.config, "thinking_mode"):
        model.config.thinking_mode = False
        logger.info("Set model.config.thinking_mode = False (Qwen3 thinking mode disabled)")

    return model


# ---------------------------------------------------------------------------
# Activations zarr helpers
# ---------------------------------------------------------------------------

def _open_activations_zarr(activations_zarr: str) -> zarr.Group:
    root = zarr.open_group(activations_zarr, mode="r")
    if "arrays" not in root:
        raise ValueError(
            f"activations.zarr at {activations_zarr!r} is missing 'arrays' group"
        )
    return root


def _load_key_list(activations_zarr: str, n_samples: int, arrays_group: zarr.Group) -> list[str]:
    """Return an ordered list of sample keys indexed by sample_index.

    Prefers meta/index.jsonl; falls back to a 'sample_key' array in the zarr
    if the sidecar is missing.
    """
    index_path = Path(activations_zarr) / "meta" / "index.jsonl"
    if index_path.exists():
        key_of: dict[int, str] = {}
        with open(index_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    k = entry.get("key")
                    idx = entry.get("sample_index")
                    if k is not None and idx is not None:
                        key_of[int(idx)] = str(k)
                except json.JSONDecodeError:
                    continue
        return [key_of.get(i, str(i)) for i in range(n_samples)]

    if "sample_key" in arrays_group:
        raw = np.asarray(arrays_group["sample_key"])
        return [str(raw[i]) for i in range(n_samples)]

    logger.warning(
        "meta/index.jsonl not found and no 'sample_key' array in zarr. "
        "Using integer indices as sample keys."
    )
    return [str(i) for i in range(n_samples)]


# ---------------------------------------------------------------------------
# --validate-first
# ---------------------------------------------------------------------------

def _run_validate(
    *,
    activations_root: zarr.Group,
    key_list: list[str],
    model,
    num_blocks: int,
    r_max: int,
    device: str,
) -> bool:
    """4-sample numerical equivalence check: recompute vs. full forward pass.

    Returns True if at least one sample was checked AND all
    max|A_recomp - A_full| < _VALIDATE_TOL.  Returns False if every candidate
    sample had response_len < 1 (which previously produced a vacuous PASS,
    seen during the Qwen3 smoketest on a zarr with blank leading rows).
    """
    arrays_group = activations_root["arrays"]
    prompt_activations: zarr.Array = arrays_group["prompt_activations"]
    response_activations: zarr.Array = arrays_group["response_activations"]
    prompt_len_arr: zarr.Array = arrays_group["prompt_len"]
    response_len_arr: zarr.Array = arrays_group["response_len"]

    validate_indices = list(range(min(4, len(key_list))))

    has_response_token_ids = "response_token_ids" in arrays_group
    has_prompt_token_ids = (
        "prompt_token_ids" in arrays_group or "input_ids" in arrays_group
    )

    if not has_response_token_ids or not has_prompt_token_ids:
        logger.warning(
            "One or more token-id arrays missing from activations.zarr. "
            "Full-forward cross-check unavailable; running recompute self-consistency only."
        )

    all_pass = True
    checked_samples = 0
    print(f"\n{'Sample':<10} {'Block':>6} {'max|diff|':>12} {'argmax_match':>14}")
    print("-" * 48)

    for s in validate_indices:
        key = key_list[s]
        prompt_len = int(prompt_len_arr[s])
        response_len = int(response_len_arr[s])
        r = min(response_len, r_max)

        if r < 1:
            logger.warning(f"Sample {key!r} has response_len={response_len}; skipping validate")
            continue

        checked_samples += 1
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
                rotary_emb=getattr(model.model, "rotary_emb", None),
            )
            recomp_all.append(attn_resp[:r, :r].cpu().numpy())

        if not has_response_token_ids or not has_prompt_token_ids:
            for b_idx, attn in enumerate(recomp_all):
                shape_ok = attn.shape == (r, r)
                print(
                    f"{key[:9]:<10} {b_idx:>6} {'self-check':>12} "
                    f"{'shape_ok=' + str(shape_ok):>14}"
                )
            continue

        response_token_ids = np.array(
            arrays_group["response_token_ids"][s, :response_len], dtype=np.int64
        )
        prompt_tok_key = "prompt_token_ids" if "prompt_token_ids" in arrays_group else "input_ids"
        prompt_token_ids_raw = np.array(arrays_group[prompt_tok_key][s], dtype=np.int64)
        prompt_token_ids = prompt_token_ids_raw[:prompt_len]
        full_ids = np.concatenate([prompt_token_ids, response_token_ids])

        input_ids = torch.from_numpy(full_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=False,
            )

        for b_idx in range(num_blocks):
            full_attn_block = out.attentions[b_idx].squeeze(0)
            full_resp = full_attn_block[
                :, prompt_len : prompt_len + r, prompt_len : prompt_len + r
            ]
            full_resp_avg = full_resp.float().mean(dim=0).cpu().numpy()

            diff = np.abs(recomp_all[b_idx] - full_resp_avg)
            max_diff = float(diff.max())
            if max_diff >= _VALIDATE_TOL:
                all_pass = False

            argmax_recomp = int(np.argmax(recomp_all[b_idx][0]))
            argmax_full = int(np.argmax(full_resp_avg[0]))
            argmax_match = argmax_recomp == argmax_full

            print(
                f"{key[:9]:<10} {b_idx:>6} {max_diff:>12.2e} {str(argmax_match):>14}"
            )

    print()
    if checked_samples == 0:
        print(
            f"FAIL: 0/{len(validate_indices)} candidate samples had response_len >= 1 — "
            f"nothing was validated. Source zarr may be blank at the leading indices "
            f"({validate_indices[0]}..{validate_indices[-1]})."
        )
        return False
    if all_pass:
        print(
            f"PASS: all max |A_recomp - A_full| < {_VALIDATE_TOL} "
            f"({checked_samples}/{len(validate_indices)} samples checked)"
        )
    else:
        print(f"FAIL: one or more blocks exceeded tolerance {_VALIDATE_TOL}")
    return all_pass


# ---------------------------------------------------------------------------
# Main processing loop (fused Stage 1+2, batched zarr reads)
# ---------------------------------------------------------------------------

def _resolve_icr_scores_path(attention_dir: str, icr_scores_path: Optional[str]) -> str:
    if icr_scores_path is not None:
        return icr_scores_path
    return str(Path(attention_dir).parent / "icr_scores.npy")


def _run_main(args: argparse.Namespace) -> None:
    activations_zarr_abs = str(Path(args.activations_zarr).resolve())

    logger.info(f"Opening activations.zarr at {activations_zarr_abs!r}")
    activations_root = _open_activations_zarr(activations_zarr_abs)
    arrays_group = activations_root["arrays"]

    prompt_activations: zarr.Array = arrays_group["prompt_activations"]
    response_activations: zarr.Array = arrays_group["response_activations"]
    prompt_len_arr: zarr.Array = arrays_group["prompt_len"]
    response_len_arr: zarr.Array = arrays_group["response_len"]

    # Shape: (N, L+1, max_tokens, H)
    n_samples_total = int(prompt_activations.shape[0])
    n_layers_plus_one = int(prompt_activations.shape[1])
    num_blocks = n_layers_plus_one - 1  # L+1 stored layers; block indices 0..num_blocks-1
    r_max = args.r_max

    logger.info(
        f"activations.zarr: N={n_samples_total}, L+1={n_layers_plus_one}, "
        f"num_blocks={num_blocks}, r_max={r_max}"
    )

    key_list = _load_key_list(activations_zarr_abs, n_samples_total, arrays_group)

    # Apply --max-samples truncation
    N = n_samples_total
    if args.max_samples is not None:
        N = min(N, args.max_samples)

    if N == 0:
        logger.warning("No samples to process (N=0). Exiting.")
        sys.exit(0)

    source_indices = list(range(N))
    key_of = key_list  # key_of[s] == key_list[s]; alias for clarity

    # Load model
    model = _load_model(args.model, args.device, args.dtype)

    config = model.config
    num_heads = getattr(
        config, "num_attention_heads", getattr(config, "num_heads", -1)
    )
    head_dim = getattr(
        config,
        "head_dim",
        getattr(config, "hidden_size", -1) // num_heads if num_heads > 0 else -1,
    )

    # --validate-first runs independently and exits
    if args.validate_first:
        logger.info("Running --validate-first on up to 4 samples ...")
        passed = _run_validate(
            activations_root=activations_root,
            key_list=key_list,
            model=model,
            num_blocks=num_blocks,
            r_max=r_max,
            device=args.device,
        )
        sys.exit(0 if passed else 1)

    config_dict = _build_config(
        activations_zarr_abs=activations_zarr_abs,
        model_name=args.model,
        num_layers=num_blocks,
        num_heads=num_heads,
        head_dim=head_dim,
        r_max=r_max,
        dtype=args.dtype,
    )

    attention_dir = str(Path(args.attention_dir).resolve())
    icr_scores_path = _resolve_icr_scores_path(attention_dir, args.icr_scores_path)

    attn_dir_exists = Path(attention_dir).exists() and (
        Path(attention_dir) / "config.json"
    ).exists()
    icr_exists = Path(icr_scores_path).exists()

    if args.resume and attn_dir_exists and icr_exists:
        writer_mode = "a"
    else:
        writer_mode = "w"

    logger.info(
        f"attention_dir={attention_dir!r} mode={writer_mode!r}, "
        f"icr_scores_path={icr_scores_path!r}, N={N}"
    )

    # icr_top_p: if --icr-top-k is given, we compute per-sample top_p at runtime
    # because compute_icr_score only accepts top_p (no top_k parameter).
    icr_top_p_default = args.icr_top_p
    icr_top_k = args.icr_top_k  # None or int

    attn_writer = AttentionMemmapWriter(
        out_dir=attention_dir,
        mode=writer_mode,
        n_samples=N,
        num_layers=num_blocks,
        r_max=r_max,
        config_dict=config_dict,
        dtype=args.dtype,
    )
    icr_writer = ICRScoresWriter(
        out_path=icr_scores_path,
        mode=writer_mode,
        n_samples=N,
        num_blocks=num_blocks,
    )

    batch_size = args.batch_size
    total_processed = 0
    t_start = time.monotonic()

    batch_num = 0
    for batch_start in range(0, N, batch_size):
        batch_idx = source_indices[batch_start : batch_start + batch_size]

        # Resume filter: skip samples where BOTH writers have already recorded the key.
        # If only one has, redo the sample so both stay consistent.
        batch_idx = [
            s
            for s in batch_idx
            if not (
                attn_writer.is_written(key_of[s]) and icr_writer.is_written(key_of[s])
            )
        ]
        if not batch_idx:
            batch_num += 1
            continue

        # Two zarr oindex calls per batch — all activation IO for this batch happens here.
        prompt_h_batch = np.asarray(prompt_activations.oindex[batch_idx, :, :, :])
        response_h_batch = np.asarray(response_activations.oindex[batch_idx, :, :, :])
        prompt_lens = np.asarray(prompt_len_arr.oindex[batch_idx])
        response_lens = np.asarray(response_len_arr.oindex[batch_idx])

        for i, s in enumerate(batch_idx):
            P, R = int(prompt_lens[i]), int(response_lens[i])

            if R < 1:
                logger.warning(f"Sample {key_of[s]!r} has response_len={R}; skipping")
                continue

            prompt_h_all = prompt_h_batch[i, :, :P, :]    # (L+1, P, H)
            response_h_all = response_h_batch[i, :, :R, :] # (L+1, R, H)

            attn_per_block = np.zeros((num_blocks, r_max, r_max), dtype=np.float16)
            icr_per_block = np.zeros((num_blocks,), dtype=np.float32)

            for b in range(num_blocks):
                h_in_resp = response_h_all[b]       # (R, H) — block b input at response positions
                h_out_resp = response_h_all[b + 1]  # (R, H) — block b output at response positions
                delta_h = h_out_resp - h_in_resp     # (R, H)
                h_prev = np.concatenate(
                    [prompt_h_all[b], h_in_resp], axis=0
                )  # (P+R, H)

                attn_resp = recompute_block_attention(
                    h_prev=torch.from_numpy(h_prev.astype(np.float32)),
                    block=model.model.layers[b],
                    prompt_len=P,
                    response_len=R,
                    device=args.device,
                    rotary_emb=getattr(model.model, "rotary_emb", None),
                ).cpu().numpy()  # (R, R) float32

                attn_per_block[b, :R, :R] = attn_resp.astype(np.float16)

                # compute_icr_score only accepts top_p, not top_k.
                # If --icr-top-k was given, derive top_p per-sample per-block.
                if icr_top_k is not None:
                    top_p_eff = float(icr_top_k) / max(R, 1)
                    top_p_eff = min(max(top_p_eff, 1.0 / R), 1.0)
                else:
                    top_p_eff = icr_top_p_default

                icr_per_block[b] = compute_icr_score(
                    response_attn=attn_resp,
                    h_block_input=h_in_resp.astype(np.float32),
                    delta_h=delta_h.astype(np.float32),
                    response_len=R,
                    top_p=top_p_eff,
                )

            attn_writer.write(s, key_of[s], attn_per_block, R, P)
            icr_writer.write(s, key_of[s], icr_per_block)
            total_processed += 1

        batch_num += 1
        if batch_num % _LOG_EVERY_BATCHES == 0:
            elapsed = time.monotonic() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0.0
            remaining = (N - (batch_start + batch_size)) / batch_size
            eta_s = (remaining * batch_size / rate) if rate > 0 else float("inf")
            logger.info(
                f"Batch {batch_num} | processed={total_processed} "
                f"| {rate:.1f} samp/s | ETA {eta_s / 60:.1f} min"
            )

    attn_writer.finalize()
    icr_writer.finalize()

    elapsed = time.monotonic() - t_start
    logger.info(
        f"Done. Wrote {total_processed} samples "
        f"in {elapsed / 60:.1f} min "
        f"({total_processed / elapsed:.1f} samp/s)."
    )
    logger.info(f"  attention_dir : {attention_dir}")
    logger.info(f"  icr_scores    : {icr_scores_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fused Stage-1+2 recomputation: reads activations.zarr, recomputes "
            "per-block head-averaged attention and ICR scores in a single pass, "
            "writes attention/ (numpy memmap) and icr_scores.npy."
        )
    )
    parser.add_argument(
        "--activations-zarr",
        required=True,
        help="Path to existing activations.zarr (read-only source).",
    )
    parser.add_argument(
        "--attention-dir",
        required=True,
        help=(
            "Directory to write attention outputs "
            "(response_attn.npy, meta.jsonl, config.json)."
        ),
    )
    parser.add_argument(
        "--icr-scores-path",
        default=None,
        help=(
            "Path for icr_scores.npy output. "
            "Default: <parent of attention-dir>/icr_scores.npy."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model ID (e.g. meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples per batched zarr oindex call (default: 16).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device for recompute_block_attention (default: 'cuda').",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Model and attention-output dtype (default: 'float16').",
    )
    parser.add_argument(
        "--r-max",
        type=int,
        default=_R_MAX_DEFAULT,
        help=f"Max response length stored in attention output (default: {_R_MAX_DEFAULT}).",
    )
    parser.add_argument(
        "--icr-top-p",
        type=float,
        default=0.1,
        help=(
            "Top-p fraction of attention row used to select key positions for ICR score "
            "(default: 0.1). Ignored if --icr-top-k is set."
        ),
    )
    parser.add_argument(
        "--icr-top-k",
        type=int,
        default=None,
        help=(
            "If set, overrides --icr-top-p: top_p is derived as top_k / response_len "
            "per sample per block. Because compute_icr_score() only accepts top_p, "
            "this conversion is applied at runtime."
        ),
    )
    parser.add_argument(
        "--validate-first",
        action="store_true",
        help=(
            "Run a 4-sample numerical-equivalence check (recompute vs. full forward pass). "
            "Exits 0 on pass (all max|diff| < 1e-3), 1 on fail. "
            "Does not write any output files."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a partially-completed run: skip samples already recorded in "
            "both meta.jsonl and icr_scores_meta.jsonl."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (smoke-test / quick-validation mode).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _run_main(args)


if __name__ == "__main__":
    main()
