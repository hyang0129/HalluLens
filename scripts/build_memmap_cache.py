#!/usr/bin/env python3
"""Build memmap cache for an activation zarr in chunks (low RAM).

The experiment runner builds memmap caches lazily on first use, holding the full
preloaded array in RAM. For large stores (90K+ samples) this can exceed
available RAM. This script streams chunks of samples through the existing
``_preload_from_zarr`` codepath and writes them directly to .npy memmaps on
disk, capping RAM at ~chunk_size × L × T × H × 2 bytes.

Output is byte-compatible with what ``_preload_all_splits`` would produce, so
the experiment runner picks it up transparently via the same fingerprint.

Usage:
    python scripts/build_memmap_cache.py \\
        --activations-path shared/hotpotqa_train_llama_3_1_8b_instruct/activations.zarr \\
        --inference-json   output/hotpotqa_train/Llama-3.1-8B-Instruct/generation.jsonl \\
        --eval-json        output/hotpotqa_train/Llama-3.1-8B-Instruct/eval_results.json \\
        --split-seed 42 \\
        --chunk-size 512

    # Verify against an existing cache (for parity testing on a small dataset):
    python scripts/build_memmap_cache.py \\
        --activations-path shared/natural_questions_llama_3_1_8b_instruct/activations.zarr \\
        --inference-json   output/natural_questions/Llama-3.1-8B-Instruct/generation.jsonl \\
        --eval-json        output/natural_questions/Llama-3.1-8B-Instruct/eval_results.json \\
        --output-dir       /tmp/memmap_parity_check \\
        --verify-against   shared/natural_questions_llama_3_1_8b_instruct/activations.zarr/_memmap_cache/<fingerprint>
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from activation_logging.activation_parser import ActivationParser  # noqa: E402

# Defaults match what scripts/run_experiment.py passes for the standard layer-14..29 sweep.
DEFAULT_RELEVANT_LAYERS = list(range(14, 30))
DEFAULT_PAD_LENGTH = 63
DEFAULT_INCLUDE_LOGPROBS = True
DEFAULT_RESPONSE_LOGPROBS_TOP_K = 20


def parse_layer_range(spec: str) -> List[int]:
    """Mirror scripts/run_experiment.py's parse_layer_range — supports '14-29' or '14,15,16'."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def build_chunked(
    parser: ActivationParser,
    relevant_layers: List[int],
    pad_length: int,
    include_logprobs: bool,
    response_logprobs_top_k: int,
    chunk_size: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """Stream chunks from zarr → on-disk memmap, write manifest + indices.

    Reuses parser._preload_from_zarr for the actual zarr → numpy conversion;
    we just slice the dataframe ourselves and write each chunk's output into
    the right slice of the on-disk array.
    """
    fingerprint = parser._memmap_cache_fingerprint(
        relevant_layers, pad_length, include_logprobs, response_logprobs_top_k
    )
    cache_dir = output_dir if output_dir is not None else parser._memmap_cache_dir(fingerprint)
    tmp_dir = cache_dir.parent / f".tmp_{fingerprint}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df_all = parser.df.reset_index(drop=True)
    train_mask = (parser.df['split'] == 'train').values
    test_mask = (parser.df['split'] == 'test').values
    val_mask = (parser.df['split'] == 'val').values if parser.split_strategy == "three_way" else None

    N = len(df_all)
    L = len(relevant_layers)
    T = pad_length
    H = int(parser.activation_logger._response_activations.shape[-1])

    print(f"[memmap-build] N={N:,} L={L} T={T} H={H} chunk_size={chunk_size}")
    print(f"[memmap-build] tmp_dir={tmp_dir}")
    print(f"[memmap-build] cache_dir={cache_dir}")

    # Pre-allocate .npy memmaps on disk. open_memmap creates a real .npy file
    # (with header) that np.load(..., mmap_mode='r') will accept later.
    act_arr = np.lib.format.open_memmap(
        tmp_dir / "activations.npy", mode='w+',
        dtype=np.float16, shape=(N, L, T, H),
    )
    label_arr = np.lib.format.open_memmap(
        tmp_dir / "labels.npy", mode='w+',
        dtype=np.int8, shape=(N,),
    )
    prompt_hashes: List[str] = []

    # Logprob arrays (allocated lazily after we read the first chunk to learn shapes).
    lp_token_ids = lp_token_lps = lp_topk_ids = lp_topk_lps = None
    top_k_use = None

    t_start = time.time()
    for start in tqdm(range(0, N, chunk_size), desc="chunks", unit="chunk"):
        end = min(start + chunk_size, N)
        chunk_df = df_all.iloc[start:end].reset_index(drop=True)
        chunk = parser._preload_from_zarr(
            df_split=chunk_df,
            relevant_layers=relevant_layers,
            pad_length=pad_length,
            include_logprobs=include_logprobs,
            response_logprobs_top_k=response_logprobs_top_k,
            split=f"chunk_{start}",
        )
        # cache: shape (chunk_size, L, T, H), dtype float16
        act_arr[start:end] = chunk['cache']
        label_arr[start:end] = chunk['labels'].astype(np.int8)
        prompt_hashes.extend(chunk['prompt_hashes'])

        if include_logprobs and 'logprob_token_ids' in chunk and chunk['logprob_token_ids'] is not None:
            if lp_token_ids is None:
                T_lp = chunk['logprob_token_ids'].shape[1]
                top_k_use = chunk['logprob_topk_ids'].shape[2]
                lp_token_ids = np.lib.format.open_memmap(
                    tmp_dir / "logprob_token_ids.npy", mode='w+',
                    dtype=np.int32, shape=(N, T_lp),
                )
                lp_token_lps = np.lib.format.open_memmap(
                    tmp_dir / "logprob_token_logprobs.npy", mode='w+',
                    dtype=np.float32, shape=(N, T_lp),
                )
                lp_topk_ids = np.lib.format.open_memmap(
                    tmp_dir / "logprob_topk_ids.npy", mode='w+',
                    dtype=np.int32, shape=(N, T_lp, top_k_use),
                )
                lp_topk_lps = np.lib.format.open_memmap(
                    tmp_dir / "logprob_topk_logprobs.npy", mode='w+',
                    dtype=np.float32, shape=(N, T_lp, top_k_use),
                )
            lp_token_ids[start:end] = chunk['logprob_token_ids']
            lp_token_lps[start:end] = chunk['logprob_token_logprobs']
            lp_topk_ids[start:end] = chunk['logprob_topk_ids']
            lp_topk_lps[start:end] = chunk['logprob_topk_logprobs']

        # Free chunk RAM aggressively.
        del chunk
        act_arr.flush()
        label_arr.flush()
        if lp_token_ids is not None:
            lp_token_ids.flush(); lp_token_lps.flush()
            lp_topk_ids.flush(); lp_topk_lps.flush()

    elapsed = time.time() - t_start
    print(f"[memmap-build] preload complete in {elapsed:.1f}s")

    # prompt_hashes — small array (N strings × 64 chars), load eagerly.
    np.save(tmp_dir / "prompt_hashes.npy", np.array(prompt_hashes, dtype='U64'))

    # Train/test/val indices — reproduce what _preload_all_splits would write.
    train_indices = np.where(train_mask)[0].astype(np.int64)
    test_indices = np.where(test_mask)[0].astype(np.int64)
    np.save(tmp_dir / "train_indices.npy", train_indices)
    np.save(tmp_dir / "test_indices.npy", test_indices)
    val_indices = None
    if val_mask is not None:
        val_indices = np.where(val_mask)[0].astype(np.int64)
        np.save(tmp_dir / "val_indices.npy", val_indices)

    manifest = {
        "fingerprint": fingerprint,
        "relevant_layers": relevant_layers,
        "pad_length": pad_length,
        "include_logprobs": include_logprobs,
        "response_logprobs_top_k": response_logprobs_top_k,
        "activation_shape": [N, L, T, H],
        "activation_dtype": "float16",
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "n_val": int(len(val_indices)) if val_indices is not None else 0,
        "n_total": N,
        "zarr_sample_count": int(parser.activation_logger._response_activations.shape[0]),
        "split_strategy": parser.split_strategy,
    }
    (tmp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Atomic rename — same filesystem assumed.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(cache_dir)
    print(f"[memmap-build] wrote cache to {cache_dir}")
    return cache_dir


def verify_parity(built_dir: Path, ref_dir: Path) -> bool:
    """Compare every .npy file across two cache dirs.  Return True if all match."""
    print(f"[verify] built : {built_dir}")
    print(f"[verify] ref   : {ref_dir}")
    ok = True
    files = sorted(set(p.name for p in built_dir.glob("*.npy")) |
                   set(p.name for p in ref_dir.glob("*.npy")))
    for fname in files:
        bp = built_dir / fname
        rp = ref_dir / fname
        if not bp.exists():
            print(f"  {fname}: MISSING in built"); ok = False; continue
        if not rp.exists():
            print(f"  {fname}: MISSING in ref"); ok = False; continue
        a = np.load(bp, mmap_mode='r')
        b = np.load(rp, mmap_mode='r')
        if a.shape != b.shape:
            print(f"  {fname}: shape mismatch {a.shape} vs {b.shape}")
            ok = False; continue
        if a.dtype != b.dtype:
            print(f"  {fname}: dtype mismatch {a.dtype} vs {b.dtype}")
            ok = False; continue
        # For floating types, allow exact equality (fp16) since both should
        # come from the same zarr → astype(fp16) path.
        if np.array_equal(a, b):
            print(f"  {fname}: OK ({a.shape}, {a.dtype})")
        else:
            n_diff = int(np.sum(a != b))
            print(f"  {fname}: MISMATCH ({n_diff:,} differing elements out of {a.size:,})")
            ok = False
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations-path", required=True, help="Path to activations.zarr")
    p.add_argument("--inference-json",   required=True, help="Path to generation.jsonl")
    p.add_argument("--eval-json",        required=True, help="Path to eval_results.json (or _for_training)")
    p.add_argument("--logger-type",      default="zarr")
    p.add_argument("--split-seed",       type=int, default=42)
    p.add_argument("--split-strategy",   default="two_way", choices=["two_way", "three_way", "none"])

    p.add_argument("--relevant-layers",  default="14-29",
                   help="Layer range like '14-29' or comma list '14,15,16' (default: 14-29)")
    p.add_argument("--pad-length",       type=int, default=DEFAULT_PAD_LENGTH)
    p.add_argument("--include-logprobs", action="store_true", default=DEFAULT_INCLUDE_LOGPROBS)
    p.add_argument("--no-logprobs",      action="store_false", dest="include_logprobs")
    p.add_argument("--response-logprobs-top-k", type=int, default=DEFAULT_RESPONSE_LOGPROBS_TOP_K)

    p.add_argument("--chunk-size",       type=int, default=512,
                   help="Samples processed per chunk (default 512). Lower = less RAM, more I/O ops.")
    p.add_argument("--output-dir",       default=None,
                   help="Override output directory (default: canonical _memmap_cache/<fingerprint>)")
    p.add_argument("--verify-against",   default=None,
                   help="If set, compare built cache to this reference cache dir and exit nonzero on mismatch")
    args = p.parse_args()

    relevant_layers = parse_layer_range(args.relevant_layers)
    activations_path = (ROOT / args.activations_path).resolve() if not Path(args.activations_path).is_absolute() else Path(args.activations_path)
    inference_json = (ROOT / args.inference_json).resolve() if not Path(args.inference_json).is_absolute() else Path(args.inference_json)
    eval_json = (ROOT / args.eval_json).resolve() if not Path(args.eval_json).is_absolute() else Path(args.eval_json)

    print(f"Building memmap cache:")
    print(f"  zarr      : {activations_path}")
    print(f"  inference : {inference_json}")
    print(f"  eval      : {eval_json}")
    print(f"  layers    : {relevant_layers}")
    print(f"  pad_length: {args.pad_length}")
    print(f"  logprobs  : {args.include_logprobs} (top_k={args.response_logprobs_top_k})")
    print(f"  split     : strategy={args.split_strategy} seed={args.split_seed}")

    parser = ActivationParser(
        inference_json=str(inference_json),
        eval_json=str(eval_json),
        activations_path=str(activations_path),
        logger_type=args.logger_type,
        random_seed=args.split_seed,
        split_strategy=args.split_strategy,
        verbose=True,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    cache_dir = build_chunked(
        parser=parser,
        relevant_layers=relevant_layers,
        pad_length=args.pad_length,
        include_logprobs=args.include_logprobs,
        response_logprobs_top_k=args.response_logprobs_top_k,
        chunk_size=args.chunk_size,
        output_dir=output_dir,
    )

    if args.verify_against:
        ref_dir = Path(args.verify_against).resolve()
        ok = verify_parity(cache_dir, ref_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
