"""
Generate dummy activation data for both JSON (NPY) and Zarr formats.

Creates 100 samples by default with:
- 32 layers
- 64 prompt tokens + 64 response tokens
- 4096 hidden size
- fp16 activations
"""
import argparse
import hashlib
import os
import time
from typing import List

import numpy as np

from activation_logging.activations_logger import JsonActivationsLogger
from activation_logging.zarr_activations_logger import ZarrActivationsLogger


def _make_layer_activations(rng: np.random.Generator, layers: int, seq_len: int, hidden: int) -> List[np.ndarray]:
    activations = []
    for _ in range(layers):
        arr = rng.standard_normal((1, seq_len, hidden), dtype=np.float32).astype(np.float16)
        activations.append(arr)
    return activations


def _make_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_json_samples(
    output_dir: str,
    num_samples: int,
    layers: int,
    prompt_len: int,
    response_len: int,
    hidden: int,
    seed: int,
):
    logger = JsonActivationsLogger(output_dir=output_dir, target_layers="all", sequence_mode="all")
    rng = np.random.default_rng(seed)

    seq_len = prompt_len + response_len
    for i in range(num_samples):
        prompt = f"Dummy prompt {i}"
        response = f"Dummy response {i}"
        prompt_hash = _make_prompt_hash(f"user: {prompt}")
        entry = {
            "prompt": prompt,
            "response": response,
            "model": "dummy-model",
            "input_length": prompt_len,
            "prompt_hash": prompt_hash,
            "all_layers_activations": _make_layer_activations(rng, layers, seq_len, hidden),
        }
        logger.log_entry(prompt_hash, entry)
    logger.close()


def generate_zarr_samples(
    zarr_path: str,
    num_samples: int,
    layers: int,
    prompt_len: int,
    response_len: int,
    hidden: int,
    seed: int,
):
    logger = ZarrActivationsLogger(
        zarr_path=zarr_path,
        target_layers="all",
        sequence_mode="all",
        prompt_max_tokens=prompt_len,
        response_max_tokens=response_len,
        prompt_chunk_tokens=prompt_len,
        response_chunk_tokens=response_len,
        dtype="float16",
        read_only=False,
    )
    rng = np.random.default_rng(seed)

    seq_len = prompt_len + response_len
    for i in range(num_samples):
        prompt = f"Dummy prompt {i}"
        response = f"Dummy response {i}"
        prompt_hash = _make_prompt_hash(f"user: {prompt}")
        entry = {
            "prompt": prompt,
            "response": response,
            "model": "dummy-model",
            "input_length": prompt_len,
            "prompt_hash": prompt_hash,
            "all_layers_activations": _make_layer_activations(rng, layers, seq_len, hidden),
        }
        logger.log_entry(prompt_hash, entry)
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Generate dummy activation data for JSON and Zarr formats.")
    parser.add_argument("--json-output", type=str, default="json_dummy_data", help="Output directory for JSON/NPY format")
    parser.add_argument("--zarr-output", type=str, default="zarr_dummy_data/activations.zarr", help="Output path for Zarr format")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--prompt-len", type=int, default=64, help="Prompt token length")
    parser.add_argument("--response-len", type=int, default=64, help="Response token length")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--skip-json", action="store_true", help="Skip JSON/NPY generation")
    parser.add_argument("--skip-zarr", action="store_true", help="Skip Zarr generation")
    args = parser.parse_args()

    os.makedirs(args.json_output, exist_ok=True)
    os.makedirs(os.path.dirname(args.zarr_output), exist_ok=True)

    start = time.time()
    if not args.skip_json:
        generate_json_samples(
            output_dir=args.json_output,
            num_samples=args.num_samples,
            layers=args.layers,
            prompt_len=args.prompt_len,
            response_len=args.response_len,
            hidden=args.hidden,
            seed=args.seed,
        )

    if not args.skip_zarr:
        generate_zarr_samples(
            zarr_path=args.zarr_output,
            num_samples=args.num_samples,
            layers=args.layers,
            prompt_len=args.prompt_len,
            response_len=args.response_len,
            hidden=args.hidden,
            seed=args.seed,
        )

    elapsed = time.time() - start
    print(f"Done. Generated {args.num_samples} samples in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
