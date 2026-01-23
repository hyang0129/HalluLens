"""
Utilities for migrating legacy JSON activation stores to Zarr.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from loguru import logger
from tqdm.autonotebook import tqdm

from .activations_logger import JsonActivationsLogger
from .zarr_activations_logger import ZarrActivationsLogger


def migrate_json_to_zarr(
    json_dir: str,
    zarr_path: str,
    *,
    overwrite: bool = False,
    resume: bool = True,
    skip_existing: bool = True,
    chunk_size: int = 1,
    activation_chunk_shape: Optional[tuple[int, int, int, int]] = None,
    max_entries: Optional[int] = None,
    prompt_max_tokens: Optional[int] = None,
    response_max_tokens: Optional[int] = None,
    prompt_chunk_tokens: Optional[int] = None,
    response_chunk_tokens: Optional[int] = None,
    dtype: str = "float16",
    progress: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Migrate a JSON activation store (including legacy formats) into a Zarr store.

    Args:
        json_dir: Path to the JSON activation directory.
        zarr_path: Destination Zarr store path.
        overwrite: If True, overwrite an existing Zarr store.
        resume: If True, allow resuming into an existing Zarr store.
        skip_existing: If True, skip entries already present in the Zarr index.
        chunk_size: Zarr chunk size (samples per chunk).
        activation_chunk_shape: Optional activation chunk shape (S, L, T, H).
            Use -1 for H to auto-match hidden size.
        max_entries: If set, only process the first N entries.
        prompt_max_tokens: Fixed max prompt tokens (P_max) for Zarr.
        response_max_tokens: Fixed max response tokens (R_max) for Zarr.
        prompt_chunk_tokens: Prompt chunk size (tokens).
        response_chunk_tokens: Response chunk size (tokens).
        dtype: Stored dtype for activations.
        progress: If True, show a progress bar.
        verbose: If True, enable verbose logging.

    Returns:
        Summary dict with migration statistics.
    """
    src_path = Path(json_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"JSON activation directory not found: {src_path}")

    dst_path = Path(zarr_path)
    if dst_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"Zarr destination already exists: {dst_path}")

    json_logger = JsonActivationsLogger(output_dir=str(src_path), read_only=True, verbose=verbose)

    zarr_logger = ZarrActivationsLogger(
        zarr_path=str(dst_path),
        mode="w" if overwrite else "a",
        chunk_size=chunk_size,
        activation_chunk_shape=activation_chunk_shape,
        read_only=False,
        prompt_max_tokens=prompt_max_tokens,
        response_max_tokens=response_max_tokens,
        prompt_chunk_tokens=prompt_chunk_tokens,
        response_chunk_tokens=response_chunk_tokens,
        dtype=dtype,
        verbose=verbose,
    )

    keys = json_logger.list_entries()
    if not keys:
        keys = _scan_activation_files(src_path)

    if max_entries is not None:
        keys = keys[:max_entries]

    existing_keys = set()
    if resume and skip_existing:
        existing_keys = set(zarr_logger.list_entries())

    stats = {
        "total": len(keys),
        "migrated": 0,
        "skipped_missing": 0,
        "skipped_no_activations": 0,
        "skipped_existing": 0,
        "failed": 0,
        "errors": [],
    }

    iterator: Iterable[str] = keys
    if progress:
        iterator = tqdm(keys, desc="Migrating JSON → Zarr", unit="entry")
        if verbose:
            logger.info("Migration progress bar enabled (tqdm.autonotebook)")

    for key in iterator:
        try:
            if skip_existing and key in existing_keys:
                stats["skipped_existing"] += 1
                continue

            entry = json_logger.get_entry_by_key(key)
            if entry is None:
                entry = _load_legacy_entry(src_path, key, json_logger)

            if entry is None:
                stats["skipped_missing"] += 1
                continue

            activations = entry.get("all_layers_activations")
            if activations is None:
                stats["skipped_no_activations"] += 1
                continue

            if entry.get("input_length") is None:
                entry["input_length"] = _infer_input_length(entry)

            zarr_logger.log_entry(key, entry)
            stats["migrated"] += 1
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append({"key": key, "error": str(exc)})
            if verbose:
                logger.exception(f"Failed to migrate entry {key}")

    return stats


def _scan_activation_files(json_dir: Path) -> List[str]:
    activations_dir = json_dir / "activations"
    if not activations_dir.exists():
        return []

    keys: List[str] = []
    for path in activations_dir.iterdir():
        if path.suffix.lower() in {".npy", ".json"}:
            keys.append(path.stem)
    return sorted(set(keys))


def _load_legacy_entry(
    json_dir: Path,
    key: str,
    json_logger: JsonActivationsLogger,
) -> Optional[Dict[str, Any]]:
    activations_dir = json_dir / "activations"
    npy_path = activations_dir / f"{key}.npy"
    json_path = activations_dir / f"{key}.json"

    if npy_path.exists():
        activation_arrays = np.load(npy_path, allow_pickle=True)
        activations = json_logger._numpy_arrays_to_tensors(activation_arrays)
        return {
            "key": key,
            "all_layers_activations": activations,
        }

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            activation_data = json.load(f)
        if "all_layers_activations" in activation_data:
            activations = json_logger._json_to_tensor(activation_data["all_layers_activations"])
            return {
                "key": key,
                "all_layers_activations": activations,
            }

    return None


def _infer_input_length(entry: Dict[str, Any]) -> int:
    logging_config = entry.get("logging_config", {})
    if logging_config.get("sequence_mode") == "response":
        return 0

    prompt_len = entry.get("prompt_len")
    if isinstance(prompt_len, int):
        return prompt_len

    activations = entry.get("all_layers_activations")
    if activations is None:
        return 0

    for layer_act in activations:
        if layer_act is None:
            continue
        if isinstance(layer_act, np.ndarray):
            seq_len = int(layer_act.shape[1]) if layer_act.ndim == 3 else int(layer_act.shape[0])
            return seq_len
        if isinstance(layer_act, torch.Tensor):
            seq_len = int(layer_act.shape[1]) if layer_act.ndim == 3 else int(layer_act.shape[0])
            return seq_len

    return 0
