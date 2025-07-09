"""
zarr_activations_logger.py
Handles Zarr-based logging for LLM activations, prompts, responses, and evaluation results.
API is similar to ActivationsLogger, but uses Zarr for scalable, chunked storage.
"""
import zarr
import numpy as np
import os
from typing import Any, Dict, Optional, List, Callable, Tuple, Union
import torch
from loguru import logger
from .compression import BaseCompressor, NoCompression, ZstdCompression, ZSTD_AVAILABLE

class ZarrActivationsLogger:
    def __init__(self, zarr_path: str = "zarr_data/activations.zarr", mode: str = "a", chunk_size: int = 1000,
                 compression: Union[str, BaseCompressor, None] = None,
                 target_layers: str = 'all', sequence_mode: str = 'all', read_only: bool = False):
        """
        Initialize the Zarr-based activations logger.
        Args:
            zarr_path: Path to the Zarr group to store activations
            mode: Zarr mode ('a' for append/update, 'w' for overwrite, etc.)
            chunk_size: Number of samples per chunk
            compression: Compression method ('zstd', None) or a BaseCompressor instance
            target_layers: Which layers to extract activations from ('all', 'first_half', or 'second_half')
            sequence_mode: Which tokens to extract activations for ('all', 'prompt', 'response')
            read_only: If True, open in read-only mode
        """
        if target_layers not in ['all', 'first_half', 'second_half']:
            raise ValueError("target_layers must be one of: 'all', 'first_half', 'second_half'")
        if sequence_mode not in ['all', 'prompt', 'response']:
            raise ValueError("sequence_mode must be one of: 'all', 'prompt', 'response'")
        self.target_layers = target_layers
        self.sequence_mode = sequence_mode
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size
        self.mode = mode
        self.read_only = read_only
        self.compressor = self._get_compressor(compression)
        self.root = zarr.open_group(zarr_path, mode=mode)
        self.activations = self.root.require_group("activations")
        self.metadata = self.root.require_group("metadata")
        logger.info(f"ZarrActivationsLogger initialized at {zarr_path}")

    def _get_compressor(self, compression: Union[str, BaseCompressor, None]) -> BaseCompressor:
        if isinstance(compression, BaseCompressor):
            return compression
        elif compression is None:
            return NoCompression()
        elif compression == 'zstd':
            if ZSTD_AVAILABLE:
                return ZstdCompression(level=19)
            else:
                raise ImportError("zstandard module not available, cannot compress")
        else:
            raise ValueError(f"Unknown compression method '{compression}', must be 'zstd' or a BaseCompressor instance")

    def extract_activations(self, model_outputs, input_length):
        """
        Extract activations from model outputs (see ActivationsLogger for details).
        """
        # Placeholder: copy logic from ActivationsLogger.extract_activations as needed
        # This should handle target_layers and sequence_mode
        raise NotImplementedError("extract_activations should be implemented as in ActivationsLogger")

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the Zarr store.
        Args:
            key: Unique identifier for the entry (e.g., prompt hash)
            entry: Dictionary containing activations, prompt, response, etc.
        """
        # Handle activations and compression
        if "all_layers_activations" in entry:
            acts = entry["all_layers_activations"]
            if isinstance(acts, torch.Tensor):
                acts = acts.cpu().numpy()
            acts_bytes = acts.tobytes()
            shape = acts.shape
            dtype = str(acts.dtype)
            # Store as a 1D uint8 array (binary blob)
            acts_np = np.frombuffer(acts_bytes, dtype=np.uint8)
            self.activations.array(name=key, data=acts_np, shape=acts_np.shape, dtype=acts_np.dtype, chunks=True, overwrite=True)
            compression_metadata = {"original_shape": shape, "original_dtype": dtype}
        else:
            compression_metadata = {}
        # Store metadata (excluding activations)
        meta = {k: v for k, v in entry.items() if k != "all_layers_activations"}
        meta["compression_metadata"] = compression_metadata
        self.metadata.attrs[key] = meta
        logger.debug(f"Logged entry {key} to Zarr store.")

    def get_entry(self, key: str) -> Dict[str, Any]:
        """
        Retrieve an entry from the Zarr store.
        Args:
            key: Unique identifier for the entry
        Returns:
            Dictionary with activations and metadata
        """
        result = {}
        if key in self.activations:
            acts_np = self.activations[key][...]
            meta = self.metadata.attrs.get(key, {})
            compression_metadata = meta.get("compression_metadata", {})
            shape = tuple(compression_metadata.get("original_shape", ()))
            dtype = compression_metadata.get("original_dtype", None)
            if shape and dtype:
                acts_bytes = acts_np.tobytes()
                acts = np.frombuffer(acts_bytes, dtype=dtype).reshape(shape)
                result["all_layers_activations"] = torch.tensor(acts)
            else:
                logger.warning(f"Missing shape or dtype in metadata for key {key}")
                result["all_layers_activations"] = None
        if key in self.metadata.attrs:
            result.update(self.metadata.attrs[key])
        else:
            logger.warning(f"No metadata found for key {key}")
        return result

    def get_entry_by_key(self, key: str, metadata_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the Zarr store by key.
        Args:
            key: Unique identifier for the entry
            metadata_only: If True, only retrieve metadata
        Returns:
            Dictionary with activations and/or metadata
        """
        if metadata_only:
            if key in self.metadata.attrs:
                return self.metadata.attrs[key]
            else:
                logger.warning(f"No metadata found for key {key}")
                return None
        return self.get_entry(key)

    def list_entries(self) -> List[str]:
        """
        List all entry keys in the Zarr store.
        Returns:
            List of entry keys as strings
        """
        return list(self.activations.array_keys())

    def search_metadata(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search through metadata entries using a filter function.
        Args:
            filter_fn: Function that takes a metadata entry and returns True if it matches
        Returns:
            List of tuples (key, metadata_entry) for matches
        """
        results = []
        for key in self.metadata.attrs:
            meta = self.metadata.attrs[key]
            if filter_fn(meta):
                results.append((key, meta))
        return results

    def close(self):
        """
        Close the Zarr store (no-op for zarr, but included for API compatibility).
        """
        pass

    def fix_cuda_tensors(self):
        """
        Placeholder for CUDA tensor fix (not directly needed for Zarr, but included for API compatibility).
        """
        logger.info("fix_cuda_tensors is not implemented for ZarrActivationsLogger.")
        pass

# Example usage (not executed):
# logger = ZarrActivationsLogger()
# logger.log_entry('somekey', {'all_layers_activations': np.random.randn(2,3,4), 'prompt': 'foo'})
# entry = logger.get_entry('somekey')
# print(entry) 