"""
activations_logger.py
Handles LMDB logging for LLM activations, prompts, responses, and evaluation results.
Used to evaluate the nature of activations when an LLM is hallucinating vs not hallucinating.
"""
import lmdb
import pickle
import os
import numpy as np
from typing import Any, Dict, Optional, List, Callable, Tuple
from loguru import logger
import torch 


class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 16 << 30,
                 compress_fn: Optional[Callable] = None, decompress_fn: Optional[Callable] = None):
        """
        Initialize the LMDB-based activations logger.
        
        Args:
            lmdb_path: Path to the LMDB file to store activations
            map_size: Maximum size of the LMDB file in bytes (default: 16GB)
            compress_fn: Optional function to compress activations before storage
            decompress_fn: Optional function to decompress activations after retrieval
        """
        self.lmdb_path = lmdb_path
        self.env = None
        self.map_size = map_size
        self.last_threshold_logged = 0  # Track the last threshold percentage logged
        
        # Set compression functions (use defaults if None provided)
        self.compress_fn = compress_fn if compress_fn is not None else self.default_compress
        self.decompress_fn = decompress_fn if decompress_fn is not None else self.default_decompress
        
        # Skip opening LMDB if path is empty
        if not lmdb_path or lmdb_path.strip() == "":
            return
            
        # Replace periods in the filename with underscores for compatibility
        base, filename = os.path.split(lmdb_path)
        safe_lmdb_path = os.path.join(base, filename)
        self.safe_lmdb_path = safe_lmdb_path
        lmdb_dir = os.path.dirname(safe_lmdb_path)
        if lmdb_dir and not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
        self.env = lmdb.open(safe_lmdb_path, map_size=map_size, subdir=True, create=True, readonly=False, lock=True)

    @staticmethod
    def default_compress(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Default compression function that doesn't actually compress.
        This serves as a placeholder for future compression implementations.
        
        Args:
            data: Dictionary containing the data to be compressed
            
        Returns:
            Tuple of (uncompressed_data, metadata)
            - uncompressed_data: The original data unchanged
            - metadata: Dictionary with metadata about the compression (empty for default)
        """
        # No compression, just return the original data and empty metadata
        return data, {"compression": "none"}
    
    @staticmethod
    def default_decompress(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default decompression function that doesn't do anything.
        This serves as a placeholder for future decompression implementations.
        
        Args:
            data: Dictionary containing the data to be decompressed
            metadata: Dictionary with metadata about how the data was compressed
            
        Returns:
            The original data unchanged
        """
        # No decompression needed, just return the original data
        return data

    def check_lmdb_size(self):
        """
        Check the current size of the LMDB file and log warnings when it passes
        certain thresholds (10%, 20%, ..., 90%) of its maximum map size.
        """
        if self.env is None or not hasattr(self, 'safe_lmdb_path'):
            return
            
        # Get the current size of the LMDB data file
        data_path = os.path.join(self.safe_lmdb_path, "data.mdb")
        if not os.path.exists(data_path):
            return
            
        current_size = os.path.getsize(data_path)
        size_percentage = (current_size / self.map_size) * 100
        
        # Check if we've passed a new 10% threshold
        current_threshold = int(size_percentage / 10) * 10
        
        if current_threshold > self.last_threshold_logged and current_threshold <= 90:
            logger.warning(f"LMDB file has reached {current_threshold}% of its maximum size "
                          f"({current_size / (1024**3):.2f}GB / {self.map_size / (1024**3):.2f}GB)")
            self.last_threshold_logged = current_threshold
            
        # Extra warning at 95%
        if size_percentage >= 95 and self.last_threshold_logged < 95:
            logger.critical(f"LMDB file is at {size_percentage:.1f}% of its maximum size! "
                           f"({current_size / (1024**3):.2f}GB / {self.map_size / (1024**3):.2f}GB)")
            self.last_threshold_logged = 95

    def extract_activations(self, model_outputs, input_length):
        """
        Extract activations from model outputs.
        
        Args:
            model_outputs: The outputs from the model's generate method
            input_length: The length of the input prompt in tokens
            
        Returns:
            Dictionary containing activations from all layers
        """
        # If model_outputs is None or doesn't have hidden_states, return None
        if model_outputs is None or not hasattr(model_outputs, 'hidden_states'):
            logger.info("No hidden states found in model outputs")
            return None
            
        
        # Get the generated tokens (excluding prompt)
        gen_sequence = model_outputs.sequences[0]
        gen_ids = gen_sequence[input_length:]
        
        all_hidden_states = model_outputs.hidden_states
        prompt_hidden = all_hidden_states[0]
        gen_hiddens = all_hidden_states[1:]

        gen_hidden_per_layer = [
            torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
            for layer_idx in range(len(prompt_hidden))
        ]

        full_hidden_states = [
            torch.cat([prompt_hidden[layer_idx], gen_hidden_per_layer[layer_idx]], dim=1)
            for layer_idx in range(len(prompt_hidden))
        ]



        return full_hidden_states

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the LMDB.
        
        Args:
            key: Unique identifier for the entry (typically a hash of the prompt)
            entry: Dictionary containing the data to log (prompt, response, model_outputs, etc.)
        """
        # Skip logging if LMDB is not initialized
        if self.env is None:
            return
            
        # Process model outputs if present
        if "model_outputs" in entry and "input_length" in entry:
            model_outputs = entry["model_outputs"]
            input_length = entry["input_length"]
            
            # Extract activations from model outputs
            all_layers_activations = self.extract_activations(model_outputs, input_length)
            
            if all_layers_activations:
                # Replace model_outputs with extracted activations to save space
                entry["all_layers_activations"] = all_layers_activations
                
            # Remove the large model_outputs object to save space
            entry.pop("model_outputs", None)
        
        # Log size before compression for debugging
        entry_size_before = len(pickle.dumps(entry))
        logger.debug(f"Compressing entry with key {key[:8]}... (size before: {entry_size_before / (1024**2):.2f} MB)")
        
        # Apply compression to the entry
        compressed_entry, compression_metadata = self.compress_fn(entry)
        
        # Log compression results
        entry_size_after = len(pickle.dumps(compressed_entry))
        compression_ratio = entry_size_before / max(1, entry_size_after)
        logger.debug(f"Compression complete: {entry_size_after / (1024**2):.2f} MB, " 
                    f"ratio: {compression_ratio:.2f}x, method: {compression_metadata.get('compression', 'unknown')}")
        
        # Add compression metadata to the entry
        final_entry = {
            "data": compressed_entry,
            "metadata": compression_metadata
        }
            
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(final_entry))
            
        # Check LMDB size after writing
        self.check_lmdb_size()

    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the LMDB.
        
        Args:
            key: Unique identifier for the entry to retrieve
            
        Returns:
            Dictionary containing the entry data if found, None otherwise
        """
        # Return None if LMDB is not initialized
        if self.env is None:
            return None
            
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is not None:
                stored_entry = pickle.loads(value)
                
                # Check if the entry uses the new format with compression
                if isinstance(stored_entry, dict) and "data" in stored_entry and "metadata" in stored_entry:
                    # Log before decompression
                    compressed_size = len(pickle.dumps(stored_entry["data"]))
                    compression_method = stored_entry["metadata"].get("compression", "unknown")
                    logger.debug(f"Decompressing entry with key {key[:8]}... "
                               f"(compressed size: {compressed_size / (1024**2):.2f} MB, method: {compression_method})")
                    
                    # Apply decompression
                    decompressed_data = self.decompress_fn(stored_entry["data"], stored_entry["metadata"])
                    
                    # Log after decompression
                    decompressed_size = len(pickle.dumps(decompressed_data))
                    logger.debug(f"Decompression complete: {decompressed_size / (1024**2):.2f} MB, "
                               f"ratio: {decompressed_size / max(1, compressed_size):.2f}x")
                    
                    return decompressed_data
                else:
                    # Handle legacy format (no compression)
                    logger.debug(f"Retrieved legacy entry with key {key[:8]} (no compression)")
                    return stored_entry
            return None

    def close(self):
        """Close the LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None 