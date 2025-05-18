"""
activations_logger.py
Handles LMDB logging for LLM activations, prompts, responses, and evaluation results.
Used to evaluate the nature of activations when an LLM is hallucinating vs not hallucinating.
"""
import lmdb
import pickle
import os
import numpy as np
from typing import Any, Dict, Optional, List


class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 16 << 30):
        """
        Initialize the LMDB-based activations logger.
        
        Args:
            lmdb_path: Path to the LMDB file to store activations
            map_size: Maximum size of the LMDB file in bytes (default: 16GB)
        """
        self.lmdb_path = lmdb_path
        self.env = None
        
        # Skip opening LMDB if path is empty
        if not lmdb_path or lmdb_path.strip() == "":
            return
            
        # Replace periods in the filename with underscores for compatibility
        base, filename = os.path.split(lmdb_path)
        safe_lmdb_path = os.path.join(base, filename)
        lmdb_dir = os.path.dirname(safe_lmdb_path)
        if lmdb_dir and not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
        self.env = lmdb.open(safe_lmdb_path, map_size=map_size, subdir=True, create=True, readonly=False, lock=True)

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
            return None
            
        all_layers_activations = {}
        
        # Get the generated tokens (excluding prompt)
        gen_sequence = model_outputs.sequences[0]
        gen_ids = gen_sequence[input_length:]
        
        # Extract activations from all layers for generated tokens
        for layer_idx, layer_hidden_states in enumerate(model_outputs.hidden_states):
            # Extract activations for this layer (shape: batch_size, seq_len, hidden_dim)
            layer_activations = layer_hidden_states[0]  # Get first batch item
            
            # Only keep activations for generated tokens (not prompt tokens)
            # We need to ensure we're getting the right tokens based on the model output format
            if len(layer_activations) >= len(gen_ids):
                generated_tokens_activations = layer_activations[-len(gen_ids):].cpu().numpy()
                all_layers_activations[f"layer_{layer_idx}"] = generated_tokens_activations
        
        return all_layers_activations

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
            
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(entry))

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
                return pickle.loads(value)
            return None

    def close(self):
        """Close the LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None 