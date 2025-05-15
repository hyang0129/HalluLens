"""
activations_logger.py
Handles LMDB logging for LLM activations, prompts, responses, and evaluation results.
Used to evaluate the nature of activations when an LLM is hallucinating vs not hallucinating.
"""
import lmdb
import pickle
import os
from typing import Any, Dict, Optional


class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 1 << 30):
        """
        Initialize the LMDB-based activations logger.
        
        Args:
            lmdb_path: Path to the LMDB file to store activations
            map_size: Maximum size of the LMDB file in bytes (default: 1GB)
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

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the LMDB.
        
        Args:
            key: Unique identifier for the entry (typically a hash of the prompt)
            entry: Dictionary containing the data to log (prompt, response, activations, etc.)
        """
        # Skip logging if LMDB is not initialized
        if self.env is None:
            return
            
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