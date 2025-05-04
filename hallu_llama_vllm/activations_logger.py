"""
activations_logger.py
Handles LMDB logging for LLM activations, prompts, responses, and evaluation results.
"""
import lmdb
import pickle
from typing import Any, Dict, Optional

import os

class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 1 << 30):
        lmdb_dir = os.path.dirname(lmdb_path)
        if lmdb_dir and not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
        self.env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, create=True, readonly=False, lock=True)

    def log_entry(self, key: str, entry: Dict[str, Any]):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(entry))

    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is not None:
                return pickle.loads(value)
            return None

    def close(self):
        self.env.close()
