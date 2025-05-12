"""
Utility script to check LMDB for a specific entry by key.
"""
import sys
import lmdb
import pickle

if len(sys.argv) < 2:
    print("Usage: python test_check_lmdb.py <prompt_hash> [lmdb_path]")
    sys.exit(1)

key = sys.argv[1]
if len(sys.argv) >= 3:
    path = sys.argv[2]
else:
    path = "lmdb_data/test_activations.lmdb"

env = lmdb.open(path, readonly=True, lock=False)
with env.begin() as txn:
    value = txn.get(key.encode("utf-8"))
    if value is None:
        print(f"No entry found for key: {key}")
        sys.exit(2)
    entry = pickle.loads(value)
    print("Entry found:")
    print("Prompt:", entry.get("prompt"))
    print("Response:", entry.get("response"))
    print("Activations shape:", getattr(entry.get("activations"), "shape", None) if entry.get("activations") is not None else None) 