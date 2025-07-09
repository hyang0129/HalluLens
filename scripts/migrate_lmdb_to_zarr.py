import os
from activation_logging.activations_logger import ActivationsLogger
from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from tqdm import tqdm
import shutil

def check_disk_space(lmdb_path, zarr_path):
    # Estimate required space as the size of the LMDB file
    lmdb_dir = os.path.dirname(os.path.abspath(lmdb_path))
    zarr_dir = os.path.dirname(os.path.abspath(zarr_path))
    lmdb_size = 0
    if os.path.isdir(lmdb_path):
        # LMDB is a directory (with data.mdb)
        for root, dirs, files in os.walk(lmdb_path):
            for f in files:
                lmdb_size += os.path.getsize(os.path.join(root, f))
    else:
        lmdb_size = os.path.getsize(lmdb_path)
    # Get available space at the Zarr destination
    usage = shutil.disk_usage(zarr_dir)
    free_space = usage.free
    print(f"Estimated required space: {lmdb_size / (1024**3):.2f} GB")
    print(f"Available space at destination: {free_space / (1024**3):.2f} GB")
    if free_space < lmdb_size:
        print("WARNING: Available disk space is less than the estimated required space!")
        resp = input("Continue anyway? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborting migration.")
            exit(1)

def migrate_lmdb_to_zarr(
    lmdb_path: str,
    zarr_path: str,
    compression: str = None,
    target_layers: str = 'all',
    sequence_mode: str = 'all',
    debug: bool = False
):
    check_disk_space(lmdb_path, zarr_path)
    # Open LMDB logger in read-only mode
    lmdb_logger = ActivationsLogger(
        lmdb_path=lmdb_path,
        read_only=True,
        target_layers=target_layers,
        sequence_mode=sequence_mode
    )
    # Open Zarr logger
    zarr_logger = ZarrActivationsLogger(
        zarr_path=zarr_path,
        compression=compression,
        target_layers=target_layers,
        sequence_mode=sequence_mode
    )

    lmdb_keys = set(lmdb_logger.list_entries())
    zarr_keys = set(zarr_logger.list_entries())
    missing_keys = sorted(lmdb_keys - zarr_keys)
    print(f"Found {len(lmdb_keys)} entries in LMDB.")
    print(f"Found {len(zarr_keys)} entries in Zarr.")
    print(f"{len(missing_keys)} entries need to be migrated.")

    if debug:
        print("DEBUG MODE: Only copying the first 200 entries.")
        missing_keys = missing_keys[:200]

    for key in tqdm(missing_keys, desc="Migrating entries"):
        try:
            entry = lmdb_logger.get_entry(key)
            zarr_logger.log_entry(key, entry)
        except Exception as e:
            print(f"Error migrating key {key}: {e}")

    print("Migration complete.")
    zarr_logger.close()
    lmdb_logger.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate LMDB activations to Zarr format.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB activations file")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to output Zarr store")
    parser.add_argument("--compression", type=str, default=None, help="Compression method (e.g., 'zstd')")
    parser.add_argument("--target_layers", type=str, default="all")
    parser.add_argument("--sequence_mode", type=str, default="all")
    parser.add_argument("--debug", action="store_true", help="If set, only copy the first 200 entries.")
    args = parser.parse_args()

    migrate_lmdb_to_zarr(
        lmdb_path=args.lmdb_path,
        zarr_path=args.zarr_path,
        compression=args.compression,
        target_layers=args.target_layers,
        sequence_mode=args.sequence_mode,
        debug=args.debug
    ) 