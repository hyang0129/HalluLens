#!/usr/bin/env python3
"""Migrate stale memmap cache directories to the current fingerprint formula.

Commit 141f090 (2026-05-01) changed the fingerprint key from:
    [..., self.random_seed, ...]
to:
    [..., n_total, ...]

Caches built with the old formula have a different hash than what the current
code expects, so they are silently rebuilt. This script renames each stale
directory to the hash the current formula would produce.

If the target hash already exists (a newer build is already correct), the stale
directory is deleted. If multiple stale dirs all map to the same target, the
most recently modified one is renamed and the rest are deleted.

Usage:
    python scripts/migrate_memmap_cache_fingerprints.py [--dry-run] [--shared-root PATH]
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def compute_fingerprint(
    zarr_path_resolved: str,
    relevant_layers: list,
    pad_length: int,
    zarr_count: int,
    n_total: int,
    include_logprobs: bool,
    response_logprobs_top_k: int,
    split_strategy: str = "two_way",
) -> str:
    key_parts: list = [
        zarr_path_resolved,
        sorted(relevant_layers),
        pad_length,
        zarr_count,
        n_total,
        include_logprobs,
        response_logprobs_top_k,
    ]
    if split_strategy != "two_way":
        key_parts.append(split_strategy)
    key_parts = tuple(key_parts)
    return hashlib.sha256(repr(key_parts).encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without renaming/deleting")
    parser.add_argument("--shared-root", default=None,
                        help="Root of shared/ dir (default: auto-detect from script location)")
    args = parser.parse_args()

    if args.shared_root:
        shared_root = Path(args.shared_root).resolve()
    else:
        repo_root = Path(__file__).parent.parent
        shared_root = (repo_root / "shared").resolve()

    print(f"Scanning: {shared_root}")
    print(f"Dry run:  {args.dry_run}")
    print()

    manifests = sorted(shared_root.glob("*/activations.zarr/_memmap_cache/*/manifest.json"))
    if not manifests:
        print("No manifest.json files found.")
        return

    # Group by (zarr_path, expected_hash) so we can handle duplicates
    # Structure: {memmap_cache_dir: [(current_hash, expected_hash, manifest_path, mtime), ...]}
    by_store: dict[Path, list] = defaultdict(list)

    for manifest_path in manifests:
        cache_dir = manifest_path.parent
        memmap_cache_dir = cache_dir.parent
        zarr_path = memmap_cache_dir.parent
        current_hash = cache_dir.name

        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {manifest_path}: {e}")
            continue

        expected_hash = compute_fingerprint(
            zarr_path_resolved=str(zarr_path),
            relevant_layers=m["relevant_layers"],
            pad_length=m["pad_length"],
            zarr_count=m["zarr_sample_count"],
            n_total=m["n_total"],
            include_logprobs=m["include_logprobs"],
            response_logprobs_top_k=m["response_logprobs_top_k"],
            split_strategy=m.get("split_strategy", "two_way"),
        )
        mtime = manifest_path.stat().st_mtime
        by_store[memmap_cache_dir].append((current_hash, expected_hash, manifest_path, mtime, m))

    n_ok = n_renamed = n_deleted = n_error = 0

    for memmap_cache_dir, entries in sorted(by_store.items()):
        store_name = memmap_cache_dir.parent.parent.name
        zarr_path = memmap_cache_dir.parent

        # Partition into already-correct vs stale
        ok_hashes = {h for h, exp, *_ in entries if h == exp}
        stale = [(h, exp, mp, mt, m) for h, exp, mp, mt, m in entries if h != exp]

        # Print OK entries
        for h in sorted(ok_hashes):
            print(f"  OK      {store_name}  {h}")
            n_ok += 1

        if not stale:
            continue

        # Group stale by their target hash
        by_target: dict[str, list] = defaultdict(list)
        for h, exp, mp, mt, m in stale:
            by_target[exp].append((h, mp, mt, m))

        for target_hash, candidates in sorted(by_target.items()):
            target_dir = memmap_cache_dir / target_hash
            target_exists = target_dir.exists() or target_hash in ok_hashes

            if target_exists:
                # Target already correct — delete all stale candidates
                for current_hash, mp, mt, m in candidates:
                    stale_dir = memmap_cache_dir / current_hash
                    print(f"  DELETE  {store_name}  {current_hash}  (target {target_hash} already exists)")
                    if not args.dry_run:
                        shutil.rmtree(stale_dir)
                    n_deleted += 1
            else:
                # Pick the most recently modified candidate to rename; delete the rest
                candidates.sort(key=lambda x: x[2], reverse=True)  # newest first
                best_hash, best_mp, best_mt, best_m = candidates[0]
                print(f"  RENAME  {store_name}  {best_hash} → {target_hash}")
                if not args.dry_run:
                    best_m["fingerprint"] = target_hash
                    with open(best_mp, "w") as f:
                        json.dump(best_m, f, indent=2)
                    (memmap_cache_dir / best_hash).rename(target_dir)
                n_renamed += 1

                for current_hash, mp, mt, m in candidates[1:]:
                    stale_dir = memmap_cache_dir / current_hash
                    print(f"  DELETE  {store_name}  {current_hash}  (duplicate → {target_hash})")
                    if not args.dry_run:
                        shutil.rmtree(stale_dir)
                    n_deleted += 1

    print()
    print(f"OK:       {n_ok}")
    print(f"Renamed:  {n_renamed}{'  (dry run)' if args.dry_run else ''}")
    print(f"Deleted:  {n_deleted}{'  (dry run)' if args.dry_run else ''}")

    if (n_renamed or n_deleted) and args.dry_run:
        print()
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
