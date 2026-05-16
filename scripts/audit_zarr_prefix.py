"""Probe the first N response_len entries of every activations.zarr under
shared/ and report how many are populated.

Lightweight alternative to `audit_datasets.py --zarr` (which can take days on
large stores): only the leading slice of `arrays/response_len` is read per
store, so the cost is O(num_zarrs) small chunk reads — typically seconds.

Intended use cases:
  * Pick a smoketest-suitable zarr (one with no blank leading rows) before
    dispatching `scripts/smoketest_attention_recompute.sh`. The smoketest
    silently skips samples with `response_len < 1`, so a blank prefix makes
    its Phase 1 PASS vacuous and its Phase 3 readback fail.
  * Spot the leading-blank-row pattern observed on
    `shared/hotpotqa_qwen3_8b/activations.zarr` (1472 contiguous zero-length
    rows at the front) across the rest of the data layout.

Example:
  python scripts/audit_zarr_prefix.py
  python scripts/audit_zarr_prefix.py --n 100 --filter qwen3_8b
  python scripts/audit_zarr_prefix.py --filter llama_3_1_8b_instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


def probe_zarr(zarr_path: Path, n: int) -> dict:
    root = zarr.open(str(zarr_path), mode="r")
    response_len = root["arrays/response_len"]
    total = int(response_len.shape[0])
    head_n = min(n, total)
    head = np.asarray(response_len[:head_n])
    nonzero_mask = head > 0
    nonzero_count = int(nonzero_mask.sum())
    first_nonzero = int(np.argmax(nonzero_mask)) if nonzero_count > 0 else -1
    # Leading contiguous-zero run length within the examined head.
    leading_blank = 0
    for v in head:
        if v > 0:
            break
        leading_blank += 1
    return {
        "total_samples": total,
        "head_examined": head_n,
        "nonzero_in_head": nonzero_count,
        "first_nonzero_idx": first_nonzero,
        "leading_blank_run": leading_blank,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shared-dir", type=Path, default=Path("shared"),
        help="Root directory containing <dataset>_<model_slug>/activations.zarr stores (default: shared)",
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of leading samples to examine per zarr (default: 100)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Substring match against the zarr parent dir name (e.g. 'qwen3_8b')",
    )
    args = parser.parse_args()

    candidates = sorted(args.shared_dir.glob("*/activations.zarr"))
    if args.filter:
        candidates = [p for p in candidates if args.filter in p.parent.name]
    if not candidates:
        print(f"No activations.zarr stores found under {args.shared_dir}/", file=sys.stderr)
        return 1

    print(
        f"{'store':<55} {'total':>8} {'head':>6} {'nonzero':>8} "
        f"{'first_nz':>9} {'lead_blank':>10}"
    )
    print("-" * 100)

    rows: list[tuple[Path, dict]] = []
    errors: list[tuple[Path, str]] = []
    for path in candidates:
        try:
            info = probe_zarr(path, args.n)
        except Exception as e:
            errors.append((path, str(e)))
            print(f"{path.parent.name:<55} ERROR: {e}")
            continue
        rows.append((path, info))
        print(
            f"{path.parent.name:<55} "
            f"{info['total_samples']:>8} "
            f"{info['head_examined']:>6} "
            f"{info['nonzero_in_head']:>8} "
            f"{info['first_nonzero_idx']:>9} "
            f"{info['leading_blank_run']:>10}"
        )

    fully_populated = [
        p for p, info in rows
        if info["head_examined"] > 0 and info["nonzero_in_head"] == info["head_examined"]
    ]
    blank_prefix = [
        (p, info) for p, info in rows
        if info["leading_blank_run"] > 0
    ]
    all_blank_in_head = [
        (p, info) for p, info in rows if info["nonzero_in_head"] == 0
    ]

    print()
    print(
        f"Summary: {len(fully_populated)}/{len(rows)} stores have all {args.n} "
        f"leading samples populated"
    )
    if blank_prefix:
        print(f"  Stores with a leading-blank prefix in the head:")
        for p, info in blank_prefix:
            tag = "ALL BLANK in head" if info["nonzero_in_head"] == 0 \
                else f"first_nonzero_idx={info['first_nonzero_idx']}"
            print(f"    {p.parent.name}: leading_blank={info['leading_blank_run']} ({tag})")
    if all_blank_in_head:
        print(
            f"  NOTE: {len(all_blank_in_head)} store(s) had zero populated rows in "
            f"the first {args.n}; first_nonzero may lie beyond — re-run with a larger --n to find it."
        )
    if errors:
        print(f"  {len(errors)} store(s) errored — see lines above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
