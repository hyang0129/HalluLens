"""CLI to convert Zarr activations to WebDataset Option A shards."""
from __future__ import annotations

import argparse
from pathlib import Path

from .webdataset_option_a import convert_zarr_to_wds_option_a, resolve_wds_output_pattern


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Zarr activations to WebDataset Option A")
    parser.add_argument("--zarr-path", required=True, help="Path to Zarr store")
    parser.add_argument("--output-pattern", default=None, help="Output shard pattern (e.g., /path/wds-%06d.tar)")
    parser.add_argument("--shard-size-mb", type=int, default=1024)
    parser.add_argument("--samples-jsonl", default=None)
    parser.add_argument("--include-prompt", action="store_true")

    args = parser.parse_args()

    samples_jsonl = args.samples_jsonl
    if samples_jsonl is not None:
        samples_jsonl = str(Path(samples_jsonl))

    output_pattern = resolve_wds_output_pattern(args.zarr_path, args.output_pattern)

    convert_zarr_to_wds_option_a(
        args.zarr_path,
        output_pattern,
        shard_size_mb=args.shard_size_mb,
        samples_jsonl_path=samples_jsonl,
        include_prompt=args.include_prompt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
