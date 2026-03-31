#!/usr/bin/env python3
"""
Setup script for SearchQA dataset.

Downloads the SearchQA dataset from HuggingFace and verifies it is accessible.
The dataset is cached by the HuggingFace datasets library, so subsequent loads
are fast.

Usage:
    python scripts/setup_searchqa.py
    python scripts/setup_searchqa.py --split test --n_samples 100
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(description="Download and verify SearchQA dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which split to download/verify (default: train)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to preview (default: show stats only)")
    args = parser.parse_args()

    print("=" * 60)
    print("SearchQA Dataset Setup")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\nDownloading/loading SearchQA ({args.split}) from HuggingFace...")
    print("  Source: kyunghyuncho/search_qa")
    print("  (This may take a few minutes on first download)\n")

    dataset = load_dataset("kyunghyuncho/search_qa", trust_remote_code=True)

    print("Available splits:")
    for split_name in dataset:
        print(f"  {split_name}: {len(dataset[split_name]):,} samples")

    split_data = dataset[args.split]
    print(f"\nSelected split: {args.split} ({len(split_data):,} samples)")

    # Show column info
    print(f"Columns: {list(split_data.column_names)}")

    # Preview samples
    n_preview = min(args.n_samples or 3, len(split_data))
    print(f"\nSample preview ({n_preview} items):")
    print("-" * 60)
    for i in range(n_preview):
        item = split_data[i]
        print(f"  Q: {item['question']}")
        print(f"  A: {item['answer']}")
        # Show snippet count but not content (closed-book usage)
        if "search_results" in item:
            snippets = item["search_results"]
            if isinstance(snippets, dict) and "snippets" in snippets:
                n_snippets = len(snippets["snippets"])
            elif isinstance(snippets, list):
                n_snippets = len(snippets)
            else:
                n_snippets = "?"
            print(f"  (search snippets: {n_snippets} -- ignored for closed-book eval)")
        print()

    print("=" * 60)
    print("Setup complete. Dataset is cached and ready for use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
