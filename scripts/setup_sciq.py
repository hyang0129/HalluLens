#!/usr/bin/env python3
"""
Download and verify the SciQ dataset from HuggingFace.

Usage:
    python scripts/setup_sciq.py
    python scripts/setup_sciq.py --split train --n_samples 100
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Download and verify SciQ dataset")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Which split to download/inspect (default: test)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to preview (default: all)")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Downloading SciQ from HuggingFace (allenai/sciq)...")
    dataset = load_dataset("allenai/sciq", trust_remote_code=True)

    print("\nDataset splits:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name:>12}: {len(split_data):>6} samples")

    split = dataset[args.split]
    n = args.n_samples or len(split)
    n = min(n, len(split))

    print(f"\nColumns: {list(split.column_names)}")
    print(f"\nFirst {min(5, n)} samples from '{args.split}':")
    print("-" * 80)
    for i, item in enumerate(split):
        if i >= min(5, n):
            break
        print(f"\n[{i}] Question : {item['question']}")
        print(f"    Answer   : {item['correct_answer']}")
        print(f"    Support  : {item['support'][:120]}..." if item.get('support') else "    Support  : (none)")
        print(f"    Distract : {item['distractor1']} / {item['distractor2']} / {item['distractor3']}")

    print(f"\nSciQ setup complete. {len(split)} samples available in '{args.split}' split.")


if __name__ == "__main__":
    main()
