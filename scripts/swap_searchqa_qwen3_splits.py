#!/usr/bin/env python3
"""
Swap Qwen3 SearchQA split directories to fix an inversion in generate_all_qwen3.sh.

What was wrong:
  output/searchqa/Qwen3-8B/       had HF train split (151K) — should be test (43K)
  output/searchqa_train/Qwen3-8B/ had HF test split  (43K)  — should be train (151K)
  shared/searchqa_qwen3_8b/       activations for train split
  shared/searchqa_train_qwen3_8b/ activations for test split

After this script:
  output/searchqa/Qwen3-8B/       ← what was output/searchqa_train/Qwen3-8B/
  output/searchqa_train/Qwen3-8B/ ← what was output/searchqa/Qwen3-8B/
  shared/searchqa_qwen3_8b/       ← what was shared/searchqa_train_qwen3_8b/
  shared/searchqa_train_qwen3_8b/ ← what was shared/searchqa_qwen3_8b/

Usage:
  python scripts/swap_searchqa_qwen3_splits.py [--dry-run]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SWAP_PAIRS = [
    (
        REPO_ROOT / "output/searchqa/Qwen3-8B",
        REPO_ROOT / "output/searchqa_train/Qwen3-8B",
    ),
    (
        REPO_ROOT / "shared/searchqa_qwen3_8b",
        REPO_ROOT / "shared/searchqa_train_qwen3_8b",
    ),
]


def swap(a: Path, b: Path, dry_run: bool):
    a_exists = a.exists()
    b_exists = b.exists()

    if not a_exists and not b_exists:
        print(f"  SKIP  neither exists: {a.relative_to(REPO_ROOT)}  /  {b.relative_to(REPO_ROOT)}")
        return

    tmp = a.parent / (a.name + "__swap_tmp")
    print(f"  SWAP  {a.relative_to(REPO_ROOT)}  <->  {b.relative_to(REPO_ROOT)}")

    if dry_run:
        return

    if a_exists:
        a.rename(tmp)
    if b_exists:
        b.rename(a)
    if a_exists:
        tmp.rename(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without doing it")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no changes will be made\n")

    for a, b in SWAP_PAIRS:
        swap(a, b, args.dry_run)

    if not args.dry_run:
        print("\nDone. Re-run eval on both dirs if eval_results.json needs refreshing.")


if __name__ == "__main__":
    main()
