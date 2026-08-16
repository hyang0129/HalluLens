"""Build the isolated 50-cell issue #151 token-wise contrastive matrix.

Grid:

    5 datasets x 2 token-pair rules x 5 training seeds = 50 cells

Each cell owns one dataset/method/seed output directory and uses
``predictions.csv`` as its completion sentinel.  High-priority cells receive a
``0_high`` filename prefix so the filesystem claim queue selects them first.
MMLU is intentionally absent from the HalluLens benchmark matrix.

This script only creates idempotent queue cells; it never starts workers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

_TARGETS = (
    ("tokenwise151_hotpotqa", "hotpotqa_memmap"),
    ("tokenwise151_nq", "nq_memmap"),
    ("tokenwise151_popqa", "popqa_memmap"),
    ("tokenwise151_sciq", "sciq_memmap"),
    ("tokenwise151_searchqa", "searchqa_memmap"),
)
_METHODS = (
    "tokenwise_contrastive_first_anchored",
    "tokenwise_contrastive_random_distinct",
)
_SEEDS = (0, 1, 2, 3, 4)


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    filename = f"{cell_id}.json"
    for subdir in ("pending", "done", "failed"):
        if (dispatch_root / subdir / filename).exists():
            return True
    claimed = dispatch_root / "claimed"
    return claimed.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed.iterdir()
    )


def build(dispatch_root: Path) -> int:
    """Create every missing issue #151 cell and return the write count."""
    init_dispatch_dirs(dispatch_root)
    written = 0
    for experiment_name, dataset_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / experiment_rel).exists():
            raise FileNotFoundError(experiment_rel)
        for method_name in _METHODS:
            method_rel = f"configs/methods/{method_name}.json"
            if not (_PROJECT_ROOT / method_rel).exists():
                raise FileNotFoundError(method_rel)
            for seed in _SEEDS:
                cell_id = (
                    f"0_high_issue151__{dataset_name}__{method_name}__seed{seed}"
                )
                if _dispatch_has_cell(dispatch_root, cell_id):
                    print(f"  skip (already queued): {cell_id}")
                    continue
                output_check = (
                    Path("runs")
                    / experiment_name
                    / dataset_name
                    / method_name
                    / f"seed_{seed}"
                    / "predictions.csv"
                )
                cell = {
                    "cell_id": cell_id,
                    "kind": "experiment",
                    "priority": "high",
                    "issue": 151,
                    "experiment_config": experiment_rel,
                    "dataset": dataset_name,
                    "method": method_name,
                    "seed": str(seed),
                    "output_check": str(output_check),
                }
                (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                    json.dumps(cell, indent=2) + "\n",
                    encoding="utf-8",
                )
                written += 1
                print(f"  queued: {cell_id}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_151_tokenwise_dispatch",
        help="Isolated queue root (default: shared/issue_151_tokenwise_dispatch)",
    )
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    count = build(dispatch_root)
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
