"""Build issue #151 KNN-validation rerun cells.

The default grid is the original five HalluLens datasets x three methods x
seed-0 comparison. ``--tokenwise-full-sweep`` instead materializes the full
five-dataset x five-seed token-wise matrix. On the live queue, the seed-0
token-wise cells from the original grid are detected and only seeds 1--4 are
added. MMLU is excluded.

This script creates queue cells only; it never starts workers.
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
    ("00", "hotpotqa_memmap", "issue151_knnval_hotpotqa"),
    ("20", "nq_memmap", "issue151_knnval_nq"),
    ("30", "popqa_memmap", "issue151_knnval_popqa"),
    ("40", "sciq_memmap", "issue151_knnval_sciq"),
    ("50", "searchqa_memmap", "issue151_knnval_searchqa"),
)
_METHODS = (
    "dual_convention_contrastive_classifier_prefix_mixed_lowk",
    "tokenwise_contrastive_first_anchored",
    "contrastive_logprob_recon_prefix_mixed_lowk",
)
_SEED = 0
_TOKENWISE_METHOD = "tokenwise_contrastive_first_anchored"
_FULL_SWEEP_SEEDS = (0, 1, 2, 3, 4)


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    filename = f"{cell_id}.json"
    for state in ("pending", "done", "failed", "cancelled"):
        if (dispatch_root / state / filename).exists():
            return True
    claimed = dispatch_root / "claimed"
    return claimed.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed.iterdir()
    )


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for priority, dataset_name, experiment_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)
        for method_index, method_name in enumerate(_METHODS):
            method_rel = f"configs/methods/{method_name}.json"
            if not (project_root / method_rel).exists():
                raise FileNotFoundError(project_root / method_rel)
            cell_id = (
                f"0_high_{priority}_{method_index}_issue151_knnval__"
                f"{dataset_name}__{method_name}__seed_{_SEED}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / method_name
                / f"seed_{_SEED}"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "high",
                "issue": 151,
                "rerun": "validation_knn_auroc",
                "worker_script": "scripts/dispatch/worker_151_knnval_rerun.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": method_name,
                "seed": _SEED,
                "seeded": True,
                "output_check": str(run_dir / "predictions.csv"),
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2) + "\n", encoding="utf-8"
            )
            written += 1
            print(f"  queued: {cell_id}")
    return written


def build_tokenwise_full_sweep(
    dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT
) -> int:
    """Queue the 25-run token-wise matrix, skipping cells already represented."""
    init_dispatch_dirs(dispatch_root)
    method_rel = f"configs/methods/{_TOKENWISE_METHOD}.json"
    if not (project_root / method_rel).exists():
        raise FileNotFoundError(project_root / method_rel)

    written = 0
    for priority, dataset_name, experiment_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)
        for seed in _FULL_SWEEP_SEEDS:
            cell_id = (
                f"0_high_{priority}_1_issue151_knnval__"
                f"{dataset_name}__{_TOKENWISE_METHOD}__seed_{seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / _TOKENWISE_METHOD
                / f"seed_{seed}"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "high",
                "issue": 151,
                "rerun": "validation_knn_auroc_full_tokenwise_sweep",
                "worker_script": "scripts/dispatch/worker_151_knnval_rerun.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _TOKENWISE_METHOD,
                "seed": seed,
                "seeded": True,
                "output_check": str(run_dir / "predictions.csv"),
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2) + "\n", encoding="utf-8"
            )
            written += 1
            print(f"  queued: {cell_id}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_151_knnval_rerun_dispatch",
    )
    parser.add_argument(
        "--tokenwise-full-sweep",
        action="store_true",
        help="Build the five-dataset x five-seed token-wise-only matrix.",
    )
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    builder = build_tokenwise_full_sweep if args.tokenwise_full_sweep else build
    count = builder(dispatch_root)
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
