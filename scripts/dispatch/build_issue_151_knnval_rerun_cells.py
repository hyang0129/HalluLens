"""Build the isolated 15-cell issue #151 KNN-validation rerun matrix.

Grid: five HalluLens datasets x three methods x seed 0. MMLU is excluded.
Every experiment name is new, so completed historical runs cannot satisfy a
completion sentinel for this corrected checkpoint-selection comparison.

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_151_knnval_rerun_dispatch",
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
