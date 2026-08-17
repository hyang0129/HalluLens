"""Build the six Issue #154 Stage-A directional-training cells.

Grid: HotpotQA, NQ, and PopQA x two new directional objectives x seed 0.
The symmetric ``first_anchored`` control is reused from the matching Issue
#155 seed-0 run, so it is recorded in every cell but is not retrained. Cells
are appended to the generic Issue #151 queue. MMLU is excluded.

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
    (
        "00",
        "hotpotqa_memmap",
        "issue154_directional_hotpotqa",
        "issue155_causal_hotpotqa",
    ),
    ("20", "nq_memmap", "issue154_directional_nq", "issue155_causal_nq"),
    (
        "30",
        "popqa_memmap",
        "issue154_directional_popqa",
        "issue155_causal_popqa",
    ),
)
_METHODS = (
    "tokenwise_causal_t0_to_later",
    "tokenwise_causal_t0_to_later_stopgrad",
)
_BASELINE_METHOD = "tokenwise_causal_temporal_positive"
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
    for priority, dataset_name, experiment_name, baseline_experiment in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)

        baseline_output = (
            Path("runs")
            / baseline_experiment
            / dataset_name
            / _BASELINE_METHOD
            / f"seed_{_SEED}"
            / "predictions.csv"
        )

        for method_index, method_name in enumerate(_METHODS):
            method_rel = f"configs/methods/{method_name}.json"
            if not (project_root / method_rel).exists():
                raise FileNotFoundError(project_root / method_rel)

            cell_id = (
                f"2_high_{priority}_{method_index}_issue154_directional__"
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
                "issue": 154,
                "experiment": "deployment_directed_temporal_gradient",
                "stage": "A",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": method_name,
                "seed": _SEED,
                "seeded": True,
                "output_check": str(run_dir / "predictions.csv"),
                "comparison_baseline_method": _BASELINE_METHOD,
                "comparison_baseline_output": str(baseline_output),
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
        help="Existing generic experiment queue to append to.",
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
