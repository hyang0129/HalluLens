"""Append the nine-cell Issue #153 token-wise v2 pilot.

The corrected Issue #151 v1 runs already provide the matched seeds 0--2 for
HotpotQA, NQ, and PopQA. This builder therefore queues only the corresponding
v2 cells. IDs sort after the Issue #155 causal controls in the shared generic
experiment queue. MMLU is excluded.
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
    ("00", "hotpotqa_memmap", "issue153_v2_hotpotqa"),
    ("20", "nq_memmap", "issue153_v2_nq"),
    ("30", "popqa_memmap", "issue153_v2_popqa"),
)
_METHOD = "tokenwise_contrastive_v2_depthnorm_projection"
_SEEDS = (0, 1, 2)


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
    method_rel = f"configs/methods/{_METHOD}.json"
    if not (project_root / method_rel).exists():
        raise FileNotFoundError(project_root / method_rel)

    written = 0
    for priority, dataset_name, experiment_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)

        for seed in _SEEDS:
            cell_id = (
                f"2_high_{priority}_{seed}_issue153_v2__{dataset_name}__"
                f"{_METHOD}__seed_{seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue

            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / _METHOD
                / f"seed_{seed}"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "high",
                "issue": 153,
                "experiment": "tokenwise_v2_depthnorm_projection",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _METHOD,
                "seed": seed,
                "seeded": True,
                "baseline_run": str(
                    Path("runs")
                    / f"issue151_knnval_{experiment_name.removeprefix('issue153_v2_')}"
                    / dataset_name
                    / "tokenwise_contrastive_first_anchored"
                    / f"seed_{seed}"
                    / "eval_metrics.json"
                ),
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
