"""Build the nine Issue #154 50/50 mixed-view ablation cells.

Grid: HotpotQA, NQ, and PopQA x matched training/split seeds 0--2. The
single method keeps two encoded views per example and samples the second view
from token zero or a same-response later token with equal probability. The
existing pure token-zero and pure later-view causal-control cells are reused as
the endpoints. MMLU is excluded.

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
    ("00", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
)
_METHOD = "tokenwise_causal_mixed_half"
_ENDPOINTS = (
    "tokenwise_causal_t0_dropout",
    "tokenwise_causal_temporal_positive",
)
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
    for priority, slug, dataset_name in _TARGETS:
        experiment_name = f"issue154_mixed_{slug}"
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)

        for seed in _SEEDS:
            cell_id = (
                f"1_high_{priority}_3_{seed}_issue154_mixed__"
                f"{dataset_name}__{_METHOD}__seed_{seed}"
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
            endpoint_outputs = {
                endpoint: str(
                    Path("runs")
                    / f"issue155_causal_{slug}"
                    / dataset_name
                    / endpoint
                    / f"seed_{seed}"
                    / "predictions.csv"
                )
                for endpoint in _ENDPOINTS
            }
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "high",
                "issue": 154,
                "experiment": "mixed_temporal_augmentation",
                "stage": "mixed_half",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _METHOD,
                "seed": seed,
                "seeded": True,
                "factor_later_view_probability": 0.5,
                "output_check": str(run_dir / "predictions.csv"),
                "comparison_endpoint_outputs": endpoint_outputs,
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
