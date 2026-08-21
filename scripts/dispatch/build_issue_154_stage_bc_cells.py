"""Build Issue #154 Stage-B/C cells for the active generic queue.

Stage B is a matched min-response-length-3 factorial over two directional
gradient modes and three view conditions: uniform one-key, stratified one-key,
and stratified two-key. Stage C adds the no-reconstruction arm for symmetric
and both directional objectives; their recon-lambda-1 controls already exist.

Grid: 18 Stage-B cells + 9 Stage-C cells across HotpotQA, NQ, and PopQA,
all at seed 0. MMLU is excluded. This script creates cells only and never
starts workers.
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
_STAGE_B_METHODS = (
    "tokenwise_b_uniform1_t0_to_later",
    "tokenwise_b_stratified1_t0_to_later",
    "tokenwise_b_stratified2_t0_to_later",
    "tokenwise_b_uniform1_t0_to_later_stopgrad",
    "tokenwise_b_stratified1_t0_to_later_stopgrad",
    "tokenwise_b_stratified2_t0_to_later_stopgrad",
)
_STAGE_C_METHODS = (
    "tokenwise_c_recon0_symmetric",
    "tokenwise_c_recon0_t0_to_later",
    "tokenwise_c_recon0_t0_to_later_stopgrad",
)
_RECON1_CONTROLS = {
    "tokenwise_c_recon0_symmetric": (
        "issue155_causal_{slug}",
        "tokenwise_causal_temporal_positive",
    ),
    "tokenwise_c_recon0_t0_to_later": (
        "issue154_directional_{slug}",
        "tokenwise_causal_t0_to_later",
    ),
    "tokenwise_c_recon0_t0_to_later_stopgrad": (
        "issue154_directional_{slug}",
        "tokenwise_causal_t0_to_later_stopgrad",
    ),
}
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


def _write_cell(dispatch_root: Path, cell_id: str, cell: dict) -> bool:
    if _dispatch_has_cell(dispatch_root, cell_id):
        print(f"  skip (already queued): {cell_id}")
        return False
    (dispatch_root / "pending" / f"{cell_id}.json").write_text(
        json.dumps(cell, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  queued: {cell_id}")
    return True


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for priority, slug, dataset_name in _TARGETS:
        stage_b_experiment = f"issue154_sampling_{slug}"
        stage_c_experiment = f"issue154_reconstruction_{slug}"

        for stage, experiment_name, methods, prefix in (
            ("B", stage_b_experiment, _STAGE_B_METHODS, "3_high"),
            ("C", stage_c_experiment, _STAGE_C_METHODS, "4_high"),
        ):
            experiment_rel = f"configs/experiments/{experiment_name}.json"
            if not (project_root / experiment_rel).exists():
                raise FileNotFoundError(project_root / experiment_rel)

            for method_index, method_name in enumerate(methods):
                method_rel = f"configs/methods/{method_name}.json"
                if not (project_root / method_rel).exists():
                    raise FileNotFoundError(project_root / method_rel)

                cell_id = (
                    f"{prefix}_{priority}_{method_index}_issue154_stage{stage.lower()}__"
                    f"{dataset_name}__{method_name}__seed_{_SEED}"
                )
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
                    "experiment": "deployment_directed_temporal_distillation",
                    "stage": stage,
                    "worker_script": "scripts/dispatch/worker_experiment.sh",
                    "experiment_config": experiment_rel,
                    "dataset": dataset_name,
                    "method": method_name,
                    "seed": _SEED,
                    "seeded": True,
                    "output_check": str(run_dir / "predictions.csv"),
                }
                if stage == "B":
                    cell.update(
                        {
                            "matched_training_cohort_min_tokens": 3,
                            "factor_gradient_mode": (
                                "t0_to_later_stopgrad"
                                if method_name.endswith("stopgrad")
                                else "t0_to_later"
                            ),
                        }
                    )
                else:
                    baseline_experiment, baseline_method = _RECON1_CONTROLS[
                        method_name
                    ]
                    baseline_output = (
                        Path("runs")
                        / baseline_experiment.format(slug=slug)
                        / dataset_name
                        / baseline_method
                        / f"seed_{_SEED}"
                        / "predictions.csv"
                    )
                    cell.update(
                        {
                            "factor_recon_lambda": 0.0,
                            "comparison_recon1_method": baseline_method,
                            "comparison_recon1_output": str(baseline_output),
                        }
                    )

                written += int(_write_cell(dispatch_root, cell_id, cell))
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
