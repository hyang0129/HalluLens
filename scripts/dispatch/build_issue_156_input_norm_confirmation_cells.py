"""Append the twenty missing v1 input-normalization confirmation cells.

The matched no-normalization v1 baseline already has five datasets x five
seeds, and Issue #156 Stage 1 supplies input-normalized seed 0. This builder
adds input-normalized seeds 1 through 4 over the five HalluLens datasets,
paired with the same-numbered split seeds. It is idempotent across every queue
state and does not launch workers. MMLU is deliberately excluded.
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
from scripts.experiment_utils import load_method_config  # noqa: E402

_METHOD = "tokenwise_arch_v1_input_norm_only"
_BASELINE_METHOD = "tokenwise_contrastive_first_anchored"
_TARGETS = (
    ("10", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)
_SEED_PAIRS = ((1, 1), (2, 2), (3, 3), (4, 4))
_EXPECTED_TRAINING_SEEDS = [1, 2, 3, 4]
_EXPECTED_SPLIT_SEEDS = [1, 2, 3, 4]


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


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def build(
    dispatch_root: Path,
    *,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
) -> int:
    """Append missing confirmation cells and return the number written."""
    method_cfg = load_method_config(_METHOD, project_root=str(project_root))
    if method_cfg.get("training_recipe") != "v1_first_anchored":
        raise ValueError(f"{_METHOD} resolved to the wrong training recipe")
    if not bool(method_cfg.get("model_params", {}).get("normalize_input")):
        raise ValueError(f"{_METHOD} must enable input normalization")

    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)
    experiments: dict[str, tuple[str, str]] = {}
    baselines: dict[tuple[str, int], Path] = {}
    for _, slug, dataset_name in _TARGETS:
        experiment_name = f"issue156_arch_v1_{slug}"
        experiment_rel = (
            f"configs/experiments/issue156_inputnorm_confirm_{slug}.json"
        )
        experiment_path = project_root / experiment_rel
        _require_file(experiment_path)
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment.get("experiment_name") != experiment_name:
            raise ValueError(
                f"{experiment_rel} must write into {experiment_name}"
            )
        if experiment.get("methods") != [_METHOD]:
            raise ValueError(f"{experiment_rel} must contain only {_METHOD}")
        if experiment.get("training_seeds") != _EXPECTED_TRAINING_SEEDS:
            raise ValueError(
                f"{experiment_name} must declare training seeds "
                f"{_EXPECTED_TRAINING_SEEDS}"
            )
        if experiment.get("split_seeds") != _EXPECTED_SPLIT_SEEDS:
            raise ValueError(
                f"{experiment_name} must pair split seeds "
                f"{_EXPECTED_SPLIT_SEEDS}"
            )
        experiments[dataset_name] = (experiment_name, experiment_rel)

        for training_seed, _ in _SEED_PAIRS:
            baseline = (
                runs_root
                / f"issue151_knnval_{slug}"
                / dataset_name
                / _BASELINE_METHOD
                / f"seed_{training_seed}"
                / "eval_metrics.json"
            )
            _require_file(baseline)
            baselines[(dataset_name, training_seed)] = baseline

    init_dispatch_dirs(dispatch_root)
    written = 0
    for dataset_order, _, dataset_name in _TARGETS:
        experiment_name, experiment_rel = experiments[dataset_name]
        for training_seed, split_seed in _SEED_PAIRS:
            cell_id = (
                f"0_high_{dataset_order}_{training_seed}_issue156_inputnorm_confirm__"
                f"{dataset_name}__{_METHOD}__seed_{training_seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue

            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / _METHOD
                / f"seed_{training_seed}"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "highest",
                "issue": 156,
                "experiment": "v1_input_norm_five_seed_confirmation",
                "stage": 2,
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _METHOD,
                "seed": training_seed,
                "split_seed": split_seed,
                "seeded": True,
                "selected_recipe": "v1",
                "comparison_baseline_method": _BASELINE_METHOD,
                "comparison_baseline_run": str(
                    baselines[(dataset_name, training_seed)]
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
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    runs_root = None if args.runs_root is None else Path(args.runs_root)
    if runs_root is not None and not runs_root.is_absolute():
        runs_root = _PROJECT_ROOT / runs_root

    count = build(dispatch_root, runs_root=runs_root)
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
