"""Append the 25-cell supervised token-zero MLP baseline sweep.

The baseline is paired with the Issue #151 token-wise sweep across five
datasets and five training/split seeds.  Cells use the generic experiment
worker and are idempotent across all queue states.  MMLU is excluded.
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

_METHOD = "token_zero_mlp_probe"
_TARGETS = (
    ("10", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)
_TRAINING_SEEDS = [0, 1, 2, 3, 4]
_SPLIT_SEEDS = [42, 1, 2, 3, 4]


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


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    """Append missing cells and return the number written."""
    method = load_method_config(_METHOD, project_root=str(project_root))
    model = method.get("model_params", {})
    if method.get("routine") != "token_zero_mlp_probe":
        raise ValueError(f"{_METHOD} resolved to the wrong routine")
    if int(model.get("parameter_floor", 0)) < 10_000_000:
        raise ValueError(f"{_METHOD} must enforce a >=10M parameter floor")
    if int(model.get("expected_total_params", 0)) != 10_531_842:
        raise ValueError(f"{_METHOD} parameter guard changed unexpectedly")
    data = method.get("data", {})
    if int(data.get("fixed_token", -1)) != 0 or int(data.get("num_views", -1)) != 1:
        raise ValueError(f"{_METHOD} must expose exactly token zero")
    if bool(data.get("include_response_logprobs", True)):
        raise ValueError(f"{_METHOD} must not use response-logprob targets")

    experiments: dict[str, tuple[str, str]] = {}
    for _, slug, dataset_name in _TARGETS:
        experiment_rel = f"configs/experiments/issue151_token_zero_mlp_{slug}.json"
        experiment_path = project_root / experiment_rel
        _require_file(experiment_path)
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected_name = f"issue151_token_zero_mlp_{slug}"
        if experiment.get("experiment_name") != expected_name:
            raise ValueError(f"{experiment_rel} must write into {expected_name}")
        if experiment.get("methods") != [_METHOD]:
            raise ValueError(f"{experiment_rel} must contain only {_METHOD}")
        if experiment.get("training_seeds") != _TRAINING_SEEDS:
            raise ValueError(f"{experiment_rel} has unmatched training seeds")
        if experiment.get("split_seeds") != _SPLIT_SEEDS:
            raise ValueError(f"{experiment_rel} has unmatched split seeds")
        experiments[dataset_name] = (expected_name, experiment_rel)

    init_dispatch_dirs(dispatch_root)
    written = 0
    for dataset_order, _, dataset_name in _TARGETS:
        experiment_name, experiment_rel = experiments[dataset_name]
        for training_seed, split_seed in zip(_TRAINING_SEEDS, _SPLIT_SEEDS):
            cell_id = (
                f"0_high_60_{dataset_order}_{training_seed}_issue151_token_zero_mlp__"
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
                "priority": "high",
                "issue": 151,
                "experiment": "token_zero_supervised_mlp_baseline",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _METHOD,
                "seed": training_seed,
                "split_seed": split_seed,
                "seeded": True,
                "model_total_params": 10_531_842,
                "primary_eval_token": 0,
                "comparison_method": "tokenwise_arch_v1_input_norm_only",
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
