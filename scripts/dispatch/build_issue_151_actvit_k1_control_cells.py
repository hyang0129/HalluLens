"""Build the matched ACT-ViT k=1 control cells for Issue #151.

Two arms are emitted into the existing cell-agnostic experiment queue:

* 15 eval-only cells: existing full-width (k=64 capture) ACT-ViT checkpoints
  evaluated at k=1 for NQ, PopQA, and SearchQA. HotpotQA and SciQ already have
  five-seed k64->k1 results and are deliberately not repeated.
* 25 training cells: ACT-ViT trained, validation-selected, and tested at k=1
  over all five benchmark datasets and paired training/split seeds.

The eval-only arm symlinks both canonical checkpoint files into additive run
directories; canonical baseline runs are never modified. MMLU is excluded.
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

_K64_EVAL_METHOD = "act_vit_k64_eval_k1"
_K1_TRAIN_METHOD = "act_vit_k1"
_SEEDS = (0, 1, 2, 3, 4)
_SPLIT_SEEDS = (42, 1, 2, 3, 4)

# HotpotQA and SciQ are already complete under prefix149_actvit_{dataset}.
_K64_EVAL_TARGETS = (
    ("10", "nq", "nq_memmap"),
    ("20", "popqa", "popqa_memmap"),
    ("30", "searchqa", "searchqa_memmap"),
)
_K1_TRAIN_TARGETS = (
    ("10", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)


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


def _load_experiment(project_root: Path, relative_path: str) -> dict:
    path = project_root / relative_path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _link_checkpoint(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if not target.is_file() or target.resolve() != source.resolve():
            raise RuntimeError(
                f"checkpoint target exists but does not resolve to source: {target}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def _validate_methods(project_root: Path) -> None:
    eval_method = load_method_config(
        _K64_EVAL_METHOD, project_root=str(project_root)
    )
    if eval_method.get("routine") != "act_vit":
        raise ValueError(f"{_K64_EVAL_METHOD} must use the ACT-ViT routine")
    if eval_method.get("training", {}).get("fixed_prefix_length") is not None:
        raise ValueError(f"{_K64_EVAL_METHOD} must retain full-width training")
    if eval_method.get("evaluation", {}).get("eval_prefix_lengths") != [1]:
        raise ValueError(f"{_K64_EVAL_METHOD} must evaluate exactly k=1")

    train_method = load_method_config(
        _K1_TRAIN_METHOD, project_root=str(project_root)
    )
    if train_method.get("routine") != "act_vit":
        raise ValueError(f"{_K1_TRAIN_METHOD} must use the ACT-ViT routine")
    if train_method.get("training", {}).get("fixed_prefix_length") != 1:
        raise ValueError(f"{_K1_TRAIN_METHOD} must train at exactly k=1")
    if bool(train_method.get("training", {}).get("prefix_training", False)):
        raise ValueError(f"{_K1_TRAIN_METHOD} must not use mixed-prefix training")


def build(
    dispatch_root: Path,
    *,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
) -> int:
    """Link prerequisites, append missing cells, and return cells written."""
    project_root = Path(project_root)
    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)
    _validate_methods(project_root)

    eval_experiments: dict[str, tuple[str, str]] = {}
    for _, slug, dataset in _K64_EVAL_TARGETS:
        relative = f"configs/experiments/issue151_actvit_k64_to_k1_{slug}.json"
        payload = _load_experiment(project_root, relative)
        expected_name = f"issue151_actvit_k64_to_k1_{slug}"
        if payload.get("experiment_name") != expected_name:
            raise ValueError(f"{relative} has the wrong experiment_name")
        if payload.get("dataset") != dataset or payload.get("methods") != [
            _K64_EVAL_METHOD
        ]:
            raise ValueError(f"{relative} has the wrong dataset or method")
        if payload.get("training_seeds") != list(_SEEDS) or payload.get(
            "split_seeds"
        ) != list(_SPLIT_SEEDS):
            raise ValueError(f"{relative} does not use paired seeds")
        eval_experiments[dataset] = (expected_name, relative)

        source_experiment = f"baseline_comparison_{slug}_memmap"
        for seed in _SEEDS:
            source_artifacts = (
                runs_root
                / source_experiment
                / dataset
                / "act_vit"
                / f"seed_{seed}"
                / "artifacts"
            )
            target_artifacts = (
                runs_root
                / expected_name
                / dataset
                / _K64_EVAL_METHOD
                / f"seed_{seed}"
                / "artifacts"
            )
            for filename in ("best_checkpoint.pt", "final_weights.pt"):
                _link_checkpoint(
                    source_artifacts / filename, target_artifacts / filename
                )

    train_experiments: dict[str, tuple[str, str]] = {}
    for _, slug, dataset in _K1_TRAIN_TARGETS:
        relative = f"configs/experiments/issue151_actvit_k1_{slug}.json"
        payload = _load_experiment(project_root, relative)
        expected_name = f"issue151_actvit_k1_{slug}"
        if payload.get("experiment_name") != expected_name:
            raise ValueError(f"{relative} has the wrong experiment_name")
        if payload.get("dataset") != dataset or payload.get("methods") != [
            _K1_TRAIN_METHOD
        ]:
            raise ValueError(f"{relative} has the wrong dataset or method")
        if payload.get("training_seeds") != list(_SEEDS) or payload.get(
            "split_seeds"
        ) != list(_SPLIT_SEEDS):
            raise ValueError(f"{relative} does not use paired seeds")
        train_experiments[dataset] = (expected_name, relative)

    init_dispatch_dirs(dispatch_root)
    written = 0
    for dataset_order, _, dataset in _K64_EVAL_TARGETS:
        experiment_name, relative = eval_experiments[dataset]
        for seed, split_seed in zip(_SEEDS, _SPLIT_SEEDS):
            cell_id = (
                f"2_eval_{dataset_order}_{seed}_issue151_actvit_k64_to_k1__"
                f"{dataset}__{_K64_EVAL_METHOD}__seed_{seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset
                / _K64_EVAL_METHOD
                / f"seed_{seed}"
            )
            absolute_run_dir = (
                runs_root
                / experiment_name
                / dataset
                / _K64_EVAL_METHOD
                / f"seed_{seed}"
            )
            if (absolute_run_dir / "predictions.csv").is_file():
                print(f"  skip (output complete): {cell_id}")
                continue
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "normal",
                "issue": 151,
                "experiment": "actvit_k64_checkpoint_eval_k1",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": relative,
                "dataset": dataset,
                "method": _K64_EVAL_METHOD,
                "seed": seed,
                "split_seed": split_seed,
                "seeded": True,
                "eval_only": True,
                "training_prefix_length": 64,
                "evaluation_prefix_length": 1,
                "checkpoint_check": str(run_dir / "artifacts/final_weights.pt"),
                "output_check": str(run_dir / "predictions.csv"),
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2) + "\n", encoding="utf-8"
            )
            written += 1
            print(f"  queued: {cell_id}")

    for dataset_order, _, dataset in _K1_TRAIN_TARGETS:
        experiment_name, relative = train_experiments[dataset]
        for seed, split_seed in zip(_SEEDS, _SPLIT_SEEDS):
            cell_id = (
                f"3_train_{dataset_order}_{seed}_issue151_actvit_k1__"
                f"{dataset}__{_K1_TRAIN_METHOD}__seed_{seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset
                / _K1_TRAIN_METHOD
                / f"seed_{seed}"
            )
            absolute_run_dir = (
                runs_root
                / experiment_name
                / dataset
                / _K1_TRAIN_METHOD
                / f"seed_{seed}"
            )
            if (absolute_run_dir / "predictions.csv").is_file():
                print(f"  skip (output complete): {cell_id}")
                continue
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "normal",
                "issue": 151,
                "experiment": "actvit_fixed_k1_train_eval",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": relative,
                "dataset": dataset,
                "method": _K1_TRAIN_METHOD,
                "seed": seed,
                "split_seed": split_seed,
                "seeded": True,
                "training_prefix_length": 1,
                "evaluation_prefix_length": 1,
                "checkpoint_selection_metric": "validation_auroc_at_k1",
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
