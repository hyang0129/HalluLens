"""Queue checkpoint-only train/validation/test embedding artifact backfills.

Issue #157 needs validation embeddings for the already completed five-dataset,
five-seed v1 and t0-control sweeps.  These cells reuse ``final_weights.pt`` and
invoke the normal evaluator with ``--eval-only``.  The runner preserves the
existing config, metrics, and predictions while writing the missing embedding
manifest.  No model training is permitted.

The builder is idempotent across all queue states.  MMLU is not in scope.
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
    ("10", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)
_RECIPES = {
    "v1": {
        "method": "tokenwise_contrastive_first_anchored",
        "experiment_prefix": "issue151_knnval",
        "config_prefix": "issue151_knnval",
    },
    "t0": {
        "method": "tokenwise_causal_t0_dropout",
        "experiment_prefix": "issue155_causal",
        "config_prefix": "issue155_causal",
    },
}
_SEEDS = (0, 1, 2, 3, 4)


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    filename = f"{cell_id}.json"
    for state in ("pending", "done", "failed", "cancelled"):
        if (dispatch_root / state / filename).exists():
            return True
    claimed_root = dispatch_root / "claimed"
    return claimed_root.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed_root.iterdir()
    )


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def build(
    dispatch_root: Path,
    *,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
    recipes: tuple[str, ...] = ("v1", "t0"),
) -> int:
    unknown = set(recipes) - set(_RECIPES)
    if unknown:
        raise ValueError(f"unknown recipes: {sorted(unknown)}")
    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)

    planned = []
    for recipe in recipes:
        spec = _RECIPES[recipe]
        for dataset_order, slug, dataset_name in _TARGETS:
            experiment_name = f"{spec['experiment_prefix']}_{slug}"
            experiment_rel = (
                f"configs/experiments/{spec['config_prefix']}_{slug}.json"
            )
            experiment_path = project_root / experiment_rel
            _require_nonempty(experiment_path)
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            if experiment.get("experiment_name") != experiment_name:
                raise ValueError(f"{experiment_rel}: experiment_name mismatch")
            if spec["method"] not in experiment.get("methods", []):
                raise ValueError(f"{experiment_rel}: missing {spec['method']}")
            if experiment.get("training_seeds") != list(_SEEDS):
                raise ValueError(f"{experiment_rel}: expected five training seeds")

            for seed in _SEEDS:
                run_dir = (
                    runs_root
                    / experiment_name
                    / dataset_name
                    / spec["method"]
                    / f"seed_{seed}"
                )
                checkpoint = run_dir / "artifacts" / "final_weights.pt"
                _require_nonempty(checkpoint)
                _require_nonempty(run_dir / "eval_metrics.json")
                _require_nonempty(run_dir / "predictions.csv")
                planned.append(
                    (
                        recipe,
                        dataset_order,
                        dataset_name,
                        seed,
                        experiment_rel,
                        run_dir,
                        checkpoint,
                    )
                )

    init_dispatch_dirs(dispatch_root)
    written = 0
    for (
        recipe,
        dataset_order,
        dataset_name,
        seed,
        experiment_rel,
        run_dir,
        checkpoint,
    ) in planned:
        method = _RECIPES[recipe]["method"]
        cell_id = (
            f"1_scorer_{recipe}_{dataset_order}_{seed}_issue157_valdump__"
            f"{dataset_name}__{method}__seed_{seed}"
        )
        if _dispatch_has_cell(dispatch_root, cell_id):
            print(f"  skip (already queued): {cell_id}")
            continue
        manifest = run_dir / "embeddings" / "manifest.json"
        if manifest.is_file() and manifest.stat().st_size > 0:
            print(f"  skip (manifest exists): {cell_id}")
            continue

        relative_run = run_dir.relative_to(runs_root.parent)
        relative_checkpoint = checkpoint.relative_to(runs_root.parent)
        cell = {
            "cell_id": cell_id,
            "kind": "experiment",
            "priority": "high",
            "issue": 157,
            "experiment": "tokenwise_validation_embedding_backfill",
            "recipe": recipe,
            "worker_script": "scripts/dispatch/worker_experiment.sh",
            "experiment_config": experiment_rel,
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "seeded": True,
            "eval_only": True,
            "embedding_backfill": True,
            "retraining_permitted": False,
            "preserve_existing_eval_outputs": True,
            "checkpoint_check": str(relative_checkpoint),
            "output_check": str(relative_run / "embeddings" / "manifest.json"),
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
    parser.add_argument("--runs-root", default=None)
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=sorted(_RECIPES),
        default=["v1", "t0"],
    )
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    runs_root = None if args.runs_root is None else Path(args.runs_root)
    if runs_root is not None and not runs_root.is_absolute():
        runs_root = _PROJECT_ROOT / runs_root
    written = build(
        dispatch_root,
        runs_root=runs_root,
        recipes=tuple(args.recipes),
    )
    print(f"\nqueued {written} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
