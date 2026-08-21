"""Build the gated Issue #156 one-factor architecture sweep.

The code/config matrix can land before Issue #154 finishes, but this builder
refuses to select a recipe or create cells until all nine mixed-half cells are
in the completed state. The caller explicitly chooses a recipe. The selected
``v1`` (t0 + tn) and ``t0`` (t0 + t0 dropout) recipes may coexist for the
matched comparison requested in Issue #156; the legacy ``mixed`` recipe may
not be combined with either one.

The v1 and t0 grids cover HotpotQA, NQ, PopQA, SciQ, and SearchQA x seed 0 x
four trained arms. The projection arm reports both its 512-d trunk and 128-d
projection scores in the same cell. MMLU is excluded.
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

_FIVE_DATASET_TARGETS = (
    ("00", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)
_THREE_DATASET_TARGETS = _FIVE_DATASET_TARGETS[:3]
_RECIPES = {
    "v1": {
        "training_recipe": "v1_first_anchored",
        "baseline_method": "tokenwise_contrastive_first_anchored",
        "baseline_experiment_prefix": "issue151_knnval",
        "experiment_prefix": "issue156_arch_v1",
        "targets": _FIVE_DATASET_TARGETS,
        "methods": (
            "tokenwise_arch_v1_input_norm_only",
            "tokenwise_arch_v1_prenorm_only",
            "tokenwise_arch_v1_attention_pool_only",
            "tokenwise_arch_v1_projection128_only",
        ),
    },
    "t0": {
        "training_recipe": "t0_same_dropout",
        "baseline_method": "tokenwise_causal_t0_dropout",
        "baseline_experiment_prefix": "issue155_causal",
        "experiment_prefix": "issue156_arch_t0",
        "targets": _FIVE_DATASET_TARGETS,
        "methods": (
            "tokenwise_arch_t0_input_norm_only",
            "tokenwise_arch_t0_prenorm_only",
            "tokenwise_arch_t0_attention_pool_only",
            "tokenwise_arch_t0_projection128_only",
        ),
    },
    "mixed": {
        "training_recipe": "mixed_half",
        "baseline_method": "tokenwise_causal_mixed_half",
        "baseline_experiment_prefix": "issue154_mixed",
        "experiment_prefix": "issue156_arch_mixed",
        "targets": _THREE_DATASET_TARGETS,
        "methods": (
            "tokenwise_arch_mixed_input_norm_only",
            "tokenwise_arch_mixed_prenorm_only",
            "tokenwise_arch_mixed_attention_pool_only",
            "tokenwise_arch_mixed_projection128_only",
        ),
    },
}
_EXPECTED_MIXED_CELLS = 9
_SELECTION_FILE = "issue156_recipe_selection.json"


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


def _mixed_cell_states(dispatch_root: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    for state in ("pending", "done", "failed", "cancelled"):
        state_dir = dispatch_root / state
        if not state_dir.exists():
            continue
        for path in state_dir.glob("*issue154_mixed*.json"):
            states[path.name] = state
    claimed = dispatch_root / "claimed"
    if claimed.exists():
        for path in claimed.glob("*/*issue154_mixed*.json"):
            states[path.name] = "claimed"
    return states


def require_completed_mixed_sweep(dispatch_root: Path) -> tuple[str, ...]:
    """Return completed mixed cell IDs or fail before mutating a queue."""
    states = _mixed_cell_states(dispatch_root)
    completed = sorted(name for name, state in states.items() if state == "done")
    incomplete = sorted(
        f"{name}:{state}" for name, state in states.items() if state != "done"
    )
    if len(states) != _EXPECTED_MIXED_CELLS or incomplete:
        raise RuntimeError(
            "Issue #156 recipe selection is gated on the completed Issue #154 "
            f"mixed 3x3 sweep: found {len(completed)}/{_EXPECTED_MIXED_CELLS} "
            f"done; incomplete={incomplete}"
        )
    return tuple(name.removesuffix(".json") for name in completed)


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def _record_recipe_selection(
    dispatch_root: Path,
    *,
    recipe: str,
    completed_mixed_cells: tuple[str, ...],
) -> None:
    selection_path = dispatch_root / _SELECTION_FILE
    recipe_entry = {
        "training_recipe": _RECIPES[recipe]["training_recipe"],
        "methods": list(_RECIPES[recipe]["methods"]),
        "datasets": [target[2] for target in _RECIPES[recipe]["targets"]],
    }
    common = {
        "issue": 156,
        "mixed_sweep_gate": "complete",
        "completed_mixed_cells": list(completed_mixed_cells),
    }
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if "recipes" in existing:
            existing_recipes = dict(existing["recipes"])
        else:
            # Migrate the original scalar selection record written by the
            # three-dataset v1/mixed builder without invalidating queued cells.
            existing_recipe = existing.get("recipe")
            existing_recipes = {
                existing_recipe: {
                    "training_recipe": existing.get("training_recipe"),
                    "methods": existing.get("methods"),
                    "datasets": [
                        target[2]
                        for target in _RECIPES.get(existing_recipe, {}).get(
                            "targets", ()
                        )
                    ],
                }
            }
        if {
            key: existing.get(key) for key in common
        } != common:
            raise RuntimeError(
                "Issue #156 selection gate metadata does not match the "
                "completed mixed sweep"
            )
        selected = set(existing_recipes)
        proposed = selected | {recipe}
        if "mixed" in proposed and len(proposed) > 1:
            raise RuntimeError(
                f"Issue #156 recipes already recorded as {sorted(selected)!r}; "
                f"refusing to mix them with {recipe!r}"
            )
        if recipe in existing_recipes and existing_recipes[recipe] != recipe_entry:
            raise RuntimeError(
                f"Issue #156 recipe {recipe!r} metadata changed after selection"
            )
        existing_recipes[recipe] = recipe_entry
    else:
        existing_recipes = {recipe: recipe_entry}
    payload = {
        **common,
        "comparison": "t0_same_dropout_vs_v1_first_anchored",
        "recipes": existing_recipes,
    }
    selection_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def build(
    dispatch_root: Path,
    *,
    recipe: str,
    mixed_sweep_root: Path,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
) -> int:
    if recipe not in _RECIPES:
        raise ValueError(f"recipe must be one of {sorted(_RECIPES)}, got {recipe!r}")
    completed_mixed_cells = require_completed_mixed_sweep(mixed_sweep_root)
    recipe_cfg = _RECIPES[recipe]
    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)

    # Validate every inherited method before writing the selection record or
    # queue cells. A malformed overlay must fail atomically.
    for method in recipe_cfg["methods"]:
        method_cfg = load_method_config(method, project_root=str(project_root))
        if method_cfg.get("training_recipe") != recipe_cfg["training_recipe"]:
            raise ValueError(f"{method} resolved to the wrong training recipe")

    baseline_paths: dict[str, Path] = {}
    experiment_paths: dict[str, Path] = {}
    for _, slug, dataset_name in recipe_cfg["targets"]:
        baseline_paths[dataset_name] = (
            runs_root
            / f"{recipe_cfg['baseline_experiment_prefix']}_{slug}"
            / dataset_name
            / recipe_cfg["baseline_method"]
            / "seed_0"
            / "eval_metrics.json"
        )
        _require_file(baseline_paths[dataset_name])
        experiment_paths[dataset_name] = (
            project_root
            / "configs"
            / "experiments"
            / f"{recipe_cfg['experiment_prefix']}_{slug}.json"
        )
        _require_file(experiment_paths[dataset_name])

    init_dispatch_dirs(dispatch_root)
    _record_recipe_selection(
        dispatch_root,
        recipe=recipe,
        completed_mixed_cells=completed_mixed_cells,
    )

    written = 0
    for dataset_order, slug, dataset_name in recipe_cfg["targets"]:
        experiment_name = f"{recipe_cfg['experiment_prefix']}_{slug}"
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        for arm_order, method in enumerate(recipe_cfg["methods"]):
            cell_id = (
                f"0_high_{dataset_order}_{arm_order}_issue156_arch__"
                f"{dataset_name}__{method}__seed_0"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / method
                / "seed_0"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "highest",
                "issue": 156,
                "experiment": "one_factor_architecture_pilot",
                "stage": 1,
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": method,
                "seed": 0,
                "seeded": True,
                "selected_recipe": recipe,
                "mixed_sweep_gate": "complete",
                "baseline_run": str(baseline_paths[dataset_name]),
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
    parser.add_argument("--recipe", choices=sorted(_RECIPES), required=True)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_151_knnval_rerun_dispatch",
        help="Generic experiment queue to append to after the gate passes.",
    )
    parser.add_argument(
        "--mixed-sweep-root",
        default=None,
        help="Queue containing the nine Issue #154 mixed cells (defaults to dispatch root).",
    )
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    mixed_sweep_root = (
        dispatch_root if args.mixed_sweep_root is None else Path(args.mixed_sweep_root)
    )
    if not mixed_sweep_root.is_absolute():
        mixed_sweep_root = _PROJECT_ROOT / mixed_sweep_root
    runs_root = None if args.runs_root is None else Path(args.runs_root)
    if runs_root is not None and not runs_root.is_absolute():
        runs_root = _PROJECT_ROOT / runs_root

    count = build(
        dispatch_root,
        recipe=args.recipe,
        mixed_sweep_root=mixed_sweep_root,
        runs_root=runs_root,
    )
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
