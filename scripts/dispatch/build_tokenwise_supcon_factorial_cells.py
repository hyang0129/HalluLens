"""Append the matched standard-SupCon view x reconstruction factorial.

The completed Issue #151 v1 run supplies the t0+tn/full-reconstruction arm.
This builder adds the other three cells in that 2x2 design over the five
HalluLens benchmarks at seed 0 (15 cells total).  Every arm uses the original
77,538,112-parameter v1 encoder/decoder, the pair-eligible response-length>=2
cohort, and the v1 optimizer, validation, and scoring recipe.  MMLU is
deliberately excluded.

The builder only appends queue files.  It does not launch workers, and it is
safe to rerun against a partially or fully populated generic queue.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402
from scripts.experiment_utils import load_method_config  # noqa: E402

_BASELINE_METHOD = "tokenwise_contrastive_first_anchored"
_REFERENCE_TOTAL_PARAMS = 77_538_112
_TARGETS = (
    ("00", "hotpotqa", "hotpotqa_memmap"),
    ("20", "nq", "nq_memmap"),
    ("30", "popqa", "popqa_memmap"),
    ("40", "sciq", "sciq_memmap"),
    ("50", "searchqa", "searchqa_memmap"),
)
_METHODS = (
    "tokenwise_supcon_t0_full_recon",
    "tokenwise_supcon_tn_no_recon",
    "tokenwise_supcon_t0_no_recon",
)
_FACTORS = {
    "tokenwise_supcon_t0_full_recon": ("t0_plus_t0_dropout", "full_response"),
    "tokenwise_supcon_tn_no_recon": (
        "t0_plus_random_same_response_tn",
        "none",
    ),
    "tokenwise_supcon_t0_no_recon": ("t0_plus_t0_dropout", "none"),
}


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


def _normalized_matched_recipe(config: dict) -> dict:
    """Remove only the two factorial dimensions and audit-only metadata."""
    payload = deepcopy(config)
    for key in (
        "name",
        "extends",
        "study",
        "factor_views",
        "factor_reconstruction",
    ):
        payload.pop(key, None)
    payload["model_params"].pop("expected_total_params", None)
    payload["model_params"].pop("reference_total_params", None)
    payload["model_params"]["recon_lambda"] = "<factor>"
    payload["training"]["contrastive_objective"] = "<fixed-legacy-supcon>"
    payload["data"]["token_pair_mode"] = "<factor>"
    return payload


def _validate_methods(project_root: Path) -> None:
    baseline = load_method_config(
        _BASELINE_METHOD, project_root=str(project_root)
    )
    if int(baseline["data"].get("min_response_tokens", 2)) != 2:
        raise ValueError("v1 baseline no longer uses the response-length>=2 cohort")
    expected = _normalized_matched_recipe(baseline)

    for method in _METHODS:
        config = load_method_config(method, project_root=str(project_root))
        if config["routine"] != "tokenwise_contrastive_logprob_recon":
            raise ValueError(f"{method} resolved to the wrong training routine")
        if config["model_class"] != "logprob_recon_progressive_compressor":
            raise ValueError(f"{method} resolved to the wrong model class")
        if config["training"].get("contrastive_objective") != "legacy_supcon":
            raise ValueError(f"{method} must use standard/legacy SupCon")
        if int(config["data"].get("min_response_tokens", -1)) != 2:
            raise ValueError(f"{method} must use the matched response-length>=2 cohort")
        if int(config["model_params"].get("expected_total_params", -1)) != (
            _REFERENCE_TOTAL_PARAMS
        ):
            raise ValueError(f"{method} is missing the fixed v1 parameter guard")
        if _normalized_matched_recipe(config) != expected:
            raise ValueError(
                f"{method} changes settings outside view construction and "
                "reconstruction weight"
            )


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    """Append missing factorial cells and return the number written."""
    _validate_methods(project_root)

    experiment_paths: dict[str, str] = {}
    for _, slug, _ in _TARGETS:
        relative = f"configs/experiments/issue151_supcon_factorial_{slug}.json"
        if not (project_root / relative).is_file():
            raise FileNotFoundError(project_root / relative)
        experiment_paths[slug] = relative

    init_dispatch_dirs(dispatch_root)
    written = 0
    for dataset_order, slug, dataset_name in _TARGETS:
        experiment_name = f"issue151_supcon_factorial_{slug}"
        for method_order, method in enumerate(_METHODS):
            cell_id = (
                f"0_high_{dataset_order}_{method_order}_supcon_factorial__"
                f"{dataset_name}__{method}__seed_0"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue

            factor_views, factor_reconstruction = _FACTORS[method]
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / method
                / "seed_0"
            )
            baseline_run = (
                Path("runs")
                / f"issue151_knnval_{slug}"
                / dataset_name
                / _BASELINE_METHOD
                / "seed_0"
                / "eval_metrics.json"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "priority": "highest",
                "issue": 151,
                "experiment": "standard_supcon_view_reconstruction_factorial",
                "worker_script": "scripts/dispatch/worker_experiment.sh",
                "experiment_config": experiment_paths[slug],
                "dataset": dataset_name,
                "method": method,
                "seed": 0,
                "seeded": True,
                "factor_views": factor_views,
                "factor_reconstruction": factor_reconstruction,
                "contrastive_objective": "legacy_supcon",
                "matched_min_response_tokens": 2,
                "expected_total_params": _REFERENCE_TOTAL_PARAMS,
                "comparison_baseline_method": _BASELINE_METHOD,
                "comparison_baseline_run": str(baseline_run),
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
