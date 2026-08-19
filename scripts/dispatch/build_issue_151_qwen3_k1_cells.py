"""Build the 75-cell Qwen3-8B first-token comparison matrix.

Grid: five datasets x five paired seeds x three methods:

* tokenwise contrastive v1 over all 36 Qwen post-block layers;
* ACT-ViT trained, validation-selected, and evaluated at k=1;
* canonical full-width ACT-ViT checkpoint evaluated at k=1 without retraining.

The eval-only arm symlinks canonical checkpoint artifacts into additive Issue
#151 run directories. Cells use the existing cell-agnostic experiment workers,
are idempotent across queue states, and deliberately exclude MMLU.
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

_TOKENWISE_METHOD = "tokenwise_contrastive_first_anchored_qwen3"
_ACTVIT_K1_METHOD = "act_vit_k1"
_ACTVIT_K64_EVAL_METHOD = "act_vit_k64_eval_k1"
_METHODS = (
    _TOKENWISE_METHOD,
    _ACTVIT_K1_METHOD,
    _ACTVIT_K64_EVAL_METHOD,
)
_TARGETS = (
    ("10", "hotpotqa", "hotpotqa_qwen3_memmap"),
    ("20", "nq", "nq_qwen3_memmap"),
    ("30", "popqa", "popqa_qwen3_memmap"),
    ("40", "sciq", "sciq_qwen3_memmap"),
    ("50", "searchqa", "searchqa_qwen3_memmap"),
)
_SEEDS = (0, 1, 2, 3, 4)
_SPLIT_SEEDS = (42, 1, 2, 3, 4)


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


def _load_json(path: Path) -> dict:
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
    tokenwise = load_method_config(
        _TOKENWISE_METHOD, project_root=str(project_root)
    )
    if tokenwise.get("routine") != "tokenwise_contrastive_logprob_recon":
        raise ValueError(f"{_TOKENWISE_METHOD} resolved to the wrong routine")
    if tokenwise.get("data", {}).get("relevant_layers") != "1-36":
        raise ValueError(f"{_TOKENWISE_METHOD} must use all 36 Qwen layers")
    if tokenwise.get("data", {}).get("token_pair_mode") != "first_anchored":
        raise ValueError(f"{_TOKENWISE_METHOD} must retain the v1 recipe")

    actvit_k1 = load_method_config(
        _ACTVIT_K1_METHOD, project_root=str(project_root)
    )
    if actvit_k1.get("training", {}).get("fixed_prefix_length") != 1:
        raise ValueError(f"{_ACTVIT_K1_METHOD} must train at k=1")
    if bool(actvit_k1.get("training", {}).get("prefix_training", False)):
        raise ValueError(f"{_ACTVIT_K1_METHOD} must not use mixed prefixes")

    actvit_eval = load_method_config(
        _ACTVIT_K64_EVAL_METHOD, project_root=str(project_root)
    )
    if actvit_eval.get("training", {}).get("fixed_prefix_length") is not None:
        raise ValueError(f"{_ACTVIT_K64_EVAL_METHOD} must retain k64 training")
    if actvit_eval.get("evaluation", {}).get("eval_prefix_lengths") != [1]:
        raise ValueError(f"{_ACTVIT_K64_EVAL_METHOD} must evaluate at k=1")


def build(
    dispatch_root: Path,
    *,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
) -> int:
    """Link Qwen checkpoints, append missing cells, and return cells written."""
    project_root = Path(project_root)
    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)
    _validate_methods(project_root)

    experiments: dict[str, tuple[str, str, str]] = {}
    for _, slug, dataset in _TARGETS:
        dataset_cfg = _load_json(
            project_root / "configs/datasets" / f"{dataset}.json"
        )
        if dataset_cfg.get("model_name") != "Qwen3-8B":
            raise ValueError(f"{dataset} is not a Qwen3-8B capture")

        relative = f"configs/experiments/issue151_qwen3_k1_controls_{slug}.json"
        experiment = _load_json(project_root / relative)
        expected_name = f"issue151_qwen3_k1_controls_{slug}"
        if experiment.get("experiment_name") != expected_name:
            raise ValueError(f"{relative} has the wrong experiment_name")
        if experiment.get("dataset") != dataset:
            raise ValueError(f"{relative} has the wrong dataset")
        if experiment.get("methods") != list(_METHODS):
            raise ValueError(f"{relative} has the wrong method matrix")
        if experiment.get("training_seeds") != list(_SEEDS) or experiment.get(
            "split_seeds"
        ) != list(_SPLIT_SEEDS):
            raise ValueError(f"{relative} does not use paired seeds")
        experiments[dataset] = (expected_name, relative, slug)

        source_experiment = f"baseline_comparison_{slug}_qwen3_memmap"
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
                / _ACTVIT_K64_EVAL_METHOD
                / f"seed_{seed}"
                / "artifacts"
            )
            for filename in ("best_checkpoint.pt", "final_weights.pt"):
                _link_checkpoint(
                    source_artifacts / filename, target_artifacts / filename
                )

    init_dispatch_dirs(dispatch_root)
    written = 0
    method_specs = (
        ("4_eval", _ACTVIT_K64_EVAL_METHOD, "actvit_qwen_k64_eval_k1"),
        ("5_v1", _TOKENWISE_METHOD, "tokenwise_v1_qwen_k1"),
        ("6_train", _ACTVIT_K1_METHOD, "actvit_qwen_k1_train_eval"),
    )
    for prefix, method, study_arm in method_specs:
        for dataset_order, _, dataset in _TARGETS:
            experiment_name, relative, _ = experiments[dataset]
            for seed, split_seed in zip(_SEEDS, _SPLIT_SEEDS):
                cell_id = (
                    f"{prefix}_{dataset_order}_{seed}_issue151_qwen3__"
                    f"{dataset}__{method}__seed_{seed}"
                )
                if _dispatch_has_cell(dispatch_root, cell_id):
                    print(f"  skip (already queued): {cell_id}")
                    continue
                run_dir = (
                    Path("runs")
                    / experiment_name
                    / dataset
                    / method
                    / f"seed_{seed}"
                )
                absolute_run_dir = (
                    runs_root
                    / experiment_name
                    / dataset
                    / method
                    / f"seed_{seed}"
                )
                if (absolute_run_dir / "predictions.csv").is_file():
                    print(f"  skip (output complete): {cell_id}")
                    continue
                eval_only = method == _ACTVIT_K64_EVAL_METHOD
                cell = {
                    "cell_id": cell_id,
                    "kind": "experiment",
                    "priority": "normal",
                    "issue": 151,
                    "experiment": study_arm,
                    "backbone": "Qwen3-8B",
                    "worker_script": "scripts/dispatch/worker_experiment.sh",
                    "experiment_config": relative,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "split_seed": split_seed,
                    "seeded": True,
                    "evaluation_prefix_length": 1,
                    "output_check": str(run_dir / "predictions.csv"),
                }
                if eval_only:
                    cell.update(
                        {
                            "eval_only": True,
                            "training_prefix_length": 64,
                            "checkpoint_check": str(
                                run_dir / "artifacts/final_weights.pt"
                            ),
                        }
                    )
                elif method == _ACTVIT_K1_METHOD:
                    cell.update(
                        {
                            "training_prefix_length": 1,
                            "checkpoint_selection_metric": (
                                "validation_auroc_at_k1"
                            ),
                        }
                    )
                else:
                    cell.update(
                        {
                            "training_view": "token0_plus_same_response_later",
                            "inference_token": 0,
                            "relevant_layers": "1-36",
                            "checkpoint_selection_metric": (
                                "validation_knn_auroc_at_token0"
                            ),
                        }
                    )
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
