"""Build the chatv1 method matrix: 5 chat-templated datasets x 2 models x the
v1 method set, seed 0.

Once the chat-template re-capture (shared/icr_capture_chat/, built by
scripts/dispatch/generate_manifest.py --chat-template) is merged (see
scripts/run_chat_capture_merges.py) and configs/datasets/*_chat_memmap.json /
configs/experiments/chatv1_*.json point at it (see the sibling configs added
alongside this builder), this queues one cell per (task, model, method) for
scripts/dispatch/worker_experiment.sh to drain.

Method set per model (seed 0 only):
  * headline tokenwise arms: tokenwise_contrastive_first_anchored(_qwen3),
    tokenwise_arch_v1_input_norm_only(_qwen3);
  * controls: act_vit (generic — layer-count agnostic, used unmodified for
    both models by the existing baseline_comparison_*_qwen3_memmap configs),
    token_zero_mlp_probe / token_zero_mlp_probe_qwen3 (the qwen3 variant
    exists solely to widen relevant_layers from Llama's 32 to Qwen's 36
    captured post-block layers);
  * cheap baselines: logprob_baseline, token_entropy (generic for both
    models — relevant_layers="14-29" fits inside both 32- and 36-layer
    captures, matching the existing baseline_comparison_* convention).

Cells use the generic experiment worker (scripts/dispatch/worker_experiment.sh)
and are idempotent across all queue states. Tokenwise cells are queued with a
"0_high" cell_id prefix so they sort and get claimed before every other
method (worker claim order is a sorted glob of pending/*.json filenames —
see scripts/dispatch/claim.py::claim_next_cell). MMLU is excluded.
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

_SEED = 0
_SPLIT_SEED = 42

_TASKS = (
    ("10", "hotpotqa"),
    ("20", "nq"),
    ("30", "popqa"),
    ("40", "sciq"),
    ("50", "searchqa"),
)

# (method, priority_prefix, priority_label) -- ordered so tokenwise methods
# (the headline arms) queue first, per model.
_LLAMA_METHODS = (
    ("tokenwise_contrastive_first_anchored", "0_high", "high"),
    ("tokenwise_arch_v1_input_norm_only", "0_high", "high"),
    ("act_vit", "5_mid", "normal"),
    ("token_zero_mlp_probe", "5_mid", "normal"),
    # Legacy CLR (full-response contrastive) — re-ranks CLR vs act_vit on the
    # chat-templated data; generic across models (layers 14-29, stored top-k
    # is sliced from 500 down to its requested 20 by the loader).
    ("contrastive_logprob_recon", "5_mid", "normal"),
    # t0-recipe (first_same, "t0+t0") arms: trained AND evaluated at token
    # zero, closing the recipe gap in the v1-vs-token_zero_mlp comparison.
    ("tokenwise_causal_t0_dropout", "6_t0", "normal"),
    ("tokenwise_arch_t0_input_norm_only", "6_t0", "normal"),
    # Probe-capacity ladder: v1 input-norm recipe at 20.9M / 5.7M / 2.1M params
    # (vs 77.5M full) — tests whether the token-zero signal needs capacity.
    ("tokenwise_v1_inputnorm_probe21m", "7_small", "normal"),
    ("tokenwise_v1_inputnorm_probe6m", "7_small", "normal"),
    ("tokenwise_v1_inputnorm_probe2m", "7_small", "normal"),
    ("logprob_baseline", "8_low", "low"),
    ("token_entropy", "8_low", "low"),
)
_QWEN_METHODS = (
    ("tokenwise_contrastive_first_anchored_qwen3", "0_high", "high"),
    ("tokenwise_arch_v1_input_norm_only_qwen3", "0_high", "high"),
    ("act_vit", "5_mid", "normal"),
    ("token_zero_mlp_probe_qwen3", "5_mid", "normal"),
    ("contrastive_logprob_recon", "5_mid", "normal"),
    ("tokenwise_causal_t0_dropout_qwen3", "6_t0", "normal"),
    ("tokenwise_arch_t0_input_norm_only_qwen3", "6_t0", "normal"),
    ("tokenwise_v1_inputnorm_probe21m_qwen3", "7_small", "normal"),
    ("tokenwise_v1_inputnorm_probe6m_qwen3", "7_small", "normal"),
    ("tokenwise_v1_inputnorm_probe2m_qwen3", "7_small", "normal"),
    ("logprob_baseline", "8_low", "low"),
    ("token_entropy", "8_low", "low"),
)

# Methods run_experiment treats as unseeded: no seed_N/ run-dir level, no
# predictions.csv — completion is signalled by eval_metrics.json.
_UNSEEDED_METHODS = frozenset({"logprob_baseline", "token_entropy"})

# (model_tag, dataset_suffix, experiment_prefix, methods)
_MODELS = (
    ("llama", "_chat_memmap", "chatv1", _LLAMA_METHODS),
    ("qwen3", "_qwen3_chat_memmap", "chatv1_qwen3", _QWEN_METHODS),
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


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_configs(project_root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Validate every dataset/experiment config referenced by the matrix.

    Returns {(model_tag, task): (experiment_name, experiment_rel)}.
    """
    experiments: dict[tuple[str, str], tuple[str, str]] = {}
    for model_tag, dataset_suffix, experiment_prefix, methods in _MODELS:
        for _, task in _TASKS:
            dataset_name = f"{task}{dataset_suffix}"
            dataset_path = project_root / "configs/datasets" / f"{dataset_name}.json"
            dataset_cfg = _load_json(dataset_path)
            if dataset_cfg.get("name") != dataset_name:
                raise ValueError(f"{dataset_path} has mismatched 'name'")

            experiment_rel = f"configs/experiments/{experiment_prefix}_{task}.json"
            experiment_path = project_root / experiment_rel
            experiment = _load_json(experiment_path)
            expected_name = f"{experiment_prefix}_{task}"
            if experiment.get("experiment_name") != expected_name:
                raise ValueError(f"{experiment_rel} has the wrong experiment_name")
            if experiment.get("dataset") != dataset_name:
                raise ValueError(f"{experiment_rel} must reference dataset {dataset_name}")
            expected_methods = [m for m, _, _ in methods]
            if experiment.get("methods") != expected_methods:
                raise ValueError(
                    f"{experiment_rel} methods {experiment.get('methods')} != "
                    f"expected {expected_methods}"
                )
            if experiment.get("training_seeds") != [_SEED]:
                raise ValueError(f"{experiment_rel} must use training_seeds=[{_SEED}]")

            for method, _, _ in methods:
                load_method_config(method, project_root=str(project_root))

            experiments[(model_tag, task)] = (expected_name, experiment_rel)
    return experiments


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    """Append missing cells and return the number written."""
    project_root = Path(project_root)
    experiments = _validate_configs(project_root)

    init_dispatch_dirs(dispatch_root)
    written = 0
    for model_tag, dataset_suffix, experiment_prefix, methods in _MODELS:
        for dataset_order, task in _TASKS:
            dataset_name = f"{task}{dataset_suffix}"
            experiment_name, experiment_rel = experiments[(model_tag, task)]
            for method, priority_prefix, priority_label in methods:
                cell_id = (
                    f"{priority_prefix}_{dataset_order}_{_SEED}_chatv1__"
                    f"{dataset_name}__{method}__seed_{_SEED}"
                )
                if _dispatch_has_cell(dispatch_root, cell_id):
                    print(f"  skip (already queued): {cell_id}")
                    continue
                # Why: logprob/entropy baselines are unseeded in run_experiment —
                # they write runs/<exp>/<ds>/<method>/eval_metrics.json with no
                # seed_N/ level and no predictions.csv. Pointing output_check at
                # the seeded path marks a successful run as failed.
                unseeded = method in _UNSEEDED_METHODS
                run_dir = Path("runs") / experiment_name / dataset_name / method
                if not unseeded:
                    run_dir = run_dir / f"seed_{_SEED}"
                output_check = run_dir / ("eval_metrics.json" if unseeded else "predictions.csv")
                cell = {
                    "cell_id": cell_id,
                    "kind": "experiment",
                    "priority": priority_label,
                    "experiment": "chatv1_matrix",
                    "backbone": "Qwen3-8B" if model_tag == "qwen3" else "Llama-3.1-8B-Instruct",
                    "worker_script": "scripts/dispatch/worker_experiment.sh",
                    "experiment_config": experiment_rel,
                    "dataset": dataset_name,
                    "method": method,
                    "seed": _SEED,
                    "split_seed": _SPLIT_SEED,
                    "seeded": not unseeded,
                    "output_check": str(output_check),
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
        default="shared/chatv1_matrix_dispatch",
        help="Dispatch queue root to create/append to.",
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
