#!/usr/bin/env python3
"""Run the validation-selected token-wise scorer study from Issue #157.

Examples:

  # Does not open test arrays.
  python scripts/run_tokenwise_scorer_study.py prepare \
    --run-dir runs/issue151_knnval_hotpotqa/hotpotqa_memmap/\
tokenwise_contrastive_first_anchored/seed_0

  # One lock for the full recipe, never one lock per dataset.
  python scripts/run_tokenwise_scorer_study.py select \
    --training-recipe tokenwise_contrastive_first_anchored \
    --validation-manifest runs/**/scorer_study/validation_manifest.json \
    --output scorer_results/v1_selection_lock.json

  # Opens test only after the lock exists; evaluates only the locked scorer.
  python scripts/run_tokenwise_scorer_study.py finalize \
    --selection scorer_results/v1_selection_lock.json \
    --output-dir scorer_results/v1_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_research.tokenwise_scorers import (  # noqa: E402
    DEFAULT_DATASETS,
    finalize_locked_scorer,
    prepare_validation_run,
    select_global_scorer,
)


def _prepare(args: argparse.Namespace) -> int:
    result = prepare_validation_run(args.run_dir, output_dir=args.output_dir)
    print(result)
    return 0


def _select(args: argparse.Namespace) -> int:
    result = select_global_scorer(
        args.validation_manifest,
        training_recipe=args.training_recipe,
        output_path=args.output,
        expected_datasets=args.expected_datasets,
    )
    print(result)
    return 0


def _finalize(args: argparse.Namespace) -> int:
    result = finalize_locked_scorer(args.selection, output_dir=args.output_dir)
    print(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="fit candidates on train and score validation only"
    )
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--output-dir", default=None)
    prepare.set_defaults(handler=_prepare)

    select = subparsers.add_parser(
        "select", help="lock one scorer by macro validation AUROC"
    )
    select.add_argument("--training-recipe", required=True)
    select.add_argument(
        "--validation-manifest", nargs="+", required=True
    )
    select.add_argument("--output", required=True)
    select.add_argument(
        "--expected-datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
    )
    select.set_defaults(handler=_select)

    finalize = subparsers.add_parser(
        "finalize", help="evaluate only the validation-locked scorer on test"
    )
    finalize.add_argument("--selection", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.set_defaults(handler=_finalize)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
