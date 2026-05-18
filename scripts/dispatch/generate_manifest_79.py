"""
generate_manifest_79.py — populate pending/ with one JSON cell per
(experiment_config × method × seed) for the Issue #79 baseline re-train sweep.

Usage:
    python scripts/dispatch/generate_manifest_79.py \
        --dispatch-root shared/issue_79_dispatch \
        [--experiment-glob 'configs/experiments/baseline_comparison_*_memmap.json'] \
        [--methods linear_probe,contrastive,...] \
        [--seeds 0,1,2,3,4] \
        [--skip-existing]

Cell shape (consumed by scripts/dispatch/worker_79.sh):

    {
      "cell_id":           "seed_0__hotpotqa_memmap__linear_probe",
      "experiment_config": "configs/experiments/baseline_comparison_hotpotqa_memmap.json",
      "dataset":           "hotpotqa_memmap",
      "method":            "linear_probe",
      "seed":              0,
      "output_check":      "runs/baseline_comparison_hotpotqa_memmap/hotpotqa_memmap/linear_probe/seed_0/eval_metrics.json"
    }

cell_id ordering is seed-first → dataset → method, so sorted filename claim
order drains one full seed across the matrix before moving on. This gives
one full results table early in the sweep.

Skip rules:
  - Cells already in pending/claimed/done/failed are never re-queued.
  - --skip-existing additionally skips cells whose output_check exists at
    manifest time (default OFF — emit all cells uniformly; the worker re-checks
    at claim time and marks already-done cells complete without re-running).
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402
from scripts.experiment_utils import (  # noqa: E402
    is_seeded_method,
    load_experiment_config,
)

_DEFAULT_GLOB = "configs/experiments/baseline_comparison_*_memmap.json"


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    fname = cell_id + ".json"
    for sub in ("pending", "done", "failed"):
        if (dispatch_root / sub / fname).exists():
            return True
    claimed = dispatch_root / "claimed"
    if claimed.exists():
        for wd in claimed.iterdir():
            if wd.is_dir() and (wd / fname).exists():
                return True
    return False


def _output_path_for(experiment_name: str, dataset: str, method: str, seed: int) -> str:
    return (
        f"runs/{experiment_name}/{dataset}/{method}/seed_{seed}/eval_metrics.json"
    )


def generate_manifest(
    dispatch_root: Path,
    experiment_glob: str,
    method_filter: list[str] | None,
    seed_filter: list[int] | None,
    skip_existing: bool,
) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0

    cfg_paths = sorted(glob(str(_PROJECT_ROOT / experiment_glob)))
    if not cfg_paths:
        print(f"no experiment configs matched: {experiment_glob}", file=sys.stderr)
        return 0

    for cfg_path_str in cfg_paths:
        cfg_path = Path(cfg_path_str)
        # Resolve method configs so we can filter out non-seeded methods.
        # Non-seeded baselines (token_entropy, logprob_baseline) are pure-CPU
        # numpy work and run once, not once-per-seed — they don't belong on the
        # GPU worker queue. Run them directly via run_experiment.py instead.
        cfg = load_experiment_config(str(cfg_path), project_root=str(_PROJECT_ROOT))

        experiment_name = cfg["experiment_name"]
        dataset = cfg["dataset"]
        cfg_methods = cfg.get("methods", [])
        cfg_seeds = cfg.get("training_seeds", [0, 1, 2, 3, 4])
        method_configs = cfg.get("method_configs", {})

        seeded_methods = []
        skipped_non_seeded = []
        for m in cfg_methods:
            if method_filter is not None and m not in method_filter:
                continue
            mcfg = method_configs.get(m, {"training": None})
            if is_seeded_method(mcfg):
                seeded_methods.append(m)
            else:
                skipped_non_seeded.append(m)

        if skipped_non_seeded:
            print(
                f"  {cfg_path.name}: skipping non-seeded methods "
                f"{skipped_non_seeded} (run directly via run_experiment.py)"
            )

        methods = seeded_methods
        seeds = [s for s in cfg_seeds if seed_filter is None or s in seed_filter]

        # Project-relative path string for the cell payload.
        cfg_rel = str(cfg_path.relative_to(_PROJECT_ROOT)).replace("\\", "/")

        for seed in seeds:
            for method in methods:
                cell_id = f"seed_{seed}__{dataset}__{method}"

                if _dispatch_has_cell(dispatch_root, cell_id):
                    continue

                output_check = _output_path_for(experiment_name, dataset, method, seed)
                if skip_existing and (_PROJECT_ROOT / output_check).exists():
                    continue

                cell = {
                    "cell_id":           cell_id,
                    "experiment_config": cfg_rel,
                    "dataset":           dataset,
                    "method":            method,
                    "seed":              int(seed),
                    "output_check":      output_check,
                }
                cell_path = dispatch_root / "pending" / f"{cell_id}.json"
                cell_path.write_text(json.dumps(cell, indent=2), encoding="utf-8")
                written += 1
                print(f"  queued: {cell_id}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate dispatch pending/ queue for Issue #79 sweep."
    )
    parser.add_argument("--dispatch-root", required=True,
                        help="Path to <root>/ for the dispatch queue.")
    parser.add_argument("--experiment-glob", default=_DEFAULT_GLOB,
                        help=f"Glob for experiment configs (default: {_DEFAULT_GLOB}).")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method filter (default: all methods in each config).")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed filter (default: training_seeds from each config).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cells whose output_check exists at manifest time. "
                             "Default OFF — emit uniformly and let the worker decide.")
    args = parser.parse_args()

    method_filter = (
        [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    )
    seed_filter = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    )

    total = generate_manifest(
        Path(args.dispatch_root),
        args.experiment_glob,
        method_filter,
        seed_filter,
        args.skip_existing,
    )
    print(f"Done — {total} cells queued in {args.dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
