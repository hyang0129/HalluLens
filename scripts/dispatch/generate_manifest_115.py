"""
generate_manifest_115.py — queue cells for the Qwen3 flipped-convention sweep
(issue #115: 5 datasets × 5 seeds of contrastive_logprob_recon_b5).

Cell shape (consumed by scripts/dispatch/worker_79.sh, same protocol as #79):

    {
      "cell_id":           "seed_0__sciq_qwen3_memmap__contrastive_logprob_recon_b5",
      "experiment_config": "configs/experiments/baseline_comparison_sciq_qwen3_flipped_memmap.json",
      "dataset":           "sciq_qwen3_memmap",
      "method":            "contrastive_logprob_recon_b5",
      "seed":              0,
      "output_check":      "runs/baseline_comparison_sciq_qwen3_flipped_memmap/sciq_qwen3_memmap/contrastive_logprob_recon_b5/seed_0/predictions.csv"
    }

cell_id is `seed_{N}__{dataset}__{method}`, so sorted-filename claim order
drains all 5 datasets at seed 0 before any seed 1 cell is claimed. That gives
a full results table at seed 0 in ~5 GPU-hours, well before the sweep is done.

output_check points at predictions.csv (not eval_metrics.json) because the
polarity fix in this PR specifically writes predictions.csv under the flipped
convention — verifying the file exists is the cheap proxy for "the polarity
fix ran". The worker re-checks this at claim time and marks pre-existing
cells complete without re-running.

Usage:
    python scripts/dispatch/generate_manifest_115.py \\
        --dispatch-root shared/issue_115_dispatch
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

# (experiment_config_basename, dataset_key) — the 5 Qwen3 flipped experiments
# created for issue #115. MMLU is excluded (cut from EMNLP paper).
_TARGETS = [
    ("baseline_comparison_hotpotqa_qwen3_flipped_memmap", "hotpotqa_qwen3_memmap"),
    ("baseline_comparison_nq_qwen3_flipped_memmap",       "nq_qwen3_memmap"),
    ("baseline_comparison_popqa_qwen3_flipped_memmap",    "popqa_qwen3_memmap"),
    ("baseline_comparison_sciq_qwen3_flipped_memmap",     "sciq_qwen3_memmap"),
    ("baseline_comparison_searchqa_qwen3_flipped_memmap", "searchqa_qwen3_memmap"),
]

_METHOD = "contrastive_logprob_recon_b5"
_SEEDS = [0, 1, 2, 3, 4]


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
        f"runs/{experiment_name}/{dataset}/{method}/seed_{seed}/predictions.csv"
    )


def build(dispatch_root: Path, skip_existing: bool) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0

    # Outer loop over seeds (not datasets) so the seed-first ordering shows up
    # in the queued log even though sorted-filename claim already enforces it.
    for seed in _SEEDS:
        for experiment_name, dataset_key in _TARGETS:
            cfg_rel = f"configs/experiments/{experiment_name}.json"
            if not (_PROJECT_ROOT / cfg_rel).exists():
                print(f"  skip (config missing): {cfg_rel}", file=sys.stderr)
                continue

            cell_id = f"seed_{seed}__{dataset_key}__{_METHOD}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue

            output_check = _output_path_for(experiment_name, dataset_key, _METHOD, seed)
            if skip_existing and (_PROJECT_ROOT / output_check).exists():
                print(f"  skip (output exists): {cell_id}")
                continue

            cell = {
                "cell_id":           cell_id,
                "experiment_config": cfg_rel,
                "dataset":           dataset_key,
                "method":            _METHOD,
                "seed":              int(seed),
                "output_check":      output_check,
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2), encoding="utf-8"
            )
            written += 1
            print(f"  queued: {cell_id}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Queue Qwen3 flipped-convention cells (issue #115)."
    )
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_115_dispatch",
        help="Dispatch root (default: shared/issue_115_dispatch).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells whose output_check exists at manifest time. "
             "Default OFF — emit uniformly; the worker re-checks at claim time.",
    )
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root

    total = build(dispatch_root, args.skip_existing)
    print(f"Done — {total} cells queued in {dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
