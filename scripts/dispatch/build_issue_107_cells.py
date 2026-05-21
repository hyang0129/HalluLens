"""
build_issue_107_cells.py — populate pending/ with one cell per (dataset, model)
to backfill per-sample KNN predictions.csv for the contrastive_logprob_recon
method (issue #107).

Each cell bundles all 5 training seeds into a single run_experiment.py
invocation. Bundling keeps the activations memmap warm in the OS page cache
across seeds and amortises Python/CUDA/HF init — modest (~10%) but free given
the inner-loop resume rule in run_experiment.py only re-runs seeds that are
missing predictions.csv.

The existing worker_79.sh routes the default task_type to:

    python scripts/run_experiment.py \
        --experiment <EXPERIMENT> --methods <METHOD> --seeds <SEED>

so we set:
    method = "contrastive_logprob_recon"
    seed   = "0,1,2,3,4"   (comma-separated string; argparse splits on ",")
    output_check = .../seed_4/predictions.csv

predictions.csv for seed_4 is the last per-seed artefact written by the
bundled run. run_experiment.py now exits non-zero if any seed errors, and
worker_79 marks the cell done only when (exit_code == 0 AND output_check
exists), so a partial failure correctly routes the cell to failed/ for
retry — even though seed_4 might have written its csv before an earlier
seed failed.

Usage:
    python scripts/dispatch/build_issue_107_cells.py \
        --dispatch-root shared/issue_107_dispatch
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

# (experiment_config_basename, dataset_key) — derived from the canonical
# baseline_comparison_*_memmap experiments in configs/experiments/.
_TARGETS = [
    ("baseline_comparison_hotpotqa_memmap",       "hotpotqa_memmap"),
    ("baseline_comparison_hotpotqa_qwen3_memmap", "hotpotqa_qwen3_memmap"),
    ("baseline_comparison_mmlu_memmap",           "mmlu_memmap"),
    ("baseline_comparison_mmlu_qwen3_memmap",     "mmlu_qwen3_memmap"),
    ("baseline_comparison_nq_memmap",             "nq_memmap"),
    ("baseline_comparison_nq_qwen3_memmap",       "nq_qwen3_memmap"),
    ("baseline_comparison_popqa_memmap",          "popqa_memmap"),
    ("baseline_comparison_popqa_qwen3_memmap",    "popqa_qwen3_memmap"),
    ("baseline_comparison_sciq_memmap",           "sciq_memmap"),
    ("baseline_comparison_sciq_qwen3_memmap",     "sciq_qwen3_memmap"),
    ("baseline_comparison_searchqa_memmap",       "searchqa_memmap"),
    ("baseline_comparison_searchqa_qwen3_memmap", "searchqa_qwen3_memmap"),
]

_METHOD = "contrastive_logprob_recon"
_SEEDS_CSV = "0,1,2,3,4"
_LAST_SEED = 4


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


def build(dispatch_root: Path) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0

    for experiment_name, dataset_key in _TARGETS:
        cfg_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / cfg_rel).exists():
            print(f"  skip (config missing): {cfg_rel}", file=sys.stderr)
            continue

        cell_id = f"1_issue107__{dataset_key}__{_METHOD}"
        if _dispatch_has_cell(dispatch_root, cell_id):
            print(f"  skip (already queued): {cell_id}")
            continue

        output_check = (
            f"runs/{experiment_name}/{dataset_key}/{_METHOD}/seed_{_LAST_SEED}/predictions.csv"
        )

        cell = {
            "cell_id":           cell_id,
            "experiment_config": cfg_rel,
            "dataset":           dataset_key,
            "method":            _METHOD,
            "seed":              _SEEDS_CSV,
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
        description="Queue contrastive_logprob_recon predictions backfill cells (issue #107)."
    )
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_107_dispatch",
        help="Dispatch root (default: shared/issue_107_dispatch).",
    )
    args = parser.parse_args()

    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root

    total = build(dispatch_root)
    print(f"Done — {total} cells queued in {dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
