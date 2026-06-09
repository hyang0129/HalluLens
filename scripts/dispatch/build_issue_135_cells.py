"""
build_issue_135_cells.py — queue cells to complete the flipped/b5 grid for the
contrastive_logprob_recon_b5 method (issue #135 complementarity work).

One cell per (flipped experiment, seed-bundle). Each cell runs
contrastive_logprob_recon_b5 for its seeds via run_experiment.py's inner-loop
resume (skips seeds whose predictions.csv already exists), so a worker that
re-claims a partially-done cell finishes only the missing seeds.

Coverage (the missing flipped cells):
  - triviaqa L / Q     : all 5 seeds (no flipped runs existed)
  - hotpotqa/popqa/sciq: seeds 2,3,4 (only seeds 0,1 existed)
nq + searchqa seeds 2,3,4 are intentionally omitted here — they are handled by
an already-running manual job; queue them too if that job dies.

Usage:
    python scripts/dispatch/build_issue_135_cells.py \
        --dispatch-root shared/issue_135_dispatch
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

_METHOD = "contrastive_logprob_recon_b5"

# (experiment_name, dataset_key, seeds_csv, last_seed)
_TARGETS = [
    ("baseline_comparison_triviaqa_flipped_memmap",        "triviaqa_memmap",       "0,1,2,3,4", 4),
    ("baseline_comparison_triviaqa_qwen3_flipped_memmap",  "triviaqa_qwen3_memmap", "0,1,2,3,4", 4),
    ("baseline_comparison_hotpotqa_flipped_memmap_fill234", "hotpotqa_memmap",      "2,3,4",     4),
    ("baseline_comparison_popqa_flipped_memmap_fill234",    "popqa_memmap",         "2,3,4",     4),
    ("baseline_comparison_sciq_flipped_memmap_fill234",     "sciq_memmap",          "2,3,4",     4),
]


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
    for experiment_name, dataset_key, seeds_csv, last_seed in _TARGETS:
        cfg_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / cfg_rel).exists():
            print(f"  skip (config missing): {cfg_rel}", file=sys.stderr)
            continue
        cell_id = f"1_issue135__{dataset_key}__{_METHOD}"
        if _dispatch_has_cell(dispatch_root, cell_id):
            print(f"  skip (already queued): {cell_id}")
            continue
        output_check = (
            f"runs/{experiment_name}/{dataset_key}/{_METHOD}/seed_{last_seed}/predictions.csv"
        )
        cell = {
            "cell_id":           cell_id,
            "experiment_config": cfg_rel,
            "dataset":           dataset_key,
            "method":            _METHOD,
            "seed":              seeds_csv,
            "output_check":      output_check,
        }
        (dispatch_root / "pending" / f"{cell_id}.json").write_text(
            json.dumps(cell, indent=2), encoding="utf-8"
        )
        written += 1
        print(f"  queued: {cell_id}  (seeds {seeds_csv})")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue b5 grid-completion cells (issue #135).")
    parser.add_argument("--dispatch-root", default="shared/issue_135_dispatch")
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    total = build(dispatch_root)
    print(f"Done — {total} cells queued in {dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
