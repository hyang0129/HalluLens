"""
build_smollm3_cells.py — queue training cells for the SmolLM3-3B headline
(issue #139). One cell per (experiment_config, method), each covering all 5
seeds, drained by scripts/dispatch/worker_79.sh via run_experiment.py's
inner-loop resume (skips seeds whose predictions.csv already exists).

Capture is a SEPARATE stage and does NOT use this builder — capture cells are
produced by scripts/dispatch/generate_manifest.py and drained by worker.sh:

    python scripts/dispatch/generate_manifest.py \
        --dispatch-root shared/icr_capture/_dispatch \
        --out-base-dir  shared/icr_capture \
        --tasks hotpotqa,mmlu,popqa,natural_questions,sciq,searchqa \
        --models HuggingFaceTB/SmolLM3-3B \
        --splits test,train

This script covers the downstream TRAINING stage once the icr_capture dirs
exist. Headline scope = the 6 core LLMsKnow datasets, 8 baseline methods,
seeds 0-4 — matching the Qwen3/Llama baseline_comparison_*_memmap grid.

Usage:
    python scripts/dispatch/build_smollm3_cells.py \
        --dispatch-root shared/smollm3_dispatch
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

# Dataset config stems (the experiment's "dataset" field) for the 6 core
# LLMsKnow datasets at SmolLM3 headline parity.
_DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]

# The 8 baseline methods, identical to the Qwen3/Llama headline grid.
_METHODS = [
    "contrastive_logprob_recon",
    "linear_probe",
    "token_entropy",
    "logprob_baseline",
    "saplma",
    "llmsknow_probe",
    "icr_probe",
    "act_vit",
]

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
    for ds in _DATASETS:
        experiment_name = f"baseline_comparison_{ds}_smollm3_memmap"
        dataset_key = f"{ds}_smollm3_memmap"
        cfg_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / cfg_rel).exists():
            print(f"  skip (config missing): {cfg_rel}", file=sys.stderr)
            continue
        for method in _METHODS:
            cell_id = f"smollm3__{dataset_key}__{method}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            # worker_79.sh marks the cell done when this exists; run_experiment's
            # inner loop resumes per-seed so a re-claimed cell finishes only the
            # missing seeds.
            output_check = (
                f"runs/{experiment_name}/{dataset_key}/{method}/"
                f"seed_{_LAST_SEED}/predictions.csv"
            )
            cell = {
                "cell_id":           cell_id,
                "experiment_config": cfg_rel,
                "dataset":           dataset_key,
                "method":            method,
                "seed":              _SEEDS_CSV,
                "output_check":      output_check,
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2), encoding="utf-8"
            )
            written += 1
            print(f"  queued: {cell_id}  (seeds {_SEEDS_CSV})")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Queue SmolLM3-3B headline training cells (issue #139)."
    )
    parser.add_argument("--dispatch-root", default="shared/smollm3_dispatch")
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    total = build(dispatch_root)
    print(f"Done — {total} cells queued in {dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
