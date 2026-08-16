"""Build the issue #149 dual-convention classifier experiment queue.

The sweep contains one independently claimable cell per dataset x seed across
the five in-scope Llama memmap datasets. MMLU is explicitly excluded. Keeping
seeds separate lets four workers make progress concurrently and makes retries
resume at exactly one run rather than replaying a five-seed bundle.
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

_TARGETS = (
    ("00", "hotpotqa_memmap", "prefix149_dual_convention_hotpotqa"),
    ("20", "nq_memmap", "prefix149_dual_convention_nq"),
    ("30", "popqa_memmap", "prefix149_dual_convention_popqa"),
    ("40", "sciq_memmap", "prefix149_dual_convention_sciq"),
    ("50", "searchqa_memmap", "prefix149_dual_convention_searchqa"),
)
_METHOD = "dual_convention_contrastive_classifier_prefix_mixed_lowk"
_SEEDS = (0, 1, 2, 3, 4)


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


def _run_dir(experiment_name: str, dataset_name: str, seed: int) -> Path:
    return (
        Path("runs")
        / experiment_name
        / dataset_name
        / _METHOD
        / f"seed_{seed}"
    )


def _seed_is_complete(project_root: Path, run_dir: Path) -> bool:
    absolute = project_root / run_dir
    return (
        not (absolute / "run_error.json").exists()
        and (absolute / "eval_metrics.json").exists()
        and (absolute / "predictions.csv").exists()
    )


def build(dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT) -> int:
    init_dispatch_dirs(dispatch_root)
    method_path = project_root / "configs" / "methods" / f"{_METHOD}.json"
    if not method_path.exists():
        raise FileNotFoundError(method_path)

    written = 0
    for priority, dataset_name, experiment_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).exists():
            raise FileNotFoundError(project_root / experiment_rel)
        for seed in _SEEDS:
            cell_id = (
                f"{priority}_issue149_dual_convention__{dataset_name}__seed_{seed}"
            )
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = _run_dir(experiment_name, dataset_name, seed)
            if _seed_is_complete(project_root, run_dir):
                print(f"  skip (seed output complete): {cell_id}")
                continue
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": _METHOD,
                "seed": seed,
                "seeded": True,
                # This routine writes eval_metrics before predictions; requiring
                # predictions is the stronger completion sentinel.
                "output_check": str(run_dir / "predictions.csv"),
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2) + "\n"
            )
            written += 1
            print(f"  queued: {cell_id}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_149_dual_convention_dispatch",
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
