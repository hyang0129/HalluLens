"""Queue highest-priority scoring cells for the Issue #153 projection head.

Each cell consumes a completed v2 seed-0 checkpoint and its dumped token-zero
trunk embeddings. Cells whose source artifacts are not ready are deliberately
skipped; rerunning this builder appends them once training finishes. MMLU is
excluded.
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
    ("00", "hotpotqa_memmap", "issue153_v2_hotpotqa"),
    ("20", "nq_memmap", "issue153_v2_nq"),
    ("30", "popqa_memmap", "issue153_v2_popqa"),
    ("40", "sciq_memmap", "issue153_v2_sciq"),
    ("50", "searchqa_memmap", "issue153_v2_searchqa"),
)
_SOURCE_METHOD = "tokenwise_contrastive_v2_depthnorm_projection"
_METHOD = "tokenwise_contrastive_v2_projection_scoring"


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


def _source_is_ready(source_run: Path) -> bool:
    required = (
        source_run / "artifacts" / "final_weights.pt",
        source_run / "eval_metrics.json",
        source_run / "embeddings" / "train_z.npy",
        source_run / "embeddings" / "train_labels.npy",
        source_run / "embeddings" / "test_z.npy",
        source_run / "embeddings" / "test_labels.npy",
        source_run / "embeddings" / "test_hashkeys.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def build(
    dispatch_root: Path,
    *,
    project_root: Path = _PROJECT_ROOT,
    runs_root: Path | None = None,
) -> int:
    init_dispatch_dirs(dispatch_root)
    runs_root = project_root / "runs" if runs_root is None else Path(runs_root)
    method_rel = f"configs/methods/{_METHOD}.json"
    if not (project_root / method_rel).is_file():
        raise FileNotFoundError(project_root / method_rel)

    written = 0
    for order, dataset_name, experiment_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (project_root / experiment_rel).is_file():
            raise FileNotFoundError(project_root / experiment_rel)

        source_run = (
            runs_root
            / experiment_name
            / dataset_name
            / _SOURCE_METHOD
            / "seed_0"
        )
        if not _source_is_ready(source_run):
            print(f"  defer (source not ready): {dataset_name}")
            continue

        cell_id = (
            f"0_high_{order}_projection_issue153__{dataset_name}__"
            f"{_METHOD}__seed_0"
        )
        if _dispatch_has_cell(dispatch_root, cell_id):
            print(f"  skip (already queued): {cell_id}")
            continue

        run_dir = Path("runs") / experiment_name / dataset_name / _METHOD / "seed_0"
        cell = {
            "cell_id": cell_id,
            "kind": "experiment",
            "priority": "highest",
            "issue": 153,
            "experiment": "projection_surface_scoring",
            "worker_script": "scripts/dispatch/worker_experiment.sh",
            "experiment_config": experiment_rel,
            "dataset": dataset_name,
            "method": _METHOD,
            "seed": 0,
            "seeded": True,
            "source_checkpoint": str(
                Path("runs")
                / experiment_name
                / dataset_name
                / _SOURCE_METHOD
                / "seed_0"
                / "artifacts"
                / "final_weights.pt"
            ),
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
