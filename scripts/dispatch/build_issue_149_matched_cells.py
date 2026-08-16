"""Queue issue #149 fair-comparison cells after the ICR caches are complete.

Grid: two datasets x seven baselines. Seeded methods bundle seeds 0-4 in one
cell for resumability; deterministic token-aggregation methods run once.
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
from scripts.experiment_utils import is_seeded_method  # noqa: E402

_TARGETS = (
    ("prefix149_matched_hotpotqa", "hotpotqa_memmap"),
    ("prefix149_matched_sciq", "sciq_memmap"),
)
_METHODS = (
    "linear_probe_prefix_multik",
    "saplma_prefix_multik",
    "act_vit_prefix_multik",
    "llmsknow_prefix_per_k",
    "icr_probe_prefix_per_k",
    "token_entropy_prefix_eval",
    "logprob_baseline_prefix_eval",
)
_SEEDS = "0,1,2,3,4"


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    filename = f"{cell_id}.json"
    for subdir in ("pending", "done", "failed"):
        if (dispatch_root / subdir / filename).exists():
            return True
    claimed = dispatch_root / "claimed"
    return claimed.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed.iterdir()
    )


def build(dispatch_root: Path) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for experiment_name, dataset_name in _TARGETS:
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / experiment_rel).exists():
            raise FileNotFoundError(experiment_rel)
        for method_name in _METHODS:
            method_path = _PROJECT_ROOT / "configs" / "methods" / f"{method_name}.json"
            method_cfg = json.loads(method_path.read_text())
            seeded = is_seeded_method(method_cfg)
            cell_id = f"issue149_matched__{dataset_name}__{method_name}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = Path("runs") / experiment_name / dataset_name / method_name
            if seeded:
                run_dir /= "seed_4"
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": method_name,
                "seed": _SEEDS if seeded else "0",
                "seeded": seeded,
                "output_check": str(run_dir / "eval_metrics.json"),
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
        "--dispatch-root", default="shared/issue_149_matched_dispatch"
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
