"""Build the symmetric low-k HalluLens/ACT-ViT queue for issue #149.

The queue covers the six canonical Llama memmap datasets.  Each cell bundles
seeds 0-4.  Numeric cell priorities put both HotpotQA methods first under the
filesystem queue's lexicographic claiming rule.
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

_DATASETS = (
    ("00", "hotpotqa_memmap"),
    ("10", "mmlu_memmap"),
    ("20", "nq_memmap"),
    ("30", "popqa_memmap"),
    ("40", "sciq_memmap"),
    ("50", "searchqa_memmap"),
)
_METHODS = (
    "act_vit_prefix_multik_lowk",
    "contrastive_logprob_recon_prefix_mixed_lowk",
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


def _experiment_name(dataset_name: str) -> str:
    return f"prefix149_lowk_{dataset_name.removesuffix('_memmap')}"


def build(dispatch_root: Path) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for priority, dataset_name in _DATASETS:
        experiment_name = _experiment_name(dataset_name)
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / experiment_rel).exists():
            raise FileNotFoundError(experiment_rel)
        for method_name in _METHODS:
            cell_id = f"{priority}_issue149_lowk__{dataset_name}__{method_name}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            run_dir = (
                Path("runs")
                / experiment_name
                / dataset_name
                / method_name
                / "seed_4"
            )
            cell = {
                "cell_id": cell_id,
                "kind": "experiment",
                "experiment_config": experiment_rel,
                "dataset": dataset_name,
                "method": method_name,
                "seed": _SEEDS,
                "seeded": True,
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
        "--dispatch-root", default="shared/issue_149_lowk_dispatch"
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
