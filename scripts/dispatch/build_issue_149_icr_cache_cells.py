"""Queue the four prefix-specific ICR cache prerequisites for issue #149.

This queue is intentionally separate from both the live pilot and the matched
training queue. Run it to completion before starting ``worker_149_matched.sh``.
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

_DATASETS = ("hotpotqa_memmap", "sciq_memmap")
_PREFIXES = "1,4,8,16,32,48,64"


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


def build(dispatch_root: Path, *, cache_root: str = "shared/prefix149_icr") -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for dataset_name in _DATASETS:
        cfg_path = _PROJECT_ROOT / "configs" / "datasets" / f"{dataset_name}.json"
        dataset_cfg = json.loads(cfg_path.read_text())
        for split_key in ("train_dir", "test_dir"):
            capture_dir = dataset_cfg["icr_capture"][split_key]
            capture_name = Path(capture_dir).name
            split = split_key.removesuffix("_dir")
            cell_id = f"issue149_icr_cache__{dataset_name}__{split}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            output_dir = str(Path(cache_root) / capture_name)
            cell = {
                "cell_id": cell_id,
                "kind": "icr_prefix_cache",
                "dataset": dataset_name,
                "split": split,
                "capture_dir": capture_dir,
                "output_dir": output_dir,
                "prefixes": _PREFIXES,
                "batch_size": 2,
                "output_check": str(Path(output_dir) / "icr_scores_k64.npy"),
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
        "--dispatch-root", default="shared/issue_149_icr_cache_dispatch"
    )
    parser.add_argument("--cache-root", default="shared/prefix149_icr")
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    count = build(dispatch_root, cache_root=args.cache_root)
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
