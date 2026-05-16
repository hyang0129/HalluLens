"""
manifest_status.py — print dispatch queue counts and failed-cell summaries.

Usage:
    python scripts/dispatch/manifest_status.py --dispatch-root shared/icr_capture/_dispatch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import count_status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Show dispatch queue status.")
    parser.add_argument("--dispatch-root", required=True,
                        help="Path to <root>/_dispatch/ directory.")
    args = parser.parse_args()

    root = Path(args.dispatch_root)
    counts = count_status(root)
    total = sum(counts.values())

    print("Dispatch queue status")
    print(f"  pending:  {counts.get('pending', 0)}")
    print(f"  claimed:  {counts.get('claimed', 0)}")
    print(f"  done:     {counts.get('done', 0)}")
    print(f"  failed:   {counts.get('failed', 0)}")
    print(f"  total:    {total}")

    failed_dir = root / "failed"
    if failed_dir.exists():
        failed_cells = sorted(failed_dir.glob("*.json"))
        if failed_cells:
            print(f"\nFailed cells ({len(failed_cells)}):")
            for cell_path in failed_cells:
                err_path = failed_dir / (cell_path.name + ".err")
                print(f"  {cell_path.name}")
                if err_path.exists():
                    lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    for line in lines[:5]:
                        print(f"    {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
