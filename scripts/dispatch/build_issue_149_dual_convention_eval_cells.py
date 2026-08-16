"""Promote recoverable issue #149 failures to highest-priority eval cells.

Only cells with a nonempty ``artifacts/final_weights.pt`` are promoted. The
original failed payload and error sidecar are retained under
``recovery_history/`` while the replacement cell is tagged ``eval_only`` so a
missing checkpoint cannot silently turn a recovery into another training run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402


def _recovery_id(cell: dict) -> str:
    return (
        "000_eval_issue149_dual_convention__"
        f"{cell['dataset']}__seed_{cell['seed']}"
    )


def _recovery_exists(dispatch_root: Path, filename: str) -> bool:
    for state in ("pending", "done", "failed", "cancelled"):
        if (dispatch_root / state / filename).exists():
            return True
    claimed = dispatch_root / "claimed"
    return claimed.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed.iterdir()
    )


def build(
    dispatch_root: Path, *, project_root: Path = _PROJECT_ROOT
) -> int:
    """Promote checkpoint-backed failed cells and return the number queued."""
    init_dispatch_dirs(dispatch_root)
    failed_root = dispatch_root / "failed"
    history_root = dispatch_root / "recovery_history"
    history_root.mkdir(parents=True, exist_ok=True)

    promoted = 0
    for source in sorted(failed_root.glob("*.json")):
        cell = json.loads(source.read_text(encoding="utf-8"))
        if "mmlu" in str(cell.get("dataset", "")).lower():
            continue

        output_path = project_root / cell["output_check"]
        run_dir = output_path.parent
        checkpoint = run_dir / "artifacts" / "final_weights.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            print(f"  skip (checkpoint missing): {cell['cell_id']}")
            continue

        recovery_id = _recovery_id(cell)
        recovery_filename = f"{recovery_id}.json"
        if _recovery_exists(dispatch_root, recovery_filename):
            print(f"  skip (recovery already queued): {recovery_id}")
            continue

        recovery = {
            **cell,
            "cell_id": recovery_id,
            "kind": "evaluation",
            "eval_only": True,
            "checkpoint_check": str(checkpoint.relative_to(project_root)),
            "recovery_of": cell["cell_id"],
            "priority": "highest",
        }
        target = dispatch_root / "pending" / recovery_filename
        temporary = dispatch_root / "pending" / f".{recovery_filename}.tmp"
        temporary.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)

        # Retain the original failure evidence without counting both the failed
        # training cell and its replacement evaluation cell as active work.
        os.replace(source, history_root / source.name)
        error_source = failed_root / f"{source.name}.err"
        if error_source.exists():
            os.replace(error_source, history_root / error_source.name)

        promoted += 1
        print(f"  promoted: {cell['cell_id']} -> {recovery_id}")

    return promoted


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
    print(f"\npromoted {count} eval-only cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
