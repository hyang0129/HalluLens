"""
_claim_cli.py — thin shell adapter for claim.py functions.

Used by worker.sh to call claim-queue operations without embedding Python
one-liners in bash. All subcommands exit 0 on success; non-zero on error.

Subcommands:
    claim      --root R --worker-id W       → prints cell path, or empty line
    complete   --root R --worker-id W --cell C
    fail       --root R --worker-id W --cell C --err-file F
    heartbeat  --root R --worker-id W
    gc         --root R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import (  # noqa: E402
    claim_next_cell,
    complete_cell,
    fail_cell,
    gc_stale_claims,
    touch_heartbeat,
)


def cmd_claim(args: argparse.Namespace) -> int:
    root = Path(args.root)
    cell = claim_next_cell(root, args.worker_id)
    print(str(cell) if cell is not None else "")
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    complete_cell(Path(args.root), args.worker_id, Path(args.cell))
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    err_path = Path(args.err_file)
    err_text = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
    # Why: spec says "last 500 lines"; slice here to keep sidecar manageable.
    lines = err_text.splitlines()
    if len(lines) > 500:
        err_text = "\n".join(lines[-500:])
    fail_cell(Path(args.root), args.worker_id, Path(args.cell), err_text)
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    touch_heartbeat(Path(args.root), args.worker_id)
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    reclaimed = gc_stale_claims(Path(args.root))
    if reclaimed:
        print(f"gc: reclaimed {len(reclaimed)} stale cells")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="claim.py shell adapter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("--root", required=True)
    p_claim.add_argument("--worker-id", required=True)

    p_complete = sub.add_parser("complete")
    p_complete.add_argument("--root", required=True)
    p_complete.add_argument("--worker-id", required=True)
    p_complete.add_argument("--cell", required=True)

    p_fail = sub.add_parser("fail")
    p_fail.add_argument("--root", required=True)
    p_fail.add_argument("--worker-id", required=True)
    p_fail.add_argument("--cell", required=True)
    p_fail.add_argument("--err-file", required=True)

    p_hb = sub.add_parser("heartbeat")
    p_hb.add_argument("--root", required=True)
    p_hb.add_argument("--worker-id", required=True)

    p_gc = sub.add_parser("gc")
    p_gc.add_argument("--root", required=True)

    args = parser.parse_args()

    dispatch = {
        "claim":     cmd_claim,
        "complete":  cmd_complete,
        "fail":      cmd_fail,
        "heartbeat": cmd_heartbeat,
        "gc":        cmd_gc,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
