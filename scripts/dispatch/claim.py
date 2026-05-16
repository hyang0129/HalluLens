"""
NFS-friendly filesystem claim queue for Issue #72 capture dispatch.

Coordinator-free work-stealing pool: multiple workers race to atomically
rename a cell file out of `pending/` and into `claimed/<worker_id>/`. POSIX
rename(2) is atomic on the same filesystem, including NFS, so exactly one
worker wins per cell.

Layout (anchored at <root>/_dispatch/):

    pending/<cell_id>.json
    claimed/<worker_id>/<cell_id>.json
    claimed/<worker_id>/heartbeat                # mtime touched while live
    done/<cell_id>.json
    failed/<cell_id>.json + <cell_id>.json.err   # error sidecar

Cell JSON format (consumed by scripts/capture_inference.py via worker.sh):

    {
      "cell_id":         "sciq_test_Llama-3.1-8B-Instruct",
      "task":            "sciq",
      "split":           "test",
      "model":           "meta-llama/Llama-3.1-8B-Instruct",
      "out_dir":         "shared/icr_capture/sciq_Llama-3.1-8B-Instruct",
      "n_samples":       null,
      "max_prompt_len":  512,
      "max_response_len":256,
      "r_max":           64,
      "top_k":           20
    }

This module is pure functions over the filesystem — no shared in-memory state,
no long-running process. Workers call these one-shot from bash.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


# Why: stale-claim threshold. Workers touch heartbeat every 60s; GC reclaims
# any claim whose heartbeat hasn't been touched in this many seconds.
STALE_CLAIM_SECONDS = 5 * 60


def init_dispatch_dirs(root: Path) -> None:
    """Create the four queue directories under <root>/_dispatch/.

    Idempotent — safe to call on an existing dispatch dir.
    """
    for sub in ("pending", "claimed", "done", "failed"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def claim_next_cell(root: Path, worker_id: str) -> Optional[Path]:
    """Atomically claim the next pending cell for this worker.

    Returns the path to the claimed cell file inside
    `claimed/<worker_id>/`, or None if no cells remain.

    Implementation: sorted scan of `pending/*.json`, attempt
    `os.rename(pending/X, claimed/<wid>/X)` on each in turn. Atomic on
    NFS (same filesystem). Loser sees FileNotFoundError and tries next.
    """
    pending = root / "pending"
    my_claimed = root / "claimed" / worker_id
    my_claimed.mkdir(parents=True, exist_ok=True)

    candidates = sorted(pending.glob("*.json"))
    for cand in candidates:
        target = my_claimed / cand.name
        try:
            os.rename(cand, target)
        except FileNotFoundError:
            continue
        return target
    return None


def complete_cell(root: Path, worker_id: str, cell_path: Path) -> Path:
    """Move a completed cell from claimed/<wid>/ to done/."""
    done = root / "done"
    done.mkdir(parents=True, exist_ok=True)
    target = done / cell_path.name
    os.rename(cell_path, target)
    return target


def fail_cell(root: Path, worker_id: str, cell_path: Path, err_text: str) -> Path:
    """Move a failed cell to failed/ with an .err sidecar."""
    failed = root / "failed"
    failed.mkdir(parents=True, exist_ok=True)
    target = failed / cell_path.name
    err_target = failed / (cell_path.name + ".err")
    err_target.write_text(err_text, encoding="utf-8")
    os.rename(cell_path, target)
    return target


def touch_heartbeat(root: Path, worker_id: str) -> None:
    """Update mtime on this worker's heartbeat file so GC won't reclaim."""
    hb = root / "claimed" / worker_id / "heartbeat"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.touch(exist_ok=True)
    # Touch alone may not update mtime if the file exists on some FS;
    # use utime to be explicit.
    now = time.time()
    os.utime(hb, (now, now))


def gc_stale_claims(root: Path, now: Optional[float] = None) -> list[Path]:
    """Move cells from dead workers' claimed/ back to pending/.

    A worker is considered dead if its `heartbeat` file is older than
    STALE_CLAIM_SECONDS, OR if no heartbeat file exists AND the claim
    directory's mtime is that old. Returns the list of cell files that
    were re-pended.
    """
    if now is None:
        now = time.time()
    claimed_root = root / "claimed"
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    reclaimed: list[Path] = []
    if not claimed_root.exists():
        return reclaimed

    for worker_dir in claimed_root.iterdir():
        if not worker_dir.is_dir():
            continue
        hb = worker_dir / "heartbeat"
        # Use heartbeat mtime if present, else the directory's own mtime.
        last_seen = hb.stat().st_mtime if hb.exists() else worker_dir.stat().st_mtime
        if now - last_seen < STALE_CLAIM_SECONDS:
            continue
        # Stale — move every cell back to pending/.
        for cell in worker_dir.glob("*.json"):
            target = pending / cell.name
            try:
                os.rename(cell, target)
                reclaimed.append(target)
            except FileNotFoundError:
                continue
        # Best-effort cleanup of the now-empty worker dir.
        try:
            if hb.exists():
                hb.unlink()
            worker_dir.rmdir()
        except OSError:
            pass
    return reclaimed


def count_status(root: Path) -> dict[str, int]:
    """Return cell counts in each queue dir, for progress monitoring."""
    out = {}
    for sub in ("pending", "done", "failed"):
        d = root / sub
        out[sub] = len(list(d.glob("*.json"))) if d.exists() else 0
    claimed_root = root / "claimed"
    out["claimed"] = (
        sum(1 for wd in claimed_root.iterdir() if wd.is_dir()
            for _ in wd.glob("*.json"))
        if claimed_root.exists() else 0
    )
    return out


def load_cell(cell_path: Path) -> dict:
    return json.loads(cell_path.read_text(encoding="utf-8"))
