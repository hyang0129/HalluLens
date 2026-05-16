"""
Tests for scripts/dispatch/claim.py — concurrency and crash-recovery contracts.

All tests are CPU-only and use tmp_path for isolation.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path

import pytest

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import (
    STALE_CLAIM_SECONDS,
    claim_next_cell,
    complete_cell,
    count_status,
    fail_cell,
    gc_stale_claims,
    init_dispatch_dirs,
    touch_heartbeat,
)


def _write_cell(pending: Path, cell_id: str) -> Path:
    cell = {
        "cell_id": cell_id,
        "task": "sciq",
        "split": "test",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "out_dir": f"shared/icr_capture/{cell_id}",
        "n_samples": None,
        "max_prompt_len": 512,
        "max_response_len": 256,
        "r_max": 64,
        "top_k": 20,
    }
    p = pending / f"{cell_id}.json"
    p.write_text(json.dumps(cell), encoding="utf-8")
    return p


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    r = tmp_path / "_dispatch"
    init_dispatch_dirs(r)
    return r


def test_claim_single_cell(root: Path) -> None:
    _write_cell(root / "pending", "sciq_test_Llama")

    claimed = claim_next_cell(root, "worker1")

    assert claimed is not None
    assert claimed.exists()
    assert claimed.parent == root / "claimed" / "worker1"
    assert not (root / "pending" / "sciq_test_Llama.json").exists()


def test_claim_returns_none_when_empty(root: Path) -> None:
    result = claim_next_cell(root, "worker1")
    assert result is None


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "os.rename atomicity guarantee on Windows differs from POSIX/NFS: "
        "MoveFileEx with different destinations can all succeed when racing, "
        "so claim.py's atomic-rename strategy is only correct on Linux. "
        "This test validates production behaviour (Linux NFS workers)."
    ),
)
def test_concurrent_claims_no_duplicate(root: Path) -> None:
    N = 10
    pending = root / "pending"
    for i in range(N):
        _write_cell(pending, f"cell_{i:03d}")

    all_claims: list[str] = []

    def worker_loop(wid: str) -> list[str]:
        mine: list[str] = []
        while True:
            cell = claim_next_cell(root, wid)
            if cell is None:
                break
            mine.append(cell.name)
        return mine

    n_workers = min(N + 5, 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(worker_loop, f"w{i}") for i in range(n_workers)]
        for f in concurrent.futures.as_completed(futs):
            all_claims.extend(f.result())

    assert len(all_claims) == N, f"expected {N} claims, got {len(all_claims)}"
    assert len(set(all_claims)) == N, "duplicate claims detected"


def test_complete_cell_moves_to_done(root: Path) -> None:
    _write_cell(root / "pending", "cell_done")
    claimed = claim_next_cell(root, "w1")
    assert claimed is not None

    done_path = complete_cell(root, "w1", claimed)

    assert done_path.exists()
    assert done_path.parent == root / "done"
    assert not claimed.exists()


def test_fail_cell_writes_err_sidecar(root: Path) -> None:
    _write_cell(root / "pending", "cell_fail")
    claimed = claim_next_cell(root, "w1")
    assert claimed is not None

    fail_cell(root, "w1", claimed, "something went wrong")

    failed_json = root / "failed" / "cell_fail.json"
    failed_err = root / "failed" / "cell_fail.json.err"
    assert failed_json.exists()
    assert failed_err.exists()
    assert "something went wrong" in failed_err.read_text(encoding="utf-8")


def test_gc_stale_claim_reclaims(root: Path) -> None:
    _write_cell(root / "pending", "cell_stale")
    claimed = claim_next_cell(root, "w_stale")
    assert claimed is not None

    hb = root / "claimed" / "w_stale" / "heartbeat"
    touch_heartbeat(root, "w_stale")
    stale_time = time.time() - STALE_CLAIM_SECONDS - 60
    os.utime(hb, (stale_time, stale_time))

    reclaimed = gc_stale_claims(root, now=time.time())

    assert len(reclaimed) == 1
    assert (root / "pending" / "cell_stale.json").exists()
    assert not (root / "claimed" / "w_stale").exists()


def test_gc_fresh_claim_not_reclaimed(root: Path) -> None:
    _write_cell(root / "pending", "cell_fresh")
    claimed = claim_next_cell(root, "w_fresh")
    assert claimed is not None
    touch_heartbeat(root, "w_fresh")

    reclaimed = gc_stale_claims(root, now=time.time())

    assert len(reclaimed) == 0
    assert claimed.exists()


def test_gc_no_heartbeat_uses_dir_mtime(root: Path) -> None:
    _write_cell(root / "pending", "cell_nohb")
    claimed = claim_next_cell(root, "w_nohb")
    assert claimed is not None

    # No heartbeat file — make worker dir mtime stale.
    worker_dir = root / "claimed" / "w_nohb"
    stale_time = time.time() - STALE_CLAIM_SECONDS - 60
    os.utime(worker_dir, (stale_time, stale_time))

    reclaimed = gc_stale_claims(root, now=time.time())

    assert len(reclaimed) == 1
    assert (root / "pending" / "cell_nohb.json").exists()


def test_count_status_counts_correctly(root: Path) -> None:
    for i in range(3):
        _write_cell(root / "pending", f"p{i}")

    # Two workers each claim one cell.
    for wid in ("wa", "wb"):
        init_dispatch_dirs(root)
        cell = claim_next_cell(root, wid)
        assert cell is not None

    for i in range(4):
        _write_cell(root / "done", f"d{i}")

    _write_cell(root / "failed", "f0")

    counts = count_status(root)

    assert counts["pending"] == 1, counts
    assert counts["claimed"] == 2, counts
    assert counts["done"] == 4, counts
    assert counts["failed"] == 1, counts
