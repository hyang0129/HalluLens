"""Tests for scripts/run_chat_capture_merges.py against synthetic capture dirs.

Uses tiny raw-byte arrays (1 byte/row) instead of real activation memmaps --
scripts/merge_icr_captures.py only cares about per-row byte width and the
written-row prefix, so a 1-byte "response_activations.npy" is a faithful
stand-in and keeps the synthetic fixtures cheap.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_chat_capture_merges as m


def _write_shard(path: Path, *, n_alloc: int, n_written: int, marker: bytes,
                  model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({
        "n_samples": n_alloc,
        "model_name": model_name,
        "chat_template": True,
    }))
    with (path / "meta.jsonl").open("w") as f:
        for i in range(n_written):
            f.write(json.dumps({"sample_index": i, "prompt_hash": f"h{i}", "hallucinated": False}) + "\n")
    data = bytearray(n_alloc)
    data[:n_written] = marker * n_written
    (path / "response_activations.npy").write_bytes(bytes(data))


def test_plan_merges_orders_shards_by_index_start_not_dir_creation_order(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "hotpotqa_test_Llama-3.1-8B-Instruct"
    # Deliberately create the LATER shard on disk first.
    _write_shard(base_dir / f"{base}_6-14", n_alloc=8, n_written=3, marker=b"B")
    _write_shard(base_dir / f"{base}_0-6", n_alloc=10, n_written=6, marker=b"A")

    plan = m.plan_merges(base_dir)
    entry = next(e for e in plan if e["base"] == base)
    assert entry["task"] == "hotpotqa" and entry["split"] == "test"
    assert entry["status"] == "merge"
    assert entry["input_starts"] == [0, 6]
    assert [p.name for p in entry["inputs"]] == [f"{base}_0-6", f"{base}_6-14"]


def test_run_merge_concatenates_written_prefixes_in_index_start_order(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "hotpotqa_test_Llama-3.1-8B-Instruct"
    _write_shard(base_dir / f"{base}_0-6", n_alloc=10, n_written=6, marker=b"A")
    _write_shard(base_dir / f"{base}_6-14", n_alloc=8, n_written=3, marker=b"B")

    plan = m.plan_merges(base_dir)
    entry = next(e for e in plan if e["base"] == base)
    m.run_merge(entry)

    out_dir = entry["out_dir"]
    assert out_dir.name == f"{base}_merged"
    merged_bytes = (out_dir / "response_activations.npy").read_bytes()
    # A's 6 written rows, then B's 3 written rows -- NOT A's full 10-row
    # alloc (which would insert 4 bytes of A's untouched tail before B).
    assert merged_bytes == b"A" * 6 + b"B" * 3
    assert m._meta_count(out_dir) == 9
    assert entry["merged_total"] == 9


def test_single_unsuffixed_dir_is_left_alone(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "popqa_test_Llama-3.1-8B-Instruct"
    _write_shard(base_dir / base, n_alloc=5, n_written=5, marker=b"C")

    plan = m.plan_merges(base_dir)
    entry = next(e for e in plan if e["base"] == base)
    assert entry["status"] == "single"
    assert not (base_dir / f"{base}_merged").exists()


def test_not_ready_when_only_one_of_several_expected_shards_present(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "searchqa_train_Qwen3-8B"
    _write_shard(base_dir / f"{base}_0-5", n_alloc=5, n_written=5, marker=b"A",
                 model_name="Qwen/Qwen3-8B")

    plan = m.plan_merges(base_dir)
    entry = next(e for e in plan if e["base"] == base)
    assert entry["status"] == "not_ready"


def test_main_dry_run_does_not_write_merged_dir(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "hotpotqa_train_Qwen3-8B"
    _write_shard(base_dir / f"{base}_0-5", n_alloc=5, n_written=5, marker=b"A",
                 model_name="Qwen/Qwen3-8B")
    _write_shard(base_dir / f"{base}_5-9", n_alloc=4, n_written=4, marker=b"B",
                 model_name="Qwen/Qwen3-8B")

    rc = m.main(["--base-dir", str(base_dir), "--dry-run", "--only", base])
    assert rc == 0
    assert not (base_dir / f"{base}_merged").exists()


def test_main_merges_and_reruns_are_idempotent_noop(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "searchqa_test_Llama-3.1-8B-Instruct"
    _write_shard(base_dir / f"{base}_0-4", n_alloc=4, n_written=4, marker=b"A")
    _write_shard(base_dir / f"{base}_4-9", n_alloc=5, n_written=5, marker=b"B")

    rc = m.main(["--base-dir", str(base_dir), "--only", base])
    assert rc == 0
    out_dir = base_dir / f"{base}_merged"
    assert m._meta_count(out_dir) == 9

    mtime_before = (out_dir / "response_activations.npy").stat().st_mtime
    rc2 = m.main(["--base-dir", str(base_dir), "--only", base])
    assert rc2 == 0
    mtime_after = (out_dir / "response_activations.npy").stat().st_mtime
    assert mtime_after == mtime_before  # untouched -- skipped, not re-copied.


def test_run_merge_raises_loudly_on_verification_mismatch(tmp_path, monkeypatch):
    """A merged dir whose row count silently disagrees with its inputs' sum
    must be a hard failure, not a quietly-accepted result."""
    base_dir = tmp_path / "icr_capture_chat"
    _write_shard(base_dir / "x_0-4", n_alloc=4, n_written=4, marker=b"A")
    _write_shard(base_dir / "x_4-9", n_alloc=5, n_written=5, marker=b"B")
    entry = {
        "base": "x",
        "inputs": [base_dir / "x_0-4", base_dir / "x_4-9"],
        "out_dir": base_dir / "x_merged",
    }

    real_meta_count = m._meta_count

    def lying_meta_count(d: Path) -> int:
        n = real_meta_count(d)
        # Simulate a merged output that silently dropped one row: lie only
        # when asked about the OUTPUT dir, after the real merge has run.
        if d == entry["out_dir"]:
            return n - 1
        return n

    monkeypatch.setattr(m, "_meta_count", lying_meta_count)

    with pytest.raises(RuntimeError, match="VERIFICATION FAILED"):
        m.run_merge(entry)


def test_main_exits_nonzero_when_a_merge_fails(tmp_path):
    base_dir = tmp_path / "icr_capture_chat"
    base = "sciq_train_Llama-3.1-8B-Instruct"
    # n_written > n_alloc is an invalid capture dir -- merge_icr_captures.py
    # itself refuses ("meta exceeds alloc"), which must propagate as a loud,
    # non-zero-exit driver failure rather than being swallowed.
    _write_shard(base_dir / f"{base}_0-4", n_alloc=4, n_written=4, marker=b"A")
    (base_dir / f"{base}_4-9").mkdir(parents=True)
    (base_dir / f"{base}_4-9" / "config.json").write_text(json.dumps({
        "n_samples": 2, "model_name": "meta-llama/Llama-3.1-8B-Instruct", "chat_template": True,
    }))
    with (base_dir / f"{base}_4-9" / "meta.jsonl").open("w") as f:
        for i in range(5):  # 5 written rows but only 2 allocated
            f.write(json.dumps({"sample_index": i}) + "\n")
    (base_dir / f"{base}_4-9" / "response_activations.npy").write_bytes(b"B" * 2)

    rc = m.main(["--base-dir", str(base_dir), "--only", base])
    assert rc == 1
    assert not (base_dir / f"{base}_merged" / "meta.jsonl").exists()
