"""Tests for the --shard-size load-balancing mode in scripts/dispatch/generate_manifest.

Unlike --cap (test_manifest_cap.py: emits at most one incomplete "next slice"
per invocation, for a sequential headline-then-appendix workflow), --shard-size
splits every dataset into ALL of its shards upfront, so many parallel dispatch
workers can drain the queue concurrently.

Covers:
- _shard_ranges_for_dataset: full upfront shard list / single unsuffixed cell
  under threshold
- generate_manifest with --shard-size emits every shard (including test splits)
- Re-invocation is idempotent: done shards and already-queued shards are skipped
- --cap and --shard-size are mutually exclusive (function-level and CLI-level)
- --max-samples composed with --shard-size: effective-size-before-sharding,
  the truncated-single-shard edge case, and --cap/--max-samples mutual
  exclusion
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dispatch.generate_manifest import (  # noqa: E402
    _shard_ranges_for_dataset,
    generate_manifest,
)


# ---------------------------------------------------------------------------
# _shard_ranges_for_dataset
# ---------------------------------------------------------------------------

def test_shard_size_none_returns_full_slice():
    ranges = _shard_ranges_for_dataset(expected_size=10_000, shard_size=None)
    assert ranges == [(None, None)]


def test_dataset_under_shard_size_returns_single_unsuffixed_slice():
    ranges = _shard_ranges_for_dataset(expected_size=1_000, shard_size=5_000)
    assert ranges == [(None, None)]


def test_dataset_exactly_at_shard_size_returns_single_unsuffixed_slice():
    ranges = _shard_ranges_for_dataset(expected_size=5_000, shard_size=5_000)
    assert ranges == [(None, None)]


def test_dataset_over_shard_size_emits_all_shards_upfront():
    # hotpotqa test split: 7405 rows, shard_size=5000 -> 2 shards.
    ranges = _shard_ranges_for_dataset(expected_size=7_405, shard_size=5_000)
    assert ranges == [(0, 5_000), (5_000, 7_405)]


def test_large_dataset_emits_many_shards():
    # hotpotqa train split: 90447 rows, shard_size=50000 -> 2 shards.
    ranges = _shard_ranges_for_dataset(expected_size=90_447, shard_size=50_000)
    assert ranges == [(0, 50_000), (50_000, 90_447)]


def test_expected_size_none_returns_single_unsuffixed_slice():
    ranges = _shard_ranges_for_dataset(expected_size=None, shard_size=5_000)
    assert ranges == [(None, None)]


# ---------------------------------------------------------------------------
# generate_manifest(shard_size=...) — full upfront emission
# ---------------------------------------------------------------------------

def test_generate_manifest_shard_size_emits_all_shards_for_over_threshold(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["hotpotqa"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"],
        n_samples=None,
        shard_size=5_000,
    )
    # hotpotqa test (validation) = 7405 rows -> 2 shards, both emitted at once.
    assert n == 2, f"expected 2 cells emitted upfront, got {n}"

    pending = list((dispatch_root / "pending").glob("*.json"))
    cell_ids = sorted(p.stem for p in pending)
    assert cell_ids == [
        "hotpotqa_test_Llama-3.1-8B-Instruct_0-5000",
        "hotpotqa_test_Llama-3.1-8B-Instruct_5000-7405",
    ]

    for cell_path in pending:
        cell = json.loads(cell_path.read_text())
        assert cell["index_start"] is not None
        assert cell["index_end"] is not None
        assert cell["shuffle_seed"] == 0
        assert cell["out_dir"].endswith(f"_{cell['index_start']}-{cell['index_end']}")


def test_generate_manifest_shard_size_under_threshold_omits_suffix(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["sciq"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"],
        n_samples=None,
        shard_size=5_000,
    )
    # sciq test = 1000 rows <= shard_size -> single unsuffixed cell.
    assert n == 1
    pending = list((dispatch_root / "pending").glob("*.json"))
    assert len(pending) == 1
    assert pending[0].stem == "sciq_test_Llama-3.1-8B-Instruct"
    cell = json.loads(pending[0].read_text())
    assert cell["index_start"] is None
    assert cell["index_end"] is None


def test_generate_manifest_shard_size_covers_both_test_and_train(tmp_path):
    """--shard-size must shard train splits too, not just test."""
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["hotpotqa"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test", "train"],
        n_samples=None,
        shard_size=50_000,
    )
    pending = list((dispatch_root / "pending").glob("*.json"))
    cell_ids = sorted(p.stem for p in pending)
    # test (7405) <= 50000 -> single unsuffixed cell.
    # train (90447) > 50000 -> 2 shards.
    assert cell_ids == [
        "hotpotqa_test_Llama-3.1-8B-Instruct",
        "hotpotqa_train_Llama-3.1-8B-Instruct_0-50000",
        "hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447",
    ]
    assert n == 3


# ---------------------------------------------------------------------------
# Idempotent re-invocation
# ---------------------------------------------------------------------------

def test_reinvocation_skips_done_shard(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n1 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"], n_samples=None, shard_size=5_000,
    )
    assert n1 == 2

    # Simulate the first shard completing and its pending cell being consumed
    # by a worker (moved out of pending/).
    first_cell = dispatch_root / "pending" / "hotpotqa_test_Llama-3.1-8B-Instruct_0-5000.json"
    first_cell.unlink()
    done_dir = out_base / "hotpotqa_test_Llama-3.1-8B-Instruct_0-5000"
    done_dir.mkdir(parents=True)
    (done_dir / "eval_results.json").write_text("{}")

    # Re-invoke: the done shard must not be re-queued; the second shard
    # (still pending from before) must not be duplicated either.
    n2 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"], n_samples=None, shard_size=5_000,
    )
    assert n2 == 0, "done shard must not be re-queued, still-pending shard must not duplicate"

    pending = list((dispatch_root / "pending").glob("*.json"))
    assert [p.stem for p in pending] == ["hotpotqa_test_Llama-3.1-8B-Instruct_5000-7405"]


def test_reinvocation_skips_shard_already_claimed(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n1 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"], n_samples=None, shard_size=5_000,
    )
    assert n1 == 2

    # Simulate a worker claiming the first shard (moved to claimed/<worker>/).
    first_cell = dispatch_root / "pending" / "hotpotqa_test_Llama-3.1-8B-Instruct_0-5000.json"
    claimed_dir = dispatch_root / "claimed" / "worker_1"
    claimed_dir.mkdir(parents=True)
    first_cell.rename(claimed_dir / first_cell.name)

    n2 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"], n_samples=None, shard_size=5_000,
    )
    assert n2 == 0, "claimed shard must not be re-queued"


# ---------------------------------------------------------------------------
# Mutual exclusion with --cap
# ---------------------------------------------------------------------------

def test_generate_manifest_rejects_cap_and_shard_size_together(tmp_path):
    with pytest.raises(ValueError):
        generate_manifest(
            dispatch_root=tmp_path / "_dispatch",
            out_base_dir=tmp_path / "icr",
            tasks=["sciq"], models=["meta-llama/Llama-3.1-8B-Instruct"],
            splits=["test"], n_samples=None,
            cap=1_000, shard_size=5_000,
        )


def test_cli_rejects_cap_and_shard_size_together(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/dispatch/generate_manifest.py"),
            "--dispatch-root", str(tmp_path / "_dispatch"),
            "--out-base-dir", str(tmp_path / "icr"),
            "--tasks", "sciq",
            "--models", "meta-llama/Llama-3.1-8B-Instruct",
            "--splits", "test",
            "--cap", "1000",
            "--shard-size", "5000",
        ],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr.lower()
    assert not (tmp_path / "_dispatch" / "pending").exists() or not list(
        (tmp_path / "_dispatch" / "pending").glob("*.json")
    )


# ---------------------------------------------------------------------------
# --max-samples composed with --shard-size
# ---------------------------------------------------------------------------

def test_max_samples_caps_effective_size_before_sharding():
    """hotpotqa train (90447) with max_samples=50000, shard_size=5000 ->
    exactly 10 shards [0-5000) ... [45000-50000), per the spec example."""
    ranges = _shard_ranges_for_dataset(expected_size=90_447, shard_size=5_000, max_samples=50_000)
    assert len(ranges) == 10
    assert ranges[0] == (0, 5_000)
    assert ranges[-1] == (45_000, 50_000)
    assert ranges == [(i * 5_000, (i + 1) * 5_000) for i in range(10)]


def test_max_samples_larger_than_expected_size_is_a_noop_when_capped_dataset_over_threshold():
    """popqa train (11413) with max_samples=50000 (>expected), shard_size=5000
    -> 3 shards [0-5000), [5000-10000), [10000-11413), per the spec example —
    max_samples doesn't shrink anything since it exceeds the real size."""
    ranges = _shard_ranges_for_dataset(expected_size=11_414, shard_size=5_000, max_samples=50_000)
    assert ranges == [(0, 5_000), (5_000, 10_000), (10_000, 11_414)]


def test_dataset_already_under_shard_size_stays_unsuffixed_regardless_of_max_samples():
    """sciq test (1000) <= shard_size (5000): unsuffixed single cell whether
    or not --max-samples is given, since max_samples (50000) doesn't truncate
    anything below the already-small raw size."""
    ranges = _shard_ranges_for_dataset(expected_size=1_000, shard_size=5_000, max_samples=50_000)
    assert ranges == [(None, None)]


def test_max_samples_smaller_than_shard_size_still_produces_explicit_range():
    """Regression guard: if max_samples truncates a dataset that would
    otherwise exceed shard_size down to <= shard_size, the single resulting
    shard must still carry an EXPLICIT (0, effective_size) range — collapsing
    it to the unsuffixed (None, None) form would mean "no index range" to
    capture_inference.py, i.e. it would capture the FULL untruncated dataset
    and silently ignore --max-samples."""
    ranges = _shard_ranges_for_dataset(expected_size=10_000, shard_size=5_000, max_samples=3_000)
    assert ranges == [(0, 3_000)]


def test_max_samples_equal_to_expected_size_is_untruncated():
    """max_samples == expected_size: no truncation occurred, so a dataset
    under shard_size still collapses to the unsuffixed full-dataset form."""
    ranges = _shard_ranges_for_dataset(expected_size=3_000, shard_size=5_000, max_samples=3_000)
    assert ranges == [(None, None)]


def test_max_samples_none_is_unaffected():
    ranges_with_none = _shard_ranges_for_dataset(expected_size=7_405, shard_size=5_000, max_samples=None)
    ranges_default = _shard_ranges_for_dataset(expected_size=7_405, shard_size=5_000)
    assert ranges_with_none == ranges_default == [(0, 5_000), (5_000, 7_405)]


def test_generate_manifest_max_samples_shard_size_hotpotqa_train_10_shards(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["hotpotqa"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"],
        n_samples=None,
        shard_size=5_000,
        max_samples=50_000,
    )
    assert n == 10

    pending = list((dispatch_root / "pending").glob("*.json"))
    cell_ids = sorted(p.stem for p in pending)
    expected_ids = sorted(
        f"hotpotqa_train_Llama-3.1-8B-Instruct_{i * 5_000}-{(i + 1) * 5_000}" for i in range(10)
    )
    assert cell_ids == expected_ids

    last_cell = json.loads((dispatch_root / "pending" / "hotpotqa_train_Llama-3.1-8B-Instruct_45000-50000.json").read_text())
    assert last_cell["index_start"] == 45_000
    assert last_cell["index_end"] == 50_000
    assert last_cell["shuffle_seed"] == 0


def test_generate_manifest_max_samples_shard_size_popqa_train_3_shards(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["popqa"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"],
        n_samples=None,
        shard_size=5_000,
        max_samples=50_000,
    )
    assert n == 3
    pending = list((dispatch_root / "pending").glob("*.json"))
    cell_ids = sorted(p.stem for p in pending)
    assert cell_ids == [
        "popqa_train_Llama-3.1-8B-Instruct_0-5000",
        "popqa_train_Llama-3.1-8B-Instruct_10000-11413",
        "popqa_train_Llama-3.1-8B-Instruct_5000-10000",
    ]


def test_generate_manifest_max_samples_reinvocation_idempotent(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n1 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["popqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"], n_samples=None, shard_size=5_000, max_samples=50_000,
    )
    assert n1 == 3

    n2 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["popqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"], n_samples=None, shard_size=5_000, max_samples=50_000,
    )
    assert n2 == 0, "re-invocation must not duplicate already-queued shards"


# ---------------------------------------------------------------------------
# --max-samples + --cap mutual exclusion
# ---------------------------------------------------------------------------

def test_generate_manifest_rejects_cap_and_max_samples_together(tmp_path):
    with pytest.raises(ValueError):
        generate_manifest(
            dispatch_root=tmp_path / "_dispatch",
            out_base_dir=tmp_path / "icr",
            tasks=["sciq"], models=["meta-llama/Llama-3.1-8B-Instruct"],
            splits=["test"], n_samples=None,
            cap=1_000, max_samples=50_000,
        )


def test_cli_rejects_cap_and_max_samples_together(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/dispatch/generate_manifest.py"),
            "--dispatch-root", str(tmp_path / "_dispatch"),
            "--out-base-dir", str(tmp_path / "icr"),
            "--tasks", "sciq",
            "--models", "meta-llama/Llama-3.1-8B-Instruct",
            "--splits", "test",
            "--cap", "1000",
            "--max-samples", "50000",
        ],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr.lower()
