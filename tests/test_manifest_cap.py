"""Tests for the --cap subsampling logic in scripts/dispatch/generate_manifest.

Covers:
- _slice_ranges_for_dataset: full / capped / multi-slice / skip-completed
- generate_manifest with --cap emits the right cells
- Re-running with a larger cap appends the next slice instead of duplicating
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dispatch.generate_manifest import (  # noqa: E402
    _slice_ranges_for_dataset,
    generate_manifest,
)


def test_no_cap_returns_full_slice(tmp_path):
    ranges = _slice_ranges_for_dataset(
        expected_size=10_000, cap=None, out_base_dir=tmp_path, base_cell_id="x",
    )
    assert ranges == [(None, None)]


def test_dataset_under_cap_returns_full_slice(tmp_path):
    ranges = _slice_ranges_for_dataset(
        expected_size=11_679, cap=50_000, out_base_dir=tmp_path, base_cell_id="x",
    )
    assert ranges == [(None, None)]


def test_dataset_over_cap_emits_capped_slices(tmp_path):
    ranges = _slice_ranges_for_dataset(
        expected_size=90_447, cap=50_000, out_base_dir=tmp_path, base_cell_id="x",
    )
    assert ranges == [(0, 50_000), (50_000, 90_447)]


def test_skip_completed_slices(tmp_path):
    (tmp_path / "x_0-50000").mkdir()
    (tmp_path / "x_0-50000" / "eval_results.json").write_text("{}")

    ranges = _slice_ranges_for_dataset(
        expected_size=90_447, cap=50_000, out_base_dir=tmp_path, base_cell_id="x",
    )
    assert ranges == [(50_000, 90_447)]


def test_generate_manifest_emits_capped_cells_with_index_fields(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["hotpotqa"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"],
        n_samples=None,
        cap=50_000,
    )
    assert n == 2, f"expected 2 cells (90447 split into 2), got {n}"

    pending = list((dispatch_root / "pending").glob("*.json"))
    cell_ids = sorted(p.stem for p in pending)
    assert cell_ids == [
        "hotpotqa_train_Llama-3.1-8B-Instruct_0-50000",
        "hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447",
    ]

    first = json.loads(pending[0].read_text())
    assert first["index_start"] == 0 or first["index_start"] == 50_000
    assert first["index_end"] in (50_000, 90_447)
    assert first["shuffle_seed"] == 0
    assert "_0-" in first["out_dir"] or "_50000-" in first["out_dir"]


def test_generate_manifest_under_cap_omits_index_suffix(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["sciq"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"],
        n_samples=None,
        cap=50_000,
    )
    assert n == 1

    pending = list((dispatch_root / "pending").glob("*.json"))
    assert len(pending) == 1
    assert pending[0].stem == "sciq_train_Llama-3.1-8B-Instruct"
    cell = json.loads(pending[0].read_text())
    assert cell["index_start"] is None
    assert cell["index_end"] is None


def test_appendix_run_emits_only_new_slice(tmp_path):
    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    # First pass: cap=50000. Should emit two cells covering [0, 50000) and [50000, 90447).
    n1 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"], n_samples=None, cap=50_000,
    )
    assert n1 == 2

    # Simulate the first slice having completed: drop a done eval_results.json
    # and remove its pending cell. The dispatcher's normal flow would do this,
    # but for this test we do it manually.
    first_cell = dispatch_root / "pending" / "hotpotqa_train_Llama-3.1-8B-Instruct_0-50000.json"
    first_cell.unlink()
    second_cell = dispatch_root / "pending" / "hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447.json"
    second_cell.unlink()
    done_dir = out_base / "hotpotqa_train_Llama-3.1-8B-Instruct_0-50000"
    done_dir.mkdir(parents=True)
    (done_dir / "eval_results.json").write_text("{}")

    # Second pass: same cap. Should emit only the second slice, not re-queue the first.
    n2 = generate_manifest(
        dispatch_root=dispatch_root, out_base_dir=out_base,
        tasks=["hotpotqa"], models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["train"], n_samples=None, cap=50_000,
    )
    assert n2 == 1
    pending = list((dispatch_root / "pending").glob("*.json"))
    assert [p.stem for p in pending] == ["hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447"]
