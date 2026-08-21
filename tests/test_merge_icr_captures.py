"""Tests for scripts/merge_icr_captures.py — N-way icr_capture dir merge.

No prior tests existed for this module (checked tests/ for anything covering
merge_icr_captures — none found), so these establish synthetic-fixture
coverage from scratch, following the style of the other CPU-only capture
tests (tmp_path fixtures, no GPU/model deps).

Each synthetic input dir models the real capture layout: raw (headerless)
memmap .npy files pre-allocated to config["n_samples"] rows, but only the
first n_written rows actually written (meta.jsonl line count) — the rest is
untouched tail, per the module's own docstring on the #142 tail-misalignment
bug this write/read discipline exists to avoid.

Covers:
- 3-way merge: written-prefix-only concatenation, in order, tail never copied
- sample_index shift by cumulative written counts across inputs
- --a/--b backward compat (== --inputs A B)
- config sanity check: chat_template mismatch across inputs -> refusal
- config sanity check: model_name mismatch across inputs -> refusal
- --skip still honored
- fewer than 2 --inputs -> refusal
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MERGE_SCRIPT = _REPO_ROOT / "scripts" / "merge_icr_captures.py"


def _make_capture_dir(
    tmp_path: Path,
    name: str,
    n_alloc: int,
    n_written: int,
    *,
    row_width_floats: int = 4,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    chat_template: bool = False,
    array_names: tuple[str, ...] = ("response_activations.npy",),
    fill_value: float | None = None,
) -> Path:
    """Build a synthetic icr_capture dir: pre-allocated arrays, written prefix
    only in meta.jsonl, config.json with n_samples=n_alloc (allocated, NOT
    written) — mirroring InferenceCaptureWriter's real on-disk contract.
    """
    d = tmp_path / name
    d.mkdir(parents=True)

    config = {
        "model_name": model_name,
        "num_layers": 2,
        "hidden_dim": row_width_floats,
        "r_max": 4,
        "dtype": "float32",
        "n_samples": n_alloc,
        "chat_template": chat_template,
    }
    (d / "config.json").write_text(json.dumps(config))

    rng = np.random.default_rng(hash(name) % (2**31))
    for arr_name in array_names:
        # Written rows get real (identifiable) values; the untouched tail
        # is filled with a sentinel so a test can assert it was never read.
        if fill_value is None:
            written_block = rng.random((n_written, row_width_floats), dtype=np.float32)
        else:
            written_block = np.full((n_written, row_width_floats), fill_value, dtype=np.float32)
        tail_sentinel = np.full((n_alloc - n_written, row_width_floats), -999.0, dtype=np.float32)
        full = np.concatenate([written_block, tail_sentinel], axis=0)
        full.tofile(d / arr_name)

    with (d / "meta.jsonl").open("w") as f:
        for i in range(n_written):
            f.write(json.dumps({
                "sample_index": i,
                "key": f"{name}_{i}",
                "prompt_len": 10,
                "response_len": 5,
                "hallucinated": bool(i % 2),
            }) + "\n")

    with (d / "generation.jsonl").open("w") as f:
        for i in range(n_written):
            f.write(json.dumps({"prompt": f"{name} prompt {i}", "generation": f"gen {i}"}) + "\n")

    return d


def _read_array(d: Path, name: str, n_rows: int, row_width: int) -> np.ndarray:
    return np.fromfile(d / name, dtype=np.float32).reshape(n_rows, row_width)


def _run_merge(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_MERGE_SCRIPT), *args],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# 3-way merge: written-prefix-only concatenation
# ---------------------------------------------------------------------------

def test_three_way_merge_concatenates_written_prefixes_in_order(tmp_path):
    row_width = 4
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, row_width_floats=row_width, fill_value=1.0)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, row_width_floats=row_width, fill_value=2.0)
    c = _make_capture_dir(tmp_path, "c", n_alloc=6, n_written=2, row_width_floats=row_width, fill_value=3.0)
    out = tmp_path / "merged"

    result = _run_merge([
        "--inputs", str(a), str(b), str(c),
        "--out", str(out),
        "--skip",  # empty skip list — merge the one test array too
    ])
    assert result.returncode == 0, result.stderr

    merged = _read_array(out, "response_activations.npy", n_rows=3 + 5 + 2, row_width=row_width)

    # Order: A's 3 written rows (value 1.0), then B's 5 (2.0), then C's 2 (3.0).
    # No tail sentinel (-999.0) rows anywhere — tail was never copied.
    assert np.all(merged[:3] == 1.0)
    assert np.all(merged[3:8] == 2.0)
    assert np.all(merged[8:10] == 3.0)
    assert not np.any(merged == -999.0), "untouched tail rows must never be copied into the merge"


def test_three_way_merge_sample_index_shifted_by_cumulative_written(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5)
    c = _make_capture_dir(tmp_path, "c", n_alloc=6, n_written=2)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), str(c), "--out", str(out)])
    assert result.returncode == 0, result.stderr

    lines = [json.loads(ln) for ln in (out / "meta.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 3 + 5 + 2

    # A's original indices [0,1,2] stay as-is (cumulative offset 0).
    assert [r["sample_index"] for r in lines[:3]] == [0, 1, 2]
    # B's original indices [0..4] shift by nA_w=3 -> [3..7].
    assert [r["sample_index"] for r in lines[3:8]] == [3, 4, 5, 6, 7]
    # C's original indices [0,1] shift by nA_w+nB_w=8 -> [8,9].
    assert [r["sample_index"] for r in lines[8:10]] == [8, 9]

    # keys carry through unmodified so provenance is traceable.
    assert lines[0]["key"] == "a_0"
    assert lines[3]["key"] == "b_0"
    assert lines[8]["key"] == "c_0"


def test_three_way_merge_config_records_provenance(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5)
    c = _make_capture_dir(tmp_path, "c", n_alloc=6, n_written=2)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), str(c), "--out", str(out)])
    assert result.returncode == 0, result.stderr

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["n_samples"] == 10
    assert cfg["_merged_from"] == ["a", "b", "c"]
    assert cfg["_merged_written_rows"] == [3, 5, 2]


def test_three_way_merge_generation_jsonl_concatenated_in_order(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5)
    c = _make_capture_dir(tmp_path, "c", n_alloc=6, n_written=2)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), str(c), "--out", str(out)])
    assert result.returncode == 0, result.stderr

    gen_lines = [json.loads(ln) for ln in (out / "generation.jsonl").read_text().splitlines() if ln.strip()]
    assert len(gen_lines) == 3 + 5 + 2
    assert gen_lines[0]["prompt"].startswith("a prompt")
    assert gen_lines[3]["prompt"].startswith("b prompt")
    assert gen_lines[8]["prompt"].startswith("c prompt")


# ---------------------------------------------------------------------------
# --a/--b backward compat
# ---------------------------------------------------------------------------

def test_a_b_flags_still_work_as_two_way_shorthand(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, fill_value=1.0)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, fill_value=2.0)
    out = tmp_path / "merged"

    result = _run_merge(["--a", str(a), "--b", str(b), "--out", str(out), "--skip"])
    assert result.returncode == 0, result.stderr

    merged = _read_array(out, "response_activations.npy", n_rows=3 + 5, row_width=4)
    assert np.all(merged[:3] == 1.0)
    assert np.all(merged[3:8] == 2.0)

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["_merged_from"] == ["a", "b"]


def test_a_only_without_b_is_rejected(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    out = tmp_path / "merged"
    result = _run_merge(["--a", str(a), "--out", str(out)])
    assert result.returncode != 0


def test_inputs_and_a_b_together_is_rejected(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5)
    out = tmp_path / "merged"
    result = _run_merge(["--inputs", str(a), str(b), "--a", str(a), "--b", str(b), "--out", str(out)])
    assert result.returncode != 0


def test_single_input_is_rejected(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3)
    out = tmp_path / "merged"
    result = _run_merge(["--inputs", str(a), "--out", str(out)])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Config sanity checks — refuse on mismatch
# ---------------------------------------------------------------------------

def test_chat_template_mismatch_refused(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, chat_template=False)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, chat_template=True)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out)])
    assert result.returncode != 0
    assert "chat_template" in result.stderr.lower() or "chat_template" in result.stdout.lower()
    # Nothing should have been written into out beyond the empty dir it created.
    assert not (out / "meta.jsonl").exists()


def test_chat_template_missing_key_treated_as_false_matches_explicit_false(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, chat_template=False)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, chat_template=False)
    # Simulate a legacy dir predating the chat_template field entirely.
    cfg_path = a / "config.json"
    cfg = json.loads(cfg_path.read_text())
    del cfg["chat_template"]
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out)])
    assert result.returncode == 0, result.stderr


def test_model_name_mismatch_refused(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, model_name="meta-llama/Llama-3.1-8B-Instruct")
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, model_name="Qwen/Qwen3-8B")
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out)])
    assert result.returncode != 0
    assert "model_name" in result.stderr.lower() or "model_name" in result.stdout.lower()


def test_three_way_one_mismatched_middle_input_refused(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, chat_template=True)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, chat_template=False)
    c = _make_capture_dir(tmp_path, "c", n_alloc=6, n_written=2, chat_template=True)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), str(c), "--out", str(out)])
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# --skip still honored
# ---------------------------------------------------------------------------

def test_skip_omits_named_array_from_output(tmp_path):
    a = _make_capture_dir(
        tmp_path, "a", n_alloc=10, n_written=3,
        array_names=("response_activations.npy", "icr_scores.npy"),
    )
    b = _make_capture_dir(
        tmp_path, "b", n_alloc=8, n_written=5,
        array_names=("response_activations.npy", "icr_scores.npy"),
    )
    out = tmp_path / "merged"

    # Default --skip includes icr_scores.npy.
    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out)])
    assert result.returncode == 0, result.stderr
    assert (out / "response_activations.npy").exists()
    assert not (out / "icr_scores.npy").exists()


def test_explicit_empty_skip_merges_everything(tmp_path):
    a = _make_capture_dir(
        tmp_path, "a", n_alloc=10, n_written=3,
        array_names=("response_activations.npy", "icr_scores.npy"),
    )
    b = _make_capture_dir(
        tmp_path, "b", n_alloc=8, n_written=5,
        array_names=("response_activations.npy", "icr_scores.npy"),
    )
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out), "--skip"])
    assert result.returncode == 0, result.stderr
    assert (out / "response_activations.npy").exists()
    assert (out / "icr_scores.npy").exists()


# ---------------------------------------------------------------------------
# Structural mismatch (row width) still refused, N-way
# ---------------------------------------------------------------------------

def test_row_width_mismatch_across_inputs_refused(tmp_path):
    a = _make_capture_dir(tmp_path, "a", n_alloc=10, n_written=3, row_width_floats=4)
    b = _make_capture_dir(tmp_path, "b", n_alloc=8, n_written=5, row_width_floats=6)
    out = tmp_path / "merged"

    result = _run_merge(["--inputs", str(a), str(b), "--out", str(out), "--skip"])
    assert result.returncode != 0
