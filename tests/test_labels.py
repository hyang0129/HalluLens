"""Tests for activation_research.labels.load_meta (dual-label contract, issue #145)."""
import json

import pytest

from activation_research.labels import load_meta

META = [
    {"sample_index": 0, "hallucinated": True},
    {"sample_index": 1, "hallucinated": False},
    {"sample_index": 2, "hallucinated": True},
]


def _write_capture(tmp_path, meta, judge=None):
    d = tmp_path / "cap"
    d.mkdir()
    with (d / "meta.jsonl").open("w") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")
    if judge is not None:
        with (d / "judge_labels.jsonl").open("w") as f:
            for r in judge:
                f.write(json.dumps(r) + "\n")
    return d


def test_substring_default(tmp_path):
    d = _write_capture(tmp_path, META)
    assert [r["hallucinated"] for r in load_meta(d, "substring")] == [True, False, True]


def test_llm_judge_overrides(tmp_path):
    judge = [
        {"sample_index": 0, "judge_verdict": "CORRECT", "hallucinated": False},
        {"sample_index": 1, "judge_verdict": "INCORRECT", "hallucinated": True},
        {"sample_index": 2, "judge_verdict": "CORRECT", "hallucinated": False},
    ]
    d = _write_capture(tmp_path, META, judge)
    assert [r["hallucinated"] for r in load_meta(d, "llm_judge")] == [False, True, False]


def test_unknown_and_missing_fall_back_to_substring(tmp_path):
    judge = [
        {"sample_index": 0, "judge_verdict": "UNKNOWN", "hallucinated": False},
        # sample_index 1 absent from the sidecar
        {"sample_index": 2, "judge_verdict": "CORRECT", "hallucinated": False},
    ]
    d = _write_capture(tmp_path, META, judge)
    # 0: UNKNOWN -> keep substring True; 1: missing -> keep substring False; 2: judge -> False
    assert [r["hallucinated"] for r in load_meta(d, "llm_judge")] == [True, False, False]


def test_join_is_by_sample_index_not_row_order(tmp_path):
    judge = [  # deliberately shuffled relative to meta
        {"sample_index": 2, "judge_verdict": "CORRECT", "hallucinated": False},
        {"sample_index": 0, "judge_verdict": "CORRECT", "hallucinated": False},
        {"sample_index": 1, "judge_verdict": "INCORRECT", "hallucinated": True},
    ]
    d = _write_capture(tmp_path, META, judge)
    assert [r["hallucinated"] for r in load_meta(d, "llm_judge")] == [False, True, False]


def test_missing_sidecar_raises(tmp_path):
    d = _write_capture(tmp_path, META)
    with pytest.raises(FileNotFoundError):
        load_meta(d, "llm_judge")


def test_invalid_label_source_raises(tmp_path):
    d = _write_capture(tmp_path, META)
    with pytest.raises(ValueError):
        load_meta(d, "bogus")
