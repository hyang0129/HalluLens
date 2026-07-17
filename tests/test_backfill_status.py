import json
from pathlib import Path

from scripts.label_audit.backfill_status import inspect_capture


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_inspect_capture_counts_final_unknown_and_missing(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    _write(cap / "generation.jsonl", [{"sample_index": i} for i in range(3)])
    _write(cap / "judge_labels.jsonl", [
        {"sample_index": 0, "judge_verdict": "CORRECT", "hallucinated": False},
        {"sample_index": 1, "judge_verdict": "UNKNOWN", "hallucinated": False},
    ])

    result = inspect_capture(cap)

    assert result["generation"] == 3
    assert result["finalized"] == 1
    assert result["missing"] == 2
    assert result["unknown"] == 1


def test_inspect_capture_detects_integrity_problems(tmp_path):
    cap = tmp_path / "capture"
    cap.mkdir()
    _write(cap / "generation.jsonl", [{"sample_index": 0}])
    _write(cap / "judge_labels.jsonl", [
        {"sample_index": 0, "judge_verdict": "CORRECT", "hallucinated": True},
        {"sample_index": 0, "judge_verdict": "INCORRECT", "hallucinated": True},
        {"sample_index": 9, "judge_verdict": "CORRECT", "hallucinated": False},
    ])

    result = inspect_capture(cap)

    assert result["duplicate_rows"] == 1
    assert result["outside_generation"] == 1
    assert result["inconsistent_rows"] == 1
