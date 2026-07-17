import json

from scripts.audit_datasets import DATASETS, capture_dir_for, judge_coverage


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_capture_dir_resolves_sharded_train_and_searchqa_flip(tmp_path):
    root = tmp_path
    capture_root = root / "shared" / "icr_capture"
    mmlu = capture_root / "mmlu_train_Qwen3-8B_0-50000"
    searchqa = capture_root / "searchqa_test_Qwen3-8B"
    mmlu.mkdir(parents=True)
    searchqa.mkdir()

    assert capture_dir_for(root, "mmlu_train", "Qwen3-8B") == mmlu
    assert capture_dir_for(root, "searchqa_train", "Qwen3-8B") == searchqa


def test_audit_includes_simpleqa_and_triviaqa_splits():
    rows = {(dataset, split, expected) for dataset, split, expected in DATASETS}

    assert ("simpleqa", "test", 866) in rows
    assert ("simpleqa_train", "train", 3_460) in rows
    assert ("triviaqa", "test", 9_954) in rows
    assert ("triviaqa_train", "train", 11_000) in rows


def test_judge_coverage_counts_finalized_and_unknown(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_jsonl(cap / "generation.jsonl", [
        {"sample_index": 0}, {"sample_index": 1}, {"sample_index": 2},
    ])
    _write_jsonl(cap / "judge_labels.jsonl", [
        {"sample_index": 0, "judge_verdict": "CORRECT", "hallucinated": False},
        {"sample_index": 1, "judge_verdict": "UNKNOWN", "hallucinated": False},
    ])

    result = judge_coverage(cap)

    assert result["finalized"] == 1
    assert result["generation"] == 3
    assert result["unknown"] == 1


def test_judge_coverage_is_absent_without_sidecar(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_jsonl(cap / "generation.jsonl", [{"sample_index": 0}])

    assert judge_coverage(cap) is None
