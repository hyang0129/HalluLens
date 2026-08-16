"""CPU-only contract tests for the issue #151 experiment matrix."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.dispatch.build_issue_151_tokenwise_cells import build
from scripts.dispatch.validate_issue_151_tokenwise import validate_cells


_ROOT = Path(__file__).resolve().parent.parent
_METHODS = (
    "tokenwise_contrastive_first_anchored",
    "tokenwise_contrastive_random_distinct",
)


def _load_json(relative: str) -> dict:
    return json.loads((_ROOT / relative).read_text(encoding="utf-8"))


def test_tokenwise_methods_match_standard_compressor_size_contract():
    baseline = _load_json("configs/methods/contrastive_logprob_recon.json")
    for method_name in _METHODS:
        method = _load_json(f"configs/methods/{method_name}.json")
        assert method["routine"] == "tokenwise_contrastive_logprob_recon"
        assert method["model_class"] == baseline["model_class"]
        assert method["model_params"] == baseline["model_params"]
        assert method["data"]["relevant_layers"] == "1-32"
        assert (
            method["data"]["view_adapter"]
            == "tokenwise_shared_contrastive_cache"
        )
        assert method["data"]["num_views"] == 2
        assert method["training"]["ignore_label"] == 1
        assert method["evaluation"]["eval_token_positions"][0] == 0
        assert "knn" in method["evaluation"]["metrics"]


def test_issue_151_builder_creates_50_idempotent_non_mmlu_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root) == 50
    assert build(dispatch_root) == 0

    cells = sorted((dispatch_root / "pending").glob("*.json"))
    assert len(cells) == 50
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in cells]
    assert {p["seed"] for p in payloads} == {"0", "1", "2", "3", "4"}
    assert {p["method"] for p in payloads} == set(_METHODS)
    assert {p["dataset"] for p in payloads} == {
        "hotpotqa_memmap",
        "nq_memmap",
        "popqa_memmap",
        "sciq_memmap",
        "searchqa_memmap",
    }
    assert all(p["priority"] == "high" for p in payloads)
    assert all(
        p["architecture"] == "tokenwise_shared_contrastive_cache_v2"
        for p in payloads
    )
    assert all("mmlu" not in json.dumps(p).lower() for p in payloads)
    assert all(p["output_check"].endswith("predictions.csv") for p in payloads)
    assert validate_cells(dispatch_root) == {
        "pending": 50,
        "claimed": 0,
        "done": 0,
        "failed": 0,
    }


def test_issue_151_builder_refreshes_stale_pending_cell(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root) == 50
    pending = next((dispatch_root / "pending").glob("*.json"))
    stale = json.loads(pending.read_text(encoding="utf-8"))
    stale.pop("architecture")
    pending.write_text(json.dumps(stale), encoding="utf-8")

    assert build(dispatch_root) == 1
    refreshed = json.loads(pending.read_text(encoding="utf-8"))
    assert refreshed["architecture"] == "tokenwise_shared_contrastive_cache_v2"
    assert build(dispatch_root) == 0
