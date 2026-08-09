"""Tests for model/dataset label derivation in scripts/results_table.py.

Pure string logic over config *names* — no run data, no GPU, no I/O beyond
reading the real config directories for the drift guard at the bottom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.results_table import _model_from_config_name  # noqa: E402


# ---------------------------------------------------------------------------
# Regression fixtures: configs that append a suffix *after* the model token.
#
# These are the cases an ``endswith("_qwen3")`` check silently attributed to
# the Llama default. Every name below is a real config in configs/.
# ---------------------------------------------------------------------------

SUFFIXED_QWEN3_NAMES = [
    "hotpotqa_qwen3_full_memmap.json",
    "sciq_qwen3_judge_memmap.json",
    "baseline_comparison_hotpotqa_qwen3_flipped_memmap.json",
    "baseline_comparison_nq_qwen3_flipped_memmap.json",
    "baseline_comparison_popqa_qwen3_flipped_memmap.json",
    "baseline_comparison_sciq_qwen3_flipped_memmap.json",
    "baseline_comparison_sciq_qwen3_judge_memmap.json",
    "baseline_comparison_searchqa_qwen3_flipped_memmap.json",
    "baseline_comparison_simpleqa_qwen3_flipped_memmap.json",
    "baseline_comparison_triviaqa_qwen3_flipped_memmap.json",
    "dataeff_hotpotqa_qwen3_frac10.json",
    "dataeff_hotpotqa_qwen3_frac50.json",
    "dataeff_hotpotqa_qwen3_full.json",
]


@pytest.mark.parametrize("cfg_name", SUFFIXED_QWEN3_NAMES)
def test_qwen3_detected_when_token_is_not_the_final_suffix(cfg_name: str):
    assert _model_from_config_name(cfg_name) == "Qwen3-8B"


@pytest.mark.parametrize(
    "cfg_name",
    [
        "hotpotqa_qwen3.json",
        "hotpotqa_qwen3_memmap.json",
        "nq_test_qwen3.json",
        "simpleqa_popqa_merged_qwen3_memmap.json",
    ],
)
def test_qwen3_detected_as_trailing_suffix(cfg_name: str):
    """The pre-existing trailing-suffix cases must keep working."""
    assert _model_from_config_name(cfg_name) == "Qwen3-8B"


@pytest.mark.parametrize(
    "cfg_name",
    ["hotpotqa_smollm3.json", "mmlu_smollm3.json", "sciq_smollm3.json"],
)
def test_smollm3_detection_unaffected(cfg_name: str):
    assert _model_from_config_name(cfg_name) == "SmolLM3"


@pytest.mark.parametrize(
    "cfg_name",
    [
        "hotpotqa_memmap.json",
        "sciq_judge_memmap.json",
        "baseline_comparison_sciq_judge_memmap.json",
        "baseline_comparison_hotpotqa_flipped_memmap_seed0.json",
        "simpleqa_popqa_merged_memmap.json",
    ],
)
def test_llama_is_the_default_when_no_model_token(cfg_name: str):
    assert _model_from_config_name(cfg_name) == "Llama-3.1-8B-Instruct"


def test_model_token_must_be_a_whole_delimited_component():
    """Substring matches must not count — only ``_``-delimited tokens.

    Guards the token check from decaying back into a substring search, which
    would misread names that merely *contain* the model string.
    """
    assert _model_from_config_name("sciq_notqwen3x_memmap.json") == "Llama-3.1-8B-Instruct"
    assert _model_from_config_name("sciq_qwen3judge_memmap.json") == "Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


def _config_names() -> list[str]:
    names: list[str] = []
    for sub in ("datasets", "experiments"):
        d = PROJECT_ROOT / "configs" / sub
        if d.is_dir():
            names.extend(p.name for p in sorted(d.glob("*.json")))
    return names


def test_detected_model_agrees_with_declared_model_name():
    """Cross-check the filename heuristic against the authoritative field.

    Every ``configs/datasets/*.json`` declares ``model_name``. The label
    derived from the filename must agree with it — if a future config breaks
    this, the heuristic (not the config) is what needs replacing, ideally by
    reading ``model_name`` directly. See the follow-up noted in PR #148.
    """
    import json

    d = PROJECT_ROOT / "configs" / "datasets"
    if not d.is_dir():
        pytest.skip("configs/datasets not present")

    mismatches = []
    for path in sorted(d.glob("*.json")):
        declared = json.loads(path.read_text()).get("model_name")
        if not declared:
            continue
        derived = _model_from_config_name(path.name)
        # ``model_name`` uses the short form ("Qwen3-8B", "SmolLM3-3B",
        # "Llama-3.1-8B-Instruct"); compare on the leading family token.
        if declared.split("-")[0].lower() != derived.split("-")[0].lower():
            mismatches.append(f"{path.name}: declared={declared} derived={derived}")

    assert not mismatches, "filename-derived model disagrees with model_name:\n" + "\n".join(mismatches)


def test_no_config_name_is_ambiguous_between_two_models():
    """No config may carry both a qwen3 and a smollm3 token."""
    ambiguous = [
        n for n in _config_names()
        if {"qwen3", "smollm3"} <= set(n.removesuffix(".json").removesuffix("_memmap").split("_"))
    ]
    assert not ambiguous, f"ambiguous config names: {ambiguous}"
