"""Tests for the ``seeded`` flag plumbing (issue #57).

Methods like ``llmsknow_probe`` have no ``training`` block but still produce
seed-dependent results (sklearn train/dev split, LogisticRegression
``random_state``). They must therefore be iterated across ``training_seeds``
the same way gradient-trained learned methods are. The runner gates this on
:func:`scripts.experiment_utils.is_seeded_method`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_utils import (
    RunSpec,
    enumerate_runs,
    is_seeded_method,
)


# ---------------------------------------------------------------------------
# is_seeded_method
# ---------------------------------------------------------------------------


def test_is_seeded_method_training_block():
    assert is_seeded_method({"training": {"max_epochs": 10}}) is True


def test_is_seeded_method_explicit_flag():
    assert is_seeded_method({"seeded": True}) is True


def test_is_seeded_method_no_training_no_flag():
    assert is_seeded_method({"sweep": {"dev_size": 2000}}) is False


def test_is_seeded_method_explicit_false_flag():
    assert is_seeded_method({"seeded": False}) is False


def test_is_seeded_method_training_block_with_falsey_seeded():
    # A method with training is always seeded, even if seeded: false sneaks in.
    assert is_seeded_method({"training": {"lr": 1.0}, "seeded": False}) is True


def test_is_seeded_method_empty():
    assert is_seeded_method({}) is False


# ---------------------------------------------------------------------------
# enumerate_runs respects the flag (regression for issue #57)
# ---------------------------------------------------------------------------


def _exp_cfg(method_configs):
    return {
        "experiment_name": "test_exp",
        "datasets": ["hotpotqa"],
        "methods": list(method_configs.keys()),
        "training_seeds": [0, 1, 2, 3, 4],
        "output_dir": "runs",
        "method_configs": method_configs,
    }


def test_enumerate_runs_seeded_flag_produces_one_run_per_seed():
    """Method with seeded:true but no training block must produce one run per seed."""
    cfg = _exp_cfg({"llmsknow_probe": {"name": "llmsknow_probe", "seeded": True}})
    specs = enumerate_runs(cfg, output_base="runs")

    seeds_seen = sorted(s.seed for s in specs)
    assert seeds_seen == [0, 1, 2, 3, 4]
    for s in specs:
        assert s.is_learned is True
        assert s.method_name == "llmsknow_probe"
        # Path must include a seed_{n} segment
        assert s.run_dir.endswith(f"seed_{s.seed}")


def test_enumerate_runs_unflagged_non_training_method_runs_once():
    """A method with neither training nor seeded:true still runs once with seed=None."""
    cfg = _exp_cfg({"token_entropy": {"name": "token_entropy"}})
    specs = enumerate_runs(cfg, output_base="runs")

    assert len(specs) == 1
    assert specs[0].seed is None
    assert specs[0].is_learned is False
    # No seed_ segment in the path
    assert "/seed_" not in specs[0].run_dir.replace("\\", "/")


def test_enumerate_runs_training_block_unchanged():
    """Regression guard: methods with a training block still produce one run per seed."""
    cfg = _exp_cfg({"linear_probe": {"name": "linear_probe", "training": {"lr": 1e-3}}})
    specs = enumerate_runs(cfg, output_base="runs")

    assert sorted(s.seed for s in specs) == [0, 1, 2, 3, 4]
    assert all(s.is_learned for s in specs)


def test_enumerate_runs_mixed_methods():
    """A realistic mix: one training-based, one seeded-flag, one neither."""
    cfg = _exp_cfg({
        "linear_probe": {"name": "linear_probe", "training": {"lr": 1e-3}},
        "llmsknow_probe": {"name": "llmsknow_probe", "seeded": True},
        "token_entropy": {"name": "token_entropy"},
    })
    specs = enumerate_runs(cfg, output_base="runs")

    by_method = {}
    for s in specs:
        by_method.setdefault(s.method_name, []).append(s)

    assert len(by_method["linear_probe"]) == 5
    assert len(by_method["llmsknow_probe"]) == 5
    assert len(by_method["token_entropy"]) == 1
    assert by_method["token_entropy"][0].seed is None


# ---------------------------------------------------------------------------
# Checked-in config: llmsknow_probe.json must carry the flag.
# ---------------------------------------------------------------------------


def test_llmsknow_probe_config_has_seeded_flag():
    """The shipped llmsknow_probe.json must declare seeded:true so all
    historical and future experiments produce per-seed probes (issue #57)."""
    repo_root = Path(__file__).parent.parent
    cfg_path = repo_root / "configs" / "methods" / "llmsknow_probe.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    assert cfg.get("seeded") is True, (
        "llmsknow_probe.json must set 'seeded: true' — without it the runner "
        "treats the method as non-learned and runs a single seed=None probe."
    )
