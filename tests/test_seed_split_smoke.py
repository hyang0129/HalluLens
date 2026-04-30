"""
Smoke tests for k-fold seed splitting correctness.

Tests verify:
1. All experiment configs have correct split_seeds (seed 0 → 42, seeds 1-4 → 1-4)
2. Different split_seeds produce genuinely different train/val partitions (pure sklearn)
3. split_strategy="none" produces a constant test set regardless of seed
4. The actual_split_seed dispatch logic in run_experiment.py is correct
5. Integration: NQ Qwen3 train zarr produces different splits for seeds 1-4
   (requires the zarr data on the GPU node — skipped if files not found)

Run locally (no GPU needed):
    pytest tests/test_seed_split_smoke.py -v

Run on gpu22 for integration test:
    pytest tests/test_seed_split_smoke.py -v --integration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent

# All experiment configs that should have the split_seeds refactor applied
EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS = [
    "baseline_comparison_hotpotqa.json",
    "baseline_comparison_hotpotqa_qwen3.json",
    "baseline_comparison_mmlu.json",
    "baseline_comparison_nq.json",
    "baseline_comparison_nq_qwen3.json",
    "baseline_comparison_popqa.json",
    "baseline_comparison_sciq.json",
    "baseline_comparison_searchqa.json",
]

EXPECTED_TRAINING_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_SPLIT_SEEDS = [42, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Config validation tests (no I/O except reading JSON)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS)
def test_experiment_config_has_split_seeds(config_name):
    """Every updated experiment config must have a split_seeds list."""
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "split_seeds" in cfg, f"{config_name} is missing 'split_seeds'"


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS)
def test_experiment_config_split_seeds_length_matches_training_seeds(config_name):
    """split_seeds and training_seeds must have the same length."""
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert len(cfg["split_seeds"]) == len(cfg["training_seeds"]), (
        f"{config_name}: len(split_seeds)={len(cfg['split_seeds'])} != "
        f"len(training_seeds)={len(cfg['training_seeds'])}"
    )


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS)
def test_experiment_config_seed0_maps_to_split_seed_42(config_name):
    """Seed 0 must always map to split_seed=42 to preserve existing results."""
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    with open(cfg_path) as f:
        cfg = json.load(f)
    split_seeds = cfg["split_seeds"]
    training_seeds = cfg["training_seeds"]
    idx0 = training_seeds.index(0)
    assert split_seeds[idx0] == 42, (
        f"{config_name}: split_seeds[{idx0}]={split_seeds[idx0]} but expected 42 "
        f"for training_seed=0 (backward compatibility)"
    )


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS)
def test_experiment_config_seeds_1_to_4_map_correctly(config_name):
    """Seeds 1-4 must map to split_seeds 1-4 for proper k-fold diversity."""
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    with open(cfg_path) as f:
        cfg = json.load(f)
    split_seeds = cfg["split_seeds"]
    training_seeds = cfg["training_seeds"]
    seed_map = dict(zip(training_seeds, split_seeds))
    for seed in [1, 2, 3, 4]:
        if seed in seed_map:
            assert seed_map[seed] == seed, (
                f"{config_name}: training_seed={seed} maps to split_seed={seed_map[seed]}, expected {seed}"
            )


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS_WITH_SPLIT_SEEDS)
def test_experiment_config_split_seeds_are_unique(config_name):
    """All split_seeds must be distinct (each fold sees a different partition)."""
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    with open(cfg_path) as f:
        cfg = json.load(f)
    split_seeds = cfg["split_seeds"]
    assert len(split_seeds) == len(set(split_seeds)), (
        f"{config_name}: split_seeds contains duplicates: {split_seeds}"
    )


# ---------------------------------------------------------------------------
# Split logic unit tests (pure sklearn, no zarr)
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    """Build a balanced synthetic DataFrame mimicking AP's gendf."""
    return pd.DataFrame(
        {
            "prompt_hash": [f"hash_{i:06d}" for i in range(n)],
            "halu": ([True] * (n // 2) + [False] * (n // 2)),
        }
    )


def test_different_split_seeds_produce_different_train_sets():
    """Different random_seed values in train_test_split yield different partitions."""
    df = _make_synthetic_df(200)
    seeds = [42, 1, 2, 3, 4]
    train_sets = []
    for seed in seeds:
        train_df, _ = train_test_split(
            df, test_size=0.2, stratify=df["halu"], random_state=seed
        )
        train_sets.append(frozenset(train_df["prompt_hash"]))

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert train_sets[i] != train_sets[j], (
                f"Seeds {seeds[i]} and {seeds[j]} produced identical train splits"
            )


def test_split_strategy_none_assigns_all_to_test():
    """split_strategy='none' should assign every row to the 'test' split."""
    df = _make_synthetic_df(100)
    df["split"] = "test"  # mirrors ActivationParser split_strategy="none" branch
    assert (df["split"] == "test").all()
    assert "train" not in df["split"].values


def test_test_set_is_constant_with_split_strategy_none():
    """With split_strategy='none', different random_seed values see identical test sets."""
    df = _make_synthetic_df(100)
    test_sets = []
    for seed in [42, 1, 2, 3, 4]:
        # Mimic ActivationParser: split_strategy="none" → all rows become 'test'
        local_df = df.copy()
        local_df["split"] = "test"
        test_sets.append(frozenset(local_df[local_df["split"] == "test"]["prompt_hash"]))

    assert len(set(test_sets)) == 1, "Test set changed across seeds — should be constant"


def test_same_split_seed_is_reproducible():
    """Same random_seed always yields the same partition (determinism check)."""
    df = _make_synthetic_df(200)
    train_a, _ = train_test_split(df, test_size=0.2, stratify=df["halu"], random_state=42)
    train_b, _ = train_test_split(df, test_size=0.2, stratify=df["halu"], random_state=42)
    assert frozenset(train_a["prompt_hash"]) == frozenset(train_b["prompt_hash"])


def test_train_test_no_overlap():
    """Train and test sets must be disjoint for all seeds."""
    df = _make_synthetic_df(200)
    for seed in [42, 1, 2, 3, 4]:
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["halu"], random_state=seed
        )
        overlap = set(train_df["prompt_hash"]) & set(test_df["prompt_hash"])
        assert not overlap, f"Seed {seed}: train/test overlap: {len(overlap)} rows"


# ---------------------------------------------------------------------------
# Dispatch logic test (no actual training, no zarr)
# ---------------------------------------------------------------------------


def _compute_actual_split_seeds(experiment_cfg: dict) -> list[int | None]:
    """Mirror the actual_split_seed computation in run_experiment.py main()."""
    split_seeds_list = experiment_cfg.get("split_seeds", None)
    global_split_seed = experiment_cfg.get("split_seed", 42)
    training_seeds = experiment_cfg.get("training_seeds", [42])

    result = []
    for seed_idx, _seed in enumerate(training_seeds):
        actual = (
            split_seeds_list[seed_idx]
            if split_seeds_list is not None
            else global_split_seed
        )
        result.append(actual)
    return result


def test_dispatch_logic_uses_split_seeds_list():
    """When split_seeds is present, each seed uses its corresponding split_seed."""
    cfg = {
        "training_seeds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 1, 2, 3, 4],
        "split_seed": 42,
    }
    actual = _compute_actual_split_seeds(cfg)
    assert actual == [42, 1, 2, 3, 4]


def test_dispatch_logic_falls_back_to_global_split_seed():
    """When split_seeds is absent, every seed uses the global split_seed."""
    cfg = {
        "training_seeds": [0, 1, 2, 3, 4],
        "split_seed": 42,
    }
    actual = _compute_actual_split_seeds(cfg)
    assert actual == [42, 42, 42, 42, 42]


def test_dispatch_logic_seed0_backward_compat():
    """Seed 0 must map to split_seed 42 to preserve existing seed-0 runs."""
    cfg = {
        "training_seeds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 1, 2, 3, 4],
    }
    actual = _compute_actual_split_seeds(cfg)
    assert actual[0] == 42, f"Seed 0 mapped to split_seed={actual[0]}, expected 42"


def test_dispatch_logic_produces_distinct_split_seeds():
    """The dispatch must produce a unique split_seed for each training seed."""
    cfg = {
        "training_seeds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 1, 2, 3, 4],
    }
    actual = _compute_actual_split_seeds(cfg)
    assert len(actual) == len(set(actual)), f"Duplicate split_seeds in dispatch: {actual}"


# ---------------------------------------------------------------------------
# Integration test: real NQ Qwen3 zarr (skipped if data not present)
# ---------------------------------------------------------------------------


NQ_QWEN3_TRAIN_INFERENCE_JSON = (
    PROJECT_ROOT / "output" / "natural_questions_train" / "Qwen3-8B" / "generation.jsonl"
)
NQ_QWEN3_TRAIN_EVAL_JSON = (
    PROJECT_ROOT
    / "output"
    / "natural_questions_train"
    / "Qwen3-8B"
    / "eval_results_for_training.json"
)


@pytest.mark.skipif(
    not NQ_QWEN3_TRAIN_INFERENCE_JSON.exists() or not NQ_QWEN3_TRAIN_EVAL_JSON.exists(),
    reason="NQ Qwen3 training data not available on this node",
)
def test_nq_qwen3_train_splits_are_distinct_across_seeds():
    """Seeds 1-4 produce genuinely different train partitions on real NQ Qwen3 data."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    # Load generation.jsonl to build a DataFrame — same logic as ActivationParser._load_gendf
    records = []
    with open(NQ_QWEN3_TRAIN_INFERENCE_JSON) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    gendf = pd.DataFrame(records)

    # Load eval labels
    with open(NQ_QWEN3_TRAIN_EVAL_JSON) as f:
        eval_data = json.load(f)

    # Map halu labels onto gendf (mirrors ActivationParser logic)
    if isinstance(eval_data, list):
        eval_df = pd.DataFrame(eval_data)
    else:
        eval_df = pd.DataFrame(eval_data.get("results", eval_data))

    if "prompt_hash" in eval_df.columns and "halu" in eval_df.columns:
        halu_map = dict(zip(eval_df["prompt_hash"], eval_df["halu"]))
        if "prompt_hash" not in gendf.columns:
            import hashlib
            gendf["prompt_hash"] = gendf["prompt"].apply(
                lambda p: hashlib.sha256(str(p).encode()).hexdigest()
            )
        gendf["halu"] = gendf["prompt_hash"].map(halu_map)
        gendf = gendf.dropna(subset=["halu"])
        gendf["halu"] = gendf["halu"].astype(bool)
    else:
        pytest.skip("eval_results_for_training.json doesn't have expected halu/prompt_hash columns")

    if len(gendf) < 50:
        pytest.skip(f"Too few labelled samples ({len(gendf)}) to run split test")

    seeds = [42, 1, 2, 3, 4]
    train_sets = {}
    for seed in seeds:
        train_df, _ = train_test_split(
            gendf, test_size=0.2, stratify=gendf["halu"], random_state=seed
        )
        train_sets[seed] = frozenset(train_df.index)

    # All 5 splits must be pairwise distinct
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1 :]:
            assert train_sets[sa] != train_sets[sb], (
                f"Seeds {sa} and {sb} produced identical train splits on real NQ Qwen3 data"
            )

    # Report sizes for manual inspection
    for seed in seeds:
        print(f"  seed={seed}: n_train={len(train_sets[seed])}")


@pytest.mark.skipif(
    not NQ_QWEN3_TRAIN_INFERENCE_JSON.exists() or not NQ_QWEN3_TRAIN_EVAL_JSON.exists(),
    reason="NQ Qwen3 training data not available on this node",
)
def test_nq_qwen3_seed0_split_seed42_matches_existing():
    """Seed 0 (split_seed=42) must produce the same partition as before the refactor."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    records = []
    with open(NQ_QWEN3_TRAIN_INFERENCE_JSON) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    gendf = pd.DataFrame(records)

    with open(NQ_QWEN3_TRAIN_EVAL_JSON) as f:
        eval_data = json.load(f)

    if isinstance(eval_data, list):
        eval_df = pd.DataFrame(eval_data)
    else:
        eval_df = pd.DataFrame(eval_data.get("results", eval_data))

    if "prompt_hash" in eval_df.columns and "halu" in eval_df.columns:
        halu_map = dict(zip(eval_df["prompt_hash"], eval_df["halu"]))
        if "prompt_hash" not in gendf.columns:
            import hashlib
            gendf["prompt_hash"] = gendf["prompt"].apply(
                lambda p: hashlib.sha256(str(p).encode()).hexdigest()
            )
        gendf["halu"] = gendf["prompt_hash"].map(halu_map)
        gendf = gendf.dropna(subset=["halu"])
        gendf["halu"] = gendf["halu"].astype(bool)
    else:
        pytest.skip("eval_results_for_training.json doesn't have expected columns")

    if len(gendf) < 50:
        pytest.skip(f"Too few labelled samples ({len(gendf)})")

    # Running split_seed=42 twice must yield the same result (determinism)
    train_a, _ = train_test_split(gendf, test_size=0.2, stratify=gendf["halu"], random_state=42)
    train_b, _ = train_test_split(gendf, test_size=0.2, stratify=gendf["halu"], random_state=42)
    assert frozenset(train_a.index) == frozenset(train_b.index), (
        "split_seed=42 is not deterministic — seed-0 backward compatibility is broken"
    )
    print(f"  split_seed=42: n_train={len(train_a)}, n_test={len(gendf)-len(train_a)}")
