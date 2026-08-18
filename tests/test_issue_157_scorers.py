"""Contracts for the validation-selected token-wise scorer study."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from activation_research.evaluation import (
    dump_embeddings_to_memmap,
    write_embedding_dump_manifest,
)
from activation_research.tokenwise_scorers import (
    ALL_SCORERS,
    PRIMARY_SCORERS,
    finalize_locked_scorer,
    prepare_validation_run,
    select_global_scorer,
)
from scripts.dispatch.build_issue_157_validation_backfill_cells import build


_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = (
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
)


def _records(split: str, n: int, offset: float = 0.0) -> list[dict]:
    records = []
    for index in range(n):
        label = index % 2
        vector = torch.tensor(
            [[
                offset + 2.0 * label + 0.01 * index,
                float(label),
                float(index % 3),
                1.0,
            ]],
            dtype=torch.float32,
        )
        records.append(
            {
                "hashkey": f"{split}-hash-{index}",
                "halu": label,
                "z_views": vector,
            }
        )
    return records


def _make_run(tmp_path: Path, dataset: str, seed: int = 0) -> Path:
    run_dir = tmp_path / dataset / f"seed_{seed}"
    emb_dir = run_dir / "embeddings"
    metas = {}
    for split, n, offset in (
        ("train", 40, 0.0),
        ("val", 20, 0.1),
        ("test", 24, 0.2),
    ):
        metas[split] = dump_embeddings_to_memmap(
            _records(split, n, offset),
            str(emb_dir),
            split,
            split_metadata={"role": split, "split_seed": seed},
        )
    write_embedding_dump_manifest(
        str(emb_dir),
        metas,
        run_metadata={
            "dataset": dataset,
            "method": "fixture_method",
            "training_recipe": "fixture_recipe",
            "training_seed": seed,
            "split_seed": seed,
            "embedding_surface": "token_zero",
        },
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "method": {
                    "name": "fixture_method",
                    "training_recipe": "fixture_recipe",
                    "training": {"select_on_val": True},
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_prepare_scores_full_matrix_without_opening_test(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, "hotpotqa_memmap")
    real_load = np.load

    def guarded_load(path, *args, **kwargs):
        if Path(path).name.startswith("test_"):
            raise AssertionError("prepare opened a test array")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    manifest_path = prepare_validation_run(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(
        (manifest_path.parent / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["test_artifacts_accessed"] is False
    assert manifest["validation_policy"]["exploratory_due_to_validation_reuse"] is True
    assert set(metrics) == set(ALL_SCORERS)
    assert {
        name for name, row in metrics.items() if row["selection_eligible"]
    } == set(PRIMARY_SCORERS)
    assert all(0.0 <= row["auroc"] <= 1.0 for row in metrics.values())
    with (manifest_path.parent / "validation_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert set(ALL_SCORERS).issubset(csv.DictReader(handle).fieldnames or [])


def test_global_lock_then_finalize_scores_only_selected_candidate(tmp_path):
    validation_manifests = []
    for dataset in _DATASETS:
        validation_manifests.append(prepare_validation_run(_make_run(tmp_path, dataset)))
    lock_path = select_global_scorer(
        validation_manifests,
        training_recipe="fixture_recipe",
        output_path=tmp_path / "selection_lock.json",
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["test_evaluated"] is False
    assert lock["selected_scorer"] in PRIMARY_SCORERS
    assert set(lock["candidates"]) == set(PRIMARY_SCORERS)
    assert lock["exploratory_due_to_checkpoint_validation_reuse"] is True

    final_path = finalize_locked_scorer(
        lock_path, output_dir=tmp_path / "final"
    )
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["test_candidates_evaluated"] == [lock["selected_scorer"]]
    assert set(final["dataset_mean_metrics"]) == set(_DATASETS)
    assert 0.0 <= final["macro_test_metrics"]["auroc"] <= 1.0
    for result in final["dataset_seed_results"]:
        score_path = (
            final_path.parent
            / "runs"
            / f"{result['dataset']}__seed_{result['training_seed']}"
            / "test_scores.csv"
        )
        with score_path.open(newline="", encoding="utf-8") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        assert lock["selected_scorer"] in fields
        assert "validation_platt_probability" in fields
        assert not any(
            scorer in fields
            for scorer in PRIMARY_SCORERS
            if scorer != lock["selected_scorer"]
        )


def test_global_selection_rejects_missing_dataset(tmp_path):
    manifests = [
        prepare_validation_run(_make_run(tmp_path, dataset))
        for dataset in _DATASETS[:-1]
    ]
    with pytest.raises(ValueError, match="requires datasets"):
        select_global_scorer(
            manifests,
            training_recipe="fixture_recipe",
            output_path=tmp_path / "lock.json",
        )


def test_validation_backfill_builder_creates_v1_and_t0_matrix(tmp_path):
    runs_root = tmp_path / "runs"
    recipes = (
        ("issue151_knnval", "tokenwise_contrastive_first_anchored"),
        ("issue155_causal", "tokenwise_causal_t0_dropout"),
    )
    slugs = ("hotpotqa", "nq", "popqa", "sciq", "searchqa")
    for experiment_prefix, method in recipes:
        for slug, dataset in zip(slugs, _DATASETS):
            for seed in range(5):
                run_dir = (
                    runs_root
                    / f"{experiment_prefix}_{slug}"
                    / dataset
                    / method
                    / f"seed_{seed}"
                )
                (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
                (run_dir / "artifacts" / "final_weights.pt").write_bytes(b"weights")
                (run_dir / "eval_metrics.json").write_text("{}", encoding="utf-8")
                (run_dir / "predictions.csv").write_text(
                    "score_halu,label_halu\n0,0\n", encoding="utf-8"
                )

    dispatch = tmp_path / "dispatch"
    assert build(dispatch, project_root=_ROOT, runs_root=runs_root) == 50
    assert build(dispatch, project_root=_ROOT, runs_root=runs_root) == 0
    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (dispatch / "pending").glob("*.json")
    ]
    assert len(cells) == 50
    assert all(cell["eval_only"] is True for cell in cells)
    assert all(cell["retraining_permitted"] is False for cell in cells)
    assert all(cell["output_check"].endswith("embeddings/manifest.json") for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)
