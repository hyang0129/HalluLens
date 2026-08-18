"""Regression tests for scoring-ready token-wise embedding artifacts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from activation_research.evaluation import (
    dump_embeddings_to_memmap,
    write_embedding_dump_manifest,
)
from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator


def _records(split: str, labels: list[int]) -> list[dict]:
    return [
        {
            "hashkey": f"{split}-hash-{index // 2}",
            "halu": label,
            "z_views": torch.full((1, 4), float(index + 1)),
        }
        for index, label in enumerate(labels)
    ]


def test_embedding_dump_keeps_train_val_test_separate_with_stable_ids(tmp_path):
    labels = {
        "train": [0, 1, 0],
        "val": [1, 0],
        "test": [0, 1, 1, 0],
    }
    metas = {}
    all_stable_ids = []
    for split_name, split_labels in labels.items():
        metas[split_name] = dump_embeddings_to_memmap(
            _records(split_name, split_labels),
            str(tmp_path),
            split_name,
            split_metadata={
                "role": {
                    "train": "scorer_reference_bank",
                    "val": "scorer_selection",
                    "test": "final_evaluation",
                }[split_name],
                "split_seed": 42,
            },
        )
    manifest = write_embedding_dump_manifest(
        str(tmp_path),
        metas,
        run_metadata={"dataset": "fixture", "training_seed": 0},
    )

    for split_name, split_labels in labels.items():
        np.testing.assert_array_equal(
            np.load(tmp_path / f"{split_name}_labels.npy"), split_labels
        )
        hashes = json.loads(
            (tmp_path / f"{split_name}_hashkeys.json").read_text()
        )
        stable_ids = json.loads(
            (tmp_path / f"{split_name}_stable_ids.json").read_text()
        )
        assert all(hashkey.startswith(f"{split_name}-") for hashkey in hashes)
        assert len(stable_ids) == len(set(stable_ids)) == len(split_labels)
        assert all(stable_id.startswith(f"{split_name}::") for stable_id in stable_ids)
        all_stable_ids.extend(stable_ids)
        assert metas[split_name]["split_name"] == split_name
        assert metas[split_name]["schema_version"] == 2
        assert metas[split_name]["files"]["stable_ids"] == (
            f"{split_name}_stable_ids.json"
        )

    assert len(all_stable_ids) == len(set(all_stable_ids))
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert manifest["label_isolation"]["val"] == (
        "training_capture_validation_rows"
    )
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_validation_embedding_labels_never_fall_back_to_test_dataframe(monkeypatch):
    class DummyLoader:
        dataset = object()

    test_df = pd.DataFrame(
        {"prompt_hash": ["shared"], "halu": [1], "split": ["test"]}
    )
    val_df = pd.DataFrame(
        {"prompt_hash": ["shared"], "halu": [0], "split": ["val"]}
    )
    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=test_df,
        train_activation_parser_df=pd.DataFrame(
            {"prompt_hash": ["train"], "halu": [1], "split": ["train"]}
        ),
        train_data_loader=DummyLoader(),
        metrics=["knn"],
        device="cpu",
        num_workers=0,
    )
    monkeypatch.setattr(
        "activation_research.metric_evaluator.inference_embeddings",
        lambda *_args, **_kwargs: [
            {"hashkey": "shared", "z_views": torch.zeros(1, 2)}
        ],
    )

    records = evaluator.compute_labeled_split_embeddings(
        DummyLoader(),
        object(),
        split_name="val",
        lookup_df=val_df,
    )
    assert [record["halu"] for record in records] == [0]

    mixed_lookup = pd.concat([val_df, test_df], ignore_index=True)
    with pytest.raises(ValueError, match="contains split values"):
        evaluator.compute_labeled_split_embeddings(
            DummyLoader(),
            object(),
            split_name="val",
            lookup_df=mixed_lookup,
        )
