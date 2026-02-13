import os
import sys

import pandas as pd
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_data():
    train_records = [
        {"hashkey": "tr_id_0", "z1": torch.tensor([0.0, 0.0]), "z2": torch.tensor([0.0, 0.0])},
        {"hashkey": "tr_id_1", "z1": torch.tensor([0.1, 0.0]), "z2": torch.tensor([0.1, 0.0])},
        {"hashkey": "tr_ood_0", "z1": torch.tensor([10.0, 10.0]), "z2": torch.tensor([10.1, 9.9])},
        {"hashkey": "tr_ood_1", "z1": torch.tensor([9.8, 10.2]), "z2": torch.tensor([9.9, 10.1])},
    ]

    test_embeddings = [
        {"hashkey": "id_0", "z1": torch.tensor([0.05, 0.05]), "z2": torch.tensor([0.06, 0.04])},
        {"hashkey": "id_1", "z1": torch.tensor([0.02, 0.04]), "z2": torch.tensor([0.03, 0.05])},
        {"hashkey": "ood_0", "z1": torch.tensor([10.0, 10.0]), "z2": torch.tensor([9.0, 11.0])},
        {"hashkey": "ood_1", "z1": torch.tensor([9.5, 10.5]), "z2": torch.tensor([8.7, 10.8])},
    ]

    activation_parser_df = pd.DataFrame(
        {
            "prompt_hash": ["tr_id_0", "tr_id_1", "tr_ood_0", "tr_ood_1", "id_0", "id_1", "ood_0", "ood_1"],
            "halu": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    return train_records, test_embeddings, activation_parser_df


def test_multi_metric_hallucination_evaluator_runs_cosine_mds_knn():
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    class DummyLoader:
        dataset = None

    train_records, test_embeddings, activation_parser_df = _make_data()

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=activation_parser_df,
        train_data_loader=DummyLoader(),
        device="cpu",
        metrics=["cosine", "mds", {"metric": "knn", "kwargs": {"k": 2}}],
    )
    evaluator._baseline_embeddings = train_records

    stats = evaluator.compute_from_embeddings(test_embeddings)

    assert "cosine_auroc" in stats
    assert "mahalanobis_auroc" in stats
    assert "knn_auroc" in stats


def test_multi_metric_hallucination_evaluator_reuses_embeddings_once():
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    class DummyLoader:
        dataset = None

    train_records, test_embeddings, activation_parser_df = _make_data()

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=activation_parser_df,
        train_data_loader=DummyLoader(),
        device="cpu",
        metrics=["cosine", "mds", {"metric": "knn", "kwargs": {"k": 2}}],
    )

    calls = {"baseline": 0, "test": 0, "label": 0}

    def fake_baseline(_model):
        calls["baseline"] += 1
        return train_records

    def fake_test(_loader, _model):
        calls["test"] += 1
        return test_embeddings

    def fake_labels(records, keep_unlabeled=False):
        calls["label"] += 1
        return [
            {**record, "halu": 0 if str(record["hashkey"]).startswith("id_") else 1}
            for record in records
        ]

    evaluator._compute_baseline_embeddings = fake_baseline
    evaluator._compute_test_embeddings = fake_test
    evaluator._assign_hallucination_labels = fake_labels

    stats = evaluator.compute(data_loader=DummyLoader(), model=object())

    assert "cosine_auroc" in stats
    assert "mahalanobis_auroc" in stats
    assert "knn_auroc" in stats
    assert calls == {"baseline": 1, "test": 1, "label": 2}


def test_multi_metric_hallucination_evaluator_uses_id_only_default_for_mds():
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    class DummyLoader:
        dataset = None

    train_records, test_embeddings, activation_parser_df = _make_data()

    def count_train_records(train_records, test_records, outlier_class=1):
        return {"n_train": len(train_records)}

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=activation_parser_df,
        train_data_loader=DummyLoader(),
        device="cpu",
        metrics=[
            {"name": "mds", "metric": count_train_records, "prefix": "mds_probe"},
            {"name": "knn", "metric": count_train_records, "prefix": "knn_probe"},
        ],
    )
    evaluator._baseline_embeddings = train_records

    stats = evaluator.compute_from_embeddings(test_embeddings)

    assert stats["mds_probe_n_train"] == 2
    assert stats["knn_probe_n_train"] == 4
