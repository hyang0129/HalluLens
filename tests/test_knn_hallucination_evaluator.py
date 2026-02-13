import os
import sys

import pandas as pd
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_knn_hallucination_evaluator_from_embeddings():
    from activation_research.metric_evaluator import KNNHallucinationEvaluator

    class DummyLoader:
        dataset = None

    train_records = [
        {"z1": torch.tensor([0.0, 0.0])},
        {"z1": torch.tensor([0.1, 0.0])},
        {"z1": torch.tensor([0.0, 0.1])},
        {"z1": torch.tensor([0.1, 0.1])},
    ]

    test_embeddings = [
        {"hashkey": "id_0", "z1": torch.tensor([0.05, 0.05])},
        {"hashkey": "id_1", "z1": torch.tensor([0.02, 0.04])},
        {"hashkey": "ood_0", "z1": torch.tensor([10.0, 10.0])},
        {"hashkey": "ood_1", "z1": torch.tensor([9.5, 10.5])},
    ]

    activation_parser_df = pd.DataFrame(
        {
            "prompt_hash": ["id_0", "id_1", "ood_0", "ood_1"],
            "halu": [0, 0, 1, 1],
        }
    )

    evaluator = KNNHallucinationEvaluator(
        activation_parser_df=activation_parser_df,
        train_data_loader=DummyLoader(),
        device="cpu",
        k=2,
    )
    evaluator._baseline_embeddings = train_records

    stats = evaluator.compute_from_embeddings(test_embeddings)

    assert "knn_auroc" in stats
    assert stats["knn_k"] == 2
    assert stats["knn_auroc"] >= 0.9
