import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_knn_metric_downsamples_large_train_set():
    from activation_research.metrics import knn_ood_stats

    train_records = []
    for i in range(1200):
        label = 0 if i < 600 else 1
        center = torch.tensor([0.0, 0.0]) if label == 0 else torch.tensor([8.0, 8.0])
        jitter = torch.tensor([float(i % 5) * 0.01, float(i % 7) * 0.01])
        train_records.append({"z1": center + jitter, "halu": label})

    test_records = [
        {"z1": torch.tensor([0.1, 0.1]), "halu": 0},
        {"z1": torch.tensor([0.2, 0.0]), "halu": 0},
        {"z1": torch.tensor([8.1, 8.2]), "halu": 1},
        {"z1": torch.tensor([7.9, 8.0]), "halu": 1},
    ]

    stats = knn_ood_stats(
        train_records=train_records,
        test_records=test_records,
        outlier_class=1,
        calibrate_k=True,
        max_train_size=100,
        sample_seed=123,
    )

    assert stats["knn_train_sampled"] is True
    assert stats["knn_train_size_used"] <= 100
    assert stats["knn_k"] in {50, 100, 200, 500, 1000} or stats["knn_k"] <= stats["knn_train_size_used"]
    assert "knn_auroc" in stats
