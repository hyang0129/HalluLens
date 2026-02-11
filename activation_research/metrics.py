import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import torch.nn.functional as F


def mahalanobis_ood_stats(train_records, test_records, outlier_class = 1):
    # Extract z1 tensors from train set and stack
    train_z = torch.stack([r['z1'] for r in train_records])

    # Compute mean and covariance from training set
    mean = train_z.mean(dim=0)
    centered = train_z - mean
    cov = torch.matmul(centered.T, centered) / (len(train_z) - 1)

    # Add small value to diagonal for numerical stability (regularization)
    cov += torch.eye(cov.shape[0]) * 1e-5
    inv_cov = torch.linalg.inv(cov)

    def mahalanobis(x):
        delta = x - mean
        return torch.sqrt((delta @ inv_cov @ delta.T).diag())

    # Prepare test set
    test_z = torch.stack([r['z1'] for r in test_records])
    test_labels = torch.tensor([r['halu'] for r in test_records], dtype=torch.int32)
    test_labels = test_labels.squeeze()

    # Compute Mahalanobis distances for test set
    with torch.no_grad():
        dists = mahalanobis(test_z)

    # Compute stats
    id_dists = dists[test_labels == 0]
    ood_dists = dists[test_labels == 1]

    if outlier_class == 0:
        id_dists = dists[test_labels == 1]
        ood_dists = dists[test_labels == 0]

    stats = {
        'mahalanobis_mean_id': id_dists.mean().item(),
        'mahalanobis_std_id': id_dists.std().item(),
        'mahalanobis_mean_ood': ood_dists.mean().item(),
        'mahalanobis_std_ood': ood_dists.std().item(),
        'mahalanobis_auroc': roc_auc_score(test_labels, dists.numpy())
    }

    return stats


def cosine_similarity_ood_stats(train_records, test_records, outlier_class = 1):
    """
    Compute OOD statistics using cosine similarity between z1 and z2 pairs.

    More similar pairs (higher cosine similarity) indicate in-distribution samples.
    Less similar pairs (lower cosine similarity) indicate out-of-distribution samples.

    Args:
        train_records: List of training records containing z1 and z2 tensors
        test_records: List of test records containing z1, z2 tensors and halu labels
        outlier_class: Which class to treat as outlier (0 or 1), default 1

    Returns:
        dict: Contains cosine similarity statistics for ID/OOD detection
    """
    # Compute cosine similarities for training set (to establish baseline)
    train_similarities = []
    for record in train_records:
        z1 = record['z1']
        z2 = record['z2']
        # Compute cosine similarity between z1 and z2
        similarity = F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
        train_similarities.append(similarity)

    train_similarities = torch.tensor(train_similarities)

    # Compute cosine similarities for test set
    test_similarities = []
    test_labels = []
    for record in test_records:
        z1 = record['z1']
        z2 = record['z2']
        halu = record['halu']

        # Compute cosine similarity between z1 and z2
        similarity = F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
        test_similarities.append(similarity)
        test_labels.append(halu)

    test_similarities = torch.tensor(test_similarities)
    test_labels = torch.tensor(test_labels, dtype=torch.int32).squeeze()

    # Separate similarities by class
    id_similarities = test_similarities[test_labels == 0]
    ood_similarities = test_similarities[test_labels == 1]

    if outlier_class == 0:
        id_similarities = test_similarities[test_labels == 1]
        ood_similarities = test_similarities[test_labels == 0]

    # For AUROC, we need to invert similarities since lower similarity = higher OOD score
    # We use negative similarities so that lower similarity gives higher "distance" score
    ood_scores = -test_similarities.numpy()

    stats = {
        'cosine_mean_train': train_similarities.mean().item(),
        'cosine_std_train': train_similarities.std().item(),
        'cosine_mean_id': id_similarities.mean().item(),
        'cosine_std_id': id_similarities.std().item(),
        'cosine_mean_ood': ood_similarities.mean().item(),
        'cosine_std_ood': ood_similarities.std().item(),
        'cosine_auroc': roc_auc_score(test_labels, ood_scores)
    }

    return stats


def classifier_ood_stats(test_records):
    # Extract hallucination probabilities and labels from test set
    test_probs = torch.tensor([r['halu_prob'] for r in test_records])
    test_labels = torch.tensor([r['halu'] for r in test_records], dtype=torch.int32)
    test_labels = test_labels.squeeze()

    # Compute stats
    id_probs = test_probs[test_labels == 0]
    ood_probs = test_probs[test_labels == 1]

    stats = {
        'classifier_mean_id': id_probs.mean().item(),
        'classifier_std_id': id_probs.std().item(),
        'classifier_mean_ood': ood_probs.mean().item(),
        'classifier_std_ood': ood_probs.std().item(),
        'classifier_auroc': roc_auc_score(test_labels, test_probs.numpy())
    }

    return stats


def knn_ood_stats(
    train_records,
    test_records,
    outlier_class: int = 1,
    k: int = 5,
    metric: str = "euclidean",
):
    """Compute OOD statistics using k-nearest-neighbor distance in embedding space.

    This uses the distance from each test embedding (z1) to its k nearest neighbors
    among training embeddings (z1). Larger distances indicate more OOD-like samples.

    Args:
        train_records: List of training records containing 'z1' tensors.
        test_records: List of test records containing 'z1' tensors and 'halu' labels.
        outlier_class: Which class to treat as outlier for ID/OOD summary stats.
        k: Number of neighbors to use.
        metric: Distance metric for NearestNeighbors (e.g. 'euclidean', 'cosine').

    Returns:
        dict: Contains KNN distance statistics for ID/OOD detection.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Stack train/test embeddings (z1). Use CPU numpy for sklearn.
    train_z = torch.stack([r["z1"] for r in train_records]).detach().cpu().numpy()
    test_z = torch.stack([r["z1"] for r in test_records]).detach().cpu().numpy()
    test_labels = torch.tensor([r["halu"] for r in test_records], dtype=torch.int32).squeeze()

    n_train = train_z.shape[0]
    n_neighbors = min(k, n_train)
    if n_neighbors < k:
        # Keep behavior reasonable for tiny train sets.
        k = n_neighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(train_z)
    distances, _ = nn.kneighbors(test_z)

    # Use mean kNN distance as the OOD score.
    knn_scores = torch.tensor(distances.mean(axis=1), dtype=torch.float32)

    id_scores = knn_scores[test_labels == 0]
    ood_scores = knn_scores[test_labels == 1]

    if outlier_class == 0:
        id_scores = knn_scores[test_labels == 1]
        ood_scores = knn_scores[test_labels == 0]

    stats = {
        "knn_k": int(k),
        "knn_metric": str(metric),
        "knn_mean_id": id_scores.mean().item() if id_scores.numel() else float("nan"),
        "knn_std_id": id_scores.std().item() if id_scores.numel() else float("nan"),
        "knn_mean_ood": ood_scores.mean().item() if ood_scores.numel() else float("nan"),
        "knn_std_ood": ood_scores.std().item() if ood_scores.numel() else float("nan"),
        # AUROC is computed with the original labels (halu==1 as positive), consistent
        # with mahalanobis_ood_stats() in this module.
        "knn_auroc": roc_auc_score(test_labels.numpy(), knn_scores.numpy()),
    }

    return stats
