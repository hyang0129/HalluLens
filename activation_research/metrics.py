import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance


def mahalanobis_ood_stats(train_records, test_records):
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

    stats = {
        'mahalanobis_mean_id': id_dists.mean().item(),
        'mahalanobis_std_id': id_dists.std().item(),
        'mahalanobis_mean_ood': ood_dists.mean().item(),
        'mahalanobis_std_ood': ood_dists.std().item(),
        'mahalanobis_auroc': roc_auc_score(test_labels, dists.numpy())
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
