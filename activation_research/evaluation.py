import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance


def evaluate(model, test_dataloader, batch_size=32, loss_fn=None, device='cuda', sub_batch_size=64, use_labels=False, ignore_label=-1):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        # Buffers to accumulate mini-batches
        buffer_x1, buffer_x2 = [], []
        buffer_labels = [] if use_labels else None
        subsinbatch = batch_size // sub_batch_size

        for i, batch in enumerate(test_dataloader):
            x1 = batch['layer1_activations'].squeeze(1).to(device, non_blocking=True)
            x2 = batch['layer2_activations'].squeeze(1).to(device, non_blocking=True)

            buffer_x1.append(x1)
            buffer_x2.append(x2)

            if use_labels:
                labels = batch['halu'].to(device, non_blocking=True)
                # Ensure labels have at least one dimension for concatenation
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                elif labels.dim() == 1 and labels.size(0) == 1:
                    # Already has batch dimension
                    pass
                else:
                    # Handle batch of labels
                    labels = labels.view(-1)
                buffer_labels.append(labels)

            # Process when buffer is full or at the end of the loop
            if len(buffer_x1) * sub_batch_size == batch_size or i == len(test_dataloader) - 1:
                x1_full = torch.cat(buffer_x1, dim=0)
                x2_full = torch.cat(buffer_x2, dim=0)
                buffer_x1 = []
                buffer_x2 = []

                z1 = model(x1_full)
                z2 = model(x2_full)

                z_stacked = torch.stack([z1, z2], dim=1)

                if use_labels:
                    labels_full = torch.cat(buffer_labels, dim=0)
                    buffer_labels = []
                    loss = loss_fn(z_stacked, labels=labels_full)
                else:
                    loss = loss_fn(z_stacked)

                acc = pairing_accuracy(z1, z2)

                total_loss += loss.item()
                total_acc += acc
                n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    return avg_loss, avg_acc

def pairing_accuracy(z1, z2):
    """
    z1, z2: (B, D) normalized embeddings
    Returns percentage of pairs correctly matched by highest cosine similarity.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute cosine similarity matrix (B x B)
    sim_matrix = torch.matmul(z1, z2.T)  # (B, B)

    # For each in z1, find index of max similarity in z2
    preds_z1 = sim_matrix.argmax(dim=1)
    # For each in z2, find index of max similarity in z1
    preds_z2 = sim_matrix.argmax(dim=0)

    batch_size = z1.size(0)
    # Correct matches: indices where preds match i
    correct_z1 = (preds_z1 == torch.arange(batch_size, device=z1.device)).sum().item()
    correct_z2 = (preds_z2 == torch.arange(batch_size, device=z2.device)).sum().item()

    # Average accuracy
    accuracy = (correct_z1 + correct_z2) / (2 * batch_size)
    return accuracy 



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

def mahalanobis_ood_stats_multilayer(train_records, test_records, layers):
    """
    Calculate Mahalanobis OOD statistics for multiple layers.
    
    Args:
        train_records: List of training records containing layer embeddings
        test_records: List of test records containing layer embeddings
        layers: List of layer indices to analyze
        
    Returns:
        dict: Contains per-layer stats and aggregated stats
    """
    # Initialize storage for per-layer stats and distances
    layer_stats = {}
    layer_dists = {}
    
    # Process each layer
    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}"
        
        # Extract embeddings for this layer from train set
        train_z = torch.stack([r['layer_embeddings'][layer_key] for r in train_records])
        
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

        # Prepare test set for this layer
        test_z = torch.stack([r['layer_embeddings'][layer_key] for r in test_records])
        test_labels = torch.tensor([r['halu'] for r in test_records], dtype=torch.int32)
        test_labels = test_labels.squeeze()

        # Compute Mahalanobis distances for test set
        with torch.no_grad():
            dists = mahalanobis(test_z)
            
        # Store distances for this layer
        layer_dists[layer_key] = dists

        # Compute per-layer stats
        id_dists = dists[test_labels == 0]
        ood_dists = dists[test_labels == 1]

        layer_stats[layer_key] = {
            'mahalanobis_mean_id': id_dists.mean().item(),
            'mahalanobis_std_id': id_dists.std().item(),
            'mahalanobis_mean_ood': ood_dists.mean().item(),
            'mahalanobis_std_ood': ood_dists.std().item(),
            'mahalanobis_auroc': roc_auc_score(test_labels, dists.numpy())
        }
    
    # Compute aggregated stats using average distance across layers
    avg_dists = torch.stack(list(layer_dists.values())).mean(dim=0)
    test_labels = torch.tensor([r['halu'] for r in test_records], dtype=torch.int32).squeeze()
    
    id_dists_avg = avg_dists[test_labels == 0]
    ood_dists_avg = avg_dists[test_labels == 1]
    
    aggregated_stats = {
        'mahalanobis_mean_id': id_dists_avg.mean().item(),
        'mahalanobis_std_id': id_dists_avg.std().item(),
        'mahalanobis_mean_ood': ood_dists_avg.mean().item(),
        'mahalanobis_std_ood': ood_dists_avg.std().item(),
        'mahalanobis_auroc': roc_auc_score(test_labels, avg_dists.numpy())
    }
    
    return {
        'per_layer_stats': layer_stats,
        'per_layer_distances': layer_dists,
        'aggregated_stats': aggregated_stats
    }

def inference_embeddings(model, dataset, batch_size=512, sub_batch_size=64, device='cuda', num_workers=16, layers=None, persistent_workers=True):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    buffer_x1, buffer_x2, buffer_hash = [], [], []
    if layers is not None:
        buffer_layers = {layer_idx: [] for layer_idx in layers}
    results = []

    subs_in_batch = batch_size // sub_batch_size

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
            if layers is None:
                x1 = batch['layer1_activations'].squeeze(1).to(device, non_blocking=True)
                x2 = batch['layer2_activations'].squeeze(1).to(device, non_blocking=True)
                buffer_x1.append(x1)
                buffer_x2.append(x2)
            else:
                all_activations = batch['all_activations']
                for layer_idx, layer_acts in zip(layers, all_activations):
                    buffer_layers[layer_idx].append(layer_acts.squeeze(1).to(device, non_blocking=True))
            
            hashkeys = batch['hashkey']
            buffer_hash.extend(hashkeys)

            # Process when buffer is full or at the end of the loop
            if (layers is None and len(buffer_x1) == subs_in_batch) or \
               (layers is not None and len(next(iter(buffer_layers.values()))) == subs_in_batch) or \
               i == len(dataloader) - 1:
                
                if layers is None:
                    x1_full = torch.cat(buffer_x1, dim=0)
                    x2_full = torch.cat(buffer_x2, dim=0)
                    buffer_x1, buffer_x2 = [], []

                    z1 = model(x1_full)
                    z2 = model(x2_full)

                    for h, z1_i, z2_i in zip(buffer_hash, z1, z2):
                        results.append({
                            "hashkey": h,
                            "z1": z1_i.cpu(),
                            "z2": z2_i.cpu()
                        })
                else:
                    layer_embeddings = {}
                    for layer_idx in layers:
                        layer_buffer = buffer_layers[layer_idx]
                        layer_full = torch.cat(layer_buffer, dim=0)
                        z = model(layer_full)
                        layer_embeddings[f"layer_{layer_idx}"] = z.cpu()
                        buffer_layers[layer_idx] = []

                    # Consistency check
                    n_hash = len(buffer_hash)
                    n_emb = next(iter(layer_embeddings.values())).shape[0]
                    assert n_hash == n_emb, f"buffer_hash length {n_hash} does not match embeddings batch size {n_emb}"
                    for idx, h in enumerate(buffer_hash):
                        results.append({
                            "hashkey": h,
                            "layer_embeddings": {k: v[idx] for k, v in layer_embeddings.items()}
                        })

                buffer_hash = []

    return results

def assign_hallucination_labels(embeddings, activation_parser_df):
    """
    Assign hallucination labels to embeddings records using the activation parser's dataframe.
    
    Args:
        embeddings: List of dicts from inference_embeddings containing 'hashkey' and 'layer_embeddings'
        activation_parser_df: DataFrame from ActivationParser containing 'prompt_hash' and 'halu' columns
        
    Returns:
        List of dicts with 'halu' key added to each record
    """
    for i, record in enumerate(embeddings):
        ishalu = activation_parser_df[activation_parser_df['prompt_hash'] == record['hashkey']]['halu']
        assert len(ishalu) == 1, f"Expected exactly 1 match for hashkey {record['hashkey']}, found {len(ishalu)}"
        embeddings[i]['halu'] = ishalu.values[0]
    
    return embeddings