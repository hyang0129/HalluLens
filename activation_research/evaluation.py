import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance


def _call_model(model, x, **kwargs):
    """Call `model` using kwargs when supported.

    This keeps backward compatibility with encoders that only accept `forward(x)`.
    """
    if not kwargs:
        return model(x)
    try:
        return model(x, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg or "got an unexpected keyword" in msg:
            return model(x)
        raise


def _normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 0:
        return labels.unsqueeze(0)
    if labels.dim() > 1:
        return labels.view(-1)
    return labels


def intra_sample_cosine_mean(z_views: torch.Tensor) -> float:
    """Average cosine similarity across all within-sample view pairs."""
    z_norm = F.normalize(z_views, dim=-1)
    batch_size, num_views, _ = z_norm.shape
    if num_views < 2:
        return 0.0
    sim = torch.matmul(z_norm, z_norm.transpose(1, 2))
    tri = torch.triu_indices(num_views, num_views, offset=1, device=z_norm.device)
    vals = sim[:, tri[0], tri[1]]
    return float(vals.mean().item())


def intra_inter_margin(z_views: torch.Tensor) -> float:
    """Margin between mean intra-sample similarity and inter-sample similarity."""
    z_norm = F.normalize(z_views, dim=-1)
    batch_size, num_views, dim = z_norm.shape
    if batch_size < 2 or num_views < 2:
        return 0.0

    sim_views = torch.matmul(z_norm, z_norm.transpose(1, 2))
    tri = torch.triu_indices(num_views, num_views, offset=1, device=z_norm.device)
    intra = sim_views[:, tri[0], tri[1]].mean()

    flat = z_norm.reshape(batch_size * num_views, dim)
    sample_ids = torch.arange(batch_size, device=z_norm.device).repeat_interleave(num_views)
    sim_all = torch.matmul(flat, flat.T)
    inter_mask = sample_ids.unsqueeze(0) != sample_ids.unsqueeze(1)
    inter_vals = sim_all[inter_mask]
    if inter_vals.numel() == 0:
        return 0.0
    return float((intra - inter_vals.mean()).item())


def evaluate(
    model,
    test_dataloader,
    batch_size=32,
    loss_fn=None,
    device='cuda',
    sub_batch_size=64,
    use_labels=False,
    ignore_label=-1,
    evaluator_manager=None,
    max_batches=None,
):
    batch_size = int(batch_size)
    sub_batch_size = int(sub_batch_size)
    if batch_size <= 0 or sub_batch_size <= 0:
        raise ValueError("batch_size and sub_batch_size must be positive")
    if batch_size % sub_batch_size != 0:
        raise ValueError("batch_size must be divisible by sub_batch_size")

    model.eval()
    total_loss = 0.0
    total_intra_cos = 0.0
    total_margin = 0.0
    n_batches = 0

    with torch.no_grad():
        # Buffers to accumulate mini-batches
        buffer_views = []
        buffer_view_indices = []
        buffer_labels = [] if use_labels else None
        buffer_hashkeys = [] if evaluator_manager is not None else None
        buffered_samples = 0

        def _flush_buffer():
            nonlocal total_loss, total_intra_cos, total_margin, n_batches
            nonlocal buffer_views, buffer_view_indices, buffer_labels, buffer_hashkeys, buffered_samples

            if not buffer_views:
                return

            views_full = torch.cat(buffer_views, dim=0)
            view_indices_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
            labels_full = torch.cat(buffer_labels, dim=0) if (use_labels and buffer_labels) else None

            bsz, num_views, seq_len, hidden_dim = views_full.shape
            x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
            layer_idx_flat = None
            if view_indices_full is not None:
                layer_idx_flat = view_indices_full.reshape(bsz * num_views)

            z_flat = _call_model(model, x_flat, layer_idx=layer_idx_flat)
            z_views = z_flat.reshape(bsz, num_views, -1)

            if evaluator_manager is not None:
                evaluator_manager.accumulate_batch(z_views, buffer_hashkeys, labels_full)
                buffer_hashkeys = []

            if use_labels:
                loss = loss_fn(z_views, labels=labels_full)
            else:
                loss = loss_fn(z_views)

            intra_cos = intra_sample_cosine_mean(z_views)
            margin = intra_inter_margin(z_views)

            total_loss += loss.item()
            total_intra_cos += intra_cos
            total_margin += margin
            n_batches += 1

            buffer_views = []
            buffer_view_indices = []
            buffer_labels = [] if use_labels else None
            buffered_samples = 0

        for i, batch in enumerate(test_dataloader):
            if max_batches is not None and i >= int(max_batches):
                break
            views = batch['views_activations'].to(device, non_blocking=True)

            buffer_views.append(views)
            buffered_samples += int(views.size(0))

            if isinstance(batch, dict) and 'view_indices' in batch:
                buffer_view_indices.append(batch['view_indices'].to(device, non_blocking=True))

            if buffer_hashkeys is not None:
                hk = None
                if hasattr(batch, 'get') and 'hashkey' in batch:
                    hk = batch['hashkey']
                elif hasattr(batch, 'get') and 'hashkeys' in batch:
                    hk = batch['hashkeys']
                if hk is not None:
                    if isinstance(hk, str):
                        hk = [hk]
                    buffer_hashkeys.extend(list(hk))

            if use_labels:
                labels = _normalize_labels(batch['halu'].to(device, non_blocking=True))
                buffer_labels.append(labels)

            if buffered_samples >= batch_size:
                _flush_buffer()

        # Process any remaining partial buffer
        if buffer_views:
            _flush_buffer()

    avg_loss = total_loss / max(1, n_batches)
    avg_intra_cos = total_intra_cos / max(1, n_batches)
    avg_margin = total_margin / max(1, n_batches)
    return avg_loss, avg_intra_cos, avg_margin





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

def inference_embeddings(model, dataset, batch_size=512, sub_batch_size=64, device='cuda', num_workers=16, layers=None, persistent_workers=False):
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

    buffer_views, buffer_hash = [], []
    buffer_view_indices = []
    if layers is not None:
        buffer_layers = {layer_idx: [] for layer_idx in layers}
    results = []

    subs_in_batch = batch_size // sub_batch_size

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
            if layers is None:
                views = batch['views_activations'].to(device, non_blocking=True)
                buffer_views.append(views)

                if isinstance(batch, dict) and 'view_indices' in batch:
                    buffer_view_indices.append(batch['view_indices'].to(device, non_blocking=True))
            else:
                all_activations = batch['all_activations']
                for layer_idx, layer_acts in zip(layers, all_activations):
                    buffer_layers[layer_idx].append(layer_acts.squeeze(1).to(device, non_blocking=True))
            
            hashkeys = batch['hashkey']
            buffer_hash.extend(hashkeys)

            # Process when buffer is full or at the end of the loop
            if (layers is None and len(buffer_views) == subs_in_batch) or \
               (layers is not None and len(next(iter(buffer_layers.values()))) == subs_in_batch) or \
               i == len(dataloader) - 1:
                
                if layers is None:
                    views_full = torch.cat(buffer_views, dim=0)
                    view_indices_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
                    buffer_views = []
                    buffer_view_indices = []

                    bsz, num_views, seq_len, hidden_dim = views_full.shape
                    x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
                    layer_idx_flat = view_indices_full.reshape(bsz * num_views) if view_indices_full is not None else None

                    z_flat = _call_model(model, x_flat, layer_idx=layer_idx_flat)
                    z_views = z_flat.reshape(bsz, num_views, -1)

                    for h, z_views_i in zip(buffer_hash, z_views):
                        results.append({
                            "hashkey": h,
                            "z_views": z_views_i.cpu(),
                        })
                else:
                    layer_embeddings = {}
                    for layer_idx in layers:
                        layer_buffer = buffer_layers[layer_idx]
                        layer_full = torch.cat(layer_buffer, dim=0)
                        layer_idx_tensor = torch.full(
                            (layer_full.shape[0],),
                            int(layer_idx),
                            dtype=torch.long,
                            device=layer_full.device,
                        )
                        z = _call_model(model, layer_full, layer_idx=layer_idx_tensor)
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