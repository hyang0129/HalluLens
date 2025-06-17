from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from .evaluation import evaluate, pairing_accuracy
from loguru import logger
from sklearn.metrics import roc_auc_score

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_contrastive(model, train_dataset, test_dataset=None,
                      epochs=10, batch_size=512, lr=1e-6,
                      temperature=0.07, device='cuda', num_workers=16, sub_batch_size=64):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SupConLoss(temperature=temperature)

    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    subsinbatch = batch_size // sub_batch_size
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        # Buffers to accumulate mini-batches
        buffer_x1, buffer_x2 = [], []

        i = 0
        for batch in loop:
            i += 1 
            x1 = batch['layer1_activations'].squeeze(1).to(device, non_blocking=True)
            x2 = batch['layer2_activations'].squeeze(1).to(device, non_blocking=True)

            buffer_x1.append(x1)
            buffer_x2.append(x2)

            # Process when buffer is full or at the end of the loop
            if len(buffer_x1) * sub_batch_size == batch_size or i == len(loop):
                x1_full = torch.cat(buffer_x1, dim=0)
                x2_full = torch.cat(buffer_x2, dim=0)
                buffer_x1 = [] 
                buffer_x2 = [] 

                z1 = model(x1_full)
                z2 = model(x2_full)

                z_stacked = torch.stack([z1, z2], dim=1)
                loss = loss_fn(z_stacked)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = pairing_accuracy(z1, z2)
                total_loss += loss.item()
                total_acc += acc
                
                avg_loss = total_loss / (i / subsinbatch)
                avg_acc = total_acc / ( i / subsinbatch)

                loop.set_postfix(loss=avg_loss, pairing_acc=avg_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Train Pairing Acc: {avg_acc:.4f}")

        if test_dataset is not None:
            test_loss, test_acc = evaluate(model, test_dataset, batch_size=batch_size, loss_fn=loss_fn, device=device, sub_batch_size= sub_batch_size)
            print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} - Test Pairing Acc: {test_acc:.4f}")


def inference_embeddings(model, dataset, batch_size=512, sub_batch_size=64, device='cuda', num_workers=16, layers=None):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    buffer_x1, buffer_x2, buffer_hash = [], [], []
    if layers is not None:
        buffer_layers = [[] for _ in layers]
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
                all_activations = batch['all_layer_activations']
                for layer_idx, layer_acts in zip(layers, all_activations):
                    buffer_layers[layer_idx].append(layer_acts.squeeze(1).to(device, non_blocking=True))
            
            hashkeys = batch['hashkey']
            buffer_hash.extend(hashkeys)

            # Process when buffer is full or at the end of the loop
            if (layers is None and len(buffer_x1) == subs_in_batch) or \
               (layers is not None and len(buffer_layers[0]) == subs_in_batch) or \
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
                    for layer_idx, layer_buffer in enumerate(buffer_layers):
                        layer_full = torch.cat(layer_buffer, dim=0)
                        z = model(layer_full)
                        layer_embeddings[f"layer_{layers[layer_idx]}"] = z.cpu()
                        buffer_layers[layer_idx] = []

                    for h in buffer_hash:
                        results.append({
                            "hashkey": h,
                            "layer_embeddings": {k: v[i] for k, v in layer_embeddings.items()}
                        })

                buffer_hash = []

    return results

def train_halu_classifier(model, train_dataset, test_dataset=None, epochs=10, batch_size=512, lr=1e-4, device='cuda', num_workers=4, sub_batch_size=64):
    """
    Train a hallucination classifier using last layer activations.
    Args:
        model: LastLayerHaluClassifier instance
        train_dataset: ActivationDataset
        test_dataset: ActivationDataset or None
        epochs: int
        batch_size: int
        lr: float
        device: str
        num_workers: int
        sub_batch_size: int, size of sub-batches to process at once
    """
    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=sub_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    subsinbatch = batch_size // sub_batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")

        # Buffers to accumulate mini-batches
        buffer_acts = []
        buffer_labels = []
        i = 0

        for batch in loop:
            i += 1
            last_layer = batch['all_activations'][-1].to(device, non_blocking=True)  # (B, L, D)
            last_layer = last_layer.squeeze()  # Remove any extra size-1 dimensions
            logger.debug(f"Last layer shape before model: {last_layer.shape}")
            labels = batch['halu'].to(device, non_blocking=True).float().view(-1, 1)  # (B, 1)

            buffer_acts.append(last_layer)
            buffer_labels.append(labels)

            # Process when buffer is full or at the end of the loop
            if len(buffer_acts) * sub_batch_size == batch_size or i == len(loop):
                acts_full = torch.cat(buffer_acts, dim=0)
                labels_full = torch.cat(buffer_labels, dim=0)
                buffer_acts = []
                buffer_labels = []

                preds = model(acts_full)
                loss = loss_fn(preds, labels_full)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels_full.size(0)
                preds_binary = (preds > 0.5).float()
                total_correct += (preds_binary == labels_full).sum().item()
                total_samples += labels_full.size(0)

                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                loop.set_postfix(loss=avg_loss, acc=avg_acc)

        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Train Acc: {avg_acc:.4f}")

        if test_dataset is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            val_preds = []
            val_labels = []
            buffer_acts = []
            buffer_labels = []
            i = 0

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Validation"):
                    i += 1
                    last_layer = batch['all_activations'][-1].to(device, non_blocking=True)
                    last_layer = last_layer.squeeze()  # Remove any extra size-1 dimensions
                    labels = batch['halu'].to(device, non_blocking=True).float().view(-1, 1)

                    buffer_acts.append(last_layer)
                    buffer_labels.append(labels)

                    if len(buffer_acts) * sub_batch_size == batch_size or i == len(test_loader):
                        acts_full = torch.cat(buffer_acts, dim=0)
                        labels_full = torch.cat(buffer_labels, dim=0)
                        buffer_acts = []
                        buffer_labels = []

                        preds = model(acts_full)
                        loss = loss_fn(preds, labels_full)
                        val_loss += loss.item() * labels_full.size(0)
                        preds_binary = (preds > 0.5).float()
                        val_correct += (preds_binary == labels_full).sum().item()
                        val_samples += labels_full.size(0)
                        
                        # Store predictions and labels for AUROC
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels_full.cpu().numpy())

            val_loss /= val_samples
            val_acc = val_correct / val_samples
            val_auroc = roc_auc_score(val_labels, val_preds)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUROC: {val_auroc:.4f}")
