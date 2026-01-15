from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from .evaluation import evaluate, pairing_accuracy, average_cosine_similarity
from loguru import logger
from sklearn.metrics import roc_auc_score
import os
import json

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, ignore_label=-1,
                 same_sample_weight=1.0, same_class_weight=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.ignore_label = ignore_label
        self.same_sample_weight = same_sample_weight
        self.same_class_weight = same_class_weight

    def forward(self, features, labels=None, mask=None, sample_ids=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz]. Labels with value equal to
                self.ignore_label are treated as belonging to no class - their
                different views will still be pulled together (positive pairs),
                but they will repel from all other samples including other
                ignore_label samples.
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            sample_ids: sample identifiers of shape [bsz]. Used to distinguish
                between same sample pairs (different views) and same class pairs
                (different samples). If provided, same sample pairs get
                same_sample_weight, same class pairs get same_class_weight.
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

            # Create base mask where samples with same labels are positive pairs
            mask = torch.eq(labels, labels.T).float().to(device)

            # For samples with ignore_label: they should not be positive with other ignore_label samples
            # but should still be positive with themselves (different views of same sample)
            ignore_cross_mask = (labels == self.ignore_label) & (labels.T == self.ignore_label)
            # Create identity mask to preserve self-positive relationships
            identity_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            # Remove cross ignore_label relationships but keep self relationships
            ignore_cross_mask = ignore_cross_mask & (~identity_mask)
            mask = mask * (~ignore_cross_mask).float()

            # Apply differential weighting if sample_ids are provided
            if sample_ids is not None:
                sample_ids = sample_ids.contiguous().view(-1, 1)
                if sample_ids.shape[0] != batch_size:
                    raise ValueError('Num of sample_ids does not match num of features')

                # Create mask for same sample pairs (different views of same sample)
                same_sample_mask = torch.eq(sample_ids, sample_ids.T).float().to(device)

                # Create mask for same class but different sample pairs
                same_class_diff_sample_mask = mask * (1 - same_sample_mask)

                # Apply weights: same sample pairs get full weight, same class pairs get reduced weight
                mask = same_sample_mask * self.same_sample_weight + same_class_diff_sample_mask * self.same_class_weight
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


def _build_balanced_sampler(dataset):
    """
    Build a WeightedRandomSampler for balanced class sampling.

    Expects dataset.df to contain a 'halu' column.
    Returns None if not applicable or if only one class is present.
    """
    if not hasattr(dataset, "df"):
        return None
    if "halu" not in dataset.df.columns:
        return None

    labels = torch.tensor(dataset.df["halu"].astype(int).values)
    if labels.numel() == 0:
        return None

    class_counts = torch.bincount(labels)
    if (class_counts > 0).sum().item() < 2:
        return None

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_contrastive(model, train_dataset, test_dataset=None,
                      epochs=10, batch_size=512, lr=1e-6,
                      temperature=0.07, device='cuda', num_workers=16, sub_batch_size=64,
                      checkpoint_dir='checkpoints', save_every=5, resume_from=None, persistent_workers=True,
                      use_labels=False, ignore_label=-1,
                      same_sample_weight=1.0, same_class_weight=1.0,
                      balanced_sampling=False):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SupConLoss(temperature=temperature, ignore_label=ignore_label,
                         same_sample_weight=same_sample_weight, same_class_weight=same_class_weight)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        checkpoint_path = os.path.join(checkpoint_dir, resume_from)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resumed training from epoch {start_epoch}")

    sampler = _build_balanced_sampler(train_dataset) if balanced_sampling and use_labels else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_cosine_sim = 0.0
        n_batches = 0  # Track number of full batches processed
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        # Buffers to accumulate mini-batches
        buffer_x1, buffer_x2 = [], []
        buffer_labels = [] if use_labels else None
        buffer_sample_ids = [] if use_labels else None

        i = 0
        for batch in loop:
            i += 1
            logger.debug(f"Adding batch {i} to buffer. Current buffer size: {len(buffer_x1)}")
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

                # Extract sample IDs (use hashkey as unique identifier)
                # Convert hashkey to numeric ID for tensor operations
                hashkeys = batch['hashkey']
                if isinstance(hashkeys, str):
                    hashkeys = [hashkeys]

                # Create numeric sample IDs from hashkeys
                sample_ids = torch.tensor([hash(hk) % 1000000 for hk in hashkeys],
                                        dtype=torch.long, device=device)
                buffer_sample_ids.append(sample_ids)

            # Process when buffer is full or at the end of the loop
            if len(buffer_x1) * sub_batch_size == batch_size or i == len(loop):
                logger.debug(f"Processing buffer at batch {i}. Buffer size: {len(buffer_x1)}")
                x1_full = torch.cat(buffer_x1, dim=0)
                x2_full = torch.cat(buffer_x2, dim=0)
                buffer_x1 = []
                buffer_x2 = []

                z1 = model(x1_full)
                z2 = model(x2_full)

                z_stacked = torch.stack([z1, z2], dim=1)

                if use_labels:
                    labels_full = torch.cat(buffer_labels, dim=0)
                    sample_ids_full = torch.cat(buffer_sample_ids, dim=0)
                    buffer_labels = []
                    buffer_sample_ids = []
                    loss = loss_fn(z_stacked, labels=labels_full, sample_ids=sample_ids_full)
                else:
                    loss = loss_fn(z_stacked)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = pairing_accuracy(z1, z2)
                cosine_sim = average_cosine_similarity(z1, z2)
                total_loss += loss.item()
                total_acc += acc
                total_cosine_sim += cosine_sim
                n_batches += 1  # Increment for each full batch processed

                avg_loss = total_loss / n_batches
                avg_acc = total_acc / n_batches
                avg_cosine_sim = total_cosine_sim / n_batches

                loop.set_postfix(loss=avg_loss, pairing_acc=avg_acc, cosine_sim=avg_cosine_sim)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Train Pairing Acc: {avg_acc:.4f} - Train Cosine Sim: {avg_cosine_sim:.4f}")

        test_loss = float('inf')
        test_cosine_sim = 0.0
        if test_dataset is not None:
            test_loss, test_acc, test_cosine_sim = evaluate(model, test_dataset, batch_size=batch_size, loss_fn=loss_fn, device=device, sub_batch_size=sub_batch_size, use_labels=use_labels, ignore_label=ignore_label)
            print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} - Test Pairing Acc: {test_acc:.4f} - Test Cosine Sim: {test_cosine_sim:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'train_acc': avg_acc,
                'train_cosine_sim': avg_cosine_sim,
                'test_loss': test_loss,
                'test_cosine_sim': test_cosine_sim,
                'best_loss': min(best_loss, test_loss),
                'temperature': temperature,
                'lr': lr
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'contrastive_checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint if test loss improved
            if test_loss < best_loss:
                best_loss = test_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, 'contrastive_best.pt')
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"New best model saved with test loss: {test_loss:.4f}")


def train_halu_classifier(model, train_dataset, test_dataset=None, epochs=10, batch_size=512, lr=1e-4, device='cuda', num_workers=4, sub_batch_size=64,
                         checkpoint_dir='checkpoints', save_every=5, resume_from=None, persistent_workers=True,
                         balanced_sampling=True):
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
        checkpoint_dir: str, directory to save checkpoints
        save_every: int, save checkpoint every N epochs
        resume_from: str, checkpoint file to resume from
        persistent_workers: bool, whether to use persistent DataLoader workers
        balanced_sampling: bool, whether to balance classes via weighted sampling
    """
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm

    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    
    start_epoch = 0
    best_auroc = 0.0
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        checkpoint_path = os.path.join(checkpoint_dir, resume_from)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_auroc = checkpoint.get('best_auroc', 0.0)
            logger.info(f"Resumed training from epoch {start_epoch}")

    sampler = _build_balanced_sampler(train_dataset) if balanced_sampling else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=sub_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )

    for epoch in range(start_epoch, epochs):
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

        val_loss = float('inf')
        val_acc = 0.0
        val_auroc = 0.0
        
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

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'train_acc': avg_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auroc': val_auroc,
                'best_auroc': max(best_auroc, val_auroc),
                'lr': lr
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'halu_classifier_checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint if AUROC improved
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_checkpoint_path = os.path.join(checkpoint_dir, 'halu_classifier_best.pt')
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"New best model saved with AUROC: {val_auroc:.4f}")
