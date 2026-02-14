from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, IterableDataset, get_worker_info
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from .evaluation import evaluate, intra_inter_margin, intra_sample_cosine_mean
from loguru import logger
from sklearn.metrics import roc_auc_score
import os
import json
import math
import re


class InfiniteIndexStream(IterableDataset):
    """Wrap a map-style dataset as an infinite IterableDataset.

    This keeps DataLoader workers alive and continuously producing samples.
    Each worker receives a disjoint shard of indices.
    """

    def __init__(self, dataset, *, shuffle: bool = True, seed: int = 0):
        super().__init__()
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def __iter__(self):
        if not hasattr(self.dataset, "__len__"):
            raise TypeError("InfiniteIndexStream requires an underlying dataset with __len__")

        n = len(self.dataset)
        if n <= 0:
            return

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        g = torch.Generator()
        epoch = 0

        while True:
            g.manual_seed(self.seed + epoch)
            if self.shuffle:
                order = torch.randperm(n, generator=g).tolist()
            else:
                order = list(range(n))

            shard = order[worker_id::num_workers]
            for idx in shard:
                yield self.dataset[idx]

            epoch += 1


def _contrastive_collate_kview(batch):
    """Collate only the fields used by contrastive training.

    This avoids default-collate trying to batch large/irregular fields like
    `all_activations` (which may include None values depending on backend).
    """
    if batch is None:
        return batch

    views = torch.stack([b["views_activations"] for b in batch], dim=0)
    halu = torch.stack([b["halu"] for b in batch], dim=0)
    hashkeys = [b["hashkey"] for b in batch]

    out = {
        "views_activations": views,
        "halu": halu,
        "hashkey": hashkeys,
    }

    if "view_indices" in batch[0]:
        out["view_indices"] = torch.stack([b["view_indices"] for b in batch], dim=0).to(dtype=torch.long)
    if "input_length" in batch[0]:
        out["input_length"] = torch.tensor([b["input_length"] for b in batch], dtype=torch.long)

    return out


# Backward compatibility alias for older imports.
_contrastive_collate_min = _contrastive_collate_kview


def _atomic_torch_save(obj, path: str) -> None:
    """Atomically write a torch checkpoint to disk.

    This prevents corrupted checkpoints if a job is pre-empted mid-write.
    """
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _cleanup_legacy_checkpoints(checkpoint_dir: str, *, keep_filenames: set[str]) -> None:
    """Remove legacy per-epoch/best checkpoint files.

    This keeps only the files required for resuming training.
    """
    if not os.path.isdir(checkpoint_dir):
        return

    legacy_prefixes = (
        "contrastive_checkpoint_epoch_",
        "halu_classifier_checkpoint_epoch_",
    )
    legacy_exact = {
        "contrastive_best.pt",
        "halu_classifier_best.pt",
    }

    for name in os.listdir(checkpoint_dir):
        if name in keep_filenames:
            continue

        if name in legacy_exact:
            try:
                os.remove(os.path.join(checkpoint_dir, name))
            except OSError:
                pass
            continue

        if name.startswith(legacy_prefixes) and name.endswith(".pt"):
            try:
                os.remove(os.path.join(checkpoint_dir, name))
            except OSError:
                pass
            continue


def _save_and_prune_snapshots(
    *,
    checkpoint_dir: str,
    snapshot_prefix: str,
    epoch_one_indexed: int,
    checkpoint: dict,
    snapshot_every: int,
    snapshot_keep_last: int,
    is_last_epoch: bool,
) -> None:
    """Optionally save periodic snapshot checkpoints and prune old ones.

    Snapshot checkpoints are useful for inspecting/rolling back training.
    Retention is handled by keeping only the newest `snapshot_keep_last`
    snapshots.
    """
    if snapshot_every is None:
        return
    snapshot_every = int(snapshot_every)
    snapshot_keep_last = int(snapshot_keep_last)
    if snapshot_every <= 0 or snapshot_keep_last <= 0:
        return

    should_snapshot = (epoch_one_indexed % snapshot_every == 0) or bool(is_last_epoch)
    if should_snapshot:
        snap_name = f"{snapshot_prefix}_snapshot_epoch_{epoch_one_indexed:05d}.pt"
        snap_path = os.path.join(checkpoint_dir, snap_name)
        _atomic_torch_save(checkpoint, snap_path)

    # Prune older snapshots beyond the retention window.
    pattern = re.compile(rf"^{re.escape(snapshot_prefix)}_snapshot_epoch_(\\d+)\\.pt$")
    snapshots: list[tuple[int, str]] = []
    try:
        for name in os.listdir(checkpoint_dir):
            m = pattern.match(name)
            if not m:
                continue
            try:
                epoch_num = int(m.group(1))
            except ValueError:
                continue
            snapshots.append((epoch_num, name))
    except FileNotFoundError:
        return

    snapshots.sort(key=lambda x: x[0])
    to_delete = snapshots[:-snapshot_keep_last] if len(snapshots) > snapshot_keep_last else []
    for _, name in to_delete:
        try:
            os.remove(os.path.join(checkpoint_dir, name))
        except OSError:
            pass

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
                      checkpoint_dir='checkpoints', save_every=1, resume_from=None, persistent_workers=True,
                      cleanup_legacy_checkpoints: bool = True,
                      snapshot_every: int = 0,
                      snapshot_keep_last: int = 5,
                      use_labels=False, ignore_label=-1,
                      same_sample_weight=1.0, same_class_weight=1.0,
                      balanced_sampling=False,
                      use_infinite_index_stream: bool = False,
                      infinite_stream_shuffle: bool = True,
                      infinite_stream_seed: int = 0,
                      use_infinite_index_stream_eval: bool = False,
                      infinite_eval_shuffle: bool = False,
                      infinite_eval_seed: int = 0):

    def _call_model_with_optional_layer_idx(m, x, layer_idx=None):
        if layer_idx is None:
            return m(x)
        try:
            return m(x, layer_idx=layer_idx)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg or "got an unexpected keyword" in msg:
                return m(x)
            raise

    base_dataset_len = None
    if use_infinite_index_stream:
        if not hasattr(train_dataset, "__len__"):
            raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
        base_dataset_len = len(train_dataset)

        if sub_batch_size != batch_size:
            logger.warning(
                "use_infinite_index_stream=True forces sub_batch_size=batch_size "
                "(disabling microbatch buffering)."
            )
        sub_batch_size = batch_size

    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SupConLoss(
        temperature=temperature,
        ignore_label=ignore_label,
        same_sample_weight=same_sample_weight,
        same_class_weight=same_class_weight,
    )

    start_epoch = 0
    best_loss = float('inf')

    if resume_from is not None:
        checkpoint_path = resume_from if os.path.isabs(resume_from) else os.path.join(checkpoint_dir, resume_from)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resumed training from epoch {start_epoch}")

    is_iterable = isinstance(train_dataset, IterableDataset)
    if use_infinite_index_stream and not is_iterable:
        train_dataset = InfiniteIndexStream(
            train_dataset,
            shuffle=bool(infinite_stream_shuffle),
            seed=int(infinite_stream_seed),
        )
        is_iterable = True
    if is_iterable and balanced_sampling and use_labels:
        logger.warning("Balanced sampling is not supported for iterable datasets; disabling sampler.")
    sampler = _build_balanced_sampler(train_dataset) if balanced_sampling and use_labels and not is_iterable else None

    use_persistent_workers = bool(persistent_workers and num_workers and num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None and not is_iterable),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        collate_fn=_contrastive_collate_kview,
    )

    test_loader = None
    eval_max_batches = None
    if test_dataset is not None:
        test_is_iterable = isinstance(test_dataset, IterableDataset)
        if use_infinite_index_stream_eval:
            if not hasattr(test_dataset, "__len__"):
                raise TypeError("use_infinite_index_stream_eval=True requires test_dataset to have __len__")
            base_test_len = len(test_dataset)
            if not test_is_iterable:
                test_dataset = InfiniteIndexStream(
                    test_dataset,
                    shuffle=bool(infinite_eval_shuffle),
                    seed=int(infinite_eval_seed),
                )
                test_is_iterable = True
            eval_max_batches = int(math.ceil(base_test_len / float(batch_size)))

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            collate_fn=_contrastive_collate_kview,
        )

    steps_per_epoch = None
    train_iter = None
    if use_infinite_index_stream:
        steps_per_epoch = int(math.ceil(base_dataset_len / float(batch_size)))
        train_iter = iter(train_loader)

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0
        total_intra_cos = 0.0
        total_intra_inter_margin = 0.0
        n_batches = 0

        if use_infinite_index_stream:
            total_steps = steps_per_epoch
            loop = tqdm(range(total_steps), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        else:
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            try:
                total_steps = len(train_loader)
            except TypeError:
                total_steps = None

        buffer_views = []
        buffer_view_indices = []
        buffer_labels = [] if use_labels else None
        buffer_sample_ids = [] if use_labels else None

        i = 0
        for batch in loop:
            if use_infinite_index_stream:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

            i += 1
            logger.debug(f"Adding batch {i} to buffer. Current buffer size: {len(buffer_views)}")
            views = batch['views_activations'].to(device, non_blocking=True)

            buffer_views.append(views)
            if 'view_indices' in batch:
                buffer_view_indices.append(batch['view_indices'].to(device, non_blocking=True))

            if use_labels:
                labels = batch['halu'].to(device, non_blocking=True)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                elif labels.dim() > 1:
                    labels = labels.view(-1)
                buffer_labels.append(labels)

                hashkeys = batch['hashkey']
                if isinstance(hashkeys, str):
                    hashkeys = [hashkeys]
                sample_ids = torch.tensor([hash(hk) % 1000000 for hk in hashkeys], dtype=torch.long, device=device)
                buffer_sample_ids.append(sample_ids)

            buffer_full = len(buffer_views) * sub_batch_size == batch_size
            last_batch = total_steps is not None and i == total_steps
            if buffer_full or last_batch:
                views_full = torch.cat(buffer_views, dim=0)
                view_idx_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
                buffer_views = []
                buffer_view_indices = []

                bsz, num_views, seq_len, hidden_dim = views_full.shape
                x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
                view_idx_flat = view_idx_full.reshape(bsz * num_views) if view_idx_full is not None else None
                z_flat = _call_model_with_optional_layer_idx(model, x_flat, layer_idx=view_idx_flat)
                z_views = z_flat.reshape(bsz, num_views, -1)

                if use_labels:
                    labels_full = torch.cat(buffer_labels, dim=0)
                    sample_ids_full = torch.cat(buffer_sample_ids, dim=0)
                    buffer_labels = []
                    buffer_sample_ids = []
                    loss = loss_fn(z_views, labels=labels_full, sample_ids=sample_ids_full)
                else:
                    loss = loss_fn(z_views)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_intra_cos += intra_sample_cosine_mean(z_views)
                total_intra_inter_margin += intra_inter_margin(z_views)
                n_batches += 1

                avg_loss = total_loss / n_batches
                avg_intra_cos = total_intra_cos / n_batches
                avg_intra_inter = total_intra_inter_margin / n_batches
                loop.set_postfix(loss=avg_loss, intra_cos=avg_intra_cos, intra_inter_margin=avg_intra_inter)

        if buffer_views:
            views_full = torch.cat(buffer_views, dim=0)
            view_idx_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None

            bsz, num_views, seq_len, hidden_dim = views_full.shape
            x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
            view_idx_flat = view_idx_full.reshape(bsz * num_views) if view_idx_full is not None else None
            z_flat = _call_model_with_optional_layer_idx(model, x_flat, layer_idx=view_idx_flat)
            z_views = z_flat.reshape(bsz, num_views, -1)

            if use_labels and buffer_labels:
                labels_full = torch.cat(buffer_labels, dim=0)
                sample_ids_full = torch.cat(buffer_sample_ids, dim=0)
                loss = loss_fn(z_views, labels=labels_full, sample_ids=sample_ids_full)
            else:
                loss = loss_fn(z_views)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_intra_cos += intra_sample_cosine_mean(z_views)
            total_intra_inter_margin += intra_inter_margin(z_views)
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_intra_cos = total_intra_cos / max(1, n_batches)
        avg_intra_inter = total_intra_inter_margin / max(1, n_batches)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} "
            f"- Train Intra Cos: {avg_intra_cos:.4f} "
            f"- Train Intra/Inter Margin: {avg_intra_inter:.4f}"
        )

        test_loss = float('inf')
        test_intra_cos = 0.0
        test_intra_inter_margin = 0.0
        if test_loader is not None:
            test_loss, test_intra_cos, test_intra_inter_margin = evaluate(
                model,
                test_loader,
                batch_size=batch_size,
                loss_fn=loss_fn,
                device=device,
                sub_batch_size=sub_batch_size,
                use_labels=use_labels,
                ignore_label=ignore_label,
                max_batches=eval_max_batches,
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} "
                f"- Test Intra Cos: {test_intra_cos:.4f} "
                f"- Test Intra/Inter Margin: {test_intra_inter_margin:.4f}"
            )

        is_last_epoch = (epoch == epochs - 1)
        if (epoch + 1) % save_every == 0 or is_last_epoch:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'train_intra_cos': avg_intra_cos,
                'train_intra_inter_margin': avg_intra_inter,
                'test_loss': test_loss,
                'test_intra_cos': test_intra_cos,
                'test_intra_inter_margin': test_intra_inter_margin,
                'best_loss': min(best_loss, test_loss),
                'temperature': temperature,
                'lr': lr,
            }

            last_path = os.path.join(checkpoint_dir, 'contrastive_last.pt')
            _atomic_torch_save(checkpoint, last_path)

            _save_and_prune_snapshots(
                checkpoint_dir=checkpoint_dir,
                snapshot_prefix='contrastive',
                epoch_one_indexed=epoch + 1,
                checkpoint=checkpoint,
                snapshot_every=snapshot_every,
                snapshot_keep_last=snapshot_keep_last,
                is_last_epoch=is_last_epoch,
            )

            if cleanup_legacy_checkpoints:
                _cleanup_legacy_checkpoints(checkpoint_dir, keep_filenames={'contrastive_last.pt'})


def train_halu_classifier(model, train_dataset, test_dataset=None, epochs=10, batch_size=512, lr=1e-4, device='cuda', num_workers=4, sub_batch_size=64,
                         checkpoint_dir='checkpoints', save_every=1, resume_from=None, persistent_workers=True,
                         cleanup_legacy_checkpoints: bool = True,
                         snapshot_every: int = 0,
                         snapshot_keep_last: int = 5,
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
    
    # Resume from checkpoint if specified. Support absolute paths; raise if missing.
    if resume_from is not None:
        checkpoint_path = resume_from if os.path.isabs(resume_from) else os.path.join(checkpoint_dir, resume_from)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auroc = checkpoint.get('best_auroc', 0.0)
        logger.info(f"Resumed training from epoch {start_epoch}")

    is_iterable = isinstance(train_dataset, IterableDataset)
    if is_iterable and balanced_sampling:
        logger.warning("Balanced sampling is not supported for iterable datasets; disabling sampler.")
    sampler = _build_balanced_sampler(train_dataset) if balanced_sampling and not is_iterable else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None and not is_iterable),
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
        try:
            total_steps = len(train_loader)
        except TypeError:
            total_steps = None

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
            buffer_full = len(buffer_acts) * sub_batch_size == batch_size
            last_batch = total_steps is not None and i == total_steps
            if buffer_full or last_batch:
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

        if buffer_acts:
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

        # Save checkpoint (keep only what is needed to resume)
        is_last_epoch = (epoch == epochs - 1)
        if (epoch + 1) % save_every == 0 or is_last_epoch:
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
                'lr': lr,
            }

            last_path = os.path.join(checkpoint_dir, 'halu_classifier_last.pt')
            _atomic_torch_save(checkpoint, last_path)

            _save_and_prune_snapshots(
                checkpoint_dir=checkpoint_dir,
                snapshot_prefix='halu_classifier',
                epoch_one_indexed=epoch + 1,
                checkpoint=checkpoint,
                snapshot_every=snapshot_every,
                snapshot_keep_last=snapshot_keep_last,
                is_last_epoch=is_last_epoch,
            )

            if cleanup_legacy_checkpoints:
                _cleanup_legacy_checkpoints(checkpoint_dir, keep_filenames={'halu_classifier_last.pt'})
