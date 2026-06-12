import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, IterableDataset, get_worker_info
from utils.progress import tqdm
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

        # compute logits (clamp to prevent overflow in exp)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits = torch.clamp(logits, min=-50.0, max=50.0)

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

        # compute log_prob (add eps to prevent log(0))
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

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

        # loss — zero out any residual NaN from anchors with no positives
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
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
                      infinite_eval_seed: int = 0,
):

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
        inferred_steps = int(math.ceil(base_dataset_len / float(batch_size)))
        steps_per_epoch = inferred_steps
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


def _contrastive_collate_with_logprob(batch):
    """Collate for logprob-reconstruction contrastive training.

    Extends ``_contrastive_collate_kview`` by also collecting the per-token
    logprob sequence.  Recognises two key names in order of preference:

    * ``"logprob"`` – explicit logprob tensor (any shape, will be flattened).
    * ``"response_token_logprobs"`` – key produced by ``ActivationDataset``
      when ``include_response_logprobs=True``; NaN-padded ``(pad_length,)``
      tensor.

    If neither key is present in all items the ``"logprob"`` output key is
    omitted so the training function can fall back to SupCon-only.
    """
    out = _contrastive_collate_kview(batch)

    logprob_src = None
    for key in ("logprob", "response_token_logprobs"):
        if all(key in b for b in batch):
            logprob_src = key
            break

    if logprob_src is not None:
        raw = [b[logprob_src] for b in batch]
        max_len = max(lp.shape[-1] if isinstance(lp, torch.Tensor) else len(lp) for lp in raw)
        padded = []
        for lp in raw:
            if not isinstance(lp, torch.Tensor):
                lp = torch.tensor(lp, dtype=torch.float32)
            lp = lp.float().reshape(-1)
            if lp.shape[0] < max_len:
                lp = torch.nn.functional.pad(lp, (0, max_len - lp.shape[0]), value=float("nan"))
            padded.append(lp)
        out["logprob"] = torch.stack(padded, dim=0)  # (B, max_len)

    return out


def train_contrastive_logprob_recon(
    model,
    train_dataset,
    test_dataset=None,
    epochs=10,
    batch_size=512,
    lr=1e-6,
    temperature=0.07,
    device="cuda",
    num_workers=16,
    sub_batch_size=64,
    checkpoint_dir="checkpoints",
    save_every=1,
    resume_from=None,
    persistent_workers=True,
    cleanup_legacy_checkpoints: bool = True,
    snapshot_every: int = 0,
    snapshot_keep_last: int = 5,
    use_labels=False,
    ignore_label=-1,
    same_sample_weight=1.0,
    same_class_weight=1.0,
    balanced_sampling=False,
    recon_lambda: float = None,
    use_infinite_index_stream: bool = False,
    infinite_stream_shuffle: bool = True,
    infinite_stream_seed: int = 0,
    steps_per_epoch_override: int = None,
    grad_clip_norm: float = None,
    augment_fn=None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    lr_schedule: str = None,
    base_temperature: float = 0.07,
    select_on_val: bool = False,
):
    """Train a ``LogprobReconProgressiveCompressor`` with auxiliary logprob reconstruction.

    Loss per step:

        L = L_SupCon(z) + λ · L_recon(g(z), ℓ)

    ``λ`` defaults to ``model.recon_lambda`` but can be overridden via
    ``recon_lambda``.  When the batch contains no ``"logprob"`` field (e.g.
    the dataset does not expose it) the reconstruction term is silently
    omitted and training degrades to standard contrastive.

    Parameters
    ----------
    model : LogprobReconProgressiveCompressor
    train_dataset, test_dataset : dataset
    epochs, batch_size, lr, temperature, device, num_workers, sub_batch_size,
    checkpoint_dir, save_every, resume_from, persistent_workers,
    cleanup_legacy_checkpoints, snapshot_every, snapshot_keep_last,
    use_labels, ignore_label, same_sample_weight, same_class_weight,
    balanced_sampling, use_infinite_index_stream, infinite_stream_shuffle,
    infinite_stream_seed :
        Same semantics as ``train_contrastive``.
    recon_lambda : float or None
        Override ``model.recon_lambda``.  Pass ``None`` to use the model default.
    steps_per_epoch_override : int or None
        When set, use this fixed step count per epoch instead of
        ``ceil(dataset_len / batch_size)``.  Requires
        ``use_infinite_index_stream=True``.
    """
    _lambda = float(recon_lambda) if recon_lambda is not None else model.recon_lambda

    if steps_per_epoch_override is not None and not use_infinite_index_stream:
        raise ValueError("steps_per_epoch_override requires use_infinite_index_stream=True")

    base_dataset_len = None
    if use_infinite_index_stream:
        if not hasattr(train_dataset, "__len__"):
            raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
        base_dataset_len = len(train_dataset)
        sub_batch_size = batch_size

    def _call_model(m, x, layer_idx=None):
        if layer_idx is None:
            return m.forward_with_recon(x)
        try:
            # LayerAware wrapper: inject layer_idx into encoder, then call decoder
            z = m.encoder(x, layer_idx=layer_idx) if hasattr(m.encoder, "forward") else m.encoder(x)
            logprob_pred = m.decoder(z)
            return z, logprob_pred
        except TypeError:
            return m.forward_with_recon(x)

    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    if str(optimizer_name).lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if str(lr_schedule).lower() == "cosine"
        else None
    )
    loss_fn = SupConLoss(
        temperature=temperature,
        base_temperature=base_temperature,
        ignore_label=ignore_label,
        same_sample_weight=same_sample_weight,
        same_class_weight=same_class_weight,
    )

    start_epoch = 0
    best_loss = float("inf")
    # Best-val checkpoint selection (opt-in): track lowest validation loss and
    # restore those weights at the end, so eval uses the best epoch instead of
    # whatever epoch the loop ended on. Mirrors run_act_vit's best-val-AUROC
    # selection (the analog here is best validation loss).
    best_val_loss = float("inf")
    best_state = None

    if resume_from is not None:
        checkpoint_path = (
            resume_from if os.path.isabs(resume_from) else os.path.join(checkpoint_dir, resume_from)
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
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
    sampler = (
        _build_balanced_sampler(train_dataset)
        if balanced_sampling and use_labels and not is_iterable
        else None
    )
    use_persistent_workers = bool(persistent_workers and num_workers and num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None and not is_iterable),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        collate_fn=_contrastive_collate_with_logprob,
    )

    steps_per_epoch = None
    train_iter = None
    if use_infinite_index_stream:
        inferred = int(math.ceil(base_dataset_len / float(batch_size)))
        if steps_per_epoch_override is not None:
            steps_per_epoch = int(steps_per_epoch_override)
        else:
            steps_per_epoch = inferred
        train_iter = iter(train_loader)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            collate_fn=_contrastive_collate_with_logprob,
        )

    from utils.progress import tqdm as _tqdm

    for epoch in _tqdm(range(start_epoch, epochs), desc="Epochs"):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()

        total_loss = total_supcon = total_recon = 0.0
        total_intra_cos = total_intra_inter = 0.0
        n_batches = 0

        if use_infinite_index_stream:
            total_steps = steps_per_epoch
            loop = _tqdm(range(total_steps), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        else:
            try:
                total_steps = len(train_loader)
            except TypeError:
                total_steps = None
            loop = _tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        buffer_views = []
        buffer_view_indices = []
        buffer_logprobs = []
        buffer_labels = [] if use_labels else None
        buffer_sample_ids = [] if use_labels else None

        i = 0
        for _loop_item in loop:
            i += 1
            batch = next(train_iter) if use_infinite_index_stream else _loop_item

            views = batch["views_activations"].to(device, non_blocking=True)
            buffer_views.append(views)

            if "view_indices" in batch:
                buffer_view_indices.append(batch["view_indices"].to(device, non_blocking=True))

            if "logprob" in batch:
                buffer_logprobs.append(batch["logprob"].to(device, non_blocking=True))

            if use_labels:
                labels = batch["halu"].to(device, non_blocking=True)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                elif labels.dim() > 1:
                    labels = labels.view(-1)
                buffer_labels.append(labels)

                hashkeys = batch["hashkey"]
                if isinstance(hashkeys, str):
                    hashkeys = [hashkeys]
                sample_ids = torch.tensor(
                    [hash(hk) % 1_000_000 for hk in hashkeys], dtype=torch.long, device=device
                )
                buffer_sample_ids.append(sample_ids)

            buffer_full = len(buffer_views) * sub_batch_size == batch_size
            last_batch = total_steps is not None and i == total_steps

            if buffer_full or last_batch:
                views_full = torch.cat(buffer_views, dim=0)
                view_idx_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
                logprob_full = torch.cat(buffer_logprobs, dim=0) if buffer_logprobs else None
                buffer_views = []
                buffer_view_indices = []
                buffer_logprobs = []

                # Assemble labels before augmentation (mixup needs them)
                labels_full = None
                sample_ids_full = None
                if use_labels:
                    labels_full = torch.cat(buffer_labels, dim=0)
                    sample_ids_full = torch.cat(buffer_sample_ids, dim=0)
                    buffer_labels = []
                    buffer_sample_ids = []

                if augment_fn is not None:
                    _aug_labels = labels_full if labels_full is not None else torch.zeros(
                        views_full.shape[0], device=device, dtype=torch.long
                    )
                    if n_batches == 0:
                        _pre = torch.nn.functional.cosine_similarity(
                            views_full[:, 0].reshape(views_full.shape[0], -1),
                            views_full[:, 1].reshape(views_full.shape[0], -1),
                            dim=1,
                        ).mean()
                    views_full = augment_fn(views_full, _aug_labels)
                    if n_batches == 0:
                        _post = torch.nn.functional.cosine_similarity(
                            views_full[:, 0].reshape(views_full.shape[0], -1),
                            views_full[:, 1].reshape(views_full.shape[0], -1),
                            dim=1,
                        ).mean()
                        logger.info(f"aug_view_cosine: pre={_pre:.4f} post={_post:.4f} delta={_post - _pre:.4f}")

                bsz, num_views, seq_len, hidden_dim = views_full.shape
                x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
                view_idx_flat = view_idx_full.reshape(bsz * num_views) if view_idx_full is not None else None

                z_flat, logprob_pred_flat = _call_model(model, x_flat, layer_idx=view_idx_flat)
                z_views = z_flat.reshape(bsz, num_views, -1)

                # SupCon loss
                if use_labels:
                    supcon = loss_fn(z_views, labels=labels_full, sample_ids=sample_ids_full)
                else:
                    supcon = loss_fn(z_views)

                # Auxiliary reconstruction loss
                recon = torch.zeros(1, device=device).squeeze()
                recon_diag = {}
                if logprob_full is not None and _lambda > 0.0:
                    # Expand logprob from (B, L) to (B*num_views, L) to match z_flat
                    logprob_expanded = logprob_full.unsqueeze(1).expand(-1, num_views, -1)
                    logprob_expanded = logprob_expanded.reshape(bsz * num_views, -1)
                    # Mask out padding NaNs with the sequence mean
                    nan_mask = logprob_expanded.isnan()
                    if nan_mask.any():
                        row_means = logprob_expanded.nanmean(dim=-1, keepdim=True)
                        logprob_expanded = logprob_expanded.masked_fill(nan_mask, 0.0)
                        logprob_expanded = logprob_expanded + nan_mask.float() * row_means
                    recon, recon_diag = model.recon_loss(logprob_pred_flat, logprob_expanded)

                loss = supcon + _lambda * recon

                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(grad_clip_norm)
                    )
                optimizer.step()

                total_loss += loss.item()
                total_supcon += supcon.item()
                total_recon += float(recon.detach())
                total_intra_cos += intra_sample_cosine_mean(z_views)
                total_intra_inter += intra_inter_margin(z_views)
                n_batches += 1

                avg_loss = total_loss / n_batches
                loop.set_postfix(
                    loss=avg_loss,
                    supcon=total_supcon / n_batches,
                    recon=total_recon / n_batches,
                    suppressed=recon_diag.get("suppressed", "N/A"),
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_supcon = total_supcon / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_intra_cos = total_intra_cos / max(1, n_batches)
        avg_intra_inter = total_intra_inter / max(1, n_batches)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
            f"(SupCon={avg_supcon:.4f}, Recon={avg_recon:.4f}) "
            f"- IntraCos: {avg_intra_cos:.4f} - IntraInterMargin: {avg_intra_inter:.4f}"
        )

        test_loss = float("inf")
        test_intra_cos = test_intra_inter = 0.0
        if test_loader is not None:
            test_loss, test_intra_cos, test_intra_inter = evaluate(
                model,
                test_loader,
                batch_size=batch_size,
                loss_fn=loss_fn,
                device=device,
                sub_batch_size=sub_batch_size,
                use_labels=use_labels,
                ignore_label=ignore_label,
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} "
                f"- Test IntraCos: {test_intra_cos:.4f} - Test IntraInterMargin: {test_intra_inter:.4f}"
            )

        is_last_epoch = epoch == epochs - 1
        if (epoch + 1) % save_every == 0 or is_last_epoch:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "train_supcon": avg_supcon,
                "train_recon": avg_recon,
                "train_intra_cos": avg_intra_cos,
                "train_intra_inter_margin": avg_intra_inter,
                "test_loss": test_loss,
                "test_intra_cos": test_intra_cos,
                "test_intra_inter_margin": test_intra_inter,
                "best_loss": min(best_loss, test_loss),
                "temperature": temperature,
                "lr": lr,
                "recon_lambda": _lambda,
            }

            last_path = os.path.join(checkpoint_dir, "contrastive_last.pt")
            _atomic_torch_save(checkpoint, last_path)

            _save_and_prune_snapshots(
                checkpoint_dir=checkpoint_dir,
                snapshot_prefix="contrastive",
                epoch_one_indexed=epoch + 1,
                checkpoint=checkpoint,
                snapshot_every=snapshot_every,
                snapshot_keep_last=snapshot_keep_last,
                is_last_epoch=is_last_epoch,
            )

            if cleanup_legacy_checkpoints:
                _cleanup_legacy_checkpoints(checkpoint_dir, keep_filenames={"contrastive_last.pt"})

        if select_on_val and test_loader is not None and test_loss < best_val_loss:
            best_val_loss = test_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()

    if select_on_val and best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best-val checkpoint (val_loss=%.4f)" % best_val_loss)


def _contrastive_collate_with_logprob_attn(batch):
    """Collate that pads/stacks logprob + attention summary fields alongside
    the standard contrastive fields.

    Same logprob behaviour as :func:`_contrastive_collate_with_logprob`.
    Adds optional ``attention_forward`` / ``attention_backward`` tensors
    (each ``(B, K, num_stat_features)``) when present in every item. If a
    direction's tensor is missing from any item it is omitted from the
    output — the trainer treats this as "no K recon for that direction
    this batch" and skips the corresponding loss term.

    Each direction is treated independently. NaN entries inside the
    attention tensors are preserved verbatim and handled by the model's
    ``recon_loss_attn`` (NaN-masked + variance-suppressed).
    """
    out = _contrastive_collate_with_logprob(batch)

    for direction_key in ("attention_forward", "attention_backward"):
        if not all(direction_key in b for b in batch):
            continue
        stacked = torch.stack([b[direction_key].float() for b in batch], dim=0)
        out[direction_key] = stacked  # (B, K, num_stat_features)

    return out


def train_contrastive_logprob_recon_dualloss(
    model,
    train_dataset,
    test_dataset=None,
    epochs=10,
    batch_size=512,
    lr=1e-6,
    temperature=0.07,
    device="cuda",
    num_workers=16,
    sub_batch_size=64,
    checkpoint_dir="checkpoints",
    save_every=1,
    resume_from=None,
    persistent_workers=True,
    cleanup_legacy_checkpoints: bool = True,
    snapshot_every: int = 0,
    snapshot_keep_last: int = 5,
    same_sample_weight=1.0,
    same_class_weight=1.0,
    balanced_sampling=False,
    recon_lambda: float = None,
    use_infinite_index_stream: bool = False,
    infinite_stream_shuffle: bool = True,
    infinite_stream_seed: int = 0,
    steps_per_epoch_override: int = None,
    grad_clip_norm: float = None,
    augment_fn=None,
    ignore_labels: tuple = (1, 0),
):
    """Train a ``SharedTrunkSplitOutputCompressor`` or ``SharedTrunkProjectionHeadCompressor``
    with a dual SupCon loss over two output heads plus auxiliary logprob reconstruction.

    Loss per step:

        L_A = SupCon(z_A, labels, ignore_label=ignore_labels[0])
        L_B = SupCon(z_B, labels, ignore_label=ignore_labels[1])
        L   = L_A + L_B + λ · L_recon(decoder(z), ℓ)

    For ``SharedTrunkSplitOutputCompressor`` (D1), ``z`` is the full 2D output,
    ``z_A`` and ``z_B`` are sliced halves.  For
    ``SharedTrunkProjectionHeadCompressor`` (D2), ``z`` is the trunk, ``z_A``
    and ``z_B`` are projection head outputs.

    All other training mechanics (checkpointing, AMP, grad clip, infinite stream,
    balanced sampler, augmentations) mirror ``train_contrastive_logprob_recon``.

    Parameters
    ----------
    model : SharedTrunkSplitOutputCompressor or SharedTrunkProjectionHeadCompressor
    train_dataset, test_dataset : dataset
    epochs, batch_size, lr, temperature, device, num_workers, sub_batch_size,
    checkpoint_dir, save_every, resume_from, persistent_workers,
    cleanup_legacy_checkpoints, snapshot_every, snapshot_keep_last,
    same_sample_weight, same_class_weight, balanced_sampling,
    use_infinite_index_stream, infinite_stream_shuffle,
    infinite_stream_seed, steps_per_epoch_override, grad_clip_norm, augment_fn :
        Same semantics as ``train_contrastive_logprob_recon``.
    recon_lambda : float or None
        Override ``model.recon_lambda``.  Pass ``None`` to use the model default.
    ignore_labels : tuple of (int, int)
        ``(ignore_label_A, ignore_label_B)`` — head A and head B respectively.
        Default ``(1, 0)``.
    """
    from activation_research.model import (
        SharedTrunkSplitOutputCompressor,
        SharedTrunkProjectionHeadCompressor,
    )

    _lambda = float(recon_lambda) if recon_lambda is not None else model.recon_lambda

    if steps_per_epoch_override is not None and not use_infinite_index_stream:
        raise ValueError("steps_per_epoch_override requires use_infinite_index_stream=True")

    base_dataset_len = None
    if use_infinite_index_stream:
        if not hasattr(train_dataset, "__len__"):
            raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
        base_dataset_len = len(train_dataset)
        sub_batch_size = batch_size

    # Choose how to call the model depending on its variant.
    _is_d1 = isinstance(model, SharedTrunkSplitOutputCompressor)

    def _call_model_dual(m, x):
        """Return (z, z_A, z_B, logprob_pred) from the appropriate helper."""
        if _is_d1:
            return m.forward_slices(x)  # (z_full, z_A, z_B, logprob_pred)
        else:
            return m.forward_with_heads(x)  # (z_trunk, z_A, z_B, logprob_pred)

    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn_A = SupConLoss(
        temperature=temperature,
        ignore_label=int(ignore_labels[0]),
        same_sample_weight=same_sample_weight,
        same_class_weight=same_class_weight,
    )
    loss_fn_B = SupConLoss(
        temperature=temperature,
        ignore_label=int(ignore_labels[1]),
        same_sample_weight=same_sample_weight,
        same_class_weight=same_class_weight,
    )

    start_epoch = 0
    best_loss = float("inf")

    if resume_from is not None:
        checkpoint_path = (
            resume_from if os.path.isabs(resume_from) else os.path.join(checkpoint_dir, resume_from)
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        logger.info(f"Resumed training from epoch {start_epoch}")

    is_iterable = isinstance(train_dataset, IterableDataset)
    if use_infinite_index_stream and not is_iterable:
        train_dataset = InfiniteIndexStream(
            train_dataset,
            shuffle=bool(infinite_stream_shuffle),
            seed=int(infinite_stream_seed),
        )
        is_iterable = True
    if is_iterable and balanced_sampling:
        logger.warning("Balanced sampling is not supported for iterable datasets; disabling sampler.")
    sampler = (
        _build_balanced_sampler(train_dataset)
        if balanced_sampling and not is_iterable
        else None
    )
    use_persistent_workers = bool(persistent_workers and num_workers and num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None and not is_iterable),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        collate_fn=_contrastive_collate_with_logprob,
    )

    steps_per_epoch = None
    train_iter = None
    if use_infinite_index_stream:
        inferred = int(math.ceil(base_dataset_len / float(batch_size)))
        if steps_per_epoch_override is not None:
            steps_per_epoch = int(steps_per_epoch_override)
        else:
            steps_per_epoch = inferred
        train_iter = iter(train_loader)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            collate_fn=_contrastive_collate_with_logprob,
        )

    from utils.progress import tqdm as _tqdm

    for epoch in _tqdm(range(start_epoch, epochs), desc="Epochs"):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()

        total_loss = total_supcon_a = total_supcon_b = total_recon = 0.0
        total_intra_cos = total_intra_inter = 0.0
        n_batches = 0
        _diag_steps = 0  # count of steps where per-head cosine diagnostics are logged

        if use_infinite_index_stream:
            total_steps = steps_per_epoch
            loop = _tqdm(range(total_steps), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        else:
            try:
                total_steps = len(train_loader)
            except TypeError:
                total_steps = None
            loop = _tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        buffer_views = []
        buffer_view_indices = []
        buffer_logprobs = []
        buffer_labels = []
        buffer_sample_ids = []

        i = 0
        for _loop_item in loop:
            i += 1
            batch = next(train_iter) if use_infinite_index_stream else _loop_item

            views = batch["views_activations"].to(device, non_blocking=True)
            buffer_views.append(views)

            if "view_indices" in batch:
                buffer_view_indices.append(batch["view_indices"].to(device, non_blocking=True))

            if "logprob" in batch:
                buffer_logprobs.append(batch["logprob"].to(device, non_blocking=True))

            labels = batch["halu"].to(device, non_blocking=True)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 1:
                labels = labels.view(-1)
            buffer_labels.append(labels)

            hashkeys = batch["hashkey"]
            if isinstance(hashkeys, str):
                hashkeys = [hashkeys]
            sample_ids = torch.tensor(
                [hash(hk) % 1_000_000 for hk in hashkeys], dtype=torch.long, device=device
            )
            buffer_sample_ids.append(sample_ids)

            buffer_full = len(buffer_views) * sub_batch_size == batch_size
            last_batch = total_steps is not None and i == total_steps

            if buffer_full or last_batch:
                views_full = torch.cat(buffer_views, dim=0)
                view_idx_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
                logprob_full = torch.cat(buffer_logprobs, dim=0) if buffer_logprobs else None
                labels_full = torch.cat(buffer_labels, dim=0)
                sample_ids_full = torch.cat(buffer_sample_ids, dim=0)

                buffer_views = []
                buffer_view_indices = []
                buffer_logprobs = []
                buffer_labels = []
                buffer_sample_ids = []

                if augment_fn is not None:
                    if n_batches == 0:
                        _pre = torch.nn.functional.cosine_similarity(
                            views_full[:, 0].reshape(views_full.shape[0], -1),
                            views_full[:, 1].reshape(views_full.shape[0], -1),
                            dim=1,
                        ).mean()
                    views_full = augment_fn(views_full, labels_full)
                    if n_batches == 0:
                        _post = torch.nn.functional.cosine_similarity(
                            views_full[:, 0].reshape(views_full.shape[0], -1),
                            views_full[:, 1].reshape(views_full.shape[0], -1),
                            dim=1,
                        ).mean()
                        logger.info(f"aug_view_cosine: pre={_pre:.4f} post={_post:.4f} delta={_post - _pre:.4f}")

                bsz, num_views, seq_len, hidden_dim = views_full.shape
                x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)

                z_flat, zA_flat, zB_flat, logprob_pred_flat = _call_model_dual(model, x_flat)

                # Reshape for SupCon: (B, num_views, dim)
                z_views = z_flat.reshape(bsz, num_views, -1)
                zA_views = zA_flat.reshape(bsz, num_views, -1)
                zB_views = zB_flat.reshape(bsz, num_views, -1)

                # Per-head view cosine diagnostic (first 100 steps)
                if _diag_steps < 100:
                    with torch.no_grad():
                        _cos_A = torch.nn.functional.cosine_similarity(
                            zA_views[:, 0], zA_views[:, 1], dim=-1
                        ).mean().item()
                        _cos_B = torch.nn.functional.cosine_similarity(
                            zB_views[:, 0], zB_views[:, 1], dim=-1
                        ).mean().item()
                    logger.debug(f"step={i} view_cos_A={_cos_A:.4f} view_cos_B={_cos_B:.4f}")
                    _diag_steps += 1

                supcon_A = loss_fn_A(zA_views, labels=labels_full, sample_ids=sample_ids_full)
                supcon_B = loss_fn_B(zB_views, labels=labels_full, sample_ids=sample_ids_full)

                # Auxiliary reconstruction loss
                recon = torch.zeros(1, device=device).squeeze()
                recon_diag = {}
                if logprob_full is not None and _lambda > 0.0:
                    logprob_expanded = logprob_full.unsqueeze(1).expand(-1, num_views, -1)
                    logprob_expanded = logprob_expanded.reshape(bsz * num_views, -1)
                    nan_mask = logprob_expanded.isnan()
                    if nan_mask.any():
                        row_means = logprob_expanded.nanmean(dim=-1, keepdim=True)
                        logprob_expanded = logprob_expanded.masked_fill(nan_mask, 0.0)
                        logprob_expanded = logprob_expanded + nan_mask.float() * row_means
                    recon, recon_diag = model.recon_loss(logprob_pred_flat, logprob_expanded)

                loss = supcon_A + supcon_B + _lambda * recon

                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(grad_clip_norm)
                    )
                optimizer.step()

                total_loss += loss.item()
                total_supcon_a += supcon_A.item()
                total_supcon_b += supcon_B.item()
                total_recon += float(recon.detach())
                total_intra_cos += intra_sample_cosine_mean(z_views)
                total_intra_inter += intra_inter_margin(z_views)
                n_batches += 1

                avg_loss = total_loss / n_batches
                loop.set_postfix(
                    loss=avg_loss,
                    supcon_a=total_supcon_a / n_batches,
                    supcon_b=total_supcon_b / n_batches,
                    recon=total_recon / n_batches,
                    suppressed=recon_diag.get("suppressed", "N/A"),
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_supcon_a = total_supcon_a / max(1, n_batches)
        avg_supcon_b = total_supcon_b / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_intra_cos = total_intra_cos / max(1, n_batches)
        avg_intra_inter = total_intra_inter / max(1, n_batches)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
            f"(SupConA={avg_supcon_a:.4f}, SupConB={avg_supcon_b:.4f}, Recon={avg_recon:.4f}) "
            f"- IntraCos: {avg_intra_cos:.4f} - IntraInterMargin: {avg_intra_inter:.4f}"
        )

        # Test evaluation uses head A's loss_fn by convention (no eval for dual loss)
        test_loss = float("inf")
        test_intra_cos = test_intra_inter = 0.0
        if test_loader is not None:
            test_loss, test_intra_cos, test_intra_inter = evaluate(
                model,
                test_loader,
                batch_size=batch_size,
                loss_fn=loss_fn_A,
                device=device,
                sub_batch_size=sub_batch_size,
                use_labels=True,
                ignore_label=int(ignore_labels[0]),
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} "
                f"- Test IntraCos: {test_intra_cos:.4f} - Test IntraInterMargin: {test_intra_inter:.4f}"
            )

        is_last_epoch = epoch == epochs - 1
        if (epoch + 1) % save_every == 0 or is_last_epoch:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "train_supcon_a": avg_supcon_a,
                "train_supcon_b": avg_supcon_b,
                "train_recon": avg_recon,
                "train_intra_cos": avg_intra_cos,
                "train_intra_inter_margin": avg_intra_inter,
                "test_loss": test_loss,
                "test_intra_cos": test_intra_cos,
                "test_intra_inter_margin": test_intra_inter,
                "best_loss": min(best_loss, test_loss),
                "temperature": temperature,
                "lr": lr,
                "recon_lambda": _lambda,
            }

            last_path = os.path.join(checkpoint_dir, "contrastive_last.pt")
            _atomic_torch_save(checkpoint, last_path)

            _save_and_prune_snapshots(
                checkpoint_dir=checkpoint_dir,
                snapshot_prefix="contrastive",
                epoch_one_indexed=epoch + 1,
                checkpoint=checkpoint,
                snapshot_every=snapshot_every,
                snapshot_keep_last=snapshot_keep_last,
                is_last_epoch=is_last_epoch,
            )

            if cleanup_legacy_checkpoints:
                _cleanup_legacy_checkpoints(checkpoint_dir, keep_filenames={"contrastive_last.pt"})


def train_contrastive_logprob_attn_recon(
    model,
    train_dataset,
    test_dataset=None,
    epochs=10,
    batch_size=512,
    lr=1e-6,
    temperature=0.07,
    device="cuda",
    num_workers=16,
    sub_batch_size=64,
    checkpoint_dir="checkpoints",
    save_every=1,
    resume_from=None,
    persistent_workers=True,
    cleanup_legacy_checkpoints: bool = True,
    snapshot_every: int = 0,
    snapshot_keep_last: int = 5,
    use_labels=False,
    ignore_label=-1,
    same_sample_weight=1.0,
    same_class_weight=1.0,
    balanced_sampling=False,
    recon_lambda: float = None,
    attn_recon_lambda: float = None,
    use_infinite_index_stream: bool = False,
    infinite_stream_shuffle: bool = True,
    infinite_stream_seed: int = 0,
    steps_per_epoch_override: int = None,
    grad_clip_norm: float = None,
):
    """Train ``LogprobAttnReconProgressiveCompressor`` with both auxes.

    Loss per step::

        L = L_SupCon(z) + λ_lp · L_lp(g_lp(z), ℓ) + Σ_d λ_attn · L_attn(g_d(z), A_d)

    Mirrors :func:`train_contrastive_logprob_recon` — microbatch buffering,
    atomic checkpoint save, snapshot pruning, optional infinite index
    stream. Extends only the per-batch loss-assembly block by adding one
    MSE term per active attention direction.

    Parameters
    ----------
    recon_lambda, attn_recon_lambda : float or None
        Override the corresponding lambdas on ``model``. Pass ``None`` to
        use the model defaults.
    """
    _lambda_lp = float(recon_lambda) if recon_lambda is not None else model.recon_lambda
    _lambda_attn = (
        float(attn_recon_lambda)
        if attn_recon_lambda is not None
        else model.attn_recon_lambda
    )

    if steps_per_epoch_override is not None and not use_infinite_index_stream:
        raise ValueError("steps_per_epoch_override requires use_infinite_index_stream=True")

    base_dataset_len = None
    if use_infinite_index_stream:
        if not hasattr(train_dataset, "__len__"):
            raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
        base_dataset_len = len(train_dataset)
        sub_batch_size = batch_size

    def _call_model(m, x, layer_idx=None):
        if layer_idx is None:
            return m.forward_with_recon(x)
        # LayerAware wrapper compatibility: inject layer_idx into encoder
        # and then re-route through the aux decoders. Currently the F+K
        # model does not wrap a layer-aware encoder, so this falls back.
        try:
            z = (
                m.encoder(x, layer_idx=layer_idx)
                if hasattr(m.encoder, "forward")
                else m.encoder(x)
            )
            lp_pred = m.lp_decoder(z)
            attn_pred = {
                d: m.attn_decoders[m._DIRECTION_TO_KEY[d]](z)
                for d in m._active_attn_dirs
            }
            return z, lp_pred, attn_pred
        except TypeError:
            return m.forward_with_recon(x)

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
    best_loss = float("inf")

    if resume_from is not None:
        checkpoint_path = (
            resume_from if os.path.isabs(resume_from) else os.path.join(checkpoint_dir, resume_from)
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
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
    sampler = (
        _build_balanced_sampler(train_dataset)
        if balanced_sampling and use_labels and not is_iterable
        else None
    )
    use_persistent_workers = bool(persistent_workers and num_workers and num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=(sampler is None and not is_iterable),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        collate_fn=_contrastive_collate_with_logprob_attn,
    )

    steps_per_epoch = None
    train_iter = None
    if use_infinite_index_stream:
        inferred = int(math.ceil(base_dataset_len / float(batch_size)))
        steps_per_epoch = int(steps_per_epoch_override) if steps_per_epoch_override is not None else inferred
        train_iter = iter(train_loader)

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            collate_fn=_contrastive_collate_with_logprob_attn,
        )

    from utils.progress import tqdm as _tqdm

    for epoch in _tqdm(range(start_epoch, epochs), desc="Epochs"):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()

        total_loss = total_supcon = total_recon_lp = 0.0
        total_recon_attn = {"forward": 0.0, "backward": 0.0}
        total_intra_cos = total_intra_inter = 0.0
        n_batches = 0

        if use_infinite_index_stream:
            total_steps = steps_per_epoch
            loop = _tqdm(range(total_steps), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        else:
            try:
                total_steps = len(train_loader)
            except TypeError:
                total_steps = None
            loop = _tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        buffer_views = []
        buffer_view_indices = []
        buffer_logprobs = []
        buffer_attn = {"forward": [], "backward": []}
        buffer_labels = [] if use_labels else None
        buffer_sample_ids = [] if use_labels else None

        i = 0
        for _loop_item in loop:
            i += 1
            batch = next(train_iter) if use_infinite_index_stream else _loop_item

            views = batch["views_activations"].to(device, non_blocking=True)
            buffer_views.append(views)

            if "view_indices" in batch:
                buffer_view_indices.append(batch["view_indices"].to(device, non_blocking=True))

            if "logprob" in batch:
                buffer_logprobs.append(batch["logprob"].to(device, non_blocking=True))

            for direction_key in ("forward", "backward"):
                field = f"attention_{direction_key}"
                if field in batch:
                    buffer_attn[direction_key].append(
                        batch[field].to(device, non_blocking=True)
                    )

            if use_labels:
                labels = batch["halu"].to(device, non_blocking=True)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                elif labels.dim() > 1:
                    labels = labels.view(-1)
                buffer_labels.append(labels)

                hashkeys = batch["hashkey"]
                if isinstance(hashkeys, str):
                    hashkeys = [hashkeys]
                sample_ids = torch.tensor(
                    [hash(hk) % 1_000_000 for hk in hashkeys], dtype=torch.long, device=device
                )
                buffer_sample_ids.append(sample_ids)

            buffer_full = len(buffer_views) * sub_batch_size == batch_size
            last_batch = total_steps is not None and i == total_steps

            if buffer_full or last_batch:
                views_full = torch.cat(buffer_views, dim=0)
                view_idx_full = torch.cat(buffer_view_indices, dim=0) if buffer_view_indices else None
                logprob_full = torch.cat(buffer_logprobs, dim=0) if buffer_logprobs else None
                attn_full = {
                    d: (torch.cat(buffer_attn[d], dim=0) if buffer_attn[d] else None)
                    for d in ("forward", "backward")
                }
                buffer_views = []
                buffer_view_indices = []
                buffer_logprobs = []
                buffer_attn = {"forward": [], "backward": []}

                bsz, num_views, seq_len, hidden_dim = views_full.shape
                x_flat = views_full.reshape(bsz * num_views, seq_len, hidden_dim)
                view_idx_flat = (
                    view_idx_full.reshape(bsz * num_views) if view_idx_full is not None else None
                )

                z_flat, lp_pred_flat, attn_pred_flat = _call_model(
                    model, x_flat, layer_idx=view_idx_flat
                )
                z_views = z_flat.reshape(bsz, num_views, -1)

                # --- SupCon ---
                if use_labels:
                    labels_full = torch.cat(buffer_labels, dim=0)
                    sample_ids_full = torch.cat(buffer_sample_ids, dim=0)
                    buffer_labels = []
                    buffer_sample_ids = []
                    supcon = loss_fn(z_views, labels=labels_full, sample_ids=sample_ids_full)
                else:
                    supcon = loss_fn(z_views)

                # --- Logprob (F) recon ---
                recon_lp = torch.zeros(1, device=device).squeeze()
                lp_diag = {}
                if logprob_full is not None and _lambda_lp > 0.0 and lp_pred_flat is not None:
                    logprob_expanded = logprob_full.unsqueeze(1).expand(-1, num_views, -1)
                    logprob_expanded = logprob_expanded.reshape(bsz * num_views, -1)
                    nan_mask = logprob_expanded.isnan()
                    if nan_mask.any():
                        row_means = logprob_expanded.nanmean(dim=-1, keepdim=True)
                        logprob_expanded = logprob_expanded.masked_fill(nan_mask, 0.0)
                        logprob_expanded = logprob_expanded + nan_mask.float() * row_means
                    recon_lp, lp_diag = model.recon_loss_lp(lp_pred_flat, logprob_expanded)

                # --- Attention (K) recon per direction ---
                recon_attn_terms = {}
                attn_diag = {}
                for direction in ("forward", "backward"):
                    target = attn_full.get(direction)
                    pred = attn_pred_flat.get(direction) if attn_pred_flat else None
                    if (
                        target is not None
                        and pred is not None
                        and _lambda_attn > 0.0
                    ):
                        # target: (B, K, F) — flatten views to (B*K, F)
                        target_flat = target.reshape(bsz * num_views, -1)
                        loss_d, diag_d = model.recon_loss_attn(pred, target_flat)
                        recon_attn_terms[direction] = loss_d
                        attn_diag[direction] = diag_d

                recon_attn_sum = (
                    sum(recon_attn_terms.values())
                    if recon_attn_terms
                    else torch.zeros(1, device=device).squeeze()
                )

                loss = supcon + _lambda_lp * recon_lp + _lambda_attn * recon_attn_sum

                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(grad_clip_norm)
                    )
                optimizer.step()

                total_loss += loss.item()
                total_supcon += supcon.item()
                total_recon_lp += float(recon_lp.detach())
                for direction, l in recon_attn_terms.items():
                    total_recon_attn[direction] += float(l.detach())
                total_intra_cos += intra_sample_cosine_mean(z_views)
                total_intra_inter += intra_inter_margin(z_views)
                n_batches += 1

                avg_loss = total_loss / n_batches
                loop.set_postfix(
                    loss=avg_loss,
                    supcon=total_supcon / n_batches,
                    recon_lp=total_recon_lp / n_batches,
                    recon_attn_fwd=total_recon_attn["forward"] / n_batches,
                    recon_attn_bwd=total_recon_attn["backward"] / n_batches,
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_supcon = total_supcon / max(1, n_batches)
        avg_recon_lp = total_recon_lp / max(1, n_batches)
        avg_recon_attn = {d: total_recon_attn[d] / max(1, n_batches) for d in total_recon_attn}
        avg_intra_cos = total_intra_cos / max(1, n_batches)
        avg_intra_inter = total_intra_inter / max(1, n_batches)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
            f"(SupCon={avg_supcon:.4f}, ReconLP={avg_recon_lp:.4f}, "
            f"ReconAttn(fwd={avg_recon_attn['forward']:.4f}, "
            f"bwd={avg_recon_attn['backward']:.4f})) "
            f"- IntraCos: {avg_intra_cos:.4f} - IntraInterMargin: {avg_intra_inter:.4f}"
        )

        test_loss = float("inf")
        test_intra_cos = test_intra_inter = 0.0
        if test_loader is not None:
            test_loss, test_intra_cos, test_intra_inter = evaluate(
                model,
                test_loader,
                batch_size=batch_size,
                loss_fn=loss_fn,
                device=device,
                sub_batch_size=sub_batch_size,
                use_labels=use_labels,
                ignore_label=ignore_label,
            )
            print(
                f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} "
                f"- Test IntraCos: {test_intra_cos:.4f} - Test IntraInterMargin: {test_intra_inter:.4f}"
            )

        is_last_epoch = epoch == epochs - 1
        if (epoch + 1) % save_every == 0 or is_last_epoch:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "train_supcon": avg_supcon,
                "train_recon_lp": avg_recon_lp,
                "train_recon_attn_forward": avg_recon_attn["forward"],
                "train_recon_attn_backward": avg_recon_attn["backward"],
                "train_intra_cos": avg_intra_cos,
                "train_intra_inter_margin": avg_intra_inter,
                "test_loss": test_loss,
                "test_intra_cos": test_intra_cos,
                "test_intra_inter_margin": test_intra_inter,
                "best_loss": min(best_loss, test_loss),
                "temperature": temperature,
                "lr": lr,
                "recon_lambda": _lambda_lp,
                "attn_recon_lambda": _lambda_attn,
            }

            last_path = os.path.join(checkpoint_dir, "contrastive_last.pt")
            _atomic_torch_save(checkpoint, last_path)

            _save_and_prune_snapshots(
                checkpoint_dir=checkpoint_dir,
                snapshot_prefix="contrastive",
                epoch_one_indexed=epoch + 1,
                checkpoint=checkpoint,
                snapshot_every=snapshot_every,
                snapshot_keep_last=snapshot_keep_last,
                is_last_epoch=is_last_epoch,
            )

            if cleanup_legacy_checkpoints:
                _cleanup_legacy_checkpoints(checkpoint_dir, keep_filenames={"contrastive_last.pt"})


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
