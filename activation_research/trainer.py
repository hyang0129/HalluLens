"""Trainer abstractions for activation-research models.

This module introduces a lightweight, Lightning-inspired training loop.

Goals:
- Provide a reusable Trainer base class.
- Implement a contrastive trainer equivalent to `train_contrastive` in
  `activation_research/training.py`, while avoiding sub-batch buffering.
- Keep the legacy function-based training code intact.

Notes:
- This code intentionally reuses a few helper utilities from
  `activation_research.training` (checkpoint atomics, collate, etc.) to keep
  behavior consistent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from tqdm.autonotebook import tqdm

from .contrastive_evaluator import (
    ContrastiveEvaluator,
    average_cosine_similarity,
    pairing_accuracy,
)
from .evaluation import evaluate
from .training import (
    InfiniteIndexStream,
    SupConLoss,
    _atomic_torch_save,
    _build_balanced_sampler,
    _cleanup_legacy_checkpoints,
    _contrastive_collate_min,
    _save_and_prune_snapshots,
)


def _resolve_device(device: str) -> torch.device:
    device = (device or "auto").lower().strip()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _select_eval_sub_batch_size(
    *,
    full_batch_size: int,
    expected_full_batches: int,
    num_workers: int,
    target_worker_multiplier: float,
    min_sub_batch_size: int,
) -> int:
    """Select a sub-batch size that increases eval loader parallelism.

    Chooses the largest divisor of ``full_batch_size`` that yields at least
    ``ceil(num_workers * target_worker_multiplier)`` sub-batches across the
    expected full-batch count.
    """
    full_batch_size = int(full_batch_size)
    expected_full_batches = int(expected_full_batches)
    num_workers = int(num_workers)
    target_worker_multiplier = float(target_worker_multiplier)
    min_sub_batch_size = max(32, int(min_sub_batch_size))

    if full_batch_size <= 0:
        raise ValueError("full_batch_size must be positive")
    if expected_full_batches <= 0 or num_workers <= 0:
        return full_batch_size

    target_sub_batches = int(math.ceil(max(1.0, target_worker_multiplier) * float(num_workers)))
    if expected_full_batches >= target_sub_batches:
        return full_batch_size

    divisors_desc = [
        size
        for size in range(full_batch_size, min_sub_batch_size - 1, -1)
        if full_batch_size % size == 0
    ]
    if not divisors_desc:
        return full_batch_size

    for sub_batch_size in divisors_desc:
        multiplier = full_batch_size // sub_batch_size
        total_sub_batches = expected_full_batches * multiplier
        if total_sub_batches >= target_sub_batches:
            return int(sub_batch_size)

    return int(divisors_desc[-1])


@dataclass
class TrainerConfig:
    """Common configuration for all trainers."""

    max_epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-6

    num_workers: int = 0
    device: str = "auto"

    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    resume_from: Optional[str] = None

    persistent_workers: bool = True
    cleanup_legacy_checkpoints: bool = True

    snapshot_every: int = 0
    snapshot_keep_last: int = 5

    eval_sub_batch_size: Optional[int] = None
    eval_auto_sub_batching: bool = True
    eval_sub_batch_target_worker_multiplier: float = 2.0
    eval_sub_batch_min_size: int = 1


class Trainer:
    """A minimal, Lightning-inspired trainer.

    Subclasses implement:
    - `training_step(batch) -> (loss, metrics)`
    - optionally `validate(...)` to override validation behavior.

    This class purposefully stays small: it owns the training loop,
    checkpointing, and device placement.
    """

    def __init__(self, model: torch.nn.Module, *, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = _resolve_device(config.device)

        self._cached_train_dataset_key: Optional[int] = None
        self._cached_train_loader: Optional[DataLoader] = None
        self._cached_train_steps_per_epoch: Optional[int] = None
        self._cached_train_infinite_iter = None

        self._cached_val_dataset_key: Optional[int] = None
        self._cached_val_loader: Optional[DataLoader] = None
        self._cached_val_max_batches: Optional[int] = None
        self._cached_val_loader_batch_size: Optional[int] = None

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.model = self.model.to(self.device)
        self.optimizer = self.configure_optimizers()

        self.start_epoch: int = 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def fit(self, train_dataset, val_dataset=None) -> None:
        """Run the training loop."""
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)

        for epoch in tqdm(range(self.start_epoch, int(self.config.max_epochs)), desc="Epochs"):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            train_metrics = self.train_epoch(epoch=epoch, train_dataset=train_dataset)

            val_metrics: Dict[str, float] = {}
            if val_dataset is not None:
                val_metrics = self.validate(epoch=epoch, val_dataset=val_dataset)

            self.maybe_save_checkpoint(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_last_epoch=(epoch == int(self.config.max_epochs) - 1),
            )

    def train_epoch(self, *, epoch: int, train_dataset) -> Dict[str, float]:
        """Train for a single epoch and return aggregate metrics."""
        self.model.train()

        train_loader, steps_per_epoch, train_iter = self._build_train_iterator(train_dataset)

        total_loss = 0.0
        total_acc = 0.0
        total_cos = 0.0
        n_steps = 0

        if steps_per_epoch is not None:
            loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{self.config.max_epochs}", leave=False)
        else:
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}", leave=False)

        for i, _ in enumerate(loop, start=1):
            if train_iter is not None:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
            else:
                batch = _

            loss, metrics = self.training_step(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_acc += float(metrics.get("pairing_acc", 0.0))
            total_cos += float(metrics.get("cosine_sim", 0.0))
            n_steps += 1

            loop.set_postfix(
                loss=(total_loss / n_steps),
                pairing_acc=(total_acc / n_steps),
                cosine_sim=(total_cos / n_steps),
            )

        out = {
            "train_loss": (total_loss / max(1, n_steps)),
            "train_acc": (total_acc / max(1, n_steps)),
            "train_cosine_sim": (total_cos / max(1, n_steps)),
        }

        logger.info(
            "Train epoch metrics: "
            f"loss={out['train_loss']:.4f}, acc={out['train_acc']:.4f}, cos={out['train_cosine_sim']:.4f}"
        )
        return out

    def validate(self, *, epoch: int, val_dataset) -> Dict[str, float]:
        """Default validation: no-op."""
        _ = epoch
        _ = val_dataset
        return {}

    def _resolve_eval_loader_batch_size(self, val_dataset) -> int:
        """Resolve eval loader batch size from config and dataset characteristics."""
        full_batch_size = int(self.config.batch_size)

        explicit_sub_batch_size = getattr(self.config, "eval_sub_batch_size", None)
        if explicit_sub_batch_size is not None:
            explicit_sub_batch_size = int(explicit_sub_batch_size)
            if explicit_sub_batch_size <= 0:
                raise ValueError("eval_sub_batch_size must be positive when provided")
            if explicit_sub_batch_size > full_batch_size:
                raise ValueError("eval_sub_batch_size cannot exceed batch_size")
            if full_batch_size % explicit_sub_batch_size != 0:
                raise ValueError("batch_size must be divisible by eval_sub_batch_size")
            return explicit_sub_batch_size

        if not bool(getattr(self.config, "eval_auto_sub_batching", True)):
            return full_batch_size

        if not hasattr(val_dataset, "__len__"):
            return full_batch_size

        num_workers = int(getattr(self.config, "num_workers", 0))
        if num_workers <= 0:
            return full_batch_size

        expected_full_batches = int(math.ceil(len(val_dataset) / float(full_batch_size)))
        sub_batch_size = _select_eval_sub_batch_size(
            full_batch_size=full_batch_size,
            expected_full_batches=expected_full_batches,
            num_workers=num_workers,
            target_worker_multiplier=float(
                getattr(self.config, "eval_sub_batch_target_worker_multiplier", 2.0)
            ),
            min_sub_batch_size=int(getattr(self.config, "eval_sub_batch_min_size", 1)),
        )
        if sub_batch_size != full_batch_size:
            logger.info(
                "Auto eval sub-batching enabled: "
                f"batch_size={full_batch_size} -> eval_sub_batch_size={sub_batch_size} "
                f"(workers={num_workers}, expected_full_batches={expected_full_batches})"
            )
        return int(sub_batch_size)

    def _build_val_loader(self, val_dataset) -> Tuple[DataLoader, Optional[int], int]:
        """Return (val_loader, max_loader_batches_or_none, eval_loader_batch_size) with caching."""
        dataset_key = id(val_dataset)
        eval_loader_batch_size = self._resolve_eval_loader_batch_size(val_dataset)
        needs_rebuild = (
            (self._cached_val_loader is None)
            or (self._cached_val_dataset_key != dataset_key)
            or (self._cached_val_loader_batch_size != int(eval_loader_batch_size))
        )

        if needs_rebuild:
            dataset = val_dataset
            eval_max_batches = None

            if bool(getattr(self.config, "use_infinite_index_stream_eval", False)):
                if not hasattr(dataset, "__len__"):
                    raise TypeError("use_infinite_index_stream_eval=True requires val_dataset to have __len__")
                base_len = len(dataset)
                if not isinstance(dataset, IterableDataset):
                    dataset = InfiniteIndexStream(
                        dataset,
                        shuffle=bool(getattr(self.config, "infinite_eval_shuffle", True)),
                        seed=int(getattr(self.config, "infinite_eval_seed", 0)),
                    )
                eval_max_batches = int(math.ceil(base_len / float(eval_loader_batch_size)))

            val_loader = self.val_dataloader(dataset, batch_size=int(eval_loader_batch_size))

            self._cached_val_loader = val_loader
            self._cached_val_dataset_key = dataset_key
            self._cached_val_max_batches = eval_max_batches
            self._cached_val_loader_batch_size = int(eval_loader_batch_size)

        return self._cached_val_loader, self._cached_val_max_batches, int(eval_loader_batch_size)

    def maybe_save_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_last_epoch: bool,
    ) -> None:
        """Default checkpointing behavior: save last checkpoint periodically."""
        if int(self.config.save_every) <= 0:
            return

        should_save = ((epoch + 1) % int(self.config.save_every) == 0) or bool(is_last_epoch)
        if not should_save:
            return

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            **train_metrics,
            **val_metrics,
            "lr": float(self.config.lr),
        }

        last_path = os.path.join(self.config.checkpoint_dir, "trainer_last.pt")
        _atomic_torch_save(checkpoint, last_path)

        _save_and_prune_snapshots(
            checkpoint_dir=self.config.checkpoint_dir,
            snapshot_prefix="trainer",
            epoch_one_indexed=epoch + 1,
            checkpoint=checkpoint,
            snapshot_every=int(self.config.snapshot_every),
            snapshot_keep_last=int(self.config.snapshot_keep_last),
            is_last_epoch=bool(is_last_epoch),
        )

        if bool(self.config.cleanup_legacy_checkpoints):
            _cleanup_legacy_checkpoints(self.config.checkpoint_dir, keep_filenames={"trainer_last.pt"})

    def load_checkpoint(self, resume_from: str) -> None:
        checkpoint_path = resume_from
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.config.checkpoint_dir, resume_from)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info(f"Resumed training from epoch {self.start_epoch}")

    def _build_train_iterator(self, train_dataset):
        """Return (train_loader, steps_per_epoch, train_iter_or_none)."""
        dataset_key = id(train_dataset)
        needs_rebuild = (self._cached_train_loader is None) or (self._cached_train_dataset_key != dataset_key)

        if needs_rebuild:
            train_loader = self.train_dataloader(train_dataset)
            self._cached_train_loader = train_loader
            self._cached_train_dataset_key = dataset_key
            self._cached_train_infinite_iter = None

            if bool(getattr(self.config, "use_infinite_index_stream", False)):
                if not hasattr(train_dataset, "__len__"):
                    raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
                self._cached_train_steps_per_epoch = int(math.ceil(len(train_dataset) / float(self.config.batch_size)))
            else:
                self._cached_train_steps_per_epoch = None

        train_loader = self._cached_train_loader

        if bool(getattr(self.config, "use_infinite_index_stream", False)):
            if self._cached_train_infinite_iter is None:
                self._cached_train_infinite_iter = iter(train_loader)

            return train_loader, self._cached_train_steps_per_epoch, self._cached_train_infinite_iter

        if isinstance(train_dataset, IterableDataset):
            # Iterable datasets may be infinite; default to iterating the loader directly.
            return train_loader, None, None

        return train_loader, None, None

    def train_dataloader(self, train_dataset) -> DataLoader:
        return DataLoader(
            train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            num_workers=int(self.config.num_workers),
            pin_memory=True,
            persistent_workers=bool(self.config.persistent_workers and int(self.config.num_workers) > 0),
        )

    def val_dataloader(self, val_dataset, *, batch_size: Optional[int] = None) -> DataLoader:
        resolved_batch_size = int(self.config.batch_size if batch_size is None else batch_size)
        return DataLoader(
            val_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            num_workers=int(self.config.num_workers),
            pin_memory=True,
            persistent_workers=bool(self.config.persistent_workers and int(self.config.num_workers) > 0),
        )


@dataclass
class ContrastiveTrainerConfig(TrainerConfig):
    """Configuration for `ContrastiveTrainer`."""

    temperature: float = 0.07

    use_labels: bool = False
    ignore_label: int = -1
    same_sample_weight: float = 1.0
    same_class_weight: float = 1.0

    balanced_sampling: bool = False

    # Optional infinite index streaming (keeps workers alive)
    use_infinite_index_stream: bool = True
    infinite_stream_shuffle: bool = True
    infinite_stream_seed: int = 0

    use_infinite_index_stream_eval: bool = True
    infinite_eval_shuffle: bool = False
    infinite_eval_seed: int = 0


@dataclass
class LayerAwareContrastiveTrainerConfig(ContrastiveTrainerConfig):
    """Configuration for `LayerAwareContrastiveTrainer`.

    Kept as a distinct type to make experiments explicit in scripts.
    """

    pass


class ContrastiveTrainer(Trainer):
    """Trainer for supervised contrastive learning on activation pairs.

    This is the class-based replacement for the legacy `train_contrastive`.

    Differences vs legacy:
    - No sub-batching / microbatch buffering. Each DataLoader batch is processed
      end-to-end.
    - Keeps checkpoint format broadly compatible (writes `contrastive_last.pt`).
    """

    def __init__(self, model: torch.nn.Module, *, config: ContrastiveTrainerConfig):
        self.contrastive_config = config
        super().__init__(model, config=config)

        self.loss_fn = SupConLoss(
            temperature=float(config.temperature),
            ignore_label=int(config.ignore_label),
            same_sample_weight=float(config.same_sample_weight),
            same_class_weight=float(config.same_class_weight),
        )

        self.best_loss: float = float("inf")

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x1 = batch["layer1_activations"].squeeze(1).to(self.device, non_blocking=True)
        x2 = batch["layer2_activations"].squeeze(1).to(self.device, non_blocking=True)

        z1 = self.model(x1)
        z2 = self.model(x2)
        z_stacked = torch.stack([z1, z2], dim=1)

        if bool(self.contrastive_config.use_labels):
            labels = batch["halu"].to(self.device, non_blocking=True)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 1:
                labels = labels.view(-1)

            hashkeys = batch.get("hashkey")
            if isinstance(hashkeys, str):
                hashkeys = [hashkeys]
            if hashkeys is None:
                # Fallback: stable ids 0..B-1 for this batch (still lets same-view pair be positive)
                sample_ids = torch.arange(labels.shape[0], device=self.device, dtype=torch.long)
            else:
                sample_ids = torch.tensor(
                    [hash(hk) % 1000000 for hk in hashkeys],
                    dtype=torch.long,
                    device=self.device,
                )

            loss = self.loss_fn(z_stacked, labels=labels, sample_ids=sample_ids)
        else:
            loss = self.loss_fn(z_stacked)

        acc = pairing_accuracy(z1, z2)
        cos = average_cosine_similarity(z1, z2)

        return loss, {"pairing_acc": float(acc), "cosine_sim": float(cos)}

    def train_dataloader(self, train_dataset) -> DataLoader:
        dataset = train_dataset

        is_iterable = isinstance(dataset, IterableDataset)
        if bool(self.contrastive_config.use_infinite_index_stream) and not is_iterable:
            if not hasattr(dataset, "__len__"):
                raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
            dataset = InfiniteIndexStream(
                dataset,
                shuffle=bool(self.contrastive_config.infinite_stream_shuffle),
                seed=int(self.contrastive_config.infinite_stream_seed),
            )
            is_iterable = True

        sampler = None
        if (
            bool(self.contrastive_config.balanced_sampling)
            and bool(self.contrastive_config.use_labels)
            and not is_iterable
        ):
            sampler = _build_balanced_sampler(dataset)

        return DataLoader(
            dataset,
            batch_size=int(self.contrastive_config.batch_size),
            shuffle=(sampler is None and not is_iterable),
            sampler=sampler,
            num_workers=int(self.contrastive_config.num_workers),
            pin_memory=True,
            persistent_workers=bool(
                self.contrastive_config.persistent_workers and int(self.contrastive_config.num_workers) > 0
            ),
            collate_fn=_contrastive_collate_min,
        )

    def val_dataloader(self, val_dataset, *, batch_size: Optional[int] = None) -> DataLoader:
        resolved_batch_size = int(self.contrastive_config.batch_size if batch_size is None else batch_size)
        return DataLoader(
            val_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            num_workers=int(self.contrastive_config.num_workers),
            pin_memory=True,
            persistent_workers=bool(
                self.contrastive_config.persistent_workers and int(self.contrastive_config.num_workers) > 0
            ),
            collate_fn=_contrastive_collate_min,
        )

    def validate(self, *, epoch: int, val_dataset) -> Dict[str, float]:
        _ = epoch
        val_loader, eval_max_batches, eval_sub_batch_size = self._build_val_loader(val_dataset)

        evaluator = ContrastiveEvaluator(
            loss_fn=self.loss_fn,
            device=self.device,
            batch_size=int(self.contrastive_config.batch_size),
            sub_batch_size=int(eval_sub_batch_size),
            use_labels=bool(self.contrastive_config.use_labels),
            ignore_label=int(self.contrastive_config.ignore_label),
        )
        val_stats = evaluator.run(
            self.model,
            val_loader,
            max_batches=eval_max_batches,
        )
        test_loss = float(val_stats["loss"])
        test_acc = float(val_stats["acc"])
        test_cos = float(val_stats["cosine_sim"])

        self.best_loss = min(float(self.best_loss), float(test_loss))

        out = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_cosine_sim": float(test_cos),
            "best_loss": float(self.best_loss),
        }

        logger.info(
            "Val metrics: "
            f"loss={out['test_loss']:.4f}, acc={out['test_acc']:.4f}, cos={out['test_cosine_sim']:.4f}"
        )
        return out

    def maybe_save_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_last_epoch: bool,
    ) -> None:
        if int(self.contrastive_config.save_every) <= 0:
            return

        should_save = ((epoch + 1) % int(self.contrastive_config.save_every) == 0) or bool(is_last_epoch)
        if not should_save:
            return

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "temperature": float(self.contrastive_config.temperature),
            "lr": float(self.contrastive_config.lr),
            **train_metrics,
            **val_metrics,
        }

        last_path = os.path.join(self.contrastive_config.checkpoint_dir, "contrastive_last.pt")
        _atomic_torch_save(checkpoint, last_path)

        _save_and_prune_snapshots(
            checkpoint_dir=self.contrastive_config.checkpoint_dir,
            snapshot_prefix="contrastive",
            epoch_one_indexed=epoch + 1,
            checkpoint=checkpoint,
            snapshot_every=int(self.contrastive_config.snapshot_every),
            snapshot_keep_last=int(self.contrastive_config.snapshot_keep_last),
            is_last_epoch=bool(is_last_epoch),
        )

        if bool(self.contrastive_config.cleanup_legacy_checkpoints):
            _cleanup_legacy_checkpoints(
                self.contrastive_config.checkpoint_dir,
                keep_filenames={"contrastive_last.pt"},
            )

    def load_checkpoint(self, resume_from: str) -> None:
        checkpoint_path = resume_from
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.contrastive_config.checkpoint_dir, resume_from)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_loss = float(checkpoint.get("best_loss", float("inf")))
        logger.info(f"Resumed training from epoch {self.start_epoch}")


class LayerAwareContrastiveTrainer(Trainer):
    """Trainer for contrastive learning with layer-aware encoders.

    Diff vs `ContrastiveTrainer`:
    - Calls the model with kwargs (`layer_idx=...`) for each view.
    - Writes distinct checkpoint files.
    """

    def __init__(self, model: torch.nn.Module, *, config: LayerAwareContrastiveTrainerConfig):
        self.contrastive_config = config
        super().__init__(model, config=config)

        self.loss_fn = SupConLoss(
            temperature=float(config.temperature),
            ignore_label=int(config.ignore_label),
            same_sample_weight=float(config.same_sample_weight),
            same_class_weight=float(config.same_class_weight),
        )

        self.best_loss: float = float("inf")

    def training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x1 = batch["layer1_activations"].squeeze(1).to(self.device, non_blocking=True)
        x2 = batch["layer2_activations"].squeeze(1).to(self.device, non_blocking=True)

        layer1_idx = batch.get("layer1_idx")
        layer2_idx = batch.get("layer2_idx")
        if isinstance(layer1_idx, torch.Tensor):
            layer1_idx = layer1_idx.to(self.device, non_blocking=True)
        if isinstance(layer2_idx, torch.Tensor):
            layer2_idx = layer2_idx.to(self.device, non_blocking=True)

        z1 = self.model(x1, layer_idx=layer1_idx)
        z2 = self.model(x2, layer_idx=layer2_idx)
        z_stacked = torch.stack([z1, z2], dim=1)

        if bool(self.contrastive_config.use_labels):
            labels = batch["halu"].to(self.device, non_blocking=True)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 1:
                labels = labels.view(-1)

            hashkeys = batch.get("hashkey")
            if isinstance(hashkeys, str):
                hashkeys = [hashkeys]
            if hashkeys is None:
                sample_ids = torch.arange(labels.shape[0], device=self.device, dtype=torch.long)
            else:
                sample_ids = torch.tensor(
                    [hash(hk) % 1000000 for hk in hashkeys],
                    dtype=torch.long,
                    device=self.device,
                )

            loss = self.loss_fn(z_stacked, labels=labels, sample_ids=sample_ids)
        else:
            loss = self.loss_fn(z_stacked)

        acc = pairing_accuracy(z1, z2)
        cos = average_cosine_similarity(z1, z2)
        return loss, {"pairing_acc": float(acc), "cosine_sim": float(cos)}

    def train_dataloader(self, train_dataset) -> DataLoader:
        dataset = train_dataset

        is_iterable = isinstance(dataset, IterableDataset)
        if bool(self.contrastive_config.use_infinite_index_stream) and not is_iterable:
            if not hasattr(dataset, "__len__"):
                raise TypeError("use_infinite_index_stream=True requires train_dataset to have __len__")
            dataset = InfiniteIndexStream(
                dataset,
                shuffle=bool(self.contrastive_config.infinite_stream_shuffle),
                seed=int(self.contrastive_config.infinite_stream_seed),
            )
            is_iterable = True

        sampler = None
        if (
            bool(self.contrastive_config.balanced_sampling)
            and bool(self.contrastive_config.use_labels)
            and not is_iterable
        ):
            sampler = _build_balanced_sampler(dataset)

        return DataLoader(
            dataset,
            batch_size=int(self.contrastive_config.batch_size),
            shuffle=(sampler is None and not is_iterable),
            sampler=sampler,
            num_workers=int(self.contrastive_config.num_workers),
            pin_memory=True,
            persistent_workers=bool(
                self.contrastive_config.persistent_workers and int(self.contrastive_config.num_workers) > 0
            ),
            collate_fn=_contrastive_collate_min,
        )

    def val_dataloader(self, val_dataset, *, batch_size: Optional[int] = None) -> DataLoader:
        resolved_batch_size = int(self.contrastive_config.batch_size if batch_size is None else batch_size)
        return DataLoader(
            val_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            num_workers=int(self.contrastive_config.num_workers),
            pin_memory=True,
            persistent_workers=bool(
                self.contrastive_config.persistent_workers and int(self.contrastive_config.num_workers) > 0
            ),
            collate_fn=_contrastive_collate_min,
        )

    def validate(self, *, epoch: int, val_dataset) -> Dict[str, float]:
        _ = epoch
        val_loader, eval_max_batches, eval_sub_batch_size = self._build_val_loader(val_dataset)

        test_loss, test_acc, test_cos = evaluate(
            self.model,
            val_loader,
            batch_size=int(self.contrastive_config.batch_size),
            sub_batch_size=int(eval_sub_batch_size),
            loss_fn=self.loss_fn,
            device=str(self.device),
            use_labels=bool(self.contrastive_config.use_labels),
            ignore_label=int(self.contrastive_config.ignore_label),
            max_batches=eval_max_batches,
        )

        self.best_loss = min(float(self.best_loss), float(test_loss))

        out = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_cosine_sim": float(test_cos),
            "best_loss": float(self.best_loss),
        }

        logger.info(
            "Val metrics: "
            f"loss={out['test_loss']:.4f}, acc={out['test_acc']:.4f}, cos={out['test_cosine_sim']:.4f}"
        )
        return out

    def maybe_save_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_last_epoch: bool,
    ) -> None:
        if int(self.contrastive_config.save_every) <= 0:
            return

        should_save = ((epoch + 1) % int(self.contrastive_config.save_every) == 0) or bool(is_last_epoch)
        if not should_save:
            return

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "temperature": float(self.contrastive_config.temperature),
            "lr": float(self.contrastive_config.lr),
            **train_metrics,
            **val_metrics,
        }

        last_path = os.path.join(self.contrastive_config.checkpoint_dir, "layer_aware_contrastive_last.pt")
        _atomic_torch_save(checkpoint, last_path)

        _save_and_prune_snapshots(
            checkpoint_dir=self.contrastive_config.checkpoint_dir,
            snapshot_prefix="layer_aware_contrastive",
            epoch_one_indexed=epoch + 1,
            checkpoint=checkpoint,
            snapshot_every=int(self.contrastive_config.snapshot_every),
            snapshot_keep_last=int(self.contrastive_config.snapshot_keep_last),
            is_last_epoch=bool(is_last_epoch),
        )

        if bool(self.contrastive_config.cleanup_legacy_checkpoints):
            _cleanup_legacy_checkpoints(
                self.contrastive_config.checkpoint_dir,
                keep_filenames={"layer_aware_contrastive_last.pt"},
            )

    def load_checkpoint(self, resume_from: str) -> None:
        checkpoint_path = resume_from
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.contrastive_config.checkpoint_dir, resume_from)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_loss = float(checkpoint.get("best_loss", float("inf")))
        logger.info(f"Resumed training from epoch {self.start_epoch}")
