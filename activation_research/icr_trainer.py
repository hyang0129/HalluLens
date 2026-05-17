"""Trainer for ICRProbe over precomputed icr_scores.npy.

Mirrors LinearProbeTrainer's interface (fit/validate/maybe_save_checkpoint)
but consumes the (B, L) ICR vector directly, with paper-style optimization:

  - BCEWithLogitsLoss                 (upstream icr_probe.py:38-46 uses BCELoss;
                                       we use the logit-stable equivalent)
  - Adam(lr, weight_decay)            per notes §7
  - ReduceLROnPlateau on val AUROC    (factor=0.5, patience=5)
  - Early stop on val AUROC           (patience=10)

Checkpoint filename is "linear_probe_last.pt" for transfer_eval.py compat
(per plan §B.2 spec §10).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.progress import tqdm

from .trainer import Trainer, TrainerConfig
from .training import _atomic_torch_save


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ICRProbeTrainerConfig(TrainerConfig):
    """Configuration for ICRProbeTrainer.

    Inherits standard fields (max_epochs, batch_size, lr, device, etc.) from
    TrainerConfig; adds ICR-specific scheduling / early-stopping knobs.
    """

    learning_rate: float = 1e-3
    """Adam learning rate.  Shadowing TrainerConfig.lr so callers can pass
    learning_rate= without renaming; both are accepted and ICRProbeTrainer
    reads this field."""

    weight_decay: float = 0.0
    """Adam weight decay.  Per upstream icr_probe.py default."""

    plateau_patience: int = 5
    """ReduceLROnPlateau patience (epochs without AUROC improvement)."""

    plateau_factor: float = 0.5
    """ReduceLROnPlateau LR reduction factor."""

    early_stop_patience: int = 10
    """Epochs without val AUROC improvement before stopping."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ICRProbeTrainer(Trainer):
    """Trainer for ICRProbe.

    Differences from the generic Trainer base:
    - Optimizer: Adam with weight_decay from config.
    - Loss: BCEWithLogitsLoss (numerically equivalent to upstream's BCELoss
      + Sigmoid, but more stable near saturation).
    - Scheduler: ReduceLROnPlateau on val AUROC (mode="max").
    - Early stopping: stops when AUROC has not improved for
      early_stop_patience consecutive epochs.
    - Checkpoint: writes "linear_probe_last.pt" for transfer_eval.py compat.

    Notes
    -----
    Does not use InfiniteIndexStream / balanced sampling — the ICR score
    array is (N, L) fp32 and small enough that a plain DataLoader is fine.
    """

    def __init__(self, model: torch.nn.Module, *, config: ICRProbeTrainerConfig) -> None:
        self.probe_config = config
        super().__init__(model, config=config)

        # Override the base Adam with paper-faithful Adam + wd from config.
        # Per notes §7 / upstream icr_probe.py:38-46.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",           # maximize val AUROC
            factor=config.plateau_factor,
            patience=config.plateau_patience,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.best_auroc: float = 0.0
        self._epochs_since_best: int = 0

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward + loss on one batch.

        Parameters
        ----------
        batch : dict with keys "icr_score" (B, L) and "halu" (B,).

        Returns
        -------
        (loss, metrics_dict)
        """
        x = batch["icr_score"].to(self.device, non_blocking=True)      # (B, L)
        labels = batch["halu"].to(self.device, non_blocking=True).float()  # (B,)
        logits = self.model(x)  # (B,)
        loss = self.loss_fn(logits, labels)
        acc = float(((logits > 0).float() == labels).float().mean().item())
        return loss, {"acc": acc}

    # ------------------------------------------------------------------
    # Epoch loop (mirrors LinearProbeTrainer.train_epoch)
    # ------------------------------------------------------------------

    def train_epoch(
        self, *, epoch: int, train_dataset
    ) -> Dict[str, float]:
        self.model.train()
        train_loader, steps_per_epoch, train_iter = self._build_train_iterator(
            train_dataset
        )

        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0

        if steps_per_epoch is not None:
            loop = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}/{self.config.max_epochs}",
                leave=False,
            )
        else:
            loop = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.max_epochs}",
                leave=False,
            )

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
            if (
                self.config.grad_clip_norm is not None
                and float(self.config.grad_clip_norm) > 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.config.grad_clip_norm),
                )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_acc += float(metrics.get("acc", 0.0))
            n_steps += 1
            loop.set_postfix(
                loss=total_loss / n_steps,
                acc=total_acc / n_steps,
            )

        out = {
            "train_loss": total_loss / max(1, n_steps),
            "train_acc": total_acc / max(1, n_steps),
        }
        logger.info(
            f"Train: loss={out['train_loss']:.4f}, acc={out['train_acc']:.4f}"
        )
        return out

    # ------------------------------------------------------------------
    # Validation (AUROC + LR scheduler)
    # ------------------------------------------------------------------

    def validate(
        self, *, epoch: int, val_dataset
    ) -> Dict[str, float]:
        """Evaluate on val split; step the LR scheduler on AUROC."""
        self.model.eval()
        val_loader, eval_max_batches, _ = self._build_val_loader(val_dataset)

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if eval_max_batches is not None and i >= eval_max_batches:
                    break
                x = batch["icr_score"].to(self.device, non_blocking=True)
                labels = batch["halu"].to(self.device, non_blocking=True).float()
                logits = self.model(x)
                loss = self.loss_fn(logits, labels)
                total_loss += float(loss.item())
                n_batches += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits_np = torch.cat(all_logits).numpy()
        all_labels_np = torch.cat(all_labels).numpy()

        # AUROC on raw logits (monotone transform of sigmoid — same ranking).
        try:
            auroc = float(roc_auc_score(all_labels_np, all_logits_np))
        except ValueError:
            auroc = float("nan")

        # Step LR scheduler (mode="max", so higher AUROC = improvement).
        if not (auroc != auroc):  # not NaN
            self.scheduler.step(auroc)
            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self._epochs_since_best = 0
            else:
                self._epochs_since_best += 1
        else:
            self._epochs_since_best += 1

        out = {
            "val_loss": total_loss / max(1, n_batches),
            "val_auroc": auroc,
            "best_auroc": self.best_auroc,
        }
        logger.info(
            f"Val: loss={out['val_loss']:.4f}, "
            f"auroc={out['val_auroc']:.4f}, "
            f"best={out['best_auroc']:.4f}, "
            f"epochs_since_best={self._epochs_since_best}"
        )
        return out

    # ------------------------------------------------------------------
    # Fit with early stopping
    # ------------------------------------------------------------------

    def fit(self, train_dataset, val_dataset=None) -> None:
        """Run the training loop with early stopping on val AUROC.

        Stops early when val AUROC has not improved for
        config.early_stop_patience consecutive epochs.
        """
        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)

        for epoch in tqdm(
            range(self.start_epoch, int(self.config.max_epochs)), desc="Epochs"
        ):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            epoch_start = time.perf_counter()

            train_start = time.perf_counter()
            train_metrics = self.train_epoch(epoch=epoch, train_dataset=train_dataset)
            train_seconds = float(time.perf_counter() - train_start)

            val_metrics: Dict[str, float] = {}
            val_seconds = 0.0
            if val_dataset is not None:
                val_start = time.perf_counter()
                val_metrics = self.validate(epoch=epoch, val_dataset=val_dataset)
                val_seconds = float(time.perf_counter() - val_start)

            is_last = epoch == int(self.config.max_epochs) - 1
            checkpoint_seconds = float(
                self.maybe_save_checkpoint(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    is_last_epoch=is_last,
                )
            )

            if bool(getattr(self.config, "log_timing", True)):
                epoch_seconds = float(time.perf_counter() - epoch_start)
                logger.info(
                    "Epoch timing: "
                    f"epoch={epoch + 1}/{self.config.max_epochs}, "
                    f"total={epoch_seconds:.2f}s, "
                    f"train={train_seconds:.2f}s, "
                    f"val={val_seconds:.2f}s, "
                    f"checkpoint={checkpoint_seconds:.2f}s"
                )

            # Early stopping.
            if (
                val_dataset is not None
                and self._epochs_since_best >= self.probe_config.early_stop_patience
            ):
                logger.info(
                    f"Early stop at epoch {epoch + 1}: "
                    f"no AUROC improvement for "
                    f"{self._epochs_since_best} epochs "
                    f"(patience={self.probe_config.early_stop_patience})"
                )
                break

    # ------------------------------------------------------------------
    # Checkpoint (linear_probe_last.pt for transfer_eval.py compat)
    # ------------------------------------------------------------------

    def maybe_save_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_last_epoch: bool,
    ) -> float:
        """Write checkpoint to linear_probe_last.pt.

        Filename matches LinearProbeTrainer convention so that
        transfer_eval.py and aggregate loaders work without special-casing.
        Per plan §B.2 / spec §10.
        """
        checkpoint_start = time.perf_counter()

        if int(self.config.save_every) <= 0:
            return 0.0

        should_save = (
            (epoch + 1) % int(self.config.save_every) == 0
        ) or bool(is_last_epoch)
        if not should_save:
            return 0.0

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, "linear_probe_last.pt")
        _atomic_torch_save(
            {
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr": float(self.probe_config.learning_rate),
                **train_metrics,
                **val_metrics,
            },
            path,
        )

        elapsed = float(time.perf_counter() - checkpoint_start)
        if bool(getattr(self.config, "log_timing", True)):
            logger.info(
                f"Checkpoint timing: epoch={epoch + 1}, "
                f"path={path}, elapsed={elapsed:.2f}s"
            )
        return elapsed

    def load_checkpoint(self, resume_from: str) -> None:
        checkpoint_path = resume_from
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, resume_from
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_auroc = float(checkpoint.get("best_auroc", 0.0))
        logger.info(
            f"Resumed ICR probe training from epoch {self.start_epoch}"
        )
