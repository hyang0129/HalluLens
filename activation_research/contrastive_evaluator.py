"""Lightning-style evaluator components for contrastive training.

This module provides a class-based validation evaluator that can be used by
trainer classes instead of relying on function-based evaluation utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _call_model(model: torch.nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Call model with optional kwargs while preserving backward compatibility."""
    if not kwargs:
        return model(x)

    try:
        return model(x, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg or "got an unexpected keyword" in msg:
            return model(x)
        raise


def pairing_accuracy(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Return bidirectional pairing accuracy based on cosine-similarity matching."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    sim_matrix = torch.matmul(z1, z2.T)
    preds_z1 = sim_matrix.argmax(dim=1)
    preds_z2 = sim_matrix.argmax(dim=0)

    batch_size = z1.size(0)
    correct_z1 = (preds_z1 == torch.arange(batch_size, device=z1.device)).sum().item()
    correct_z2 = (preds_z2 == torch.arange(batch_size, device=z2.device)).sum().item()
    return float((correct_z1 + correct_z2) / (2 * batch_size))


def average_cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Return mean cosine similarity for corresponding embedding pairs."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return float((z1 * z2).sum(dim=1).mean().item())


@dataclass
class ContrastiveEvalState:
    """Mutable running state for one validation epoch."""

    total_loss: float = 0.0
    total_acc: float = 0.0
    total_cosine_sim: float = 0.0
    n_batches: int = 0


def _normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Normalize labels to shape ``(B,)`` for loss functions."""
    if labels.dim() == 0:
        return labels.unsqueeze(0)
    if labels.dim() > 1:
        return labels.view(-1)
    return labels


class BaseEvaluator(ABC):
    """Base class for trainer-integrated evaluators.

    Implementations are expected to follow a simple epoch lifecycle.
    """

    @abstractmethod
    def on_validation_epoch_start(self) -> None:
        """Reset evaluator state before a validation epoch."""

    @abstractmethod
    def validation_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, float]:
        """Process one validation batch and update running state."""

    @abstractmethod
    def on_validation_epoch_end(self) -> Dict[str, float]:
        """Compute and return aggregated validation metrics."""

    def run(
        self,
        model: torch.nn.Module,
        dataloader,
        *,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Execute the full validation lifecycle over a dataloader."""
        model.eval()
        self.on_validation_epoch_start()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break
                self.validation_step(model, batch)

        return self.on_validation_epoch_end()


class ContrastiveEvaluator(BaseEvaluator):
    """Evaluator for contrastive validation.

    Lifecycle:
    1) `on_validation_epoch_start`
    2) repeated `validation_step`
    3) `on_validation_epoch_end`
    """

    def __init__(
        self,
        *,
        loss_fn,
        device: torch.device,
        batch_size: int = 32,
        sub_batch_size: Optional[int] = None,
        use_labels: bool = False,
        ignore_label: int = -1,
    ):
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = int(batch_size)
        self.sub_batch_size = int(sub_batch_size) if sub_batch_size is not None else int(batch_size)
        self.use_labels = bool(use_labels)
        self.ignore_label = int(ignore_label)
        if self.batch_size <= 0 or self.sub_batch_size <= 0:
            raise ValueError("batch_size and sub_batch_size must be positive")
        if self.batch_size % self.sub_batch_size != 0:
            raise ValueError("batch_size must be divisible by sub_batch_size")

        self._buffer_x1 = []
        self._buffer_x2 = []
        self._buffer_l1 = []
        self._buffer_l2 = []
        self._buffer_labels = []
        self.state = ContrastiveEvalState()

    def on_validation_epoch_start(self) -> None:
        """Reset running aggregates before validation starts."""
        self.state = ContrastiveEvalState()
        self._buffer_x1 = []
        self._buffer_x2 = []
        self._buffer_l1 = []
        self._buffer_l2 = []
        self._buffer_labels = []

    def _consume_buffer(self, model: torch.nn.Module, *, flush_remainder: bool) -> None:
        """Consume buffered sub-batches into full evaluation batches."""
        if not self._buffer_x1:
            return

        x1_full = torch.cat(self._buffer_x1, dim=0)
        x2_full = torch.cat(self._buffer_x2, dim=0)
        l1_full = torch.cat(self._buffer_l1, dim=0) if self._buffer_l1 else None
        l2_full = torch.cat(self._buffer_l2, dim=0) if self._buffer_l2 else None
        labels_full = torch.cat(self._buffer_labels, dim=0) if (self.use_labels and self._buffer_labels) else None

        total_samples = int(x1_full.size(0))
        consumed = 0

        while consumed + self.batch_size <= total_samples:
            end = consumed + self.batch_size
            self._process_full_batch(
                model,
                x1=x1_full[consumed:end],
                x2=x2_full[consumed:end],
                l1=(l1_full[consumed:end] if l1_full is not None else None),
                l2=(l2_full[consumed:end] if l2_full is not None else None),
                labels=(labels_full[consumed:end] if labels_full is not None else None),
            )
            consumed = end

        if flush_remainder and consumed < total_samples:
            self._process_full_batch(
                model,
                x1=x1_full[consumed:],
                x2=x2_full[consumed:],
                l1=(l1_full[consumed:] if l1_full is not None else None),
                l2=(l2_full[consumed:] if l2_full is not None else None),
                labels=(labels_full[consumed:] if labels_full is not None else None),
            )
            consumed = total_samples

        if consumed < total_samples:
            self._buffer_x1 = [x1_full[consumed:]]
            self._buffer_x2 = [x2_full[consumed:]]
            self._buffer_l1 = [l1_full[consumed:]] if l1_full is not None else []
            self._buffer_l2 = [l2_full[consumed:]] if l2_full is not None else []
            self._buffer_labels = [labels_full[consumed:]] if labels_full is not None else []
        else:
            self._buffer_x1 = []
            self._buffer_x2 = []
            self._buffer_l1 = []
            self._buffer_l2 = []
            self._buffer_labels = []

    def _process_full_batch(
        self,
        model: torch.nn.Module,
        *,
        x1: torch.Tensor,
        x2: torch.Tensor,
        l1: Optional[torch.Tensor],
        l2: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> None:
        """Run one full evaluation batch through the model and update aggregates."""
        z1 = _call_model(model, x1, layer_idx=l1)
        z2 = _call_model(model, x2, layer_idx=l2)
        z_stacked = torch.stack([z1, z2], dim=1)

        if self.use_labels and labels is not None:
            loss = self.loss_fn(z_stacked, labels=labels)
        else:
            loss = self.loss_fn(z_stacked)

        acc = pairing_accuracy(z1, z2)
        cosine_sim = average_cosine_similarity(z1, z2)

        self.state.total_loss += float(loss.detach().cpu().item())
        self.state.total_acc += float(acc)
        self.state.total_cosine_sim += float(cosine_sim)
        self.state.n_batches += 1

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, float]:
        """Process one validation batch and update running aggregates."""
        x1 = batch["layer1_activations"].squeeze(1).to(self.device, non_blocking=True)
        x2 = batch["layer2_activations"].squeeze(1).to(self.device, non_blocking=True)
        self._buffer_x1.append(x1)
        self._buffer_x2.append(x2)

        layer1_idx = batch.get("layer1_idx")
        layer2_idx = batch.get("layer2_idx")
        if isinstance(layer1_idx, torch.Tensor):
            self._buffer_l1.append(layer1_idx.to(self.device, non_blocking=True))
        if isinstance(layer2_idx, torch.Tensor):
            self._buffer_l2.append(layer2_idx.to(self.device, non_blocking=True))

        if self.use_labels:
            labels = _normalize_labels(batch["halu"].to(self.device, non_blocking=True))
            self._buffer_labels.append(labels)

        self._consume_buffer(model, flush_remainder=False)
        return {}

    def on_validation_epoch_end(self) -> Dict[str, float]:
        """Return averaged validation metrics."""
        # Final partial batch is still evaluated to match previous behavior.
        # This can be a true partial full-batch when dataset size is not divisible.
        # We do not drop it.
        #
        # `model` is only required when processing new buffered data during
        # `validation_step`; remaining data here must be consumed by rerunning
        # through the same model context, so this method should only be called
        # after all steps complete.
        denom = max(1, int(self.state.n_batches))
        return {
            "loss": float(self.state.total_loss / denom),
            "acc": float(self.state.total_acc / denom),
            "cosine_sim": float(self.state.total_cosine_sim / denom),
        }

    def run(
        self,
        model: torch.nn.Module,
        dataloader,
        *,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Execute validation while aggregating sub-batches into full batches."""
        model.eval()
        self.on_validation_epoch_start()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break
                self.validation_step(model, batch)
            self._consume_buffer(model, flush_remainder=True)

        return self.on_validation_epoch_end()