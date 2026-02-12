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
        use_labels: bool = False,
        ignore_label: int = -1,
    ):
        self.loss_fn = loss_fn
        self.device = device
        self.use_labels = bool(use_labels)
        self.ignore_label = int(ignore_label)
        self.state = ContrastiveEvalState()

    def on_validation_epoch_start(self) -> None:
        """Reset running aggregates before validation starts."""
        self.state = ContrastiveEvalState()

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, float]:
        """Process one validation batch and update running aggregates."""
        x1 = batch["layer1_activations"].squeeze(1).to(self.device, non_blocking=True)
        x2 = batch["layer2_activations"].squeeze(1).to(self.device, non_blocking=True)

        z1 = _call_model(model, x1)
        z2 = _call_model(model, x2)
        z_stacked = torch.stack([z1, z2], dim=1)

        if self.use_labels:
            labels = batch["halu"].to(self.device, non_blocking=True)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            elif labels.dim() > 1:
                labels = labels.view(-1)
            loss = self.loss_fn(z_stacked, labels=labels)
        else:
            loss = self.loss_fn(z_stacked)

        acc = pairing_accuracy(z1, z2)
        cosine_sim = average_cosine_similarity(z1, z2)

        self.state.total_loss += float(loss.detach().cpu().item())
        self.state.total_acc += float(acc)
        self.state.total_cosine_sim += float(cosine_sim)
        self.state.n_batches += 1

        return {
            "loss": float(loss.detach().cpu().item()),
            "acc": float(acc),
            "cosine_sim": float(cosine_sim),
        }

    def on_validation_epoch_end(self) -> Dict[str, float]:
        """Return averaged validation metrics."""
        denom = max(1, int(self.state.n_batches))
        return {
            "loss": float(self.state.total_loss / denom),
            "acc": float(self.state.total_acc / denom),
            "cosine_sim": float(self.state.total_cosine_sim / denom),
        }