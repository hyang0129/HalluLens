"""Lightning-style evaluator components for K-view contrastive training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .evaluation import intra_inter_margin, intra_sample_cosine_mean


def _call_model(model: torch.nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
    if not kwargs:
        return model(x)

    try:
        return model(x, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg or "got an unexpected keyword" in msg:
            return model(x)
        raise


def _normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 0:
        return labels.unsqueeze(0)
    if labels.dim() > 1:
        return labels.view(-1)
    return labels


@dataclass
class ContrastiveEvalState:
    total_loss: float = 0.0
    total_intra_cos: float = 0.0
    total_intra_inter_margin: float = 0.0
    n_batches: int = 0


class BaseEvaluator(ABC):
    @abstractmethod
    def on_validation_epoch_start(self) -> None:
        pass

    @abstractmethod
    def validation_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, float]:
        pass

    @abstractmethod
    def on_validation_epoch_end(self) -> Dict[str, float]:
        pass

    def run(self, model: torch.nn.Module, dataloader, *, max_batches: Optional[int] = None) -> Dict[str, float]:
        model.eval()
        self.on_validation_epoch_start()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break
                self.validation_step(model, batch)

        return self.on_validation_epoch_end()


class ContrastiveEvaluator(BaseEvaluator):
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

        self._buffer_views = []
        self._buffer_view_indices = []
        self._buffer_labels = []
        self.state = ContrastiveEvalState()

    def on_validation_epoch_start(self) -> None:
        self.state = ContrastiveEvalState()
        self._buffer_views = []
        self._buffer_view_indices = []
        self._buffer_labels = []

    def _consume_buffer(self, model: torch.nn.Module, *, flush_remainder: bool) -> None:
        if not self._buffer_views:
            return

        views_full = torch.cat(self._buffer_views, dim=0)
        view_indices_full = torch.cat(self._buffer_view_indices, dim=0) if self._buffer_view_indices else None
        labels_full = torch.cat(self._buffer_labels, dim=0) if (self.use_labels and self._buffer_labels) else None

        total_samples = int(views_full.size(0))
        consumed = 0

        while consumed + self.batch_size <= total_samples:
            end = consumed + self.batch_size
            self._process_full_batch(
                model,
                views=views_full[consumed:end],
                view_indices=(view_indices_full[consumed:end] if view_indices_full is not None else None),
                labels=(labels_full[consumed:end] if labels_full is not None else None),
            )
            consumed = end

        if flush_remainder and consumed < total_samples:
            self._process_full_batch(
                model,
                views=views_full[consumed:],
                view_indices=(view_indices_full[consumed:] if view_indices_full is not None else None),
                labels=(labels_full[consumed:] if labels_full is not None else None),
            )
            consumed = total_samples

        if consumed < total_samples:
            self._buffer_views = [views_full[consumed:]]
            self._buffer_view_indices = [view_indices_full[consumed:]] if view_indices_full is not None else []
            self._buffer_labels = [labels_full[consumed:]] if labels_full is not None else []
        else:
            self._buffer_views = []
            self._buffer_view_indices = []
            self._buffer_labels = []

    def _process_full_batch(
        self,
        model: torch.nn.Module,
        *,
        views: torch.Tensor,
        view_indices: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> None:
        bsz, num_views, seq_len, hidden_dim = views.shape
        x_flat = views.reshape(bsz * num_views, seq_len, hidden_dim)
        layer_idx_flat = view_indices.reshape(bsz * num_views) if view_indices is not None else None

        z_flat = _call_model(model, x_flat, layer_idx=layer_idx_flat)
        z_views = z_flat.reshape(bsz, num_views, -1)

        if self.use_labels and labels is not None:
            loss = self.loss_fn(z_views, labels=labels)
        else:
            loss = self.loss_fn(z_views)

        self.state.total_loss += float(loss.detach().cpu().item())
        self.state.total_intra_cos += float(intra_sample_cosine_mean(z_views))
        self.state.total_intra_inter_margin += float(intra_inter_margin(z_views))
        self.state.n_batches += 1

    def validation_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, float]:
        views = batch["views_activations"].to(self.device, non_blocking=True)
        self._buffer_views.append(views)

        view_indices = batch.get("view_indices")
        if isinstance(view_indices, torch.Tensor):
            self._buffer_view_indices.append(view_indices.to(self.device, non_blocking=True))

        if self.use_labels:
            labels = _normalize_labels(batch["halu"].to(self.device, non_blocking=True))
            self._buffer_labels.append(labels)

        self._consume_buffer(model, flush_remainder=False)
        return {}

    def on_validation_epoch_end(self) -> Dict[str, float]:
        denom = max(1, int(self.state.n_batches))
        return {
            "loss": float(self.state.total_loss / denom),
            "intra_cos": float(self.state.total_intra_cos / denom),
            "intra_inter_margin": float(self.state.total_intra_inter_margin / denom),
        }

    def run(
        self,
        model: torch.nn.Module,
        dataloader,
        *,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        self.on_validation_epoch_start()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= int(max_batches):
                    break
                self.validation_step(model, batch)
            self._consume_buffer(model, flush_remainder=True)

        return self.on_validation_epoch_end()