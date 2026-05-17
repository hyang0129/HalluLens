"""Tests for ICRProbeTrainer: fit / checkpoint / early stopping.

All tests use synthetic in-memory datasets — no GPU, no real capture data.

Run with:
    pytest tests/test_icr_trainer.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from activation_research.icr_probe import ICRProbe
from activation_research.icr_trainer import ICRProbeTrainer, ICRProbeTrainerConfig


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class _SyntheticICRDataset(Dataset):
    """In-memory dataset mimicking ICRDataset's return dict shape.

    Returns {"hashkey": str, "halu": float32 Tensor, "icr_score": Tensor (L,)}.
    """

    def __init__(
        self,
        n: int,
        L: int,
        *,
        separable: bool = False,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n).astype(np.float32)
        if separable:
            scores = rng.standard_normal((n, L)).astype(np.float32)
            scores[labels == 1] += 3.0   # class 1 shifted up → linearly separable
        else:
            scores = rng.standard_normal((n, L)).astype(np.float32)

        self._scores = torch.from_numpy(scores)
        self._labels = torch.from_numpy(labels)
        self._n = n
        self._L = L

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "hashkey": f"fake_{idx}",
            "halu": self._labels[idx],
            "icr_score": self._scores[idx],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(
    tmp_path: Path,
    L: int = 16,
    max_epochs: int = 2,
    batch_size: int = 32,
    early_stop_patience: int = 100,
    num_workers: int = 0,
) -> tuple[ICRProbeTrainer, ICRProbe]:
    model = ICRProbe(input_dim=L)
    config = ICRProbeTrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        lr=1e-3,
        weight_decay=0.0,
        plateau_patience=3,
        plateau_factor=0.5,
        early_stop_patience=early_stop_patience,
        device="cpu",
        num_workers=num_workers,
        persistent_workers=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every=1,
    )
    trainer = ICRProbeTrainer(model, config=config)
    return trainer, model


# ---------------------------------------------------------------------------
# 1. Single fit run (sanity)
# ---------------------------------------------------------------------------

def test_trainer_runs_two_epochs_on_synthetic_dataset(tmp_path):
    """Synthetic 100-sample dataset → trainer.fit completes 2 epochs without error."""
    L = 16
    train_ds = _SyntheticICRDataset(100, L, seed=0)
    val_ds = _SyntheticICRDataset(30, L, seed=1)

    trainer, model = _make_trainer(tmp_path, L=L, max_epochs=2)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    # Model should still be usable after fit.
    model.eval()
    x = torch.randn(4, L)
    out = model(x)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# 2. Checkpoint filename
# ---------------------------------------------------------------------------

def test_checkpoint_filename_is_linear_probe_last_pt(tmp_path):
    """Per plan §B.2 / spec §10: checkpoint filename must be 'linear_probe_last.pt'.

    This ensures transfer_eval.py and aggregate loaders work without changes.
    """
    L = 16
    train_ds = _SyntheticICRDataset(100, L, seed=0)
    val_ds = _SyntheticICRDataset(30, L, seed=1)

    trainer, _ = _make_trainer(tmp_path, L=L, max_epochs=1)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    checkpoint_dir = Path(tmp_path) / "checkpoints"
    checkpoint_path = checkpoint_dir / "linear_probe_last.pt"
    assert checkpoint_path.exists(), (
        f"Expected checkpoint at {checkpoint_path}, but it does not exist.\n"
        f"Files in checkpoint_dir: {list(checkpoint_dir.iterdir())}"
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    assert "model_state_dict" in ckpt, "Checkpoint missing 'model_state_dict'"
    assert "optimizer_state_dict" in ckpt, "Checkpoint missing 'optimizer_state_dict'"
    assert "epoch" in ckpt, "Checkpoint missing 'epoch'"


def test_checkpoint_contains_auroc_metrics(tmp_path):
    """Checkpoint should include val_auroc and best_auroc from validation."""
    L = 16
    train_ds = _SyntheticICRDataset(200, L, separable=True, seed=0)
    val_ds = _SyntheticICRDataset(60, L, separable=True, seed=1)

    trainer, _ = _make_trainer(tmp_path, L=L, max_epochs=2)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    ckpt = torch.load(
        str(Path(tmp_path) / "checkpoints" / "linear_probe_last.pt"),
        map_location="cpu",
    )
    assert "val_auroc" in ckpt, "Checkpoint missing 'val_auroc'"
    assert "best_auroc" in ckpt, "Checkpoint missing 'best_auroc'"


# ---------------------------------------------------------------------------
# 3. Early stopping
# ---------------------------------------------------------------------------

def test_early_stop_triggers_before_max_epochs(tmp_path):
    """Force val AUROC to flat-line → fit exits before max_epochs.

    We set early_stop_patience=2 and max_epochs=20.  With a random dataset
    (no signal), AUROC should not improve consistently and early stopping
    should trigger well before epoch 20.
    """
    L = 16
    # Random data: no learnable signal — AUROC hovers around 0.5.
    train_ds = _SyntheticICRDataset(200, L, separable=False, seed=42)
    val_ds = _SyntheticICRDataset(60, L, separable=False, seed=99)

    trainer, model = _make_trainer(
        tmp_path,
        L=L,
        max_epochs=20,
        batch_size=32,
        early_stop_patience=2,
    )
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    # trainer._epochs_since_best >= early_stop_patience means it stopped early.
    # We assert that we did NOT run all 20 epochs.
    # start_epoch is never updated in a non-resumed run; check epoch count via
    # checkpoint instead.
    ckpt = torch.load(
        str(Path(tmp_path) / "checkpoints" / "linear_probe_last.pt"),
        map_location="cpu",
    )
    last_epoch = int(ckpt["epoch"])
    assert last_epoch < 19, (
        f"Expected early stop before epoch 19, but ran to epoch {last_epoch}"
    )


# ---------------------------------------------------------------------------
# 4. Val AUROC improves on separable data
# ---------------------------------------------------------------------------

def test_val_auroc_improves_on_separable_data(tmp_path):
    """On linearly-separable data, best_auroc after fit should exceed 0.80."""
    L = 16
    train_ds = _SyntheticICRDataset(400, L, separable=True, seed=0)
    val_ds = _SyntheticICRDataset(100, L, separable=True, seed=1)

    trainer, _ = _make_trainer(
        tmp_path, L=L, max_epochs=15, early_stop_patience=100
    )
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    assert trainer.best_auroc > 0.80, (
        f"Expected best_auroc > 0.80 on separable data, got {trainer.best_auroc:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Training step returns finite loss and acc in [0, 1]
# ---------------------------------------------------------------------------

def test_training_step_returns_finite_metrics(tmp_path):
    """Single training_step returns finite loss and acc in [0, 1]."""
    L = 16
    trainer, model = _make_trainer(tmp_path, L=L)
    model.train()
    model.to(trainer.device)

    batch = {
        "icr_score": torch.randn(8, L),
        "halu": torch.randint(0, 2, (8,)).float(),
    }
    loss, metrics = trainer.training_step(batch)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert 0.0 <= metrics["acc"] <= 1.0, f"Acc out of range: {metrics['acc']}"
