"""
act_vit_dataset.py — Memmap-backed Dataset for ACT-ViT.

Reads directly from an icr_capture directory (produced by
InferenceCaptureWriter) without going through MemmapContrastiveDataset.
Returns full (L, N, D) activation tensors suitable for ACTViT.forward().

Capture layout expected:
    config.json           — {"n_samples": N, "num_layers": L, "max_response_len": R, "hidden_dim": D, ...}
    response_activations.npy — memmap (N, num_layers+1, max_response_len, hidden_dim) fp16
    response_len.npy      — memmap (N,) int32
    meta.jsonl            — one JSON obj per sample; field "hallucinated": bool
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ACTViTDataset(Dataset):
    """Dataset returning full activation tensors (L, N, D) per sample.

    Parameters
    ----------
    capture_dir : str | Path
        Path to an icr_capture directory.
    indices : array-like of int
        Row indices into the capture to expose (allows train/val/test slicing).
    cfg : dict
        Optional overrides; currently unused — shape info is read from
        config.json in the capture directory.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        indices: Sequence[int],
        *,
        cfg: dict | None = None,
    ) -> None:
        self._capture_dir = Path(capture_dir)

        # Load capture config for shape info.
        with (self._capture_dir / "config.json").open() as fh:
            cap_cfg: dict = json.load(fh)

        n_samples: int = cap_cfg["n_samples"]
        # response_activations.npy has shape (N, num_layers+1, max_response_len, hidden_dim).
        # The +1 includes the embedding layer (layer 0); transformer layers are 1..num_layers.
        # ACT-ViT uses transformer layers only (layers 1..num_layers), so we drop layer 0.
        num_layers_plus1: int = cap_cfg["num_layers"] + 1
        max_response_len: int = cap_cfg["max_response_len"]
        hidden_dim: int = cap_cfg["hidden_dim"]

        # Open memmap arrays.
        self._acts = np.memmap(
            str(self._capture_dir / "response_activations.npy"),
            dtype=np.float16,
            mode="r",
            shape=(n_samples, num_layers_plus1, max_response_len, hidden_dim),
        )
        self._resp_lens = np.memmap(
            str(self._capture_dir / "response_len.npy"),
            dtype=np.int32,
            mode="r",
            shape=(n_samples,),
        )

        # Load labels from meta.jsonl.
        labels: List[int] = []
        with (self._capture_dir / "meta.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    labels.append(int(bool(obj["hallucinated"])))
        self._labels = np.array(labels, dtype=np.int32)

        # Store the index subset.
        self._indices = np.asarray(indices, dtype=np.int64)

        # Record shape metadata (dropping embedding layer → L transformer layers).
        self.num_layers = num_layers_plus1 - 1
        self.max_response_len = max_response_len
        self.hidden_dim = hidden_dim

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict:
        idx = int(self._indices[i])

        # Load activation for this sample; shape (num_layers+1, R, D) fp16.
        # Drop the embedding layer (index 0) → (num_layers, R, D).
        act_raw = self._acts[idx]                  # (num_layers+1, R, D) fp16  — numpy memmap slice
        act = np.array(act_raw[1:], dtype=np.float16)  # drop embed layer first, materialize off memmap, stay fp16
        act_t = torch.from_numpy(act)              # (L, R, D) fp16 — ACTViT.forward casts to fp32 on GPU

        label = int(self._labels[idx])
        resp_len = int(self._resp_lens[idx])

        return {
            "activations": act_t,       # Tensor (L, R, D) float16
            "label": label,              # int 0/1 (1 = hallucinated)
            "response_len": resp_len,    # int
        }
