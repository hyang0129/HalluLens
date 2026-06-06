"""
contrastive_actvit_dataset.py — grid-view dataset for Contrastive ACT-ViT (#134).

Returns, per sample, ``num_views`` augmented copies of the full
``(L × N × D)`` activation grid, each **flattened to a sequence** ``(L*N, D)``
so the existing 4-D contrastive trainer / evaluator can consume them unchanged
(:class:`activation_research.act_vit.ContrastiveACTViT` reshapes back to the grid
internally). Emits the same dict keys the SupCon+recon pipeline expects:
``views_activations`` (num_views, L*N, D), ``halu``, ``hashkey``, ``logprob``.

The four view-augmentation schemes (issue #134):
    "noise"      CAV-0 — additive Gaussian (independent per view); no structural crop.
    "token_crop" CAV-1 — keep a random ~``keep_frac`` of token columns.
    "layer_band" CAV-2 — keep a contiguous random ~``keep_frac`` band of layers.
    "patch_mask" CAV-3 — mask ~``mask_frac`` of (patch_h × patch_w) layer×token patches.

Masked cells are filled with the per-sample minimum so the model's adaptive
max-pool ignores them (rather than letting injected zeros win the pool).

Capture layout (icr_capture dir):
    config.json                  {n_samples, num_layers, max_response_len, hidden_dim}
    response_activations.npy     memmap (N, num_layers+1, max_response_len, hidden_dim) fp16
    response_token_logprobs.npy  memmap (N, max_response_len) float32 (NaN-padded)
    response_len.npy             memmap (N,) int32
    meta.jsonl                   one JSON/line with "hallucinated": bool
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveACTViTDataset(Dataset):
    def __init__(
        self,
        capture_dir: str | Path,
        indices: Sequence[int],
        *,
        relevant_layers: Optional[Sequence[int]] = None,
        num_views: int = 2,
        view_aug: str = "noise",
        keep_frac: float = 0.6,
        mask_frac: float = 0.5,
        noise_scale: float = 0.1,
        patch_h: int = 2,
        patch_w: int = 10,
        include_logprob: bool = True,
        hashkey_map: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        self._dir = Path(capture_dir)
        with (self._dir / "config.json").open() as fh:
            cap = json.load(fh)
        n_samples = int(cap["n_samples"])
        n_layers_plus1 = int(cap["num_layers"]) + 1
        self.max_response_len = int(cap["max_response_len"])
        self.hidden_dim = int(cap["hidden_dim"])

        self._acts = np.memmap(
            str(self._dir / "response_activations.npy"), dtype=np.float16, mode="r",
            shape=(n_samples, n_layers_plus1, self.max_response_len, self.hidden_dim),
        )
        self._resp_lens = np.memmap(
            str(self._dir / "response_len.npy"), dtype=np.int32, mode="r", shape=(n_samples,),
        )
        self._logprob = None
        lp_path = self._dir / "response_token_logprobs.npy"
        if include_logprob and lp_path.exists():
            self._logprob = np.memmap(
                str(lp_path), dtype=np.float32, mode="r",
                shape=(n_samples, self.max_response_len),
            )

        labels: List[int] = []
        with (self._dir / "meta.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    labels.append(int(bool(json.loads(line)["hallucinated"])))
        self._labels = np.array(labels, dtype=np.int32)

        # Transformer layers are 1..num_layers (drop embedding layer 0).
        n_layers_total = n_layers_plus1 - 1
        if relevant_layers is None:
            self._layer_rows = list(range(1, n_layers_plus1))           # all transformer layers
        else:
            # relevant_layers are model-layer ids; +1 to index past the embedding row.
            self._layer_rows = [int(l) + 1 for l in relevant_layers]
        self.n_layers = len(self._layer_rows)
        self.n_tokens = self.max_response_len

        eff = len(self._labels)
        idx = np.asarray(indices, dtype=np.int64)
        self._indices = idx[idx < eff]

        self.num_views = int(num_views)
        self.view_aug = str(view_aug).lower()
        self.keep_frac = float(keep_frac)
        self.mask_frac = float(mask_frac)
        self.noise_scale = float(noise_scale)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self._rng = random.Random(int(seed))
        # Map sample_index -> prompt_hash so eval embeddings align with the
        # parser df (the evaluator looks up labels by df['prompt_hash'] == hashkey).
        self._hashkey_map = {int(k): str(v) for k, v in hashkey_map.items()} if hashkey_map else None
        _ = n_layers_total  # documented; layer selection handled above

    def __len__(self) -> int:
        return len(self._indices)

    def _augment(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (L, N, D) float32 → one augmented view (L, N, D) float32."""
        L, N, D = grid.shape
        aug = self.view_aug
        if aug == "noise":
            std = float(grid.std()) + 1e-6
            return grid + torch.randn_like(grid) * (self.noise_scale * std)
        fill = float(grid.min())
        if aug == "token_crop":
            k = max(1, int(round(self.keep_frac * N)))
            keep = self._rng.sample(range(N), k)
            mask = torch.zeros(N, dtype=torch.bool)
            mask[keep] = True
            out = grid.clone()
            out[:, ~mask, :] = fill
            return out
        if aug == "layer_band":
            band = max(1, int(round(self.keep_frac * L)))
            start = self._rng.randint(0, max(0, L - band))
            out = torch.full_like(grid, fill)
            out[start:start + band] = grid[start:start + band]
            return out
        if aug == "patch_mask":
            pH, pW = self.patch_h, self.patch_w
            nH = (L + pH - 1) // pH
            nW = (N + pW - 1) // pW
            out = grid.clone()
            for ih in range(nH):
                for iw in range(nW):
                    if self._rng.random() < self.mask_frac:
                        out[ih * pH:(ih + 1) * pH, iw * pW:(iw + 1) * pW, :] = fill
            return out
        raise ValueError(f"unknown view_aug: {self.view_aug}")

    def __getitem__(self, i: int) -> dict:
        idx = int(self._indices[i])
        # (n_layers, N, D) float32 grid for the selected transformer layers.
        grid = torch.from_numpy(
            np.array(self._acts[idx, self._layer_rows, :, :], dtype=np.float32)
        )
        views = []
        for _ in range(self.num_views):
            v = self._augment(grid)                          # (L, N, D)
            views.append(v.reshape(self.n_layers * self.n_tokens, self.hidden_dim).half())
        hashkey = self._hashkey_map.get(idx, str(idx)) if self._hashkey_map else str(idx)
        out = {
            "views_activations": torch.stack(views, dim=0),  # (num_views, L*N, D) fp16
            "halu": torch.tensor(int(self._labels[idx]), dtype=torch.long),
            "hashkey": hashkey,
        }
        if self._logprob is not None:
            out["logprob"] = torch.from_numpy(
                np.array(self._logprob[idx], dtype=np.float32)
            )                                                # (max_response_len,)
        return out
