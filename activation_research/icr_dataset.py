"""
icr_dataset.py — PyTorch Dataset wrapper for ICR-probe training data.

Three modes:

  mode="icr"
      Reads from legacy layout:
          icr_scores.npy          (N, num_layers) fp32
          meta.jsonl              one line per sample, must have "prompt_hash"
                                  and "hallucinated"
      Returns: {"hashkey": str, "halu": float32, "icr_score": Tensor (num_layers,)}

  mode="raw"
      Reads from zarr + attention.zarr layout (original activation_parser path).
      Returns: {"hashkey": str, "halu": float32,
                "response_attn": Tensor, "h_block_input": Tensor,
                "delta_h": Tensor, "response_len": int}

  mode="memmap"
      Reads from the new InferenceCaptureWriter layout (Issue #72):
          config.json             num_layers, hidden_dim, r_max, max_response_len,
                                  max_prompt_len, n_samples, dtype, ...
          meta.jsonl              one line per valid sample, must have "prompt_hash",
                                  "hallucinated", and "sample_index"
          icr_scores.npy          (n_samples, num_layers) fp32  — regular np.save
          response_activations.npy  memmap (n_samples, num_layers+1, max_response_len, hidden_dim) fp16
          response_attention.npy    memmap (n_samples, num_layers, r_max, r_max) fp16
          prompt_activations.npy    memmap (n_samples, num_layers+1, max_prompt_len, hidden_dim) fp16
          response_len.npy          (n_samples,) int32
          prompt_len.npy            (n_samples,) int32
          eval_results.json         {"halu_test_res": [...], "abstantion": [...]}
      Returns: same dict as mode="icr"

  mode="memmap-raw"
      Same layout source as mode="memmap" but returns the same dict shape as
      mode="raw" — lets ablation paths that consume (response_attn, h_block_input,
      delta_h, response_len) work against new-layout captures without reimplementing
      ICR score elsewhere.
      Returns: {"hashkey": str, "halu": float32,
                "response_attn": Tensor (num_layers, r_max, r_max),
                "h_block_input": Tensor (num_layers, hidden_dim),
                "delta_h": Tensor (num_layers, hidden_dim),
                "response_len": int}

Split logic (all modes):
    _make_split_indices(labels, split, val_fraction, random_seed) does a two-stage
    stratified split (train+val vs test, then train vs val) and returns the index
    array for the requested split.  The "val" boundary is 0.125 * (1 - test_frac)
    of total data (i.e. ~10% of total when test_frac=0.2 and val_fraction=None).
    When val_fraction is provided, that fraction of total data becomes the val set
    and the remainder of non-test data becomes train.  All four modes share this
    helper — no forked split path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared split helper
# ---------------------------------------------------------------------------

def _make_split_indices(
    labels: np.ndarray,
    split: Literal["train", "val", "test", "all"],
    val_fraction: float | None,
    random_seed: int,
) -> np.ndarray:
    """Return the integer indices (into `labels`) that belong to `split`.

    Two-stage stratified split:
      stage 1: 80 / 20  train+val vs test
      stage 2: within train+val, carve out val_fraction (of total) as val

    When val_fraction is None, val defaults to 0.125 of the 80% block
    (= 10% of total), matching the existing ActivationParser three_way logic.
    """
    all_idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=random_seed,
    )

    if split == "test":
        return test_idx

    # How much of the trainval block becomes val?
    trainval_size = len(trainval_idx)
    total_size = len(labels)
    if val_fraction is None:
        # 0.125 of trainval = 10% of total (mirrors three_way behaviour)
        val_of_trainval = 0.125
    else:
        # val_fraction is expressed as a fraction of total
        # Why: caller specifies val as % of full dataset, not % of trainval block
        val_of_total = max(val_fraction, 1 / trainval_size)
        val_of_trainval = min(val_of_total * total_size / trainval_size, 0.9)

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_of_trainval,
        stratify=labels[trainval_idx],
        random_state=random_seed + 1,
    )

    if split == "val":
        return val_idx
    return train_idx


# ---------------------------------------------------------------------------
# mode="icr"
# ---------------------------------------------------------------------------

class _ICRMode:
    """Reads icr_scores.npy + meta.jsonl from a legacy flat directory."""

    def __init__(self, capture_dir: Path):
        meta_path = capture_dir / "meta.jsonl"
        scores_path = capture_dir / "icr_scores.npy"

        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
        if not scores_path.exists():
            raise FileNotFoundError(f"icr_scores.npy not found: {scores_path}")

        meta_rows: list[dict] = []
        with meta_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    meta_rows.append(json.loads(line))

        self._icr_scores = np.load(scores_path)  # eager — small array
        self._meta = meta_rows

    def labels(self) -> np.ndarray:
        return np.array(
            [int(bool(r["hallucinated"])) for r in self._meta], dtype=np.int32
        )

    def __len__(self) -> int:
        return len(self._meta)

    def get(self, idx: int) -> dict:
        row = self._meta[idx]
        score = torch.from_numpy(self._icr_scores[idx].astype(np.float32))
        return {
            "hashkey": row["prompt_hash"],
            "halu": torch.tensor(float(bool(row["hallucinated"])), dtype=torch.float32),
            "icr_score": score,
        }


# ---------------------------------------------------------------------------
# mode="memmap"  and  mode="memmap-raw"
# ---------------------------------------------------------------------------

class _MemmapMode:
    """Reads from InferenceCaptureWriter layout (Issue #72)."""

    def __init__(self, capture_dir: Path):
        config_path = capture_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found: {config_path}")

        with config_path.open() as fh:
            cfg = json.load(fh)

        self._cfg = cfg
        n_samples: int = cfg["n_samples"]
        num_layers: int = cfg["num_layers"]
        hidden_dim: int = cfg["hidden_dim"]
        r_max: int = cfg["r_max"]
        max_resp: int = cfg["max_response_len"]
        max_prompt: int = cfg["max_prompt_len"]
        act_dtype = np.float16

        # meta.jsonl — authoritative valid-rows list
        meta_path = capture_dir / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")

        meta_rows: list[dict] = []
        with meta_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    meta_rows.append(json.loads(line))

        # icr_scores.npy — eager load (small: ~1.4 MB / 10k samples).
        # Loaded before filtering meta_rows so we can clip to the actual
        # scores array size (guards against partial captures and
        # restart-appended duplicate rows in meta.jsonl).
        scores_path = capture_dir / "icr_scores.npy"
        if not scores_path.exists():
            raise FileNotFoundError(f"icr_scores.npy not found: {scores_path}")
        self._icr_scores = np.load(scores_path)  # (n_samples, num_layers) fp32

        scores_n = self._icr_scores.shape[0]
        bad = [r for r in meta_rows if r["sample_index"] >= scores_n]
        if bad:
            import warnings
            warnings.warn(
                f"{capture_dir}: dropping {len(bad)} meta row(s) whose "
                f"sample_index >= icr_scores size ({scores_n}). "
                "The capture is incomplete for these samples.",
                UserWarning,
                stacklevel=4,
            )
            meta_rows = [r for r in meta_rows if r["sample_index"] < scores_n]

        # Why: only rows listed in meta.jsonl were fully committed. Pre-allocated
        # rows beyond meta.jsonl's sample_indices may be zero-filled garbage from
        # a partial write or pre-allocation; they must never be reachable.
        valid_sample_indices = np.array(
            [r["sample_index"] for r in meta_rows], dtype=np.int64
        )
        self._meta = meta_rows
        self._valid_sample_indices = valid_sample_indices

        # Large arrays — keep as memmap (no full RAM load)
        def _mm(name: str, shape: tuple, dtype) -> np.memmap:
            p = capture_dir / name
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")
            return np.memmap(str(p), dtype=dtype, mode="r", shape=shape)

        self._resp_act = _mm(
            "response_activations.npy",
            (n_samples, num_layers + 1, max_resp, hidden_dim),
            act_dtype,
        )
        self._resp_attn = _mm(
            "response_attention.npy",
            (n_samples, num_layers, r_max, r_max),
            act_dtype,
        )
        self._prompt_act = _mm(
            "prompt_activations.npy",
            (n_samples, num_layers + 1, max_prompt, hidden_dim),
            act_dtype,
        )
        self._resp_len = _mm("response_len.npy", (n_samples,), np.int32)
        self._prompt_len = _mm("prompt_len.npy", (n_samples,), np.int32)

        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

    def labels(self) -> np.ndarray:
        return np.array(
            [int(bool(r["hallucinated"])) for r in self._meta], dtype=np.int32
        )

    def __len__(self) -> int:
        return len(self._meta)

    def get_icr(self, idx: int) -> dict:
        """Return mode="icr" compatible dict."""
        row = self._meta[idx]
        si = self._valid_sample_indices[idx]
        score = torch.from_numpy(self._icr_scores[si].astype(np.float32))
        return {
            "hashkey": row["prompt_hash"],
            "halu": torch.tensor(float(bool(row["hallucinated"])), dtype=torch.float32),
            "icr_score": score,
        }

    def get_raw(self, idx: int) -> dict:
        """Return mode="raw" compatible dict sourced from memmap arrays."""
        row = self._meta[idx]
        si = self._valid_sample_indices[idx]
        resp_len = int(self._resp_len[si])

        # response_attn: (num_layers, r_max, r_max) fp32
        resp_attn = torch.from_numpy(
            self._resp_attn[si].astype(np.float32)
        )

        # h_block_input: input hidden states to each transformer block.
        # Layer 0 is the embedding; layers 1..num_layers are block inputs.
        # Shape: (num_layers, hidden_dim), averaging over response tokens
        # Why: mode="raw" consumers use per-sample averaged hidden states for
        # h_block_input; we follow the same convention here.
        resp_act = self._resp_act[si]  # (num_layers+1, max_resp, hidden_dim) fp16
        # Use first valid response token's representation per layer
        h_block_input = torch.from_numpy(
            resp_act[:-1, 0, :].astype(np.float32)
        )  # (num_layers, hidden_dim)

        # delta_h: residual update per block = output_hs - input_hs
        # Shape: (num_layers, hidden_dim) at first response token position
        delta_h = torch.from_numpy(
            (resp_act[1:, 0, :] - resp_act[:-1, 0, :]).astype(np.float32)
        )  # (num_layers, hidden_dim)

        return {
            "hashkey": row["prompt_hash"],
            "halu": torch.tensor(float(bool(row["hallucinated"])), dtype=torch.float32),
            "response_attn": resp_attn,
            "h_block_input": h_block_input,
            "delta_h": delta_h,
            "response_len": resp_len,
        }


# ---------------------------------------------------------------------------
# mode="raw"  (zarr-backed — thin wrapper to keep interface consistent)
# ---------------------------------------------------------------------------

class _RawMode:
    """Placeholder — zarr-backed raw mode. Raises NotImplementedError until
    the full zarr reader is wired; exists so mode="raw" appears in the type
    hierarchy and future implementers have a defined slot.

    Why separate class: keeps _MemmapMode clean and avoids importing zarr
    unless mode="raw" is actually requested.
    """

    def __init__(self, capture_dir: Path):
        raise NotImplementedError(
            "mode='raw' (zarr backend) is not yet implemented in ICRDataset. "
            "Use mode='memmap-raw' against a new-layout capture dir."
        )

    def labels(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def get(self, idx: int) -> dict:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Public Dataset class
# ---------------------------------------------------------------------------

class ICRDataset(Dataset):
    """PyTorch Dataset for ICR probe training and evaluation.

    Parameters
    ----------
    capture_dir : str or Path
        Directory containing the data files (layout depends on mode).
    mode : {"icr", "memmap", "memmap-raw", "raw"}
        Read mode.  "icr" and "memmap" return the same dict shape
        (interchangeable for the probe).  "memmap-raw" and "raw" return the
        raw-activation dict shape (for ablations).
    split : {"train", "val", "test", "all"}
        Which partition to expose.  "all" bypasses the stratified split and
        exposes every sample in the capture directory — intended for the
        test-cell case where a separate capture directory holds only test
        samples and no internal split is needed.
    val_fraction : float or None
        If provided, fraction of *total* samples reserved for val.
        None uses the ActivationParser three_way default (~10% of total).
        Ignored when split="all".
    random_seed : int
        RNG seed for the stratified split.  Ignored when split="all".
    """

    def __init__(
        self,
        capture_dir: str | Path,
        mode: Literal["icr", "memmap", "memmap-raw", "raw"] = "icr",
        split: Literal["train", "val", "test", "all"] = "train",
        val_fraction: float | None = None,
        random_seed: int = 42,
    ) -> None:
        capture_dir = Path(capture_dir)
        if not capture_dir.exists():
            raise FileNotFoundError(f"capture_dir not found: {capture_dir}")

        if mode == "icr":
            self._backend = _ICRMode(capture_dir)
            self._get_item = self._backend.get
        elif mode == "memmap":
            self._backend = _MemmapMode(capture_dir)
            self._get_item = self._backend.get_icr
        elif mode == "memmap-raw":
            self._backend = _MemmapMode(capture_dir)
            self._get_item = self._backend.get_raw
        elif mode == "raw":
            self._backend = _RawMode(capture_dir)  # raises immediately
            self._get_item = self._backend.get  # unreachable
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. "
                "Valid modes: 'icr', 'memmap', 'memmap-raw', 'raw'."
            )

        self._mode = mode
        labels = self._backend.labels()

        if split == "all":
            # Bypass stratified split — expose every sample.  Intended for a
            # separate test-cell capture directory where all samples are test
            # data and an internal train/val/test split is not meaningful.
            self._split_indices = np.arange(len(labels), dtype=np.int64)
        else:
            self._split_indices = _make_split_indices(
                labels, split, val_fraction, random_seed
            )

    def __len__(self) -> int:
        return len(self._split_indices)

    def __getitem__(self, idx: int) -> dict:
        backend_idx = int(self._split_indices[idx])
        return self._get_item(backend_idx)
