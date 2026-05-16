"""
attention_memmap_writer.py

Writer class for the numpy-memmap attention store produced by the fused
Stage-1+2 recomputation pipeline (Issue #69, Wave 4).  Stores head-averaged
response-to-response attention sub-blocks alongside per-sample metadata, with
resume semantics consistent with the existing zarr infrastructure.

Storage layout written by this class:

    <out_dir>/
      response_attn.npy   np.memmap, shape (N, num_layers, R_max, R_max), float16
                          Pre-allocated zero-filled at creation; unwritten rows
                          remain zero.  The file is NOT sparse — all N×… bytes
                          are present on disk from the moment the writer is
                          constructed in mode='w'.
      meta.jsonl          One JSON object per *written* sample (append-only):
                            {"key": "...", "sample_index": int,
                             "response_len": int, "prompt_len": int}
                          This file is the authoritative record of what is
                          written.  response_attn.npy is bulk data only.
      config.json         Verbatim config_dict provided at init, plus
                          storage_format, n_samples, num_layers, r_max, dtype.

This writer does NOT accept torch.Tensor inputs — tensor→numpy conversion
must happen at the call site in recompute_attention.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class AttentionMemmapWriter:
    """Incremental writer for numpy-memmap attention stores.

    Stores head-averaged response-to-response attention sub-blocks (one row per
    sample) with resume semantics: re-opening an existing store in mode='a'
    skips samples that are already recorded in meta.jsonl.

    ``write()`` takes an explicit ``sample_index`` (the row in the pre-allocated
    memmap to write to) rather than auto-incrementing.  This makes it possible
    to resume a partially-complete run at any sample position without disturbing
    rows that were already written.

    NOTE: ``response_attn`` must be a numpy ndarray.  torch.Tensor is not
    accepted; convert to numpy before calling this class.

    Args:
        out_dir: Directory that will contain response_attn.npy, meta.jsonl, and
                 config.json.  Created (with parents) if it does not exist.
        mode: ``'w'`` to create/overwrite, ``'a'`` to append/resume.
        n_samples: Total number of samples the memmap is pre-allocated for.
                   Required at creation; must match the stored value on resume.
        num_layers: Number of transformer blocks stored (e.g. 32 for Llama-3.1-8B).
        r_max: Maximum response length padded into storage (default 64).
        config_dict: Written verbatim (plus storage metadata) as config.json on
                     creation; verified field-by-field on resume.
        dtype: Stored dtype for response_attn (default ``"float16"``).
    """

    def __init__(
        self,
        out_dir: str,
        mode: Literal["w", "a"],
        n_samples: int,
        num_layers: int,
        r_max: int = 64,
        config_dict: Optional[dict] = None,
        dtype: str = "float16",
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

        self._out_dir = Path(out_dir)
        self._mode = mode
        self._n_samples = n_samples
        self._num_layers = num_layers
        self._r_max = r_max
        self._user_config: dict = config_dict or {}
        self._dtype = np.dtype(dtype)
        self._finalized = False

        self._npy_path = self._out_dir / "response_attn.npy"
        self._meta_path = self._out_dir / "meta.jsonl"
        self._config_path = self._out_dir / "config.json"

        self._written_keys: set[str] = set()
        self._written_indices: set[int] = set()
        self._mm: Optional[np.memmap] = None

        if mode == "w":
            self._init_write()
        else:
            self._init_append()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _full_config(self) -> dict:
        """Return the config dict that is written to / verified against disk."""
        cfg = dict(self._user_config)
        cfg.update(
            storage_format="numpy_memmap_v1",
            n_samples=self._n_samples,
            num_layers=self._num_layers,
            r_max=self._r_max,
            dtype=str(self._dtype),
        )
        return cfg

    def _write_config(self) -> None:
        self._config_path.write_text(
            json.dumps(self._full_config(), indent=2), encoding="utf-8"
        )

    def _init_write(self) -> None:
        """Create out_dir, pre-allocate memmap, write empty meta.jsonl + config."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        shape = (self._n_samples, self._num_layers, self._r_max, self._r_max)
        # Pre-allocate; np.memmap 'w+' creates the file and fills with zeros.
        self._mm = np.memmap(
            str(self._npy_path), dtype=self._dtype, mode="w+", shape=shape
        )
        self._meta_path.write_text("", encoding="utf-8")
        self._write_config()

    def _init_append(self) -> None:
        """Open an existing store for resume; rebuild written-key sets from meta.jsonl."""
        if not self._config_path.exists():
            raise ValueError(
                f"config.json not found at {self._config_path}. "
                "Cannot resume a store that was not initialised with mode='w'."
            )
        stored = json.loads(self._config_path.read_text(encoding="utf-8"))

        # Structural fields must match exactly.
        for field in ("storage_format", "n_samples", "num_layers", "r_max", "dtype"):
            if stored.get(field) != self._full_config().get(field):
                raise ValueError(
                    f"config.json field {field!r} mismatch: "
                    f"stored={stored.get(field)!r}, "
                    f"expected={self._full_config().get(field)!r}"
                )

        if not self._npy_path.exists():
            raise ValueError(
                f"response_attn.npy not found at {self._npy_path}. "
                "Cannot resume a store whose memmap file is missing."
            )

        shape = (self._n_samples, self._num_layers, self._r_max, self._r_max)
        self._mm = np.memmap(
            str(self._npy_path), dtype=self._dtype, mode="r+", shape=shape
        )

        # Rebuild written-key and written-index sets from the authoritative log.
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = entry.get("key")
                        idx = entry.get("sample_index")
                        if key is not None:
                            self._written_keys.add(str(key))
                        if idx is not None:
                            self._written_indices.add(int(idx))
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_written(self, sample_key: str) -> bool:
        """Return True if this sample key has already been written (O(1))."""
        return sample_key in self._written_keys

    def write(
        self,
        sample_index: int,
        sample_key: str,
        response_attn: np.ndarray,
        response_len: int,
        prompt_len: int,
    ) -> None:
        """Write one sample's attention block to the pre-allocated memmap.

        Args:
            sample_index: Row index in the memmap to write to.  Must match the
                          source activations.zarr row index so that alignment
                          between response_attn.npy and icr_scores.npy is exact.
            sample_key: Unique string identifier for this sample.
            response_attn: numpy ndarray, shape (num_layers, r_max, r_max).
                           Will be cast to float16 and padded/truncated to match
                           the stored shape.  torch.Tensor is not accepted.
            response_len: Actual response length (used by readers to mask padding).
            prompt_len: Prompt length (stored for projection-target indexing).

        Raises:
            IndexError: If sample_index is outside [0, n_samples).
            ValueError: If sample_index is already mapped to a *different* key
                        (would clobber an existing row).
        """
        if not (0 <= sample_index < self._n_samples):
            raise IndexError(
                f"sample_index {sample_index} out of range [0, {self._n_samples})"
            )

        # Idempotent on exact retry.
        if sample_key in self._written_keys:
            return

        # Protect against index collisions from different keys.
        if sample_index in self._written_indices:
            raise ValueError(
                f"sample_index {sample_index} is already mapped to a different "
                f"sample key; cannot write key {sample_key!r} at this index."
            )

        arr = np.asarray(response_attn).astype(self._dtype, copy=False)

        target_shape = (self._num_layers, self._r_max, self._r_max)
        if arr.shape != target_shape:
            padded = np.zeros(target_shape, dtype=self._dtype)
            l_in = min(arr.shape[0], self._num_layers)
            r_in = min(arr.shape[1], self._r_max) if arr.ndim > 1 else 0
            c_in = min(arr.shape[2], self._r_max) if arr.ndim > 2 else 0
            padded[:l_in, :r_in, :c_in] = arr[:l_in, :r_in, :c_in]
            arr = padded

        self._mm[sample_index] = arr

        # Crash-safe append: open, write, flush, fsync, close on every call.
        # This guarantees meta.jsonl is durable even if the process dies
        # mid-run; the memmap flush is best-effort (OS page-cache), but the
        # log entry is the authoritative "was written" signal.
        record = json.dumps(
            {
                "key": sample_key,
                "sample_index": sample_index,
                "response_len": int(response_len),
                "prompt_len": int(prompt_len),
            }
        )
        with open(self._meta_path, "a", encoding="utf-8") as fh:
            fh.write(record + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        self._written_keys.add(sample_key)
        self._written_indices.add(sample_index)

    def finalize(self) -> None:
        """Flush the memmap and confirm completion by rewriting config.json.

        Safe to call multiple times.  Does not truncate the pre-allocated
        memmap (callers can read unwritten rows; they will be zero-filled and
        absent from meta.jsonl).
        """
        if self._finalized:
            return
        if self._mm is not None:
            self._mm.flush()
        self._write_config()
        self._finalized = True

    def __len__(self) -> int:
        """Return the number of samples recorded in meta.jsonl."""
        return len(self._written_keys)

    def __enter__(self) -> "AttentionMemmapWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.finalize()
