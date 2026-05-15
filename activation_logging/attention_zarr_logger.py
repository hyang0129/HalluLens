"""
attention_zarr_logger.py

Writer class for attention.zarr stores produced by the attention recomputation
pipeline (Issue #69).  Stores head-averaged response-to-response attention
sub-blocks alongside per-sample metadata, with resume semantics consistent
with the existing activation zarr infrastructure.

Zarr layout written by this class:
    attention.zarr/
      arrays/
        response_attn   (N, num_layers, r_max, r_max)  float16  zstd-compressed
        sample_key      (N,)                            |S64
        response_len    (N,)                            int32
        prompt_len      (N,)                            int32
      meta/
        index.jsonl     one JSON object per sample
        config.json     verbatim copy of config_dict provided at init
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import zarr

# Import from compression.py to honour the spec's sourcing requirement.
# ZSTD_AVAILABLE is used to decide whether to apply chunk-level compression.
from .compression import ZstdCompression, ZSTD_AVAILABLE  # noqa: F401

try:
    import numcodecs
    _NUMCODECS_AVAILABLE = True
except ImportError:
    _NUMCODECS_AVAILABLE = False


def _make_zstd_compressor() -> Optional[object]:
    """Return a numcodecs Zstd compressor if available, else None."""
    if ZSTD_AVAILABLE and _NUMCODECS_AVAILABLE:
        return numcodecs.Zstd(level=19)
    return None


class AttentionZarrLogger:
    """Incremental writer for attention.zarr stores.

    Stores head-averaged response-to-response attention sub-blocks (one per
    sample per layer) with resume semantics: re-opening an existing store in
    mode='a' skips samples that are already written.

    Args:
        zarr_path: Path to the attention.zarr directory to create or open.
        mode: ``'w'`` to create/overwrite, ``'a'`` to append/resume.
        num_layers: Number of transformer blocks stored (excludes the embedding
                    layer; e.g. 32 for Llama-3.1-8B).
        r_max: Maximum response length padded into storage (default 64).
        config_dict: Written verbatim as ``meta/config.json`` on creation;
                     verified against the stored file on resume.
        expected_samples: Pre-allocate arrays for this many samples to avoid
                          repeated O(n) zarr resizes.
        dtype: Stored dtype for response_attn (default ``"float16"``).
    """

    def __init__(
        self,
        zarr_path: str,
        mode: str,
        num_layers: int,
        r_max: int = 64,
        config_dict: Optional[Dict] = None,
        expected_samples: Optional[int] = None,
        dtype: str = "float16",
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

        self._zarr_path = Path(zarr_path)
        self._mode = mode
        self._num_layers = num_layers
        self._r_max = r_max
        self._config_dict: Dict = config_dict or {}
        self._expected_samples = expected_samples
        self._dtype = np.dtype(dtype)
        self._finalized = False

        self._meta_dir = self._zarr_path / "meta"
        self._config_path = self._meta_dir / "config.json"
        self._index_path = self._meta_dir / "index.jsonl"

        # Open zarr store FIRST — mode='w' clears the directory, so meta/ must
        # be created afterwards.
        self._root = zarr.open_group(str(self._zarr_path), mode=mode)
        self._arrays_group = self._root.require_group("arrays")
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        self._response_attn: Optional[zarr.Array] = None
        self._sample_key_arr: Optional[zarr.Array] = None
        self._response_len_arr: Optional[zarr.Array] = None
        self._prompt_len_arr: Optional[zarr.Array] = None

        self._written_keys: set[str] = set()
        self._next_idx: int = 0

        if mode == "w":
            self._write_config()
            init_n = expected_samples or 0
            self._create_arrays(init_n)
        else:  # mode == "a"
            self._verify_config()
            self._load_existing()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _write_config(self) -> None:
        self._config_path.write_text(
            json.dumps(self._config_dict, indent=2), encoding="utf-8"
        )

    def _verify_config(self) -> None:
        if not self._config_path.exists():
            raise ValueError(
                f"config.json not found at {self._config_path}. "
                "Cannot resume a store that was not initialised with mode='w'."
            )
        stored = json.loads(self._config_path.read_text(encoding="utf-8"))
        if stored != self._config_dict:
            raise ValueError(
                f"config.json mismatch between stored and provided config.\n"
                f"Stored:   {stored}\n"
                f"Provided: {self._config_dict}"
            )

    # ------------------------------------------------------------------
    # Array creation / loading
    # ------------------------------------------------------------------

    def _create_arrays(self, n: int) -> None:
        compressor = _make_zstd_compressor()

        self._response_attn = self._arrays_group.require_dataset(
            "response_attn",
            shape=(n, self._num_layers, self._r_max, self._r_max),
            chunks=(1, 1, self._r_max, self._r_max),
            dtype=self._dtype,
            fill_value=0,
            compressor=compressor,
            overwrite=True,
        )
        self._sample_key_arr = self._arrays_group.require_dataset(
            "sample_key",
            shape=(n,),
            chunks=(max(1, min(n, 4096)),) if n > 0 else (4096,),
            dtype="|S64",
            fill_value=b"",
            compressor=None,
            overwrite=True,
        )
        self._response_len_arr = self._arrays_group.require_dataset(
            "response_len",
            shape=(n,),
            chunks=(max(1, min(n, 4096)),) if n > 0 else (4096,),
            dtype=np.int32,
            fill_value=0,
            compressor=None,
            overwrite=True,
        )
        self._prompt_len_arr = self._arrays_group.require_dataset(
            "prompt_len",
            shape=(n,),
            chunks=(max(1, min(n, 4096)),) if n > 0 else (4096,),
            dtype=np.int32,
            fill_value=0,
            compressor=None,
            overwrite=True,
        )

    def _load_existing(self) -> None:
        """Load existing arrays and rebuild the written-key set from index.jsonl."""
        if "response_attn" in self._arrays_group:
            self._response_attn = self._arrays_group["response_attn"]
            self._sample_key_arr = self._arrays_group["sample_key"]
            self._response_len_arr = self._arrays_group["response_len"]
            self._prompt_len_arr = self._arrays_group["prompt_len"]

        # Build written key set from index.jsonl (authoritative record).
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = entry.get("key")
                        if key:
                            self._written_keys.add(str(key))
                    except json.JSONDecodeError:
                        continue

        self._next_idx = len(self._written_keys)

        # If pre-allocation was requested and the arrays are too small, resize now.
        if (
            self._expected_samples is not None
            and self._response_attn is not None
            and self._response_attn.shape[0] < self._expected_samples
        ):
            self._ensure_capacity(self._expected_samples)

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _ensure_capacity(self, needed: int) -> None:
        """Resize all arrays to hold at least `needed` rows."""
        if self._response_attn is None:
            self._create_arrays(needed)
            return
        if self._response_attn.shape[0] >= needed:
            return
        self._response_attn.resize(
            (needed, self._num_layers, self._r_max, self._r_max)
        )
        self._sample_key_arr.resize((needed,))
        self._response_len_arr.resize((needed,))
        self._prompt_len_arr.resize((needed,))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_written(self, sample_key: str) -> bool:
        """Return True if this sample key is already stored (O(1))."""
        return sample_key in self._written_keys

    def write(
        self,
        sample_key: str,
        response_attn: Union[torch.Tensor, np.ndarray],
        response_len: int,
        prompt_len: int,
    ) -> None:
        """Append one sample to the store.

        Args:
            sample_key: Unique string identifier for this sample.
            response_attn: (num_layers, r_max, r_max) float16 attention data.
                           Pads with zeros or truncates to fit the stored shape.
            response_len: Actual response length (used by readers to mask padding).
            prompt_len: Prompt length (stored for projection-target indexing).
        """
        if isinstance(response_attn, torch.Tensor):
            arr = response_attn.detach().cpu().numpy()
        else:
            arr = np.asarray(response_attn)

        arr = arr.astype(self._dtype, copy=False)

        # Pad or truncate to (num_layers, r_max, r_max).
        target_shape = (self._num_layers, self._r_max, self._r_max)
        if arr.shape != target_shape:
            padded = np.zeros(target_shape, dtype=self._dtype)
            l_in = min(arr.shape[0], self._num_layers)
            r_in = min(arr.shape[1], self._r_max) if arr.ndim > 1 else 0
            c_in = min(arr.shape[2], self._r_max) if arr.ndim > 2 else 0
            padded[:l_in, :r_in, :c_in] = arr[:l_in, :r_in, :c_in]
            arr = padded

        idx = self._next_idx
        self._ensure_capacity(idx + 1)

        self._response_attn[idx] = arr
        key_bytes = sample_key.encode("utf-8")[:64]
        self._sample_key_arr[idx] = key_bytes
        self._response_len_arr[idx] = int(response_len)
        self._prompt_len_arr[idx] = int(prompt_len)

        with open(self._index_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"key": sample_key, "sample_index": idx}) + "\n"
            )

        self._written_keys.add(sample_key)
        self._next_idx = idx + 1

    def finalize(self) -> None:
        """Flush and finalise the store.

        Truncates pre-allocated array rows to the number actually written,
        then rewrites config.json.  Safe to call multiple times.
        """
        if self._finalized:
            return
        n = self._next_idx
        if self._response_attn is not None and self._response_attn.shape[0] > n:
            self._response_attn.resize(
                (n, self._num_layers, self._r_max, self._r_max)
            )
            self._sample_key_arr.resize((n,))
            self._response_len_arr.resize((n,))
            self._prompt_len_arr.resize((n,))
        self._write_config()
        self._finalized = True

    def __len__(self) -> int:
        return self._next_idx

    def __enter__(self) -> "AttentionZarrLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.finalize()
