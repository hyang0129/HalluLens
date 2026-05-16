"""
attention_parser.py

Reader API for numpy-memmap attention stores produced by the fused Stage-1+2
recomputation pipeline (Issue #69, Wave 4).  Pairs head-averaged
response-to-response attention data from the memmap store with the cached
hidden states from the source ``activations.zarr`` for use by the ICR Probe
(Issue #70).

Layer-index alignment (per ``notes/icr_probe_paper_notes.md`` §6 and
``icr_score.py:42-51``):

    activations.zarr stores L+1 layer entries per sample:
      index 0 → embedding output  (= block 0's input, h^{-1} in HF notation)
      index b → block b's output  (= block b+1's input)

    For transformer block b ∈ [0, num_blocks):
      h_block_input[b]  = response_activations[key, b,   :response_len, :]  (HF index b)
      h_block_output[b] = response_activations[key, b+1, :response_len, :]  (HF index b+1)
      delta_h[b]        = h_block_output[b] − h_block_input[b]

All returned tensors are CPU float32 regardless of stored dtype (float16).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .zarr_activations_logger import ZarrActivationsLogger


class AttentionParser:
    """Reads recomputed attention probs from a numpy-memmap attention store,
    paired with the activations zarr that they were derived from.

    Args:
        attention_zarr_path: Path to a directory written by
            :class:`~activation_logging.attention_memmap_writer.AttentionMemmapWriter`.
            Must contain ``response_attn.npy``, ``meta.jsonl``, and
            ``config.json``.  The parameter name preserves backward-compat
            with :class:`~activation_research.icr_dataset.ICRDataset` (Wave 3).
        activations_logger: An already-constructed :class:`ZarrActivationsLogger`
            opened in read-only mode on the source ``activations.zarr`` store.
            If ``None``, the cross-check is skipped.
        activations_parser: Legacy alias for *activations_logger* accepted by
            :class:`~activation_research.icr_dataset.ICRDataset` (Wave 3).

    Raises:
        FileNotFoundError: If ``attention_dir`` does not exist, or if
            ``response_attn.npy`` is missing.
        ValueError: If ``config.json`` or ``meta.jsonl`` is missing, or if the
            ``model_name`` in the attention store does not match the one in the
            activations store's ``meta/config.json``.
    """

    def __init__(
        self,
        attention_zarr_path: str,
        activations_logger: Optional[ZarrActivationsLogger] = None,
        # Legacy alias accepted by ICRDataset (Wave 3) — maps to activations_logger.
        activations_parser: Optional[ZarrActivationsLogger] = None,
    ) -> None:
        # Support the old kwarg name used by ICRDataset and the test suite.
        if activations_logger is None and activations_parser is not None:
            activations_logger = activations_parser
        self._attn_path = Path(attention_zarr_path)
        if not self._attn_path.exists():
            raise FileNotFoundError(
                f"attention store not found at {self._attn_path}"
            )

        # ------------------------------------------------------------------
        # Load config.json from the attention store.
        # ------------------------------------------------------------------
        config_path = self._attn_path / "config.json"
        if not config_path.exists():
            raise ValueError(
                f"config.json missing from attention store at {self._attn_path}. "
                "The store may not have been created with AttentionMemmapWriter."
            )
        self._config: dict = json.loads(config_path.read_text(encoding="utf-8"))

        # ------------------------------------------------------------------
        # Cache shape parameters for downstream shape-checks.
        # ------------------------------------------------------------------
        self._num_layers: int = int(self._config["num_layers"])
        self._r_max: int = int(self._config["r_max"])

        # ------------------------------------------------------------------
        # Read meta.jsonl → key-to-meta map, preserving insertion order.
        # ------------------------------------------------------------------
        meta_path = self._attn_path / "meta.jsonl"
        if not meta_path.exists():
            raise ValueError(
                f"meta.jsonl missing from attention store at {self._attn_path}."
            )
        self._keys: list[str] = []
        self._key_to_meta: dict[str, dict] = {}
        with open(meta_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key: str = str(entry["key"])
                if key not in self._key_to_meta:
                    self._keys.append(key)
                self._key_to_meta[key] = {
                    "sample_index": int(entry["sample_index"]),
                    "response_len": int(entry["response_len"]),
                    "prompt_len": int(entry["prompt_len"]),
                }

        # ------------------------------------------------------------------
        # Open the pre-allocated memmap read-only.
        # ------------------------------------------------------------------
        npy_path = self._attn_path / "response_attn.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"response_attn.npy missing from attention store at {self._attn_path}."
            )
        n_samples: int = int(self._config["n_samples"])
        dtype_str: str = str(self._config.get("dtype", "float16"))
        self._mm: np.memmap = np.memmap(
            str(npy_path),
            dtype=np.dtype(dtype_str),
            mode="r",
            shape=(n_samples, self._num_layers, self._r_max, self._r_max),
        )

        # ------------------------------------------------------------------
        # Adopt the activations logger and validate model_name if provided.
        # ------------------------------------------------------------------
        self._act_logger: Optional[ZarrActivationsLogger] = activations_logger
        if activations_logger is not None:
            self._validate_model_name(activations_logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_attention(self, key: str) -> dict:
        """Return the stored attention data for one sample.

        Args:
            key: Sample key (must match a key written by AttentionMemmapWriter).

        Returns:
            A dict with:

            * ``"response_attn"`` — Tensor ``(L, R_max, R_max)`` float32.
              Positions beyond ``response_len`` are zero-padded.
              The caller is responsible for masking positions ≥ response_len.
            * ``"response_len"`` — int, actual response length.
            * ``"prompt_len"`` — int, prompt token count.

        Raises:
            KeyError: If *key* is not present in the attention store.
        """
        meta = self._resolve_meta(key)
        attn_np = self._mm[meta["sample_index"]].copy()   # (L, R_max, R_max) float16
        return {
            "response_attn": torch.from_numpy(attn_np).float(),
            "response_len": meta["response_len"],
            "prompt_len": meta["prompt_len"],
        }

    def get_paired(self, key: str, relevant_layers: list[int]) -> dict:
        """Return paired attention and hidden-state data for one sample.

        Layer-index alignment follows ``icr_score.py:42-51`` (see module
        docstring):

        * ``h_block_input[b]``  ← ``response_activations[key, b,   :R, :]``
        * ``h_block_output[b]`` ← ``response_activations[key, b+1, :R, :]``
        * ``delta_h[b]``        ← ``h_block_output[b] − h_block_input[b]``
        * ``response_attn[b]``  ← ``attention store[key, b, :R, :R]``

        where ``R = response_len``.

        Args:
            key: Sample key present in both the attention store and
                activations.zarr.
            relevant_layers: Block indices to include (0-based, matching the ICR
                Probe convention where block 0's input is the embedding output).

        Returns:
            A dict with:

            * ``"h_block_input"``  — ``dict[int → Tensor(response_len, H)]`` float32
            * ``"delta_h"``        — ``dict[int → Tensor(response_len, H)]`` float32
            * ``"response_attn"``  — ``dict[int → Tensor(response_len, response_len)]`` float32
            * ``"response_len"``   — int
            * ``"prompt_len"``     — int

        Raises:
            KeyError: If *key* is absent from either the attention store or the
                activations store.
        """
        meta = self._resolve_meta(key)
        response_len: int = meta["response_len"]
        prompt_len: int = meta["prompt_len"]
        attn_idx: int = meta["sample_index"]

        # --- activations side ---
        if self._act_logger is None:
            raise KeyError(
                f"Cannot call get_paired('{key}') without an activations_logger. "
                "Provide activations_logger at __init__ time."
            )

        act_meta = self._act_logger._index.get(key)
        if act_meta is None:
            raise KeyError(
                f"Key '{key}' not found in activations.zarr at "
                f"{self._act_logger.zarr_path!r}. "
                "The key exists in the attention store but is missing from the source store."
            )
        act_idx: int = act_meta.get("sample_index")
        if act_idx is None:
            raise KeyError(
                f"Key '{key}' has no 'sample_index' in the activations index. "
                "The store may be corrupt."
            )

        resp_arr = self._act_logger._response_activations
        if resp_arr is None:
            raise KeyError(
                f"activations.zarr at {self._act_logger.zarr_path!r} has no "
                "'response_activations' array. Cannot construct paired data."
            )

        act_L_plus_1: int = int(resp_arr.shape[1])

        h_block_input: dict[int, torch.Tensor] = {}
        delta_h: dict[int, torch.Tensor] = {}
        response_attn: dict[int, torch.Tensor] = {}

        for b in relevant_layers:
            if b >= act_L_plus_1:
                raise KeyError(
                    f"relevant_layer {b} exceeds activations.zarr layer axis "
                    f"size {act_L_plus_1} for key '{key}'."
                )
            if b + 1 >= act_L_plus_1:
                raise KeyError(
                    f"relevant_layer {b}: need activations index b+1={b+1} but "
                    f"activations.zarr layer axis size is {act_L_plus_1} for key '{key}'."
                )

            h_in_np = np.array(resp_arr[act_idx, b, :response_len, :])       # (R, H) float16
            h_out_np = np.array(resp_arr[act_idx, b + 1, :response_len, :])  # (R, H) float16

            h_in = torch.from_numpy(h_in_np).float()
            h_out = torch.from_numpy(h_out_np).float()

            h_block_input[b] = h_in
            delta_h[b] = h_out - h_in

            # Attention: slice to actual response_len × response_len.
            attn_layer_np = np.array(
                self._mm[attn_idx, b, :response_len, :response_len]
            )
            response_attn[b] = torch.from_numpy(attn_layer_np).float()

        return {
            "h_block_input": h_block_input,
            "delta_h": delta_h,
            "response_attn": response_attn,
            "response_len": response_len,
            "prompt_len": prompt_len,
        }

    def list_keys(self) -> list[str]:
        """Return all sample keys stored in the attention store, in insertion order."""
        return list(self._keys)

    def __len__(self) -> int:
        """Number of samples recorded in meta.jsonl."""
        return len(self._keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_meta(self, key: str) -> dict:
        """Resolve a sample key to its meta dict.

        Raises:
            KeyError: If *key* is not present in the attention store.
        """
        meta = self._key_to_meta.get(key)
        if meta is None:
            raise KeyError(
                f"Sample key {key!r} not found in attention store at {self._attn_path!r}."
            )
        return meta

    def _validate_model_name(self, activations_logger: ZarrActivationsLogger) -> None:
        """Cross-check model_name between this attention store and activations.zarr.

        Reads activations.zarr's ``meta/config.json`` from disk via the public
        ``zarr_path`` attribute on the logger.  If the file is absent, raises
        rather than silently skipping — that was the dead-code bug in the
        previous zarr-based parser (attention_parser.py:109, Wave 3).

        Raises:
            FileNotFoundError: If activations.zarr/meta/config.json does not exist.
            ValueError: If model_name fields disagree.
        """
        attn_model: Optional[str] = self._config.get("model_name")
        if attn_model is None:
            # No model_name in the attention store config — nothing to cross-check.
            return

        # zarr_path is the public str attribute on ZarrActivationsLogger.
        act_config_path = Path(activations_logger.zarr_path) / "meta" / "config.json"
        if not act_config_path.exists():
            raise FileNotFoundError(
                f"activations.zarr meta/config.json not found at {act_config_path}. "
                "Cannot verify model_name consistency between the two stores."
            )

        act_config: dict = json.loads(act_config_path.read_text(encoding="utf-8"))
        act_model: Optional[str] = act_config.get("model_name")
        if act_model is not None and act_model != attn_model:
            raise ValueError(
                f"model_name mismatch between attention store ({attn_model!r}) "
                f"and activations store ({act_model!r})."
            )
