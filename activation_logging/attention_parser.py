"""
attention_parser.py

Reader API for attention.zarr stores produced by the attention recomputation
pipeline (Issue #69).  Pairs head-averaged response-to-response attention data
from ``attention.zarr`` with the cached hidden states from the source
``activations.zarr`` for use by the ICR Probe (Issue #70).

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
import zarr

from .zarr_activations_logger import ZarrActivationsLogger


class AttentionParser:
    """Reads recomputed attention probs from attention.zarr, paired with
    the activations zarr that they were derived from.

    Args:
        attention_zarr_path: Path to an ``attention.zarr`` directory written by
            :class:`~activation_logging.attention_zarr_logger.AttentionZarrLogger`.
        activations_parser: An already-constructed :class:`ZarrActivationsLogger`
            opened in read-only mode on the source ``activations.zarr`` store.
            If ``None``, the parser reads ``meta/config.json`` inside the
            attention store to find the source path and constructs one.

    Raises:
        FileNotFoundError: If ``attention_zarr_path`` or the derived
            ``source_activations_zarr`` path does not exist.
        ValueError: If ``meta/config.json`` is missing, or if the
            ``model_name`` field in the attention config does not match the
            ``model_name`` recorded in ``activations.zarr``'s own config.
    """

    def __init__(
        self,
        attention_zarr_path: str,
        activations_parser: Optional[ZarrActivationsLogger] = None,
    ) -> None:
        self._attn_path = Path(attention_zarr_path)
        if not self._attn_path.exists():
            raise FileNotFoundError(
                f"attention.zarr not found at {self._attn_path}"
            )

        # ------------------------------------------------------------------
        # Load config.json from the attention store.
        # ------------------------------------------------------------------
        config_path = self._attn_path / "meta" / "config.json"
        if not config_path.exists():
            raise ValueError(
                f"meta/config.json missing from attention store at {self._attn_path}. "
                "The store may not have been created with AttentionZarrLogger."
            )
        self._config: dict = json.loads(config_path.read_text(encoding="utf-8"))

        # ------------------------------------------------------------------
        # Open (or adopt) the activations.zarr reader.
        # ------------------------------------------------------------------
        if activations_parser is not None:
            self._act_logger = activations_parser
        else:
            source_path_raw: Optional[str] = self._config.get("source_activations_zarr")
            if not source_path_raw:
                raise ValueError(
                    "config.json does not contain 'source_activations_zarr'. "
                    "Pass activations_parser explicitly."
                )
            # Resolve relative paths against the attention store's parent dir.
            source_path = Path(source_path_raw)
            if not source_path.is_absolute():
                source_path = (self._attn_path.parent / source_path).resolve()
            if not source_path.exists():
                raise FileNotFoundError(
                    f"source_activations_zarr '{source_path}' does not exist "
                    f"(resolved from '{source_path_raw}' relative to {self._attn_path.parent})."
                )
            self._act_logger = ZarrActivationsLogger(
                zarr_path=str(source_path),
                read_only=True,
            )

        # ------------------------------------------------------------------
        # Validate model_name if activations store exposes it.
        # ------------------------------------------------------------------
        attn_model: Optional[str] = self._config.get("model_name")
        if attn_model is not None:
            act_config = self._act_logger._config if hasattr(self._act_logger, "_config") else {}
            act_model: Optional[str] = act_config.get("model_name") if act_config else None
            if act_model is not None and act_model != attn_model:
                raise ValueError(
                    f"model_name mismatch: attention.zarr config says '{attn_model}' "
                    f"but activations.zarr config says '{act_model}'."
                )

        # ------------------------------------------------------------------
        # Open the attention zarr store and build the key→sample_index map.
        # ------------------------------------------------------------------
        self._attn_root = zarr.open_group(str(self._attn_path), mode="r")
        self._attn_arrays = self._attn_root["arrays"]

        self._response_attn: zarr.Array = self._attn_arrays["response_attn"]
        self._sample_key_arr: zarr.Array = self._attn_arrays["sample_key"]
        self._response_len_arr: zarr.Array = self._attn_arrays["response_len"]
        self._prompt_len_arr: zarr.Array = self._attn_arrays["prompt_len"]

        # Build O(1) key→row-index map from the stored sample_key array.
        self._key_to_idx: dict[str, int] = {}
        n = int(self._sample_key_arr.shape[0])
        for i in range(n):
            raw = self._sample_key_arr[i]
            key = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            if key:
                self._key_to_idx[key] = i

        # Cache derived metadata.
        self._num_layers: int = int(self._config.get("num_layers", self._response_attn.shape[1]))
        self._r_max: int = int(self._config.get("r_max", self._response_attn.shape[2]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_attention(self, key: str) -> dict:
        """Return the stored attention data for one sample.

        Args:
            key: Sample key (must match a key written by AttentionZarrLogger).

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
        idx = self._resolve_attn_idx(key)

        attn_np = np.array(self._response_attn[idx])          # (L, R_max, R_max) float16
        response_len = int(self._response_len_arr[idx])
        prompt_len = int(self._prompt_len_arr[idx])

        return {
            "response_attn": torch.from_numpy(attn_np).float(),
            "response_len": response_len,
            "prompt_len": prompt_len,
        }

    def get_paired(self, key: str, relevant_layers: list[int]) -> dict:
        """Return paired attention and hidden-state data for one sample.

        Layer-index alignment follows ``icr_score.py:42-51`` (see module
        docstring):

        * ``h_block_input[b]``  ← ``response_activations[key, b,   :R, :]``
        * ``h_block_output[b]`` ← ``response_activations[key, b+1, :R, :]``
        * ``delta_h[b]``        ← ``h_block_output[b] − h_block_input[b]``
        * ``response_attn[b]``  ← ``attention.zarr[key, b, :R, :R]``

        where ``R = response_len``.

        Args:
            key: Sample key present in both attention.zarr and activations.zarr.
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
        # --- attention side ---
        attn_idx = self._resolve_attn_idx(key)
        response_len = int(self._response_len_arr[attn_idx])
        prompt_len = int(self._prompt_len_arr[attn_idx])

        # --- activations side ---
        act_meta = self._act_logger._index.get(key)
        if act_meta is None:
            raise KeyError(
                f"Key '{key}' not found in activations.zarr at "
                f"{self._act_logger.zarr_path!r}. "
                "The key exists in attention.zarr but is missing from the source store."
            )
        act_idx: int = act_meta.get("sample_index")
        if act_idx is None:
            raise KeyError(
                f"Key '{key}' has no 'sample_index' in the activations index. "
                "The store may be corrupt."
            )

        # Lazy-access the underlying zarr arrays via ZarrActivationsLogger attributes.
        resp_arr: zarr.Array = self._act_logger._response_activations
        if resp_arr is None:
            raise KeyError(
                f"activations.zarr at {self._act_logger.zarr_path!r} has no "
                "'response_activations' array. Cannot construct paired data."
            )

        # activations.zarr layout: (N, L+1, R_max, H) — index b = block-b input.
        act_L_plus_1 = int(resp_arr.shape[1])

        h_block_input: dict[int, torch.Tensor] = {}
        delta_h: dict[int, torch.Tensor] = {}
        response_attn: dict[int, torch.Tensor] = {}

        for b in relevant_layers:
            # Block b's input hidden state lives at activations index b.
            if b >= act_L_plus_1:
                raise KeyError(
                    f"relevant_layer {b} exceeds activations.zarr layer axis "
                    f"size {act_L_plus_1} for key '{key}'."
                )
            # Block b's output hidden state lives at activations index b+1.
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
            attn_layer_np = np.array(self._response_attn[attn_idx, b, :response_len, :response_len])
            response_attn[b] = torch.from_numpy(attn_layer_np).float()

        return {
            "h_block_input": h_block_input,
            "delta_h": delta_h,
            "response_attn": response_attn,
            "response_len": response_len,
            "prompt_len": prompt_len,
        }

    def list_keys(self) -> list[str]:
        """Return all sample keys stored in the attention store."""
        return list(self._key_to_idx.keys())

    def __len__(self) -> int:
        """Number of samples stored in the attention store."""
        return len(self._key_to_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_attn_idx(self, key: str) -> int:
        """Resolve a sample key to its row index in the attention arrays.

        Raises:
            KeyError: If *key* is not present in the attention store.
        """
        idx = self._key_to_idx.get(key)
        if idx is None:
            raise KeyError(
                f"Key '{key}' not found in attention.zarr at {self._attn_path!r}."
            )
        return idx
