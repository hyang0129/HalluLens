"""
inference_capture_writer.py — memmap-native writer for the single-process
inference-capture pipeline (Issue #72).

Layout produced (see specs/issue_72_inference_capture_rewrite.md "Data layout"):

    shared/icr_capture/{dataset}_{model_slug}/
      config.json
      meta.jsonl                          # authoritative valid-rows list
      response_activations.npy            # (N, num_layers+1, max_response_len, hidden_dim) fp16
      response_attention.npy              # (N, num_layers, r_max, r_max) fp16
      prompt_activations.npy              # (N, num_layers+1, max_prompt_len, hidden_dim) fp16
      prompt_token_ids.npy                # (N, max_prompt_len) int32  (-1 padded)
      response_token_ids.npy              # (N, max_response_len) int32  (-1 padded)
      response_token_logprobs.npy         # (N, max_response_len) float32  (NaN padded)
      response_topk_token_ids.npy         # (N, max_response_len, top_k) int32  (-1 padded)
      response_topk_logprobs.npy          # (N, max_response_len, top_k) float32  (NaN padded)
      prompt_len.npy                      # (N,) int32
      response_len.npy                    # (N,) int32
      icr_scores.npy                      # (N, num_layers) fp32   (regular np.save at finalize)
      generation.jsonl                    # one JSON object per written sample
      eval_results.json                   # synthesized at finalize() — array format

Design constraints (must hold):

1. All large arrays are np.memmap pre-allocated on first open (mode='w').
   On resume (mode='a'), config.json is verified field-by-field and the
   memmaps are re-opened with mode='r+'.

2. meta.jsonl is the authoritative source of "was sample N written".
   Sequence per sample.append():
       a. Write each array's row N (memmap[idx] = arr).
       b. Append one line to generation.jsonl (flush + fsync).
       c. Append one line to meta.jsonl (flush + fsync) — this is the
          commit point. A crash between (b) and (c) leaves a dangling
          generation.jsonl line that is ignored on resume (because there
          is no matching meta.jsonl line).

3. Resume reads meta.jsonl on open, builds a set of written sample_indices
   and prompt_hashes; .is_written(prompt_hash) returns True for any sample
   already present. .next_index() returns max(sample_index) + 1.

4. finalize() saves icr_scores.npy (regular np.save, not memmap, since the
   array is small) and synthesizes eval_results.json from meta.jsonl:
       {"halu_test_res": [bool, ...], "abstantion": [false, ...]}
   indexed by sample_index, matching ActivationParser._load_metadata's
   expected array format.

5. config.json minimum required keys (the writer enforces these on
   construction and re-verifies on resume):
       model_name (str), num_layers (int), hidden_dim (int), r_max (int),
       dtype (str), response_logprobs_top_k (int),
       max_prompt_len (int), max_response_len (int)

6. The append() signature accepts numpy ndarrays only — torch.Tensor must
   be converted by the caller (matches the existing
   AttentionMemmapWriter.write convention in attention_memmap_writer.py).

Public API:

    class InferenceCaptureWriter:
        def __init__(self, out_dir, mode, n_samples, config_dict):
            ...
        def is_written(self, prompt_hash: str) -> bool: ...
        def next_index(self) -> int: ...
        def append(self, *, sample_index: int, prompt_hash: str, key: str,
                   prompt_len: int, response_len: int,
                   prompt_activations: np.ndarray,
                   response_activations: np.ndarray,
                   response_attention: np.ndarray,
                   prompt_token_ids: np.ndarray,
                   response_token_ids: np.ndarray,
                   response_token_logprobs: np.ndarray,
                   response_topk_token_ids: np.ndarray,
                   response_topk_logprobs: np.ndarray,
                   icr_score_per_layer: np.ndarray,    # (num_layers,) fp32
                   hallucinated: bool,
                   generation_record: dict) -> None: ...
        def finalize(self) -> None: ...

The writer must NOT compute ICR or labels itself — those are passed in by
the orchestrator (scripts/capture_inference.py). It is a dumb sink.

Unit-test target: tests/test_inference_capture_writer.py
- Round-trip a tiny fake sample (synthesize arrays of the right shapes,
  write, close, re-open, read back; assert equality).
- Resume semantics: write 3 samples, close, re-open in 'a' mode, assert
  next_index() == 3 and is_written() returns True for all 3.
- Crash-during-meta-append simulation: write data + generation.jsonl but
  not meta.jsonl, then re-open — that sample must NOT be reported as
  written.
- finalize() synthesizes eval_results.json with halu_test_res in the right
  order (sample_index = row position).
- config.json field mismatch on resume raises ValueError (catch model_name
  swap, num_layers swap, dtype swap).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np


_REQUIRED_CONFIG_KEYS = (
    "model_name",
    "num_layers",
    "hidden_dim",
    "r_max",
    "dtype",
    "response_logprobs_top_k",
    "max_prompt_len",
    "max_response_len",
)


class InferenceCaptureWriter:
    """Memmap-native writer for the single-process inference-capture pipeline.

    See module docstring for full contract.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike,
        mode: Literal["w", "a"],
        n_samples: int,
        config_dict: dict,
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")

        for key in _REQUIRED_CONFIG_KEYS:
            if key not in config_dict:
                raise ValueError(
                    f"config_dict is missing required key {key!r}. "
                    f"Required keys: {_REQUIRED_CONFIG_KEYS}"
                )

        self._out_dir = Path(out_dir)
        self._mode = mode
        self._n_samples = n_samples
        self._config_dict = dict(config_dict)

        self._num_layers: int = int(config_dict["num_layers"])
        self._hidden_dim: int = int(config_dict["hidden_dim"])
        self._r_max: int = int(config_dict["r_max"])
        self._dtype = np.dtype(config_dict["dtype"])
        self._top_k: int = int(config_dict["response_logprobs_top_k"])
        self._max_prompt_len: int = int(config_dict["max_prompt_len"])
        self._max_response_len: int = int(config_dict["max_response_len"])

        self._config_path = self._out_dir / "config.json"
        self._meta_path = self._out_dir / "meta.jsonl"
        self._gen_path = self._out_dir / "generation.jsonl"

        # Memmaps keyed by logical name.
        self._mm: dict[str, np.memmap] = {}

        # In-memory sets rebuilt from meta.jsonl on append-mode open.
        self._written_hashes: set[str] = set()
        self._written_indices: set[int] = set()

        # ICR score accumulator; flushed to npy at finalize().
        # Each entry is (sample_index, icr_score_per_layer ndarray).
        self._icr_buffer: list[tuple[int, np.ndarray]] = []

        self._finalized = False

        if mode == "w":
            self._init_write()
        else:
            self._init_append()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_config(self) -> dict:
        cfg = dict(self._config_dict)
        cfg["storage_format"] = "inference_capture_v1"
        cfg["n_samples"] = self._n_samples
        return cfg

    def _write_config(self) -> None:
        self._config_path.write_text(
            json.dumps(self._full_config(), indent=2), encoding="utf-8"
        )

    def _memmap_specs(self) -> list[tuple[str, tuple, np.dtype, object]]:
        """Return (filename_stem, shape, dtype, fill_value) for each memmap."""
        fp16 = np.dtype("float16")
        int32 = np.dtype("int32")
        fp32 = np.dtype("float32")
        N = self._n_samples
        L = self._num_layers
        H = self._hidden_dim
        R = self._r_max
        P = self._max_prompt_len
        S = self._max_response_len
        K = self._top_k
        return [
            ("response_activations", (N, L + 1, S, H), fp16, 0),
            ("response_attention",   (N, L,     R, R), fp16, 0),
            ("prompt_activations",   (N, L + 1, P, H), fp16, 0),
            ("prompt_token_ids",     (N, P),            int32, -1),
            ("response_token_ids",   (N, S),            int32, -1),
            ("response_token_logprobs", (N, S),         fp32, np.nan),
            ("response_topk_token_ids", (N, S, K),     int32, -1),
            ("response_topk_logprobs",  (N, S, K),     fp32, np.nan),
            ("prompt_len",           (N,),              int32, 0),
            ("response_len",         (N,),              int32, 0),
        ]

    def _npy_path(self, stem: str) -> Path:
        return self._out_dir / f"{stem}.npy"

    def _init_write(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        for stem, shape, dtype, fill in self._memmap_specs():
            mm = np.memmap(
                str(self._npy_path(stem)),
                dtype=dtype,
                mode="w+",
                shape=shape,
            )
            if fill != 0:
                # Why: zero-fill is the default for memmap 'w+', but token-ID and
                # logprob arrays have non-zero sentinel padding values.
                mm[:] = fill
                mm.flush()
            self._mm[stem] = mm

        self._meta_path.write_text("", encoding="utf-8")
        self._gen_path.write_text("", encoding="utf-8")
        self._write_config()

    def _init_append(self) -> None:
        if not self._config_path.exists():
            raise ValueError(
                f"config.json not found at {self._config_path}. "
                "Cannot resume a store that was not initialised with mode='w'."
            )
        stored = json.loads(self._config_path.read_text(encoding="utf-8"))
        expected = self._full_config()

        for key in _REQUIRED_CONFIG_KEYS:
            if stored.get(key) != expected.get(key):
                raise ValueError(
                    f"config.json field {key!r} mismatch: "
                    f"stored={stored.get(key)!r}, "
                    f"expected={expected.get(key)!r}"
                )

        for stem, shape, dtype, _fill in self._memmap_specs():
            path = self._npy_path(stem)
            if not path.exists():
                raise ValueError(
                    f"{stem}.npy not found at {path}. "
                    "Cannot resume a store whose memmap file is missing."
                )
            mm = np.memmap(str(path), dtype=dtype, mode="r+", shape=shape)
            self._mm[stem] = mm

        # Rebuild written-set from meta.jsonl (the authoritative log).
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ph = entry.get("prompt_hash")
                    idx = entry.get("sample_index")
                    if ph is not None:
                        self._written_hashes.add(str(ph))
                    if idx is not None:
                        self._written_indices.add(int(idx))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_written(self, prompt_hash: str) -> bool:
        """O(1) lookup: True if this prompt_hash is already in meta.jsonl."""
        return prompt_hash in self._written_hashes

    def next_index(self) -> int:
        """Return max(written_sample_indices) + 1, or 0 if none written."""
        if not self._written_indices:
            return 0
        return max(self._written_indices) + 1

    def append(
        self,
        *,
        sample_index: int,
        prompt_hash: str,
        key: str,
        prompt_len: int,
        response_len: int,
        prompt_activations: np.ndarray,
        response_activations: np.ndarray,
        response_attention: np.ndarray,
        prompt_token_ids: np.ndarray,
        response_token_ids: np.ndarray,
        response_token_logprobs: np.ndarray,
        response_topk_token_ids: np.ndarray,
        response_topk_logprobs: np.ndarray,
        icr_score_per_layer: np.ndarray,
        hallucinated: bool,
        generation_record: dict,
    ) -> None:
        """Write one sample to all memmaps + generation.jsonl + meta.jsonl.

        Write order is crash-safe:
          1. Memmap rows (may be partially durable).
          2. generation.jsonl line (fsync).
          3. meta.jsonl line (fsync) — this is the commit point.

        Idempotent: if prompt_hash already in meta.jsonl, returns immediately.
        """
        if not (0 <= sample_index < self._n_samples):
            raise IndexError(
                f"sample_index {sample_index} out of range [0, {self._n_samples})"
            )

        if prompt_hash in self._written_hashes:
            return

        # Why: guard against two different prompts mapping to the same row,
        # which would silently corrupt previously-written data.
        if sample_index in self._written_indices:
            raise ValueError(
                f"sample_index {sample_index} is already mapped to a different "
                f"prompt; cannot write prompt_hash {prompt_hash!r} at this index."
            )

        # --- Step 1: write all memmap rows ---
        def _put(stem: str, arr: np.ndarray) -> None:
            mm = self._mm[stem]
            target_dtype = mm.dtype
            arr_np = np.asarray(arr).astype(target_dtype, copy=False)
            row = mm[sample_index]
            if arr_np.shape == row.shape:
                mm[sample_index] = arr_np
            else:
                # Pad/truncate to stored shape.
                padded = np.empty(row.shape, dtype=target_dtype)
                # Apply the sentinel fill for this stream.
                fill = _FILL_FOR[stem]
                if fill == 0:
                    padded[:] = 0
                elif fill == -1:
                    padded[:] = -1
                else:
                    padded[:] = fill  # NaN handled here
                # Copy as much of the incoming data as fits.
                slices = tuple(slice(0, min(a, b)) for a, b in zip(arr_np.shape, row.shape))
                padded[slices] = arr_np[slices]
                mm[sample_index] = padded

        _put("response_activations", response_activations)
        _put("response_attention", response_attention)
        _put("prompt_activations", prompt_activations)
        _put("prompt_token_ids", prompt_token_ids)
        _put("response_token_ids", response_token_ids)
        _put("response_token_logprobs", response_token_logprobs)
        _put("response_topk_token_ids", response_topk_token_ids)
        _put("response_topk_logprobs", response_topk_logprobs)
        self._mm["prompt_len"][sample_index] = np.int32(prompt_len)
        self._mm["response_len"][sample_index] = np.int32(response_len)

        # --- Step 2: generation.jsonl (fsync before meta commit) ---
        gen_line = dict(generation_record)
        gen_line["sample_index"] = sample_index
        gen_line["prompt_hash"] = prompt_hash
        with open(self._gen_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(gen_line) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        # --- Step 3: meta.jsonl — the commit point ---
        meta_line = {
            "sample_index": sample_index,
            "key": key,
            "prompt_hash": prompt_hash,
            "prompt_len": int(prompt_len),
            "response_len": int(response_len),
            "hallucinated": bool(hallucinated),
            "wrote_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._meta_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta_line) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        # Update in-memory sets only after the durable commit.
        self._written_hashes.add(prompt_hash)
        self._written_indices.add(sample_index)
        self._icr_buffer.append((sample_index, np.asarray(icr_score_per_layer, dtype=np.float32)))

    def finalize(self) -> None:
        """Flush memmaps, save icr_scores.npy, synthesize eval_results.json.

        Idempotent: safe to call multiple times.
        """
        if self._finalized:
            return

        for mm in self._mm.values():
            mm.flush()

        # ICR scores: stack in sample_index order and np.save.
        if self._icr_buffer:
            sorted_buf = sorted(self._icr_buffer, key=lambda t: t[0])
            icr_arr = np.stack([arr for _, arr in sorted_buf], axis=0)
            np.save(str(self._out_dir / "icr_scores.npy"), icr_arr)

        # eval_results.json — read meta.jsonl in sample_index order.
        entries: list[dict] = []
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        entries.sort(key=lambda e: e["sample_index"])
        halu = [bool(e["hallucinated"]) for e in entries]
        abstain = [False] * len(entries)
        eval_results = {"halu_test_res": halu, "abstantion": abstain}
        (self._out_dir / "eval_results.json").write_text(
            json.dumps(eval_results, indent=2), encoding="utf-8"
        )

        self._finalized = True

    def __enter__(self) -> "InferenceCaptureWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.finalize()


# Sentinel fill values per memmap stream (used by _put helper inside append).
_FILL_FOR: dict[str, object] = {
    "response_activations": 0,
    "response_attention": 0,
    "prompt_activations": 0,
    "prompt_token_ids": -1,
    "response_token_ids": -1,
    "response_token_logprobs": np.nan,
    "response_topk_token_ids": -1,
    "response_topk_logprobs": np.nan,
    "prompt_len": 0,
    "response_len": 0,
}
