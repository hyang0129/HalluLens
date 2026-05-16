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
from pathlib import Path
from typing import Literal

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
    """STUB — to be implemented by sonnet B. See module docstring for full contract."""

    def __init__(
        self,
        out_dir: str | os.PathLike,
        mode: Literal["w", "a"],
        n_samples: int,
        config_dict: dict,
    ) -> None:
        raise NotImplementedError(
            "Implement per inference_capture_writer.py module docstring."
        )

    def is_written(self, prompt_hash: str) -> bool:
        raise NotImplementedError

    def next_index(self) -> int:
        raise NotImplementedError

    def append(self, **kwargs) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "InferenceCaptureWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.finalize()
