"""
memmap_activation_parser.py — ActivationParser-compatible wrapper over icr_capture
directories for issue #79 (50k memmap re-train).

Exposes the same surface that scripts/run_experiment.py's method runners consume:
  .get_dataset(split, **kwargs)  — returns a MemmapContrastiveDataset
  .df                            — full-capture DataFrame
  .split_strategy                — "three_way" (train) or "none" (test)

For train captures (split_strategy="three_way"): uses a 90/10 stratified
train/val split on the capture's halu labels. random_seed must be passed
explicitly (no default) to prevent silent seed collapse across folds.

For test captures (split_strategy="none"): all rows exposed as "test".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset

if TYPE_CHECKING:
    import pandas as pd
    import torch

logger = logging.getLogger(__name__)


class MemmapActivationParser:
    """ActivationParser-compatible wrapper over an icr_capture directory.

    Each instance wraps exactly one capture directory (train OR test — not both).
    Callers in run_experiment.py construct two instances per dataset:
      test_ap = MemmapActivationParser(test_dir, split_strategy="none", ...)
      ap      = MemmapActivationParser(train_dir, split_strategy="three_way", ...)
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        random_seed: int,
        split_strategy: Literal["none", "three_way"] = "three_way",
        verbose: bool = False,
    ) -> None:
        self._capture_dir = Path(capture_dir)
        if not self._capture_dir.exists():
            raise FileNotFoundError(
                f"MemmapActivationParser: capture_dir not found: {self._capture_dir}"
            )
        self._split_strategy = split_strategy
        self._random_seed = random_seed

        # Load config and meta.
        with (self._capture_dir / "config.json").open() as fh:
            self._cfg: dict = json.load(fh)
        n_samples: int = self._cfg["n_samples"]

        meta_rows: List[dict] = []
        with (self._capture_dir / "meta.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    meta_rows.append(json.loads(line))
        if not meta_rows:
            raise ValueError(f"meta.jsonl is empty in {self._capture_dir}")
        self._meta = meta_rows
        N = len(meta_rows)

        # Labels from meta (hallucinated → halu).
        labels = np.array([int(bool(r["hallucinated"])) for r in meta_rows], dtype=np.int32)

        # Prompt/response lengths from memmaps.
        prompt_len_arr = np.memmap(
            str(self._capture_dir / "prompt_len.npy"),
            dtype=np.int32, mode="r", shape=(n_samples,),
        )
        resp_len_arr = np.memmap(
            str(self._capture_dir / "response_len.npy"),
            dtype=np.int32, mode="r", shape=(n_samples,),
        )
        valid_sample_indices = np.array(
            [r["sample_index"] for r in meta_rows], dtype=np.int64
        )

        # Compute 90/10 train/val split on the full N meta-rows.
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        all_idx = np.arange(N, dtype=np.int64)

        if split_strategy == "three_way":
            self._train_idx, self._val_idx = train_test_split(
                all_idx,
                test_size=0.1,
                stratify=labels,
                random_state=random_seed,
            )
            split_col = np.full(N, "train", dtype=object)
            split_col[self._val_idx] = "val"
        elif split_strategy == "none":
            split_col = np.full(N, "test", dtype=object)
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

        # Build full-capture DataFrame (parser-level, covers all N rows).
        import pandas as pd
        sample_rows = valid_sample_indices  # meta[i].sample_index = memmap row
        self._df = pd.DataFrame({
            "prompt_hash": [r["prompt_hash"] for r in meta_rows],
            "halu": labels.astype(int),
            "split": split_col,
            "sample_index": sample_rows,
            "prompt_len": [int(prompt_len_arr[sr]) for sr in sample_rows],
            "response_len": [int(resp_len_arr[sr]) for sr in sample_rows],
        })

        if verbose:
            logger.info(
                f"MemmapActivationParser: {self._capture_dir.name}  "
                f"N={N}  split_strategy={split_strategy}  "
                f"seed={random_seed}"
            )
            if split_strategy == "three_way":
                logger.info(
                    f"  train={len(self._train_idx)}  "
                    f"val={len(self._val_idx)}"
                )

    # ------------------------------------------------------------------ #
    @property
    def df(self) -> "pd.DataFrame":
        """Full-capture DataFrame (all N rows with split column assigned)."""
        return self._df

    @property
    def split_strategy(self) -> str:
        return self._split_strategy

    # ------------------------------------------------------------------ #
    def get_dataset(
        self,
        split: Literal["train", "val", "test"],
        *,
        relevant_layers: Optional[List[int]] = None,
        num_views: int = 2,
        fixed_layer: Optional[int] = None,
        pad_length: Optional[int] = None,
        preload: bool = False,          # accepted; ignored — memmap IS the store
        check_ram: bool = False,        # accepted; ignored
        min_target_layers: Optional[int] = None,  # accepted; ignored
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        deterministic: bool = False,    # accepted; MemmapMultiLayerDataset deferred
        **_ignored: Any,
    ) -> "MemmapContrastiveDataset":
        """Return a MemmapContrastiveDataset for the requested split.

        For split_strategy="none" (test captures): all rows are returned
        regardless of the `split` argument; the dataset's .df reports split="test".

        For split_strategy="three_way" (train captures): "train" returns the 90%
        subset, "val" returns the 10% subset. Requesting "test" raises ValueError
        because test rows live in the separate test capture directory.
        """
        if self._split_strategy == "none":
            return MemmapContrastiveDataset(
                self._capture_dir,
                split="all",
                random_seed=self._random_seed,
                relevant_layers=relevant_layers,
                num_views=num_views,
                fixed_layer=fixed_layer,
                pad_length=pad_length,
                include_response_logprobs=include_response_logprobs,
                response_logprobs_top_k=response_logprobs_top_k,
                _override_split_name="test",
            )

        # three_way
        if split == "test":
            raise ValueError(
                "MemmapActivationParser was constructed over a train capture "
                "(split_strategy='three_way'). Test rows live in the separate "
                "test capture directory — construct a second parser with "
                "split_strategy='none' for test data."
            )
        if split == "train":
            indices = self._train_idx
        elif split == "val":
            indices = self._val_idx
        else:
            raise ValueError(f"Unknown split: {split!r}")

        return MemmapContrastiveDataset(
            self._capture_dir,
            split="all",
            random_seed=self._random_seed,
            relevant_layers=relevant_layers,
            num_views=num_views,
            fixed_layer=fixed_layer,
            pad_length=pad_length,
            include_response_logprobs=include_response_logprobs,
            response_logprobs_top_k=response_logprobs_top_k,
            _override_indices=indices,
            _override_split_name=split,
        )
