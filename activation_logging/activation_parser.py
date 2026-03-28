"""
activation_parser.py
Handles parsing of metadata from JSON files and looking up corresponding activations from Zarr.
"""
import json
import os
import hashlib
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from loguru import logger
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import random
import numpy as np

from .activations_logger import ActivationsLogger

class ActivationDataset(Dataset):
    """PyTorch Dataset for loading activation data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        activations_path: str,
        split: Literal['train', 'test'],
        relevant_layers: Optional[List[int]] = None,
        logger_type: str = "zarr",
        fixed_layer: Optional[int] = None,
        random_seed: int = 42,
        verbose: bool = False,
        pad_length: int = 63,
        min_target_layers: int = 2,
        num_views: int = 2,
        view_sampling_with_replacement: bool = False,
        return_all_activations: bool = False,
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame containing the metadata
            activations_path: Path to the activations storage (.zarr store)
            split: Which split to use ('train' or 'test')
            relevant_layers: List of layer indices to use (default: layers 16-29)
            logger_type: Activation storage type. Only 'zarr' is supported.
            fixed_layer: If specified, one activation will always be from this layer (index in relevant_layers)
            random_seed: Random seed for train/test split (default: 42)
            verbose: Whether to log initialization messages (default: False for datasets)
            num_views: Number of contrastive views to sample per sample (K-view, K>=2)
            view_sampling_with_replacement: Whether to sample view indices with replacement
            return_all_activations: If True, include all relevant layers in all_activations
            include_response_logprobs: If True, include per-token top-k response logprobs in each sample
            response_logprobs_top_k: Number of top logprobs to expose per response token
        """
        self.activations_path = activations_path
        normalized_logger_type = str(logger_type).strip().lower()
        if normalized_logger_type != "zarr":
            raise ValueError(
                f"Unsupported logger_type='{logger_type}'. "
                "Only Zarr activation storage is supported. "
                "LMDB/JSON/WDS backends are deprecated and no longer supported."
            )
        self.logger_type = normalized_logger_type
        self.random_seed = random_seed
        self.verbose = verbose
        # For backward compatibility
        self.lmdb_path = activations_path
        self._activation_parser = None
        self.split = split
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.relevant_layers = relevant_layers if relevant_layers is not None else list(range(16,30))
        self.fixed_layer = fixed_layer
        self.pad_length = pad_length
        self.min_target_layers = min_target_layers
        self.num_views = int(num_views)
        self.view_sampling_with_replacement = bool(view_sampling_with_replacement)
        self.return_all_activations = return_all_activations
        self.include_response_logprobs = bool(include_response_logprobs)
        if int(response_logprobs_top_k) <= 0:
            raise ValueError("response_logprobs_top_k must be >= 1")
        self.response_logprobs_top_k = int(response_logprobs_top_k)
        self.response_logprobs_top_k = int(
            os.environ.get("ACTIVATION_LOGPROBS_TOPK", self.response_logprobs_top_k)
        )
        if self.response_logprobs_top_k <= 0:
            self.response_logprobs_top_k = 1
        if self.num_views < 2:
            raise ValueError("num_views must be >= 2")
        self._use_zarr_fast_path = self._is_zarr_path(activations_path)
        if not self._use_zarr_fast_path:
            raise ValueError(
                f"Unsupported activations_path='{activations_path}'. "
                "Only .zarr activation stores are supported."
            )

    @staticmethod
    def _is_zarr_path(path: str) -> bool:
        return str(path).endswith(".zarr")

    @property
    def activation_parser(self):
        if self._activation_parser is None:
            self._activation_parser = ActivationParser("", "", self.activations_path,
                                                     df=self.df, logger_type=self.logger_type,
                                                     random_seed=self.random_seed, verbose=self.verbose)
        return self._activation_parser

    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data point.

        Args:
            idx: Index of the data point

        Returns:
            Dictionary containing:
            - hashkey: The prompt hash
            - halu: Whether this is a hallucination
            - all_activations: All padded activations (missing layers zero-filled)
            - views_activations: Activations from selected K views, shape (K, T, H)
            - view_indices: Indices of selected views in relevant_layers, shape (K,)
            - input_length: Length of the input prompt
        """
        if self._use_zarr_fast_path:
            return self._getitem_zarr(idx)
        return self._getitem_standard(idx)

    def _getitem_standard(self, idx: int) -> Dict[str, Any]:
        row, result, activations, input_length = self.activation_parser.get_activations(idx)
        entry_key = self.activation_parser.select_primary_key(row['prompt_hash'])

        # Filter to relevant layers and pad if necessary
        padded_activations = []
        for layer_idx in self.relevant_layers:
            act = activations[layer_idx]
            if act is not None:
                if act.ndim == 2:
                    act = act.unsqueeze(0)
                seq_len = act.shape[1]

                if seq_len < self.pad_length:
                    # Generate random noise with same shape as activations
                    noise = torch.randn(act.shape[0], self.pad_length - seq_len, act.shape[2],
                                      device=act.device, dtype=act.dtype)
                    # Concatenate original activations with noise
                    act = torch.cat([act, noise], dim=1)
                elif seq_len > self.pad_length:
                    # Truncate if longer than pad_length
                    act = act[:, :self.pad_length, :]
            # If act is None, keep it as None
            padded_activations.append(act)

        # Select K layers, with optional fixed layer
        available_layers = [i for i, act in enumerate(padded_activations) if act is not None]
        if len(available_layers) < self.min_target_layers:
            raise ValueError(
                f"Not enough targeted layers available (found {len(available_layers)} layers; "
                f"need at least {self.min_target_layers})."
            )

        selected_view_indices = self._select_view_indices(available_layers)
        filled_activations = self._fill_missing_layers(padded_activations)
        selected_views = [filled_activations[i] for i in selected_view_indices]
        selected_views = [act.squeeze(0) if act.ndim == 3 and act.shape[0] == 1 else act for act in selected_views]
        views_activations = torch.stack(selected_views, dim=0)

        sample = {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': filled_activations,
            'views_activations': views_activations,
            'view_indices': torch.tensor(selected_view_indices, dtype=torch.long),
            'input_length': input_length
        }
        return self._attach_response_logprobs(sample, entry_key)

    def _fill_missing_layers(self, layers: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        reference = next((act for act in layers if act is not None), None)
        if reference is None:
            raise ValueError("No available layers found for this sample")
        return [torch.zeros_like(reference) if act is None else act for act in layers]

    def _select_view_indices(self, available_layers: List[int]) -> List[int]:
        if self.fixed_layer is not None and self.fixed_layer not in available_layers:
            raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

        if self.fixed_layer is not None:
            other_layers = [i for i in available_layers if i != self.fixed_layer]
            if self.view_sampling_with_replacement:
                if not other_layers and self.num_views > 1:
                    raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
                sampled = random.choices(other_layers, k=self.num_views - 1) if self.num_views > 1 else []
            else:
                if len(other_layers) < self.num_views - 1:
                    raise ValueError(
                        f"Not enough layers for strict K-view sampling with fixed layer: "
                        f"need {self.num_views - 1} additional layers, found {len(other_layers)}"
                    )
                sampled = random.sample(other_layers, self.num_views - 1)
            return [self.fixed_layer, *sampled]

        if self.view_sampling_with_replacement:
            if not available_layers:
                raise ValueError("No available layers found for this sample")
            return random.choices(available_layers, k=self.num_views)

        if len(available_layers) < self.num_views:
            raise ValueError(
                f"Not enough layers for strict K-view sampling: "
                f"need {self.num_views}, found {len(available_layers)}"
            )
        return random.sample(available_layers, self.num_views)

    @staticmethod
    def _to_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _attach_response_logprobs(self, sample: Dict[str, Any], entry_key: str) -> Dict[str, Any]:
        if not self.include_response_logprobs:
            return sample

        target_len = int(self.pad_length)
        target_top_k = int(self.response_logprobs_top_k)

        token_ids_out = torch.full((target_len,), -1, dtype=torch.int32)
        token_logprobs_out = torch.full((target_len,), float("nan"), dtype=torch.float32)
        topk_ids_out = torch.full((target_len, target_top_k), -1, dtype=torch.int32)
        topk_logprobs_out = torch.full((target_len, target_top_k), float("nan"), dtype=torch.float32)
        token_mask = torch.zeros((target_len,), dtype=torch.bool)

        copied_len = 0
        copied_top_k = 0

        payload = self.activation_parser.get_response_logprobs(entry_key)
        if payload is not None:
            token_ids = self._to_tensor(payload.get("response_token_ids"), dtype=torch.int32)
            token_logprobs = self._to_tensor(payload.get("response_token_logprobs"), dtype=torch.float32)
            topk_ids = self._to_tensor(payload.get("response_topk_token_ids"), dtype=torch.int32)
            topk_logprobs = self._to_tensor(payload.get("response_topk_logprobs"), dtype=torch.float32)

            if (
                token_ids is not None
                and token_logprobs is not None
                and topk_ids is not None
                and topk_logprobs is not None
                and token_ids.ndim == 1
                and token_logprobs.ndim == 1
                and topk_ids.ndim == 2
                and topk_logprobs.ndim == 2
            ):
                copied_len = min(
                    target_len,
                    int(token_ids.shape[0]),
                    int(token_logprobs.shape[0]),
                    int(topk_ids.shape[0]),
                    int(topk_logprobs.shape[0]),
                )
                copied_top_k = min(target_top_k, int(topk_ids.shape[1]), int(topk_logprobs.shape[1]))
                if copied_len > 0 and copied_top_k > 0:
                    token_ids_out[:copied_len] = token_ids[:copied_len]
                    token_logprobs_out[:copied_len] = token_logprobs[:copied_len]
                    topk_ids_out[:copied_len, :copied_top_k] = topk_ids[:copied_len, :copied_top_k]
                    topk_logprobs_out[:copied_len, :copied_top_k] = topk_logprobs[:copied_len, :copied_top_k]
                    token_mask[:copied_len] = True

        sample["response_token_ids"] = token_ids_out
        sample["response_token_logprobs"] = token_logprobs_out
        sample["response_topk_token_ids"] = topk_ids_out
        sample["response_topk_logprobs"] = topk_logprobs_out
        sample["response_logprob_mask"] = token_mask
        sample["response_logprob_len"] = torch.tensor(copied_len, dtype=torch.long)
        sample["response_logprobs_top_k"] = torch.tensor(target_top_k, dtype=torch.long)
        sample["response_logprobs_available_top_k"] = torch.tensor(copied_top_k, dtype=torch.long)
        return sample

    def _getitem_zarr(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        entry_key = self.activation_parser.select_primary_key(row['prompt_hash'])
        metadata = self.activation_parser.get_entry_metadata(entry_key)
        input_length = metadata.get('input_length') or metadata.get('prompt_len', 0)

        padded_activations: List[Optional[torch.Tensor]] = [None] * len(self.relevant_layers)

        def load_layer(layer_pos: int) -> Optional[torch.Tensor]:
            actual_layer = self.relevant_layers[layer_pos]
            act = self.activation_parser.get_layer_activation(entry_key, actual_layer, sequence_mode="response")
            if act is None:
                return None
            if act.ndim == 2:
                act = act.unsqueeze(0)
            seq_len = act.shape[1]
            if seq_len < self.pad_length:
                noise = torch.randn(act.shape[0], self.pad_length - seq_len, act.shape[2],
                                  device=act.device, dtype=act.dtype)
                act = torch.cat([act, noise], dim=1)
            elif seq_len > self.pad_length:
                act = act[:, :self.pad_length, :]
            return act

        if self.return_all_activations:
            for layer_pos in range(len(self.relevant_layers)):
                padded_activations[layer_pos] = load_layer(layer_pos)

            available_layers = [i for i, act in enumerate(padded_activations) if act is not None]
            if len(available_layers) < self.min_target_layers:
                raise ValueError(
                    f"Not enough targeted layers available (found {len(available_layers)} layers; "
                    f"need at least {self.min_target_layers})."
                )

            selected_view_indices = self._select_view_indices(available_layers)
        else:
            # Lazy path: load as few layers as needed to select K views.
            # We avoid eagerly loading every relevant layer for each sample.
            candidate_positions = list(range(len(self.relevant_layers)))

            if self.fixed_layer is not None and self.fixed_layer not in candidate_positions:
                raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

            # Load fixed layer first (if configured), then progressively load others.
            if self.fixed_layer is not None:
                padded_activations[self.fixed_layer] = load_layer(self.fixed_layer)
                if padded_activations[self.fixed_layer] is None:
                    raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

            remaining = [pos for pos in candidate_positions if pos != self.fixed_layer]
            random.shuffle(remaining)

            def current_available() -> List[int]:
                return [i for i, act in enumerate(padded_activations) if act is not None]

            selected_view_indices: Optional[List[int]] = None
            last_error: Optional[Exception] = None

            # Attempt selection with progressively more loaded candidates.
            for candidate in [None, *remaining]:
                if candidate is not None:
                    padded_activations[candidate] = load_layer(candidate)

                available_layers = current_available()
                if len(available_layers) < self.min_target_layers:
                    continue

                try:
                    selected_view_indices = self._select_view_indices(available_layers)
                    break
                except ValueError as exc:
                    last_error = exc

            if selected_view_indices is None:
                available_layers = current_available()
                if len(available_layers) < self.min_target_layers:
                    raise ValueError(
                        f"Not enough targeted layers available (found {len(available_layers)} layers; "
                        f"need at least {self.min_target_layers})."
                    )
                if last_error is not None:
                    raise last_error
                raise ValueError("Could not select K views from available layers")

        filled_activations = self._fill_missing_layers(padded_activations)
        selected_views = [filled_activations[i] for i in selected_view_indices]
        selected_views = [act.squeeze(0) if act.ndim == 3 and act.shape[0] == 1 else act for act in selected_views]
        views_activations = torch.stack(selected_views, dim=0)

        sample = {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': filled_activations,
            'views_activations': views_activations,
            'view_indices': torch.tensor(selected_view_indices, dtype=torch.long),
            'input_length': input_length
        }
        return self._attach_response_logprobs(sample, entry_key)
    



class SingleLayerDataset(Dataset):
    """Deterministic single-layer dataset for linear probing.

    Returns activations for exactly one layer — no view sampling, no
    randomness.  ``__getitem__`` returns ``views_activations`` with shape
    ``(1, T, H)`` for compatibility with ``LinearProbeTrainer``.
    """

    def __init__(
        self,
        cache: np.ndarray,
        labels: np.ndarray,
        prompt_hashes: List[str],
        layer_pos: int,
        layer_id: int,
        _row_indices: Optional[np.ndarray] = None,
    ):
        self.cache = cache                    # (N, L, T, H)
        self.labels = labels
        self.prompt_hashes = prompt_hashes
        self.layer_pos = layer_pos            # positional index into dim-1
        self.layer_id = layer_id              # model layer number (for metadata)
        self._row_indices = _row_indices

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_idx = int(self._row_indices[idx]) if self._row_indices is not None else idx
        act = torch.from_numpy(np.array(self.cache[cache_idx, self.layer_pos]))  # (T, H)
        return {
            'hashkey': self.prompt_hashes[idx],
            'halu': torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            'views_activations': act.unsqueeze(0),  # (1, T, H)
            'view_indices': torch.tensor([self.layer_id], dtype=torch.long),
        }


class PreloadedActivationDataset(Dataset):
    """RAM-resident activation dataset.

    All activations are loaded from zarr into a numpy array at construction time
    via :meth:`ActivationParser.get_dataset` with ``preload=True``.
    ``__getitem__`` performs zero-copy views via ``torch.from_numpy`` — no disk
    I/O during training.

    Safe for multi-worker DataLoaders: on Linux the default fork start method
    means workers inherit the array via copy-on-write and physical pages are
    never duplicated (validated by PSS measurement).
    """

    def __init__(
        self,
        cache: np.ndarray,
        labels: np.ndarray,
        prompt_hashes: List[str],
        num_views: int,
        pad_length: int,
        fixed_layer: Optional[int] = None,
        view_sampling_with_replacement: bool = False,
        min_target_layers: int = 2,
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        relevant_layers: Optional[List[int]] = None,
        _row_indices: Optional[np.ndarray] = None,
        _logprob_token_ids: Optional[np.ndarray] = None,
        _logprob_token_logprobs: Optional[np.ndarray] = None,
        _logprob_topk_ids: Optional[np.ndarray] = None,
        _logprob_topk_logprobs: Optional[np.ndarray] = None,
    ):
        self.cache = cache                    # (N, L, T, H) float16, never written after init
        self.labels = labels                  # (N,) or (N_split,) when using _row_indices
        self.prompt_hashes = prompt_hashes    # len N_split
        self.num_views = int(num_views)
        self.pad_length = int(pad_length)
        self.fixed_layer = fixed_layer
        self.view_sampling_with_replacement = bool(view_sampling_with_replacement)
        self.min_target_layers = int(min_target_layers)
        self.L = cache.shape[1]
        self.relevant_layers = relevant_layers if relevant_layers is not None else list(range(self.L))
        self.include_response_logprobs = bool(include_response_logprobs)
        self.response_logprobs_top_k = int(response_logprobs_top_k)
        self._row_indices = _row_indices      # None = direct indexing, array = memmap indirection
        self._logprob_token_ids = _logprob_token_ids
        self._logprob_token_logprobs = _logprob_token_logprobs
        self._logprob_topk_ids = _logprob_topk_ids
        self._logprob_topk_logprobs = _logprob_topk_logprobs

    def __len__(self) -> int:
        return len(self.labels)

    def slice_layers(
        self,
        layers: List[int],
        num_views: Optional[int] = None,
    ) -> "PreloadedActivationDataset":
        """Return a new dataset backed by a subset of layers from this cache.

        The returned dataset shares the underlying numpy memory — no copy is
        made.  ``layers`` are specified as model layer indices (matching
        ``self.relevant_layers``), not positional indices.

        Args:
            layers: Model layer indices to keep (must be a subset of
                ``self.relevant_layers``).
            num_views: Number of views to sample.  Defaults to
                ``len(layers)`` so that all selected layers are used.
        """
        pos_indices = []
        for layer in layers:
            if layer not in self.relevant_layers:
                raise ValueError(
                    f"Layer {layer} not in preloaded relevant_layers {self.relevant_layers}"
                )
            pos_indices.append(self.relevant_layers.index(layer))

        sliced_cache = self.cache[:, pos_indices, :, :]  # numpy fancy index — view when contiguous
        if num_views is None:
            num_views = len(layers)

        return PreloadedActivationDataset(
            cache=sliced_cache,
            labels=self.labels,
            prompt_hashes=self.prompt_hashes,
            num_views=num_views,
            pad_length=self.pad_length,
            fixed_layer=self.fixed_layer,
            view_sampling_with_replacement=self.view_sampling_with_replacement,
            min_target_layers=min(self.min_target_layers, len(layers)),
            include_response_logprobs=self.include_response_logprobs,
            response_logprobs_top_k=self.response_logprobs_top_k,
            relevant_layers=list(layers),
            _row_indices=self._row_indices,
            _logprob_token_ids=self._logprob_token_ids,
            _logprob_token_logprobs=self._logprob_token_logprobs,
            _logprob_topk_ids=self._logprob_topk_ids,
            _logprob_topk_logprobs=self._logprob_topk_logprobs,
        )

    def get_single_layer_dataset(self, layer: int) -> "SingleLayerDataset":
        """Return a deterministic single-layer dataset for linear probing.

        Unlike ``slice_layers``, this returns a ``SingleLayerDataset`` that
        always yields the activation for exactly one layer — no view sampling.

        Args:
            layer: Model layer index (must be in ``self.relevant_layers``).
        """
        if layer not in self.relevant_layers:
            raise ValueError(
                f"Layer {layer} not in preloaded relevant_layers {self.relevant_layers}"
            )
        layer_pos = self.relevant_layers.index(layer)
        return SingleLayerDataset(
            cache=self.cache,
            labels=self.labels,
            prompt_hashes=self.prompt_hashes,
            layer_pos=layer_pos,
            layer_id=layer,
            _row_indices=self._row_indices,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_idx = int(self._row_indices[idx]) if self._row_indices is not None else idx
        acts = torch.from_numpy(np.array(self.cache[cache_idx]))  # (L, T, H)
        view_indices = self._select_view_indices(list(range(self.L)))
        views = acts[view_indices]                 # (K, T, H)
        sample: Dict[str, Any] = {
            'hashkey': self.prompt_hashes[idx],
            'halu': torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            'views_activations': views,
            'view_indices': torch.tensor(view_indices, dtype=torch.long),
        }
        if self.include_response_logprobs:
            sample.update(self._get_logprobs(idx, cache_idx))
        return sample

    def _select_view_indices(self, available_layers: List[int]) -> List[int]:
        if self.fixed_layer is not None and self.fixed_layer not in available_layers:
            raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

        if self.fixed_layer is not None:
            other_layers = [i for i in available_layers if i != self.fixed_layer]
            if self.view_sampling_with_replacement:
                if not other_layers and self.num_views > 1:
                    raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
                sampled = random.choices(other_layers, k=self.num_views - 1) if self.num_views > 1 else []
            else:
                if len(other_layers) < self.num_views - 1:
                    raise ValueError(
                        f"Not enough layers for strict K-view sampling with fixed layer: "
                        f"need {self.num_views - 1} additional layers, found {len(other_layers)}"
                    )
                sampled = random.sample(other_layers, self.num_views - 1)
            return [self.fixed_layer, *sampled]

        if self.view_sampling_with_replacement:
            if not available_layers:
                raise ValueError("No available layers found for this sample")
            return random.choices(available_layers, k=self.num_views)

        if len(available_layers) < self.num_views:
            raise ValueError(
                f"Not enough layers for strict K-view sampling: "
                f"need {self.num_views}, found {len(available_layers)}"
            )
        return random.sample(available_layers, self.num_views)

    def _get_logprobs(self, idx: int, cache_idx: Optional[int] = None) -> Dict[str, Any]:
        if cache_idx is None:
            cache_idx = idx
        target_len = self.pad_length
        target_top_k = self.response_logprobs_top_k

        token_ids_out = torch.full((target_len,), -1, dtype=torch.int32)
        token_logprobs_out = torch.full((target_len,), float("nan"), dtype=torch.float32)
        topk_ids_out = torch.full((target_len, target_top_k), -1, dtype=torch.int32)
        topk_logprobs_out = torch.full((target_len, target_top_k), float("nan"), dtype=torch.float32)
        token_mask = torch.zeros((target_len,), dtype=torch.bool)
        copied_len = 0
        copied_top_k = 0

        if (
            self._logprob_token_ids is not None
            and self._logprob_token_logprobs is not None
            and self._logprob_topk_ids is not None
            and self._logprob_topk_logprobs is not None
        ):
            raw_ids = self._logprob_token_ids[cache_idx]
            raw_lps = self._logprob_token_logprobs[cache_idx]
            raw_topk_ids = self._logprob_topk_ids[cache_idx]
            raw_topk_lps = self._logprob_topk_logprobs[cache_idx]

            copied_len = min(target_len, raw_ids.shape[0])
            copied_top_k = min(target_top_k, raw_topk_ids.shape[1])
            if copied_len > 0 and copied_top_k > 0:
                token_ids_out[:copied_len] = torch.from_numpy(raw_ids[:copied_len].astype(np.int32))
                token_logprobs_out[:copied_len] = torch.from_numpy(raw_lps[:copied_len].astype(np.float32))
                topk_ids_out[:copied_len, :copied_top_k] = torch.from_numpy(
                    raw_topk_ids[:copied_len, :copied_top_k].astype(np.int32)
                )
                topk_logprobs_out[:copied_len, :copied_top_k] = torch.from_numpy(
                    raw_topk_lps[:copied_len, :copied_top_k].astype(np.float32)
                )
                token_mask[:copied_len] = True

        return {
            "response_token_ids": token_ids_out,
            "response_token_logprobs": token_logprobs_out,
            "response_topk_token_ids": topk_ids_out,
            "response_topk_logprobs": topk_logprobs_out,
            "response_logprob_mask": token_mask,
            "response_logprob_len": torch.tensor(copied_len, dtype=torch.long),
            "response_logprobs_top_k": torch.tensor(target_top_k, dtype=torch.long),
            "response_logprobs_available_top_k": torch.tensor(copied_top_k, dtype=torch.long),
        }


class TokenTrajectoryDataset(Dataset):
    """Token-trajectory contrastive dataset.

    Wraps a :class:`PreloadedActivationDataset` (cache shape ``(N, L, T, H)``).
    Instead of sampling K layers per sample (the existing approach), each call to
    ``__getitem__`` samples K token positions from the response and returns their
    full layer-stack trajectory.

    Each view has shape ``(L, H)`` — the residual stream across all L layers for
    one response token.  ``views_activations`` has shape ``(num_views, L, H)``.

    The shape contract is identical to :class:`PreloadedActivationDataset`, so
    ``_contrastive_collate_kview`` and ``train_contrastive`` work unchanged:
    the ``seq_len`` dimension is now ``L`` (layer count) rather than ``T`` (token
    count padded to ``pad_length``).

    Args:
        preloaded: A :class:`PreloadedActivationDataset` whose ``.cache`` array
            has shape ``(N, L, T, H)``.  The dataset's labels, prompt_hashes, and
            optional ``_row_indices`` are reused directly.
        response_lengths: Integer array of shape ``(N,)`` giving the number of
            real (non-padded) response tokens for each cache row.  Obtain from
            the zarr store: ``zarr.open(path)["response_len"][:]``.
        num_views: Number of token positions to sample per example.
        min_response_len: Samples whose actual response length is shorter than
            this are skipped during dataset construction (filtered out of the
            index).  Must be >= ``num_views``.
    """

    def __init__(
        self,
        preloaded: "PreloadedActivationDataset",
        response_lengths: np.ndarray,
        num_views: int = 2,
        min_response_len: Optional[int] = None,
    ):
        self.preloaded = preloaded
        self.response_lengths = np.asarray(response_lengths, dtype=np.int32)
        self.num_views = int(num_views)
        min_len = int(min_response_len) if min_response_len is not None else self.num_views

        # Build a valid-index list: logical indices into the preloaded dataset
        # where the actual response is long enough to sample num_views tokens.
        valid = []
        for logical_idx in range(len(preloaded)):
            cache_idx = (
                int(preloaded._row_indices[logical_idx])
                if preloaded._row_indices is not None
                else logical_idx
            )
            actual_len = min(
                int(self.response_lengths[cache_idx]),
                preloaded.cache.shape[2],  # T dimension cap
            )
            if actual_len >= min_len:
                valid.append(logical_idx)

        self._valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        logical_idx = int(self._valid_indices[idx])
        cache_idx = (
            int(self.preloaded._row_indices[logical_idx])
            if self.preloaded._row_indices is not None
            else logical_idx
        )

        acts = torch.from_numpy(
            np.array(self.preloaded.cache[cache_idx], dtype=np.float32)
        )  # (L, T, H)

        actual_len = min(
            int(self.response_lengths[cache_idx]),
            acts.shape[1],
        )

        token_positions = random.sample(range(actual_len), self.num_views)

        # acts[:, pos, :] → (L, H); stack K of them → (K, L, H)
        trajectories = acts[:, token_positions, :].permute(1, 0, 2).contiguous()

        return {
            "views_activations": trajectories,
            "view_indices": torch.tensor(token_positions, dtype=torch.long),
            "halu": torch.tensor(
                float(self.preloaded.labels[logical_idx]), dtype=torch.float32
            ),
            "hashkey": self.preloaded.prompt_hashes[logical_idx],
        }


class ActivationParser:
    def __init__(self, inference_json: str, eval_json: str, activations_path: str,
                 df: Optional[pd.DataFrame] = None, logger_type: str = "zarr",
                 random_seed: int = 42, verbose: bool = True):
        """
        Initialize the ActivationParser.

        Args:
            inference_json: Path to the inference JSON file
            eval_json: Path to the evaluation JSON file
            activations_path: Path to the activations storage (.zarr store)
            df: Optional DataFrame to use instead of loading from JSON files
            logger_type: Activation storage type. Only 'zarr' is supported.
            random_seed: Random seed for train/test split (default: 42)
            verbose: Whether to log initialization and metadata loading messages (default: True)
        """
        self.inference_json = Path(inference_json)
        if not self.inference_json.exists():
            raise FileNotFoundError(f"JSON file not found: {inference_json}")

        self.eval_json = Path(eval_json)
        if not self.eval_json.exists():
            raise FileNotFoundError(f"JSON file not found: {eval_json}")

        if not str(activations_path).endswith(".zarr"):
            raise ValueError(
                f"Unsupported activations_path='{activations_path}'. "
                "Only .zarr activation stores are supported."
            )

        self.activations_path = activations_path
        normalized_logger_type = str(logger_type).strip().lower()
        if normalized_logger_type != "zarr":
            raise ValueError(
                f"Unsupported logger_type='{logger_type}'. "
                "Only Zarr activation storage is supported. "
                "LMDB/JSON/WDS backends are deprecated and no longer supported."
            )
        self.logger_type = normalized_logger_type
        self.random_seed = random_seed
        self.verbose = verbose
        self._activation_logger = None
        self._group_index = None
        self._wds_shards = None
        self._preloaded_splits = None   # cached result of _preload_all_splits

        # For backward compatibility, also store as lmdb_path
        self.lmdb_path = activations_path

        # Load metadata from JSON or use provided DataFrame
        self.df = df if df is not None else self._load_metadata()

    @staticmethod
    def _detect_wds_shards(activations_path: str) -> Optional[str]:
        path = Path(activations_path)
        if path.is_dir():
            candidate = path / "webdataset"
            if candidate.exists() and candidate.is_dir():
                return str(candidate / "*.tar")
        return None

    @property
    def activation_logger(self):
        if self._activation_logger is None:
            # Add staggered initialization to reduce filesystem contention with many workers
            import time
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None and worker_info.num_workers > 8:
                # Stagger by 50ms per worker for large worker counts
                time.sleep(worker_info.id * 0.05)

            self._activation_logger = ActivationsLogger(lmdb_path=self.activations_path, read_only=True, verbose=self.verbose)
        return self._activation_logger

    def get_entry_metadata(self, entry_key: str) -> Dict[str, Any]:
        """Retrieve metadata only for a given entry key."""
        if self.activation_logger is None:
            return {}
        metadata = self.activation_logger.get_entry_by_key(entry_key, metadata_only=True)
        return metadata or {}

    def get_layer_activation(self, entry_key: str, layer_idx: int, sequence_mode: str = "response") -> Optional[torch.Tensor]:
        """Retrieve a single layer activation for the given entry key and sequence mode."""
        if self.activation_logger is None:
            return None
        if hasattr(self.activation_logger, "get_layer_activation"):
            return self.activation_logger.get_layer_activation(entry_key, layer_idx, sequence_mode=sequence_mode)
        return None

    def get_response_logprobs(self, entry_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve response token-level logprob features for the given entry key."""
        if self.activation_logger is None:
            return None
        if hasattr(self.activation_logger, "get_response_logprobs"):
            return self.activation_logger.get_response_logprobs(entry_key)
        return None

    def _load_metadata(self) -> Dict[str, Any]:
        gendf = pd.read_json(self.inference_json, lines=True)

        with open(self.eval_json, 'r') as f:
            data = json.loads(f.read())
            
        gendf['abstain'] = data['abstantion']
        gendf['halu'] = data['halu_test_res']

        # note that previous versions may have used a different variation (eg prepending "user")
        gendf['prompt_hash'] = gendf['prompt'].apply(lambda x :
                                                    hashlib.sha256(x.encode("utf-8")).hexdigest())

        gendf = gendf[~gendf['prompt_hash'].duplicated(keep=False)]

        keys = self.activation_logger.list_entries()
        base_keys = {key.split("_")[0] for key in keys}
        gendf = gendf[gendf['prompt_hash'].isin(base_keys)]

        gendf = gendf[~gendf['abstain']]

        # Apply train/test split
        train_df, test_df = train_test_split(gendf, test_size=0.2,
                                           stratify=gendf['halu'], random_state=self.random_seed)

        gendf['split'] = 'unassigned'
        gendf.loc[train_df.index, 'split'] = 'train'
        gendf.loc[test_df.index, 'split'] = 'test'

        if self.verbose:
            logger.info(f"Found {len(gendf)} prompts with activations")
            logger.info(f"Found {len(gendf[gendf['halu']])} hallucinations")
            logger.info(f"Found {len(gendf[~gendf['halu']])} non-hallucinations")
            logger.info(f"Found {gendf['halu'].sum()/len(gendf)}% hallucinations")
            logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

        # gen df contains these columns ['index', 'title', 'h_score_cat', 'pageid', 'revid', 'description',
        # 'categories', 'reference', 'prompt', 'answer', 'generation', 'abstain',
        # 'halu', 'prompt_hash', 'split'] 

        return gendf

            
    def _hash_prompt(self, prompt: str) -> str:
        """
        Hash a prompt string to match the format used in ActivationsLogger.
        
        Args:
            prompt: The prompt string to hash
            
        Returns:
            SHA-256 hash of the prompt
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _build_group_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build an index of activation entries grouped by prompt hash or sample group ID.

        Returns:
            Dictionary mapping group ID to a list of entry metadata dicts.
        """
        if self._group_index is not None:
            return self._group_index

        group_index: Dict[str, List[Dict[str, Any]]] = {}
        keys = self.activation_logger.list_entries()

        for key in keys:
            try:
                metadata = self.activation_logger.get_entry_by_key(key, metadata_only=True)
            except Exception:
                metadata = None

            if metadata is None:
                group_id = key.split("_")[0]
                entry = {"key": key, "sample_index": None, "request_id": None}
            else:
                group_id = (
                    metadata.get("sample_group_id")
                    or metadata.get("prompt_hash")
                    or key.split("_")[0]
                )
                entry = {
                    "key": key,
                    "sample_index": metadata.get("sample_index"),
                    "request_id": metadata.get("request_id"),
                }

            group_index.setdefault(group_id, []).append(entry)

        self._group_index = group_index
        return group_index

    def get_group_keys(self, prompt_hash: str) -> List[str]:
        """
        Get all activation entry keys associated with a prompt hash or group ID.

        Args:
            prompt_hash: Prompt hash or sample group ID

        Returns:
            List of activation entry keys
        """
        group_index = self._build_group_index()
        entries = group_index.get(prompt_hash, [])
        return [entry["key"] for entry in entries]

    def select_primary_key(self, prompt_hash: str) -> str:
        """
        Select a primary activation entry key for a prompt hash.

        Priority order:
        1) sample_index == 0
        2) entry without sample_index
        3) first key in sorted order

        Args:
            prompt_hash: Prompt hash or sample group ID

        Returns:
            Selected activation entry key
        """
        group_index = self._build_group_index()
        entries = group_index.get(prompt_hash, [])

        if not entries:
            return prompt_hash

        for entry in entries:
            if entry.get("sample_index") == 0:
                return entry["key"]

        for entry in entries:
            if entry.get("sample_index") is None:
                return entry["key"]

        return sorted(entry["key"] for entry in entries)[0]
        
    def get_activations(self, idx) -> Optional[Dict[str, Any]]:
        row = self.df.iloc[idx]
        entry_key = self.select_primary_key(row['prompt_hash'])
        result = self.activation_logger.get_entry(entry_key)
        input_length = result['input_length']
        
        # Check how the activations were logged
        logging_config = result.get('logging_config', {})
        sequence_mode = logging_config.get('sequence_mode', 'all')
        
        # Only trim if the activations weren't already logged in response mode
        if sequence_mode != 'response':
            # Trim activations to only include generated tokens
            activations = []
            for act in result['all_layers_activations']:
                if act is not None:
                    activations.append(act[:, input_length:, :])
                else:
                    activations.append(None)
        else:
            activations = result['all_layers_activations']

        return row, result, activations, input_length

    @staticmethod
    def _estimate_preload_bytes(
        N: int,
        L: int,
        T: int,
        H: int,
        include_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
    ) -> int:
        """Estimate bytes required to hold a preloaded activation cache in RAM."""
        act_bytes = N * L * T * H * 2  # float16
        logprob_bytes = 0
        if include_logprobs:
            # token_ids (int32) + token_logprobs (float32) + topk_ids (int32) + topk_logprobs (float32)
            logprob_bytes = N * T * (4 + 4 + response_logprobs_top_k * 4 + response_logprobs_top_k * 4)
        return act_bytes + logprob_bytes

    @staticmethod
    def _check_ram_available(estimated_bytes: int, buffer: float = 0.10) -> None:
        """Raise MemoryError if estimated_bytes exceed available RAM minus a safety buffer.

        Args:
            estimated_bytes: Expected allocation size in bytes.
            buffer: Fractional headroom required beyond the allocation (default 10%).
        """
        available = psutil.virtual_memory().available
        required = estimated_bytes * (1.0 + buffer)
        if required > available:
            raise MemoryError(
                f"Preload would require ~{estimated_bytes / 1024**3:.1f} GB but only "
                f"{available / 1024**3:.1f} GB available "
                f"(including {buffer * 100:.0f}% safety buffer). "
                "Pass check_ram=False to skip this check."
            )

    # ------------------------------------------------------------------
    # Memmap preload cache
    # ------------------------------------------------------------------

    def _memmap_cache_fingerprint(
        self,
        relevant_layers: List[int],
        pad_length: int,
        include_logprobs: bool,
        response_logprobs_top_k: int,
    ) -> str:
        """Deterministic cache key for a set of preload parameters."""
        zarr_path_resolved = str(Path(self.activations_path).resolve())
        zarr_count = int(self.activation_logger._response_activations.shape[0])
        key_parts = (
            zarr_path_resolved,
            sorted(relevant_layers),
            pad_length,
            zarr_count,
            self.random_seed,
            include_logprobs,
            response_logprobs_top_k,
        )
        return hashlib.sha256(repr(key_parts).encode()).hexdigest()[:16]

    def _memmap_cache_dir(self, fingerprint: str) -> Path:
        """Resolve cache directory path under the zarr store."""
        zarr_path = Path(self.activations_path).resolve()
        return zarr_path / "_memmap_cache" / fingerprint

    def _save_memmap_cache(
        self,
        cache_dir: Path,
        data: Dict[str, Any],
        fingerprint: str,
        relevant_layers: List[int],
        pad_length: int,
        include_logprobs: bool,
        response_logprobs_top_k: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> None:
        """Write preloaded arrays as .npy files for subsequent memmap loading."""
        import shutil

        tmp_dir = cache_dir.parent / f".tmp_{fingerprint}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(tmp_dir / "activations.npy", data['cache'])
            np.save(tmp_dir / "labels.npy", data['labels'].astype(np.int8))
            np.save(tmp_dir / "prompt_hashes.npy",
                    np.array(data['prompt_hashes'], dtype='U64'))
            np.save(tmp_dir / "train_indices.npy", train_indices)
            np.save(tmp_dir / "test_indices.npy", test_indices)

            if include_logprobs:
                for key in ('logprob_token_ids', 'logprob_token_logprobs',
                            'logprob_topk_ids', 'logprob_topk_logprobs'):
                    if key in data and data[key] is not None:
                        np.save(tmp_dir / f"{key}.npy", data[key])

            manifest = {
                "fingerprint": fingerprint,
                "relevant_layers": relevant_layers,
                "pad_length": pad_length,
                "include_logprobs": include_logprobs,
                "response_logprobs_top_k": response_logprobs_top_k,
                "activation_shape": list(data['cache'].shape),
                "activation_dtype": str(data['cache'].dtype),
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "n_total": int(data['cache'].shape[0]),
                "zarr_sample_count": int(
                    self.activation_logger._response_activations.shape[0]
                ),
            }
            (tmp_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            # Atomic rename (same filesystem)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            tmp_dir.rename(cache_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _load_memmap_cache(
        self,
        cache_dir: Path,
        include_logprobs: bool,
    ) -> Optional[Dict[str, Any]]:
        """Load cached arrays via memmap.  Returns None on cache miss."""
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Staleness check: zarr sample count must match
        zarr_count = int(self.activation_logger._response_activations.shape[0])
        if manifest.get("zarr_sample_count") != zarr_count:
            logger.warning(
                f"Memmap cache stale: zarr has {zarr_count} samples, "
                f"cache has {manifest.get('zarr_sample_count')}. Rebuilding."
            )
            return None

        act_path = cache_dir / "activations.npy"
        if not act_path.exists():
            return None

        data: Dict[str, Any] = {
            'cache': np.load(act_path, mmap_mode='r'),
            'labels': np.load(cache_dir / "labels.npy", mmap_mode='r'),
            'prompt_hashes': np.load(
                cache_dir / "prompt_hashes.npy"
            ).tolist(),  # small array, load eagerly
            'train_indices': np.load(cache_dir / "train_indices.npy"),
            'test_indices': np.load(cache_dir / "test_indices.npy"),
        }

        if include_logprobs and manifest.get("include_logprobs"):
            for key in ('logprob_token_ids', 'logprob_token_logprobs',
                        'logprob_topk_ids', 'logprob_topk_logprobs'):
                npy_path = cache_dir / f"{key}.npy"
                if npy_path.exists():
                    data[key] = np.load(npy_path, mmap_mode='r')

        return data

    def _split_cached_data(
        self,
        data: Dict[str, Any],
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        is_memmap: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Split the full preloaded data into per-split dicts.

        When *is_memmap* is True, activation and logprob arrays are kept as the
        full memmap and a ``_row_indices`` array is attached for indirection in
        ``__getitem__``.  This avoids materialising a copy of the large array.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for split_name, indices in [('train', train_indices), ('test', test_indices)]:
            if is_memmap:
                split_data: Dict[str, Any] = {
                    'cache': data['cache'],               # full memmap
                    'labels': np.asarray(data['labels'])[indices],
                    'prompt_hashes': [data['prompt_hashes'][i] for i in indices],
                    '_row_indices': indices,
                }
                for key in ('logprob_token_ids', 'logprob_token_logprobs',
                            'logprob_topk_ids', 'logprob_topk_logprobs'):
                    if key in data:
                        split_data[key] = data[key]       # full memmap
            else:
                split_data = {
                    'cache': data['cache'][indices],
                    'labels': data['labels'][indices],
                    'prompt_hashes': [data['prompt_hashes'][i] for i in indices],
                }
                for key in ('logprob_token_ids', 'logprob_token_logprobs',
                            'logprob_topk_ids', 'logprob_topk_logprobs'):
                    if key in data and data[key] is not None:
                        split_data[key] = data[key][indices]
            result[split_name] = split_data
        return result

    def _preload_all_splits(
        self,
        relevant_layers: List[int],
        pad_length: int,
        include_logprobs: bool,
        response_logprobs_top_k: int,
        check_ram: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Load all splits, using memmap cache when available.

        First call reads from zarr and writes a cache.  Subsequent calls memmap
        the cache files — effectively instant.
        """
        fingerprint = self._memmap_cache_fingerprint(
            relevant_layers, pad_length, include_logprobs, response_logprobs_top_k
        )
        cache_dir = self._memmap_cache_dir(fingerprint)

        # Try cache
        cached = self._load_memmap_cache(cache_dir, include_logprobs)
        if cached is not None:
            train_idx = cached.pop('train_indices')
            test_idx = cached.pop('test_indices')
            logger.info(
                f"Loaded memmap cache from {cache_dir} "
                f"(train={len(train_idx)}, test={len(test_idx)})"
            )
            return self._split_cached_data(cached, train_idx, test_idx, is_memmap=True)

        # Cache miss — full preload from zarr
        logger.info("Memmap cache miss; preloading all samples from zarr ...")
        df_all = self.df.reset_index(drop=True)
        train_mask = (self.df['split'] == 'train').values
        test_mask = (self.df['split'] == 'test').values

        H = int(self.activation_logger._response_activations.shape[-1])
        if check_ram:
            est = self._estimate_preload_bytes(
                N=len(df_all), L=len(relevant_layers), T=pad_length, H=H,
                include_logprobs=include_logprobs,
                response_logprobs_top_k=response_logprobs_top_k,
            )
            est_gb = est / 1024**3
            avail_gb = psutil.virtual_memory().available / 1024**3
            logger.info(
                f"Preload RAM estimate (all splits): "
                f"{est_gb:.2f} GB required, {avail_gb:.2f} GB available"
            )
            self._check_ram_available(est)

        data = self._preload_from_zarr(
            df_split=df_all,
            relevant_layers=relevant_layers,
            pad_length=pad_length,
            include_logprobs=include_logprobs,
            response_logprobs_top_k=response_logprobs_top_k,
            split="all",
        )

        train_indices = np.where(train_mask)[0].astype(np.int64)
        test_indices = np.where(test_mask)[0].astype(np.int64)

        # Write cache (best-effort)
        try:
            self._save_memmap_cache(
                cache_dir, data, fingerprint, relevant_layers,
                pad_length, include_logprobs, response_logprobs_top_k,
                train_indices, test_indices,
            )
            logger.info(f"Saved memmap cache to {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to save memmap cache: {e}")

        return self._split_cached_data(
            data, train_indices, test_indices, is_memmap=False
        )

    def _preload_from_zarr(
        self,
        df_split: pd.DataFrame,
        relevant_layers: List[int],
        pad_length: int,
        include_logprobs: bool,
        response_logprobs_top_k: int,
        split: str = "data",
    ) -> Dict[str, Any]:
        """Bulk-read activations from zarr into a RAM-resident numpy array.

        Reads all relevant layers for one sample at a time, matching the default
        zarr chunk layout (1, 1, T, H) where each chunk covers one sample × one
        layer.  This access pattern is maximally sequential on disk: chunks for
        sample i are stored contiguously before chunks for sample i+1.  Positions
        with no zarr entry are zero-filled and a warning is emitted.

        Returns:
            dict with keys: cache (N,L,T,H), labels (N,), prompt_hashes (N,)
            and optionally logprob arrays when include_logprobs is True.
        """
        log_inst = self.activation_logger  # ZarrActivationsLogger

        # Resolve zarr integer row index for each df row (in split order)
        df_positions: List[int] = []
        zr_indices: List[int] = []
        missing: List[int] = []

        for pos, (_, row) in enumerate(df_split.iterrows()):
            key = self.select_primary_key(row['prompt_hash'])
            meta = log_inst._index.get(key)
            if meta is not None and meta.get('sample_index') is not None:
                df_positions.append(pos)
                zr_indices.append(int(meta['sample_index']))
            else:
                missing.append(pos)

        if missing:
            logger.warning(
                f"Preload: {len(missing)} samples have no zarr entry — those rows will be zero-filled"
            )

        N = len(df_split)
        L = len(relevant_layers)
        zarr_resp = log_inst._response_activations   # (N_total, L_total, T_zarr, H)
        H = int(zarr_resp.shape[-1])
        T_zarr = int(zarr_resp.shape[2])
        T_read = min(pad_length, T_zarr)

        cache = np.zeros((N, L, pad_length, H), dtype=np.float16)

        logger.info(
            f"Preloading {N} samples × {L} layers × {T_read} tokens × {H} hidden "
            f"({N * L * T_read * H * 2 / 1024**3:.2f} GB) ..."
        )

        # Sample-first loop: for each sample read all L relevant layers in one
        # zarr slice.  With (1, 1, T, H) chunk shape the chunks for sample i are
        # laid out contiguously on disk, so this is maximally sequential I/O.
        from tqdm.auto import tqdm
        with tqdm(total=len(zr_indices), desc=f"Preloading {split}", unit="sample") as pbar:
            for df_pos, zr_idx in zip(df_positions, zr_indices):
                # zarr_resp[zr_idx, relevant_layers, :T_read, :] → (L, T_read, H)
                cache[df_pos, :, :T_read, :] = np.asarray(
                    zarr_resp[zr_idx, relevant_layers, :T_read, :]
                )
                pbar.update(1)

        result: Dict[str, Any] = {
            'cache': cache,
            'labels': df_split['halu'].to_numpy(),
            'prompt_hashes': df_split['prompt_hash'].tolist(),
        }

        if include_logprobs and log_inst._response_token_ids is not None:
            top_k_avail = int(log_inst._response_topk_logprobs.shape[2])
            top_k_use = min(response_logprobs_top_k, top_k_avail)
            T_lp = min(pad_length, int(log_inst._response_token_ids.shape[1]))

            lp_ids = np.full((N, T_lp), -1, dtype=np.int32)
            lp_lps = np.full((N, T_lp), np.nan, dtype=np.float32)
            lp_topk_ids = np.full((N, T_lp, top_k_use), -1, dtype=np.int32)
            lp_topk_lps = np.full((N, T_lp, top_k_use), np.nan, dtype=np.float32)

            if zr_indices:
                zr_arr = np.array(zr_indices, dtype=np.intp)
                lp_ids[df_positions] = np.asarray(log_inst._response_token_ids)[zr_arr, :T_lp]
                lp_lps[df_positions] = np.asarray(log_inst._response_token_logprobs)[zr_arr, :T_lp]
                lp_topk_ids[df_positions] = np.asarray(log_inst._response_topk_token_ids)[zr_arr, :T_lp, :top_k_use]
                lp_topk_lps[df_positions] = np.asarray(log_inst._response_topk_logprobs)[zr_arr, :T_lp, :top_k_use]

            result.update({
                'logprob_token_ids': lp_ids,
                'logprob_token_logprobs': lp_lps,
                'logprob_topk_ids': lp_topk_ids,
                'logprob_topk_logprobs': lp_topk_lps,
            })

        logger.info("Preload complete.")
        return result

    def get_dataset(
        self,
        split: Literal['train', 'test'],
        relevant_layers: Optional[List[int]] = None,
        fixed_layer: Optional[int] = None,
        pad_length: int = 63,
        min_target_layers: int = 2,
        num_views: int = 2,
        view_sampling_with_replacement: bool = False,
        return_all_activations: bool = False,
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        backend: Literal["auto", "zarr"] = "auto",
        preload: bool = False,
        check_ram: bool = True,
    ) -> Dataset:
        """
        Get a PyTorch Dataset for the specified split.

        Args:
            split: Which split to use ('train' or 'test')
            relevant_layers: List of layer indices to use (default: layers 16-29)
            fixed_layer: If specified, one activation will always be from this layer (index in relevant_layers)
            num_views: Number of views to sample per example
            view_sampling_with_replacement: Whether to sample with replacement when selecting views
            return_all_activations: If True, include all relevant layers in all_activations
            include_response_logprobs: Whether to include token-level response logprob features
            response_logprobs_top_k: Number of top logprobs to expose per response token
            preload: If True, bulk-read all activations into RAM at construction time and return a
                PreloadedActivationDataset.  __getitem__ becomes a pure in-memory op — no zarr I/O
                during training.  Recommended when RAM >> dataset size.
            check_ram: When preload=True, estimate required RAM and raise MemoryError if available
                RAM (with a 10% safety buffer) is insufficient.  Pass False to skip the check.

        Returns:
            ActivationDataset (zarr-backed) or PreloadedActivationDataset (RAM-backed) instance.
        """
        if backend not in {"auto", "zarr"}:
            raise ValueError(
                "backend must be one of: 'auto', 'zarr'. "
                "WDS backend is deprecated and no longer supported."
            )

        _relevant_layers = relevant_layers if relevant_layers is not None else list(range(16, 30))

        if preload:
            cache_key = (
                tuple(_relevant_layers), pad_length,
                include_response_logprobs, response_logprobs_top_k,
            )
            if self._preloaded_splits is None or self._preloaded_splits[0] != cache_key:
                splits = self._preload_all_splits(
                    relevant_layers=_relevant_layers,
                    pad_length=pad_length,
                    include_logprobs=include_response_logprobs,
                    response_logprobs_top_k=response_logprobs_top_k,
                    check_ram=check_ram,
                )
                self._preloaded_splits = (cache_key, splits)
            else:
                splits = self._preloaded_splits[1]

            data = splits[split]
            return PreloadedActivationDataset(
                cache=data['cache'],
                labels=data['labels'] if isinstance(data['labels'], np.ndarray) else np.array(data['labels']),
                prompt_hashes=data['prompt_hashes'],
                num_views=num_views,
                pad_length=pad_length,
                fixed_layer=fixed_layer,
                view_sampling_with_replacement=view_sampling_with_replacement,
                min_target_layers=min_target_layers,
                include_response_logprobs=include_response_logprobs,
                response_logprobs_top_k=response_logprobs_top_k,
                relevant_layers=_relevant_layers,
                _row_indices=data.get('_row_indices'),
                _logprob_token_ids=data.get('logprob_token_ids'),
                _logprob_token_logprobs=data.get('logprob_token_logprobs'),
                _logprob_topk_ids=data.get('logprob_topk_ids'),
                _logprob_topk_logprobs=data.get('logprob_topk_logprobs'),
            )

        return ActivationDataset(
            self.df,
            self.activations_path,
            split,
            relevant_layers,
            "zarr",
            fixed_layer,
            self.random_seed,
            verbose=False,
            pad_length=pad_length,
            min_target_layers=min_target_layers,
            num_views=num_views,
            view_sampling_with_replacement=view_sampling_with_replacement,
            return_all_activations=return_all_activations,
            include_response_logprobs=include_response_logprobs,
            response_logprobs_top_k=response_logprobs_top_k,
        )

    def close(self):
        """Close the underlying activation logger connection."""
        self.logger.close() 


