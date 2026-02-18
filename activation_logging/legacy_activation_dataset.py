"""Legacy two-view activation dataset for performance comparisons.

This module preserves the pre-K-view loading behavior so it can be benchmarked
against the current K-view loader implementation.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class LegacyActivationDataset(Dataset):
    """Pre-K-view dataset behavior (two selected layers per sample)."""

    def __init__(
        self,
        df: pd.DataFrame,
        activations_path: str,
        split: Literal["train", "test"],
        relevant_layers: Optional[List[int]] = None,
        logger_type: str = "zarr",
        fixed_layer: Optional[int] = None,
        random_seed: int = 42,
        verbose: bool = False,
        pad_length: int = 63,
        min_target_layers: int = 2,
        return_all_activations: bool = False,
    ):
        """Initialize the legacy dataset.

        Args:
            df: Metadata dataframe used by ``ActivationParser``.
            activations_path: Path to the Zarr activation store.
            split: Split name (``train`` or ``test``).
            relevant_layers: Candidate layer ids from the full model.
            logger_type: Backend type. Only ``zarr`` is supported.
            fixed_layer: Optional fixed index in ``relevant_layers``.
            random_seed: Seed forwarded into parser behavior.
            verbose: Whether to print parser/loader logs.
            pad_length: Sequence length used for pad/truncate.
            min_target_layers: Required count of available target layers.
            return_all_activations: If true, eagerly loads all relevant layers.
        """
        normalized_logger_type = str(logger_type).strip().lower()
        if normalized_logger_type != "zarr":
            raise ValueError(
                f"Unsupported logger_type='{logger_type}'. "
                "Only Zarr activation storage is supported."
            )

        if not str(activations_path).endswith(".zarr"):
            raise ValueError(
                f"Unsupported activations_path='{activations_path}'. "
                "Only .zarr activation stores are supported."
            )

        self.activations_path = activations_path
        self.logger_type = normalized_logger_type
        self.random_seed = int(random_seed)
        self.verbose = bool(verbose)
        self.split = split
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.relevant_layers = (
            relevant_layers if relevant_layers is not None else list(range(16, 30))
        )
        self.fixed_layer = fixed_layer
        self.pad_length = int(pad_length)
        self.min_target_layers = int(min_target_layers)
        self.return_all_activations = bool(return_all_activations)
        self._activation_parser = None

    @property
    def activation_parser(self):
        """Lazy parser creation to avoid heavy setup in object init."""
        if self._activation_parser is None:
            from .activation_parser import ActivationParser

            self._activation_parser = ActivationParser(
                inference_json="",
                eval_json="",
                activations_path=self.activations_path,
                df=self.df,
                logger_type=self.logger_type,
                random_seed=self.random_seed,
                verbose=self.verbose,
            )
        return self._activation_parser

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_truncate(self, act: torch.Tensor) -> torch.Tensor:
        if act.ndim == 2:
            act = act.unsqueeze(0)
        seq_len = act.shape[1]
        if seq_len < self.pad_length:
            noise = torch.randn(
                act.shape[0],
                self.pad_length - seq_len,
                act.shape[2],
                device=act.device,
                dtype=act.dtype,
            )
            return torch.cat([act, noise], dim=1)
        if seq_len > self.pad_length:
            return act[:, : self.pad_length, :]
        return act

    def _fill_missing_layers(self, layers: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        reference = next((act for act in layers if act is not None), None)
        if reference is None:
            raise ValueError("No available layers found for this sample")
        return [torch.zeros_like(reference) if act is None else act for act in layers]

    def _select_two_layers(self, available_layers: List[int]) -> tuple[int, int]:
        if len(available_layers) < self.min_target_layers:
            raise ValueError(
                f"Not enough targeted layers available (found {len(available_layers)} layers; "
                f"need at least {self.min_target_layers})."
            )

        if self.fixed_layer is not None:
            if self.fixed_layer not in available_layers:
                raise ValueError(
                    f"Fixed layer {self.fixed_layer} is not available in the relevant layers"
                )
            other_layers = [i for i in available_layers if i != self.fixed_layer]
            if not other_layers:
                raise ValueError(
                    f"No other layers available besides fixed layer {self.fixed_layer}"
                )
            return self.fixed_layer, random.choice(other_layers)

        if len(available_layers) == 1:
            return available_layers[0], available_layers[0]
        return tuple(random.sample(available_layers, 2))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a legacy two-view sample.

        Returns keys compatible with old two-view codepaths:
        ``layer1_activations``, ``layer2_activations``, ``layer1_idx``, ``layer2_idx``.
        """
        row = self.df.iloc[idx]
        entry_key = self.activation_parser.select_primary_key(row["prompt_hash"])
        metadata = self.activation_parser.get_entry_metadata(entry_key)
        input_length = metadata.get("input_length") or metadata.get("prompt_len", 0)

        padded_activations: List[Optional[torch.Tensor]] = [None] * len(self.relevant_layers)

        def load_layer(layer_pos: int) -> Optional[torch.Tensor]:
            actual_layer = self.relevant_layers[layer_pos]
            act = self.activation_parser.get_layer_activation(
                entry_key,
                actual_layer,
                sequence_mode="response",
            )
            if act is None:
                return None
            return self._pad_or_truncate(act)

        if self.return_all_activations:
            for layer_pos in range(len(self.relevant_layers)):
                padded_activations[layer_pos] = load_layer(layer_pos)
            available_layers = [
                i for i, act in enumerate(padded_activations) if act is not None
            ]
            layer1_idx, layer2_idx = self._select_two_layers(available_layers)
            filled = self._fill_missing_layers(padded_activations)
            layer1_activations = filled[layer1_idx]
            layer2_activations = filled[layer2_idx]
            return {
                "hashkey": row["prompt_hash"],
                "halu": torch.tensor(row["halu"], dtype=torch.float32),
                "all_activations": filled,
                "layer1_activations": layer1_activations,
                "layer2_activations": layer2_activations,
                "layer1_idx": layer1_idx,
                "layer2_idx": layer2_idx,
                "input_length": input_length,
            }

        all_positions = list(range(len(self.relevant_layers)))

        if self.fixed_layer is not None:
            layer1_idx = self.fixed_layer
            layer1_activations = load_layer(layer1_idx)
            if layer1_activations is None:
                raise ValueError(
                    f"Fixed layer {self.fixed_layer} is not available in the relevant layers"
                )
            other_layers = [i for i in all_positions if i != layer1_idx]
            random.shuffle(other_layers)
            layer2_idx = None
            layer2_activations = None
            for candidate in other_layers:
                layer2_activations = load_layer(candidate)
                if layer2_activations is not None:
                    layer2_idx = candidate
                    break
            if layer2_idx is None:
                raise ValueError(
                    f"No other layers available besides fixed layer {self.fixed_layer}"
                )
        else:
            random.shuffle(all_positions)
            layer1_idx = None
            layer2_idx = None
            layer1_activations = None
            layer2_activations = None

            for candidate in all_positions:
                layer1_activations = load_layer(candidate)
                if layer1_activations is not None:
                    layer1_idx = candidate
                    break
            if layer1_idx is None:
                raise ValueError("No available layers found for this sample")

            for candidate in all_positions:
                if candidate == layer1_idx:
                    continue
                layer2_activations = load_layer(candidate)
                if layer2_activations is not None:
                    layer2_idx = candidate
                    break
            if layer2_idx is None:
                if self.min_target_layers == 1:
                    layer2_idx = layer1_idx
                    layer2_activations = layer1_activations
                else:
                    raise ValueError("Not enough targeted layers available for this sample")

        padded_activations[layer1_idx] = layer1_activations
        padded_activations[layer2_idx] = layer2_activations
        filled = self._fill_missing_layers(padded_activations)

        return {
            "hashkey": row["prompt_hash"],
            "halu": torch.tensor(row["halu"], dtype=torch.float32),
            "all_activations": filled,
            "layer1_activations": layer1_activations,
            "layer2_activations": layer2_activations,
            "layer1_idx": layer1_idx,
            "layer2_idx": layer2_idx,
            "input_length": input_length,
        }
