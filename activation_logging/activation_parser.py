"""
activation_parser.py
Handles parsing of metadata from JSON files and looking up corresponding activations from Zarr.
"""
import json
import hashlib
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import random

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

        return {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': filled_activations,
            'views_activations': views_activations,
            'view_indices': torch.tensor(selected_view_indices, dtype=torch.long),
            'input_length': input_length
        }

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

        for layer_pos in range(len(self.relevant_layers)):
            padded_activations[layer_pos] = load_layer(layer_pos)

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

        return {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': filled_activations,
            'views_activations': views_activations,
            'view_indices': torch.tensor(selected_view_indices, dtype=torch.long),
            'input_length': input_length
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

    def _load_metadata(self) -> Dict[str, Any]:
        gendf = pd.read_json(self.inference_json, lines=True)

        with open(self.eval_json, 'r') as f:
            data = json.loads(f.read())
            
        gendf['abstain'] = data['abstantion']
        gendf['halu'] = data['halu_test_res']

        gendf['prompt_hash'] = gendf['prompt'].apply(lambda x : 
                                                    hashlib.sha256(('user: ' + 
                                                                    x).encode("utf-8")).hexdigest())

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
        backend: Literal["auto", "zarr"] = "auto",
    ) -> ActivationDataset:
        """
        Get a PyTorch Dataset for the specified split.

        Args:
            split: Which split to use ('train' or 'test')
            relevant_layers: List of layer indices to use (default: layers 16-29)
            fixed_layer: If specified, one activation will always be from this layer (index in relevant_layers)
            num_views: Number of views to sample per example
            view_sampling_with_replacement: Whether to sample with replacement when selecting views

        Returns:
            ActivationDataset instance for the specified split
        """
        if backend not in {"auto", "zarr"}:
            raise ValueError(
                "backend must be one of: 'auto', 'zarr'. "
                "WDS backend is deprecated and no longer supported."
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
        )

    def close(self):
        """Close the underlying activation logger connection."""
        self.logger.close() 


