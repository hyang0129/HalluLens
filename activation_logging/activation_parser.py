"""
activation_parser.py
Handles parsing of metadata from JSON files and looking up corresponding activations from LMDB.
"""
import json
import hashlib
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from loguru import logger
import pandas as pd 
import json 
from loguru import logger
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import random

from .activations_logger import ActivationsLogger, JsonActivationsLogger
from .webdataset_option_a import WDSOptionAConfig, WDSOptionAIterableDataset

class ActivationDataset(Dataset):
    """PyTorch Dataset for loading activation data."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        activations_path: str,
        split: Literal['train', 'test'],
        relevant_layers: Optional[List[int]] = None,
        logger_type: str = "lmdb",
        fixed_layer: Optional[int] = None,
        random_seed: int = 42,
        verbose: bool = False,
        pad_length: int = 63,
        min_target_layers: int = 2,
        return_all_activations: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            df: DataFrame containing the metadata
            activations_path: Path to the activations storage (LMDB file, JSON directory, or .zarr store)
            split: Which split to use ('train' or 'test')
            relevant_layers: List of layer indices to use (default: layers 16-29)
            logger_type: Type of logger to use ('lmdb' or 'json')
            fixed_layer: If specified, one activation will always be from this layer (index in relevant_layers)
            random_seed: Random seed for train/test split (default: 42)
            verbose: Whether to log initialization messages (default: False for datasets)
            return_all_activations: If True, load all relevant layers for Zarr instead of only two
        """
        self.activations_path = activations_path
        self.logger_type = logger_type
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
        self.return_all_activations = return_all_activations
        self._use_zarr_fast_path = self._is_zarr_path(activations_path) and logger_type != "json"

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
            - all_activations: All padded activations (None for non-targeted layers)
            - layer1_activations: Activations from first layer (fixed layer if specified, otherwise random)
            - layer2_activations: Activations from second layer (random, different from layer1)
            - layer1_idx: Index of first selected layer
            - layer2_idx: Index of second selected layer
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

        # Select two different layers, with optional fixed layer
        available_layers = [i for i, act in enumerate(padded_activations) if act is not None]
        if len(available_layers) < self.min_target_layers:
            raise ValueError(
                f"Not enough targeted layers available (found {len(available_layers)} layers; "
                f"need at least {self.min_target_layers})."
            )

        if self.fixed_layer is not None:
            # Ensure fixed_layer is valid and available
            if self.fixed_layer not in available_layers:
                raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

            # Set one layer to the fixed layer
            layer1_idx = self.fixed_layer
            # Select a random different layer for the second activation
            other_layers = [i for i in available_layers if i != self.fixed_layer]
            if len(other_layers) == 0:
                raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
            layer2_idx = random.choice(other_layers)
        else:
            # Original behavior: randomly select two different layers.
            # For classifier-style usage, allow min_target_layers == 1 and return the same layer twice.
            if len(available_layers) == 1:
                layer1_idx = available_layers[0]
                layer2_idx = available_layers[0]
            else:
                layer1_idx, layer2_idx = random.sample(available_layers, 2)

        layer1_activations = padded_activations[layer1_idx]
        layer2_activations = padded_activations[layer2_idx]

        return {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': padded_activations,
            'layer1_activations': layer1_activations,
            'layer2_activations': layer2_activations,
            'layer1_idx': layer1_idx,
            'layer2_idx': layer2_idx,
            'input_length': input_length
        }

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

        def _fill_missing_layers(layers: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
            reference = next((act for act in layers if act is not None), None)
            if reference is None:
                raise ValueError("No available layers found for this sample")
            filled: List[torch.Tensor] = []
            for act in layers:
                if act is None:
                    filled.append(torch.zeros_like(reference))
                else:
                    filled.append(act)
            return filled

        if self.return_all_activations:
            for layer_pos in range(len(self.relevant_layers)):
                padded_activations[layer_pos] = load_layer(layer_pos)
            available_layers = [i for i, act in enumerate(padded_activations) if act is not None]
            if len(available_layers) < self.min_target_layers:
                raise ValueError(
                    f"Not enough targeted layers available (found {len(available_layers)} layers; "
                    f"need at least {self.min_target_layers})."
                )
            if self.fixed_layer is not None:
                if self.fixed_layer not in available_layers:
                    raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")
                layer1_idx = self.fixed_layer
                other_layers = [i for i in available_layers if i != layer1_idx]
                if not other_layers:
                    raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
                layer2_idx = random.choice(other_layers)
            else:
                if len(available_layers) == 1:
                    layer1_idx = available_layers[0]
                    layer2_idx = available_layers[0]
                else:
                    layer1_idx, layer2_idx = random.sample(available_layers, 2)

            padded_activations = _fill_missing_layers(padded_activations)
            layer1_activations = padded_activations[layer1_idx]
            layer2_activations = padded_activations[layer2_idx]

            return {
                'hashkey': row['prompt_hash'],
                'halu': torch.tensor(row['halu'], dtype=torch.float32),
                'all_activations': padded_activations,
                'layer1_activations': layer1_activations,
                'layer2_activations': layer2_activations,
                'layer1_idx': layer1_idx,
                'layer2_idx': layer2_idx,
                'input_length': input_length
            }

        all_positions = list(range(len(self.relevant_layers)))

        if self.fixed_layer is not None:
            layer1_idx = self.fixed_layer
            layer1_activations = load_layer(layer1_idx)
            if layer1_activations is None:
                raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")

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
                raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
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
        padded_activations = _fill_missing_layers(padded_activations)

        return {
            'hashkey': row['prompt_hash'],
            'halu': torch.tensor(row['halu'], dtype=torch.float32),
            'all_activations': padded_activations,
            'layer1_activations': layer1_activations,
            'layer2_activations': layer2_activations,
            'layer1_idx': layer1_idx,
            'layer2_idx': layer2_idx,
            'input_length': input_length
        }
    



class ActivationParser:
    def __init__(self, inference_json: str, eval_json: str, activations_path: str,
                 df: Optional[pd.DataFrame] = None, logger_type: str = "lmdb",
                 random_seed: int = 42, verbose: bool = True):
        """
        Initialize the ActivationParser.

        Args:
            inference_json: Path to the inference JSON file
            eval_json: Path to the evaluation JSON file
            activations_path: Path to the activations storage (LMDB file or JSON directory)
            df: Optional DataFrame to use instead of loading from JSON files
            logger_type: Type of logger to use ('lmdb' or 'json')
            random_seed: Random seed for train/test split (default: 42)
            verbose: Whether to log initialization and metadata loading messages (default: True)
        """
        self.inference_json = Path(inference_json)
        if not self.inference_json.exists():
            raise FileNotFoundError(f"JSON file not found: {inference_json}")

        self.eval_json = Path(eval_json)
        if not self.eval_json.exists():
            raise FileNotFoundError(f"JSON file not found: {eval_json}")

        self.activations_path = activations_path
        self.logger_type = logger_type
        self.random_seed = random_seed
        self.verbose = verbose
        self._activation_logger = None
        self._group_index = None
        self._wds_shards = self._detect_wds_shards(activations_path)

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

            if self.logger_type == "wds" or self._wds_shards is not None:
                self._activation_logger = None
            elif self.logger_type == "json":
                self._activation_logger = JsonActivationsLogger(output_dir=self.activations_path, read_only=True, verbose=self.verbose)
            else:  # default to lmdb
                self._activation_logger = ActivationsLogger(lmdb_path=self.activations_path, read_only=True, verbose=self.verbose)
        return self._activation_logger

    def get_entry_metadata(self, entry_key: str) -> Dict[str, Any]:
        """Retrieve metadata only for a given entry key."""
        if self.logger_type == "wds" or self.activation_logger is None:
            return {}
        metadata = self.activation_logger.get_entry_by_key(entry_key, metadata_only=True)
        return metadata or {}

    def get_layer_activation(self, entry_key: str, layer_idx: int, sequence_mode: str = "response") -> Optional[torch.Tensor]:
        """Retrieve a single layer activation for the given entry key and sequence mode."""
        if self.logger_type == "wds" or self.activation_logger is None:
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

        if self.logger_type != "wds" and self._wds_shards is None:
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
        return_all_activations: bool = False,
    ) -> ActivationDataset:
        """
        Get a PyTorch Dataset for the specified split.

        Args:
            split: Which split to use ('train' or 'test')
            relevant_layers: List of layer indices to use (default: layers 16-29)
            fixed_layer: If specified, one activation will always be from this layer (index in relevant_layers)

        Returns:
            ActivationDataset instance for the specified split
        """
        if self.logger_type == "wds" or self._wds_shards is not None:
            config = WDSOptionAConfig(
                shards=self._wds_shards or self.activations_path,
                split=split,
                shuffle_buffer=10_000,
                pad_length=pad_length,
                min_target_layers=min_target_layers,
                relevant_layers=relevant_layers,
                fixed_layer=fixed_layer,
            )
            return WDSOptionAIterableDataset(self.df, config)

        return ActivationDataset(
            self.df,
            self.activations_path,
            split,
            relevant_layers,
            self.logger_type,
            fixed_layer,
            self.random_seed,
            verbose=False,
            pad_length=pad_length,
            min_target_layers=min_target_layers,
            return_all_activations=return_all_activations,
        )

    def close(self):
        """Close the LMDB connection."""
        self.logger.close() 


