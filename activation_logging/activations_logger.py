"""
activations_logger.py
Handles LMDB logging for LLM activations, prompts, responses, and evaluation results.
Used to evaluate the nature of activations when an LLM is hallucinating vs not hallucinating.
"""
import lmdb
import pickle
import os
import numpy as np
from typing import Any, Dict, Optional, List, Callable, Tuple, Union, Type
from loguru import logger
import torch 
from tqdm.autonotebook import tqdm
from .compression import BaseCompressor, NoCompression, ZstdCompression, ZSTD_AVAILABLE
import json
from pathlib import Path
import uuid
import shutil


class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 16 << 30,
                 compression: Union[str, BaseCompressor, None] = None,
                 read_only: bool = False,
                 target_layers: str = 'all',
                 sequence_mode: str = 'all',
                 verbose: bool = True):
        """
        Initialize the LMDB-based activations logger.

        Args:
            lmdb_path: Path to the LMDB file to store activations
            map_size: Maximum size of the LMDB file in bytes (default: 16GB)
            compression: Compression method ('zstd', None) or a BaseCompressor instance
            read_only: if True, the LMDB will be opened in read-only mode
            target_layers: Which layers to extract activations from ('all', 'first_half', or 'second_half')
            sequence_mode: Which tokens to extract activations for ('all' for full sequence, 'prompt' for prompt tokens only, or 'response' for response tokens only)
            verbose: Whether to log detailed initialization and processing messages (default: True)
        """
        if target_layers not in ['all', 'first_half', 'second_half']:
            raise ValueError("target_layers must be one of: 'all', 'first_half', 'second_half'")
        if sequence_mode not in ['all', 'prompt', 'response']:
            raise ValueError("sequence_mode must be one of: 'all', 'prompt', 'response'")
            
        self.target_layers = target_layers
        self.sequence_mode = sequence_mode
        self.verbose = verbose
        if self.verbose:
            logger.info(f"ActivationsLogger initialized to target '{target_layers}' layers with '{sequence_mode}' sequence mode")
        self.lmdb_path = lmdb_path
        self.env = None
        self.metadata_env = None  # Separate environment for metadata
        self.map_size = map_size
        self.last_threshold_logged = 0  # Track the last threshold percentage logged
        
        # Set up the compressor based on the compression parameter
        self.compressor = self._get_compressor(compression)
        
        # Skip opening LMDB if path is empty
        if not lmdb_path or lmdb_path.strip() == "":
            return
            
        safe_lmdb_path = lmdb_path
        self.safe_lmdb_path = safe_lmdb_path
        lmdb_dir = os.path.dirname(safe_lmdb_path)
        if lmdb_dir and not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
            
        # Do NOT open main LMDB or metadata LMDB here; rely on lazy initialization
        # self.env = lmdb.open(safe_lmdb_path, map_size=map_size, subdir=True, create=True, readonly=read_only, lock=True, max_readers=2048)
        # self.env.reader_check()
        
        # metadata_path = os.path.join(os.path.dirname(safe_lmdb_path), f"{os.path.basename(safe_lmdb_path)}_metadata")
        # os.makedirs(metadata_path, exist_ok=True)
        # self.metadata_env = lmdb.open(metadata_path, map_size, subdir=True, create=True, readonly=read_only, lock=True, max_readers=2048)
        # self.metadata_env.reader_check()
        # logger.info(f"Opened metadata store at {metadata_path}")

        self.read_only = read_only

        if self.read_only:
            self.keys = self.list_entries()
    
    def _get_compressor(self, compression: Union[str, BaseCompressor, None]) -> BaseCompressor:
        """
        Get the appropriate compressor based on the compression parameter.
        
        Args:
            compression: Compression method ('zstd', None) or a BaseCompressor instance
            
        Returns:
            BaseCompressor instance to use for compression/decompression
        """
        if isinstance(compression, BaseCompressor):
            # Use the provided compressor
            return compression
        elif compression is None:
            # Use no compression
            return NoCompression()
        elif compression == 'zstd':
            # Use zstandard compression if available
            if ZSTD_AVAILABLE:
                return ZstdCompression(level=19)
            else:
                raise ImportError("zstandard module not available, cannot compress")
        else:
            # Unknown compression method, fall back to no compression
            raise ValueError(f"Unknown compression method '{compression}', must be 'zstd' or a BaseCompressor instance")

    def check_lmdb_size(self):
        """
        Check the current size of the LMDB file and log warnings when it passes
        certain thresholds (10%, 20%, ..., 90%) of its maximum map size.
        """
        if self.env is None or not hasattr(self, 'safe_lmdb_path'):
            return
            
        # Get the current size of the LMDB data file
        data_path = os.path.join(self.safe_lmdb_path, "data.mdb")
        if not os.path.exists(data_path):
            return
            
        current_size = os.path.getsize(data_path)
        size_percentage = (current_size / self.map_size) * 100
        
        # Check if we've passed a new 10% threshold
        current_threshold = int(size_percentage / 10) * 10
        
        if current_threshold > self.last_threshold_logged and current_threshold <= 90:
            logger.warning(f"LMDB file has reached {current_threshold}% of its maximum size "
                          f"({current_size / (1024**3):.2f}GB / {self.map_size / (1024**3):.2f}GB)")
            self.last_threshold_logged = current_threshold
            
        # Extra warning at 95%
        if size_percentage >= 95 and self.last_threshold_logged < 95:
            logger.critical(f"LMDB file is at {size_percentage:.1f}% of its maximum size! "
                           f"({current_size / (1024**3):.2f}GB / {self.map_size / (1024**3):.2f}GB)")
            self.last_threshold_logged = 95

    def extract_activations(self, model_outputs, input_length):
        """
        Extract activations from model outputs.
        
        Args:
            model_outputs: The outputs from the model's generate method
            input_length: The length of the input prompt in tokens
            
        Returns:
            List of tensors containing activations from the specified layers based on target_layers setting:
            - 'all': all layers with their original activations
            - 'first_half': first half of layers with original activations, second half as None
            - 'second_half': second half of layers with original activations, first half as None
            
            For each layer, activations are extracted based on sequence_mode:
            - 'all': activations for both prompt and generated tokens
            - 'prompt': activations for prompt tokens only
            - 'response': activations for response tokens only
        """
        # If model_outputs is None or doesn't have hidden_states, return None
        if model_outputs is None or not hasattr(model_outputs, 'hidden_states'):
            if self.verbose:
                logger.info("No hidden states found in model outputs")
            return None
            
        # Get the generated tokens (excluding prompt)
        gen_sequence = model_outputs.sequences[0]
        gen_ids = gen_sequence[input_length:]
        
        all_hidden_states = model_outputs.hidden_states
        prompt_hidden = all_hidden_states[0]  # first set of tokens, the prompt tokens. It has len = num_layers
        gen_hiddens = all_hidden_states[1:]  # all subsequent sets of tokens, the generated tokens. The structure of this list is [token_num, layer_num, ...]

        # If there's a trim position, only use activations up to that point
        trim_pos = None
        if hasattr(model_outputs, 'trim_position'):
            trim_pos = model_outputs.trim_position
            if trim_pos is not None:
                if self.verbose:
                    logger.info(f"Trimming activations at position {trim_pos}")
                gen_hiddens = gen_hiddens[:trim_pos]
                        
        # Determine which layers to extract based on target_layers setting
        num_layers = len(prompt_hidden)
        if self.target_layers == 'first_half':
            target_layer_indices = range(num_layers // 2)
            logger.debug(f"Extracting first {num_layers // 2} layers")
        elif self.target_layers == 'second_half':
            start_idx = num_layers // 2
            target_layer_indices = range(start_idx, num_layers)
            logger.debug(f"Extracting second half of layers ({start_idx} to {num_layers-1})")
        else:  # 'all'
            target_layer_indices = range(num_layers)
            logger.debug(f"Extracting all {num_layers} layers")

        if self.sequence_mode == 'prompt':
            # Only return prompt activations
            logger.debug("Extracting activations for prompt tokens only")
            full_hidden_states = []
            for layer_idx in range(num_layers):
                if layer_idx in target_layer_indices:
                    full_hidden_states.append(prompt_hidden[layer_idx])
                else:
                    full_hidden_states.append(None)
        elif self.sequence_mode == 'response':
            # Only return response activations
            logger.debug("Extracting activations for response tokens only")
            full_hidden_states = []
            for layer_idx in range(num_layers):
                if layer_idx in target_layer_indices:
                    # Only concatenate steps for target layers
                    layer_acts = torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
                    full_hidden_states.append(layer_acts)
                else:
                    full_hidden_states.append(None)
        else:  # 'all'
            # Concatenate prompt and generated token activations
            logger.debug("Extracting activations for full sequence (prompt + generated tokens)")
            
            full_hidden_states = []
            for layer_idx in range(num_layers):
                if layer_idx in target_layer_indices:
                    # Only concatenate steps for target layers
                    gen_layer_acts = torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
                    # Concatenate prompt and generated activations
                    layer_acts = torch.cat([prompt_hidden[layer_idx], gen_layer_acts], dim=1)
                    full_hidden_states.append(layer_acts)
                else:
                    full_hidden_states.append(None)

        return full_hidden_states

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the LMDB.
        
        Args:
            key: Unique identifier for the entry (typically a hash of the prompt)
            entry: Dictionary containing the data to log (prompt, response, model_outputs, etc.)
        """
        # Skip logging if LMDB is not initialized
        if self.env is None or self.metadata_env is None:
            return
            
        # Process model outputs if present
        if "model_outputs" in entry and "input_length" in entry:
            model_outputs = entry["model_outputs"]
            input_length = entry["input_length"]
            
            # Add trim position to model_outputs for use in extract_activations
            if "trim_position" in entry:
                model_outputs.trim_position = entry["trim_position"]
            
            # Extract activations from model outputs
            all_layers_activations = self.extract_activations(model_outputs, input_length)
            
            entry.pop("model_outputs", None)
            metadata_entry = entry.copy()
            metadata_entry.pop("all_layers_activations", None)

            if all_layers_activations:
                # Replace model_outputs with extracted activations to save space
                entry["all_layers_activations"] = all_layers_activations
                
                # Store the logging configuration in metadata
                metadata_entry["logging_config"] = {
                    "target_layers": self.target_layers,
                    "sequence_mode": self.sequence_mode
                }

        else: 
            raise ValueError("No model_outputs or input_length found in entry")
        
        import time
        metadata_entry["timestamp"] = time.time()
        
        # Log size before compression for debugging
        entry_size_before = len(pickle.dumps(entry))
        logger.debug(f"Compressing entry with key {key[:8]}... (size before: {entry_size_before / (1024**2):.2f} MB)")
        
        # Apply compression to the entry using the compressor
        compressed_entry, compression_metadata = self.compressor.compress(entry)
        
        # Log compression results
        entry_size_after = len(pickle.dumps(compressed_entry))
        compression_ratio = entry_size_before / max(1, entry_size_after)
        logger.debug(f"Compression complete: {entry_size_after / (1024**2):.2f} MB, " 
                    f"ratio: {compression_ratio:.2f}x, method: {compression_metadata.get('compression', 'unknown')}")
        
        # Add compression metadata to both entries
        final_entry = {
            "data": compressed_entry,
            "metadata": compression_metadata
        }
        
        # Add reference to the main data in metadata entry
        metadata_entry["has_activations"] = True
        metadata_entry["compression_metadata"] = compression_metadata
            
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(final_entry))
            
        # Store metadata separately without the large activations
        with self.metadata_env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(metadata_entry))
            
        # Check LMDB size after writing
        self.check_lmdb_size()

    def get_entry(self, lookup: Union[str,int]) -> Dict[str, Any]:
        """
        Retrieve an entry from the LMDB.
        
        Args:
            lookup: either a key or an index
        """

        if isinstance(lookup, int):
            if lookup >= len(self.keys):
                raise ValueError(f"Index {lookup} is out of range for {len(self.keys)} entries")
            if self.read_only:
                key = self.keys[lookup]
            else:
                logger.warning("LMDB is not read-only, listing entries to find key")
                key = self.list_entries()[lookup]
        else:
            key = lookup

        return self.get_entry_by_key(key)

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, map_size=self.map_size, subdir=True, create=True, readonly=self.read_only, lock=True, max_readers=2048)
        return self.env

    def _get_metadata_env(self):
        if self.metadata_env is None:
            metadata_path = os.path.join(os.path.dirname(self.lmdb_path), f"{os.path.basename(self.lmdb_path)}_metadata")
            os.makedirs(metadata_path, exist_ok=True)
            self.metadata_env = lmdb.open(metadata_path, self.map_size, subdir=True, create=True, readonly=self.read_only, lock=True, max_readers=2048)
        return self.metadata_env

    def get_entry_by_key(self, key: str, metadata_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the LMDB.
        
        Args:
            key: Unique identifier for the entry to retrieve
            metadata_only: If True, only retrieve metadata without activations
            
        Returns:
            Dictionary containing the entry data if found, None otherwise
        """
        # Raise an error if LMDB is not initialized
        if self._get_env() is None:
            raise ValueError("LMDB is not initialized")
            
        # If only metadata is requested and metadata store is available
        if metadata_only:
            metadata_env = self._get_metadata_env()
            if metadata_env is not None:
                with metadata_env.begin(write=False) as txn:
                    value = txn.get(key.encode("utf-8"))
                    if value is not None:
                        return pickle.loads(value)
                    else:
                        raise ValueError("No metadata found for key")
        
        # Otherwise retrieve full entry including activations
        with self._get_env().begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
            if value is not None:
                stored_entry = pickle.loads(value)
                
                # Check if the entry uses the new format with compression
                if isinstance(stored_entry, dict) and "data" in stored_entry and "metadata" in stored_entry:
                    # Log before decompression
                    compressed_size = len(pickle.dumps(stored_entry["data"]))
                    compression_method = stored_entry["metadata"].get("compression", "unknown")
                    logger.debug(f"Decompressing entry with key {key[:8]}... "
                               f"(compressed size: {compressed_size / (1024**2):.2f} MB, method: {compression_method})")
                    
                    # Apply decompression using the compressor
                    decompressed_data = self.compressor.decompress(stored_entry["data"], stored_entry["metadata"])
                    
                    # Log after decompression
                    decompressed_size = len(pickle.dumps(decompressed_data))
                    logger.debug(f"Decompression complete: {decompressed_size / (1024**2):.2f} MB, "
                               f"ratio: {decompressed_size / max(1, compressed_size):.2f}x")
                    
                    return decompressed_data
                else:
                    # Handle legacy format (no compression)
                    raise ValueError("No compression metadata found for key")
                    
            else:
                raise ValueError("No entry found for key")

    def list_entries(self) -> List[str]:
        """
        List all entry keys in the metadata store.
        
        Returns:
            List of entry keys as strings
        """
        metadata_env = self._get_metadata_env()
        if metadata_env is None:
            return []
            
        keys = []
        with metadata_env.begin(write=False) as txn:
            cursor = txn.cursor()
            # Use iternext with keys=True, values=False to avoid reading values
            for key in cursor.iternext(keys=True, values=False):
                keys.append(key.decode("utf-8"))
        return keys

    def search_metadata(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search through metadata entries using a filter function.
        
        Args:
            filter_fn: Function that takes a metadata entry and returns True if it matches the search criteria
            
        Returns:
            List of tuples containing (key, metadata_entry) for matching entries
        """
        if self.metadata_env is None:
            return []
            
        results = []
        with self.metadata_env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, value in cursor:
                key = key_bytes.decode("utf-8")
                metadata = pickle.loads(value)
                if filter_fn(metadata):
                    results.append((key, metadata))
        return results

    def close(self):
        """Close the LMDB environments."""
        if self._get_env() is not None:
            self._get_env().close()
            self.env = None 
            
        if self.metadata_env is not None:
            self.metadata_env.close()
            self.metadata_env = None

    def fix_cuda_tensors(self):
        """
        Fix existing entries by ensuring all tensors are on CPU.
        This is useful for entries that were saved with CUDA tensors.
        """
        if self._get_env() is None:
            raise ValueError("LMDB is not initialized")
            
        if self.read_only:
            raise ValueError("Cannot fix CUDA tensors in read-only mode. Initialize ActivationsLogger with read_only=False")
            
        logger.info("Starting to fix CUDA tensors in existing entries...")
        fixed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Get all keys
        keys = self.list_entries()
        total_entries = len(keys)
        logger.info(f"Found {total_entries} entries to process")
        
        for key in tqdm(keys, desc="Fixing CUDA tensors"):
            try:
                # Use a single transaction for both reading and writing
                with self._get_env().begin(write=True) as txn:
                    value = txn.get(key.encode("utf-8"))
                    if value is None:
                        continue
                        
                    stored_entry = pickle.loads(value)
                    
                    # Check if this is a compressed entry
                    if isinstance(stored_entry, dict) and "data" in stored_entry and "metadata" in stored_entry:
                        data = stored_entry["data"]
                        metadata = stored_entry["metadata"]
                        
                        # Check if this has activations
                        if "all_layers_activations" in data:
                            # Decompress if needed
                            if metadata.get("compression") == "zstd":
                                data = self.compressor.decompress(data, metadata)
                            
                            # Check if any tensors need to be moved to CPU
                            needs_fixing = False
                            fixed_tensors = []
                            for tensor in data["all_layers_activations"]:
                                if isinstance(tensor, torch.Tensor):
                                    if tensor.device.type != 'cpu':
                                        needs_fixing = True
                                        fixed_tensors.append(tensor.cpu())
                                    else:
                                        fixed_tensors.append(tensor)
                                else:
                                    fixed_tensors.append(tensor)
                            
                            # Only write back if tensors needed fixing
                            if needs_fixing:
                                # Update the data
                                data["all_layers_activations"] = fixed_tensors
                                
                                # Recompress
                                compressed_data, compression_metadata = self.compressor.compress(data)
                                
                                # Create new entry
                                new_entry = {
                                    "data": compressed_data,
                                    "metadata": compression_metadata
                                }
                                
                                # Save back to LMDB using the same transaction
                                txn.put(key.encode("utf-8"), pickle.dumps(new_entry))
                                
                                fixed_count += 1
                            else:
                                skipped_count += 1
                                
            except Exception as e:
                logger.error(f"Error fixing entry {key[:8]}: {e}")
                error_count += 1
                
        logger.info(f"Finished fixing CUDA tensors. Fixed {fixed_count} entries, skipped {skipped_count} entries (already on CPU), encountered {error_count} errors")

    def export_to_json_npy_format(self, output_dir: str, indices: Optional[List[int]] = None):
        """
        Export entries to the JsonActivationsLogger NPY format.
        Args:
            output_dir: Directory to write the JsonActivationsLogger files
            indices: Optional list of indices to export (default: all entries)
        """
        # Disk space check
        keys = self.list_entries()
        if indices is None:
            indices = list(range(len(keys)))
        entries_to_export = [keys[idx] for idx in indices]
        # Estimate required space (sum of pickled entry sizes)
        import pickle
        total_size = 0
        for key in entries_to_export:
            entry = self.get_entry_by_key(key)
            total_size += len(pickle.dumps(entry))
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(output_dir_path))
        free_space = usage.free
        print(f"Estimated required space: {total_size / (1024**3):.2f} GB")
        print(f"Available space at destination: {free_space / (1024**3):.2f} GB")
        if free_space < total_size:
            print("ERROR: Available disk space is less than the estimated required space! Aborting export.")
            return
        # Proceed with export using JsonActivationsLogger with NPY storage
        json_logger = JsonActivationsLogger(output_dir=output_dir, target_layers=self.target_layers, sequence_mode=self.sequence_mode)
        for key in entries_to_export:
            entry = self.get_entry_by_key(key)
            json_logger.log_entry(key, entry)
        json_logger.close()





class JsonActivationsLogger:
    """
    Handles logging for LLM activations using JSON metadata and NPY activation files.
    Structure:
    - Folder/
      - metadata.json (contains metadata for all entries)
      - activations/
        - <key1>.npy (activation tensor data for key1)
        - <key2>.npy (activation tensor data for key2)
        - ...
    """
    def __init__(self, output_dir: str = "json_data/", target_layers: str = 'all',
                 sequence_mode: str = 'all', read_only: bool = False, verbose: bool = True):
        """
        Initialize the JSON-based activations logger with NPY tensor storage.

        Args:
            output_dir: Directory to store the JSON metadata and NPY activation files
            target_layers: Which layers to extract activations from ('all', 'first_half', or 'second_half')
            sequence_mode: Which tokens to extract activations for ('all', 'prompt', 'response')
            read_only: If True, open in read-only mode
            verbose: Whether to log detailed initialization and processing messages (default: True)
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(f"JsonActivationsLogger.__init__ starting - output_dir: {output_dir}, target_layers: {target_layers}, sequence_mode: {sequence_mode}, read_only: {read_only}")

        if target_layers not in ['all', 'first_half', 'second_half']:
            raise ValueError("target_layers must be one of: 'all', 'first_half', 'second_half'")
        if sequence_mode not in ['all', 'prompt', 'response']:
            raise ValueError("sequence_mode must be one of: 'all', 'prompt', 'response'")

        if self.verbose:
            logger.info(f"JsonActivationsLogger.__init__ - Setting up paths")
        self.output_dir = Path(output_dir)
        self.activations_dir = self.output_dir / "activations"
        self.metadata_path = self.output_dir / "metadata.json"
        self.target_layers = target_layers
        self.sequence_mode = sequence_mode
        self.read_only = read_only

        # Create directories if they don't exist
        if not read_only:
            logger.info(f"JsonActivationsLogger.__init__ - Creating directories")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.activations_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"JsonActivationsLogger.__init__ - Directories created successfully")
            except Exception as e:
                logger.error(f"JsonActivationsLogger.__init__ - Failed to create directories: {e}")
                raise

        # Load or initialize the metadata
        logger.info(f"JsonActivationsLogger.__init__ - Loading metadata from {self.metadata_path}")
        try:
            if self.metadata_path.exists():
                logger.info(f"JsonActivationsLogger.__init__ - Metadata file exists, loading...")
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"JsonActivationsLogger.__init__ - Metadata loaded successfully")
            else:
                logger.info(f"JsonActivationsLogger.__init__ - Creating new metadata")
                self.metadata = {
                    "logger_config": {
                        "target_layers": target_layers,
                        "sequence_mode": sequence_mode,
                        "version": "2.0",  # Updated version for NPY format
                        "storage_format": "npy"
                    },
                    "entries": {}
                }
                logger.info(f"JsonActivationsLogger.__init__ - New metadata created")
        except Exception as e:
            logger.error(f"JsonActivationsLogger.__init__ - Failed to load/create metadata: {e}")
            raise

        logger.info(f"JsonActivationsLogger (NPY format) initialized at {output_dir} with {len(self.metadata['entries'])} existing entries")
        logger.info(f"JsonActivationsLogger.__init__ completed successfully")

    def _tensors_to_numpy_arrays(self, obj):
        """Convert tensors to numpy arrays for NPY storage."""
        if isinstance(obj, torch.Tensor):
            logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Converting tensor of shape {obj.shape} to numpy")
            try:
                result = obj.cpu().numpy()
                logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Tensor conversion successful")
                return result
            except Exception as e:
                logger.error(f"JsonActivationsLogger._tensors_to_numpy_arrays - Tensor conversion failed: {e}")
                raise
        elif isinstance(obj, np.ndarray):
            logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Object is already numpy array")
            return obj
        elif isinstance(obj, list):
            logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Converting list of {len(obj)} items")
            try:
                result = [self._tensors_to_numpy_arrays(item) for item in obj]
                logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - List conversion successful")
                return result
            except Exception as e:
                logger.error(f"JsonActivationsLogger._tensors_to_numpy_arrays - List conversion failed: {e}")
                raise
        elif isinstance(obj, dict):
            logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Converting dict with {len(obj)} keys")
            try:
                result = {k: self._tensors_to_numpy_arrays(v) for k, v in obj.items()}
                logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Dict conversion successful")
                return result
            except Exception as e:
                logger.error(f"JsonActivationsLogger._tensors_to_numpy_arrays - Dict conversion failed: {e}")
                raise
        elif obj is None:
            return None
        else:
            logger.debug(f"JsonActivationsLogger._tensors_to_numpy_arrays - Object type {type(obj)} passed through unchanged")
            return obj

    def _get_activation_shape_info(self, activation_arrays):
        """Get shape information for activation arrays to store in metadata."""
        if isinstance(activation_arrays, np.ndarray):
            return {
                "shape": list(activation_arrays.shape),
                "dtype": str(activation_arrays.dtype)
            }
        elif isinstance(activation_arrays, list):
            return [self._get_activation_shape_info(item) for item in activation_arrays]
        elif isinstance(activation_arrays, dict):
            return {k: self._get_activation_shape_info(v) for k, v in activation_arrays.items()}
        elif activation_arrays is None:
            return None
        else:
            return {"type": type(activation_arrays).__name__}

    def _tensor_to_json_serializable(self, obj):
        """Convert tensors and other non-JSON-serializable objects to JSON-serializable format.

        Note: This method is kept for backward compatibility but is no longer used
        for activation storage in NPY format.
        """
        if isinstance(obj, torch.Tensor):
            return {
                "_type": "torch.Tensor",
                "data": obj.cpu().numpy().tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, np.ndarray):
            return {
                "_type": "numpy.ndarray",
                "data": obj.tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, list):
            return [self._tensor_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._tensor_to_json_serializable(v) for k, v in obj.items()}
        elif obj is None:
            return None
        else:
            return obj

    def _numpy_arrays_to_tensors(self, obj):
        """Convert numpy arrays back to tensors."""
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, list):
            return [self._numpy_arrays_to_tensors(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._numpy_arrays_to_tensors(v) for k, v in obj.items()}
        elif obj is None:
            return None
        else:
            return obj

    def _json_to_tensor(self, obj):
        """Convert JSON-serializable format back to tensors.

        Note: This method is kept for backward compatibility with old JSON activation files.
        """
        if isinstance(obj, dict) and "_type" in obj:
            if obj["_type"] == "torch.Tensor":
                data = np.array(obj["data"])
                tensor = torch.from_numpy(data)
                # Convert dtype string back to torch dtype
                if "dtype" in obj:
                    dtype_str = obj["dtype"].replace("torch.", "")
                    if hasattr(torch, dtype_str):
                        tensor = tensor.to(getattr(torch, dtype_str))
                return tensor
            elif obj["_type"] == "numpy.ndarray":
                return np.array(obj["data"], dtype=obj.get("dtype", "float32"))
        elif isinstance(obj, list):
            return [self._json_to_tensor(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._json_to_tensor(v) for k, v in obj.items()}
        else:
            return obj

    def extract_activations(self, model_outputs, input_length):
        """
        Extract activations from model outputs.
        This method is identical to the one in ActivationsLogger for compatibility.
        """
        logger.info(f"JsonActivationsLogger.extract_activations starting - input_length: {input_length}")

        # If model_outputs is None or doesn't have hidden_states, return None
        if model_outputs is None or not hasattr(model_outputs, 'hidden_states'):
            logger.info("JsonActivationsLogger.extract_activations - No hidden states found in model outputs")
            return None

        logger.info(f"JsonActivationsLogger.extract_activations - Processing model outputs")

        # Get the generated tokens (excluding prompt)
        try:
            gen_sequence = model_outputs.sequences[0]
            gen_ids = gen_sequence[input_length:]
            logger.info(f"JsonActivationsLogger.extract_activations - Generated sequence length: {len(gen_ids)}")
        except Exception as e:
            logger.error(f"JsonActivationsLogger.extract_activations - Failed to extract generated sequence: {e}")
            raise

        try:
            all_hidden_states = model_outputs.hidden_states
            prompt_hidden = all_hidden_states[0]  # first set of tokens, the prompt tokens. It has len = num_layers
            gen_hiddens = all_hidden_states[1:]  # all subsequent sets of tokens, the generated tokens. The structure of this list is [token_num, layer_num, ...]
            logger.info(f"JsonActivationsLogger.extract_activations - Hidden states structure: {len(all_hidden_states)} steps, {len(prompt_hidden)} layers")
        except Exception as e:
            logger.error(f"JsonActivationsLogger.extract_activations - Failed to extract hidden states: {e}")
            raise

        # If there's a trim position, only use activations up to that point
        trim_pos = None
        if hasattr(model_outputs, 'trim_position'):
            trim_pos = model_outputs.trim_position
            if trim_pos is not None:
                logger.info(f"JsonActivationsLogger.extract_activations - Trimming activations at position {trim_pos}")
                gen_hiddens = gen_hiddens[:trim_pos]

        # Determine which layers to extract based on target_layers setting
        num_layers = len(prompt_hidden)
        if self.target_layers == 'first_half':
            target_layer_indices = range(num_layers // 2)
            logger.info(f"JsonActivationsLogger.extract_activations - Extracting first {num_layers // 2} layers")
        elif self.target_layers == 'second_half':
            start_idx = num_layers // 2
            target_layer_indices = range(start_idx, num_layers)
            logger.info(f"JsonActivationsLogger.extract_activations - Extracting second half of layers ({start_idx} to {num_layers-1})")
        else:  # 'all'
            target_layer_indices = range(num_layers)
            logger.info(f"JsonActivationsLogger.extract_activations - Extracting all {num_layers} layers")

        logger.info(f"JsonActivationsLogger.extract_activations - Processing sequence mode: {self.sequence_mode}")

        if self.sequence_mode == 'prompt':
            # Only return prompt activations
            logger.info("JsonActivationsLogger.extract_activations - Extracting activations for prompt tokens only")
            full_hidden_states = []
            try:
                for layer_idx in range(num_layers):
                    if layer_idx in target_layer_indices:
                        full_hidden_states.append(prompt_hidden[layer_idx])
                    else:
                        full_hidden_states.append(None)
                logger.info(f"JsonActivationsLogger.extract_activations - Prompt activations extracted successfully")
            except Exception as e:
                logger.error(f"JsonActivationsLogger.extract_activations - Failed to extract prompt activations: {e}")
                raise

        elif self.sequence_mode == 'response':
            # Only return response activations
            logger.info("JsonActivationsLogger.extract_activations - Extracting activations for response tokens only")
            full_hidden_states = []
            try:
                for layer_idx in range(num_layers):
                    if layer_idx in target_layer_indices:
                        # Only concatenate steps for target layers
                        logger.debug(f"JsonActivationsLogger.extract_activations - Concatenating response activations for layer {layer_idx}")
                        layer_acts = torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
                        full_hidden_states.append(layer_acts)
                    else:
                        full_hidden_states.append(None)
                logger.info(f"JsonActivationsLogger.extract_activations - Response activations extracted successfully")
            except Exception as e:
                logger.error(f"JsonActivationsLogger.extract_activations - Failed to extract response activations: {e}")
                raise

        else:  # 'all'
            # Concatenate prompt and generated token activations
            logger.info("JsonActivationsLogger.extract_activations - Extracting activations for full sequence (prompt + generated tokens)")

            full_hidden_states = []
            try:
                for layer_idx in range(num_layers):
                    if layer_idx in target_layer_indices:
                        # Only concatenate steps for target layers
                        logger.debug(f"JsonActivationsLogger.extract_activations - Processing layer {layer_idx}/{num_layers}")
                        gen_layer_acts = torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
                        # Concatenate prompt and generated activations
                        layer_acts = torch.cat([prompt_hidden[layer_idx], gen_layer_acts], dim=1)
                        full_hidden_states.append(layer_acts)
                    else:
                        full_hidden_states.append(None)
                logger.info(f"JsonActivationsLogger.extract_activations - Full sequence activations extracted successfully")
            except Exception as e:
                logger.error(f"JsonActivationsLogger.extract_activations - Failed to extract full sequence activations: {e}")
                raise

        logger.info(f"JsonActivationsLogger.extract_activations completed - extracted {len([x for x in full_hidden_states if x is not None])} layers")
        return full_hidden_states

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the JSON store with NPY activation files.

        Args:
            key: Unique identifier for the entry (typically a hash of the prompt)
            entry: Dictionary containing the data to log (prompt, response, model_outputs, etc.)
        """
        logger.info(f"JsonActivationsLogger.log_entry starting - key: {key[:8]}...")

        if self.read_only:
            raise ValueError("Cannot log entries in read-only mode")

        # Process model outputs if present
        if "model_outputs" in entry and "input_length" in entry:
            logger.info(f"JsonActivationsLogger.log_entry - Processing model outputs")
            model_outputs = entry["model_outputs"]
            input_length = entry["input_length"]

            # Add trim position to model_outputs for use in extract_activations
            if "trim_position" in entry:
                model_outputs.trim_position = entry["trim_position"]
                logger.info(f"JsonActivationsLogger.log_entry - Added trim position: {entry['trim_position']}")

            # Extract activations from model outputs
            logger.info(f"JsonActivationsLogger.log_entry - Starting activation extraction")
            try:
                import time
                extraction_start = time.time()
                all_layers_activations = self.extract_activations(model_outputs, input_length)
                extraction_time = time.time() - extraction_start
                logger.info(f"JsonActivationsLogger.log_entry - Activation extraction completed in {extraction_time:.3f}s")
            except Exception as e:
                logger.error(f"JsonActivationsLogger.log_entry - Activation extraction failed: {e}")
                raise

            # Create metadata entry (without activations)
            logger.info(f"JsonActivationsLogger.log_entry - Creating metadata entry")
            metadata_entry = entry.copy()
            metadata_entry.pop("model_outputs", None)
            metadata_entry.pop("all_layers_activations", None)

            if all_layers_activations:
                logger.info(f"JsonActivationsLogger.log_entry - Processing activations for storage")

                # Store the logging configuration in metadata
                metadata_entry["logging_config"] = {
                    "target_layers": self.target_layers,
                    "sequence_mode": self.sequence_mode
                }

                # Convert activations to numpy arrays and save as NPY file
                logger.info(f"JsonActivationsLogger.log_entry - Converting tensors to numpy arrays")
                try:
                    conversion_start = time.time()
                    activation_arrays = self._tensors_to_numpy_arrays(all_layers_activations)
                    conversion_time = time.time() - conversion_start
                    logger.info(f"JsonActivationsLogger.log_entry - Tensor conversion completed in {conversion_time:.3f}s")
                except Exception as e:
                    logger.error(f"JsonActivationsLogger.log_entry - Tensor conversion failed: {e}")
                    raise

                activation_file_path = self.activations_dir / f"{key}.npy"
                logger.info(f"JsonActivationsLogger.log_entry - Saving NPY file to {activation_file_path}")

                # Save as NPY file with allow_pickle=True to handle list of arrays
                try:
                    save_start = time.time()
                    np.save(activation_file_path, activation_arrays, allow_pickle=True)
                    save_time = time.time() - save_start
                    file_size = activation_file_path.stat().st_size / 1024**2  # MB
                    logger.info(f"JsonActivationsLogger.log_entry - NPY file saved in {save_time:.3f}s, size: {file_size:.2f}MB")
                except Exception as e:
                    logger.error(f"JsonActivationsLogger.log_entry - NPY file save failed: {e}")
                    raise

                # Add reference to activation file in metadata
                metadata_entry["has_activations"] = True
                metadata_entry["activation_file"] = f"activations/{key}.npy"
                metadata_entry["activation_shape_info"] = self._get_activation_shape_info(activation_arrays)
                logger.info(f"JsonActivationsLogger.log_entry - Added activation metadata")
            else:
                metadata_entry["has_activations"] = False
                logger.info(f"JsonActivationsLogger.log_entry - No activations to store")
        else:
            raise ValueError("No model_outputs or input_length found in entry")

        metadata_entry["timestamp"] = time.time()

        # Store metadata in the main metadata file
        logger.info(f"JsonActivationsLogger.log_entry - Updating metadata file")
        self.metadata["entries"][key] = metadata_entry

        # Save updated metadata
        try:
            metadata_save_start = time.time()
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            metadata_save_time = time.time() - metadata_save_start
            logger.info(f"JsonActivationsLogger.log_entry - Metadata saved in {metadata_save_time:.3f}s")
        except Exception as e:
            logger.error(f"JsonActivationsLogger.log_entry - Metadata save failed: {e}")
            raise

        logger.info(f"JsonActivationsLogger.log_entry completed successfully - key: {key[:8]}...")

    def get_entry(self, key: str, metadata_only: bool = False) -> Dict[str, Any]:
        """
        Retrieve an entry from the JSON store with NPY activation loading.

        Args:
            key: Unique identifier for the entry to retrieve
            metadata_only: If True, only retrieve metadata without activations

        Returns:
            Dictionary containing the entry data if found
        """
        if key not in self.metadata["entries"]:
            raise KeyError(f"Key {key} not found in metadata.")

        entry_metadata = self.metadata["entries"][key].copy()

        # If only metadata is requested or no activations exist
        if metadata_only or not entry_metadata.get("has_activations", False):
            return entry_metadata

        # Load activations from NPY file
        activation_file = entry_metadata.get("activation_file")
        if activation_file:
            activation_path = self.output_dir / activation_file
            if activation_path.exists():
                # Check if it's an NPY file (new format) or JSON file (old format)
                if activation_path.suffix == '.npy':
                    # Load NPY file and convert back to tensors
                    activation_arrays = np.load(activation_path, allow_pickle=True)
                    activations = self._numpy_arrays_to_tensors(activation_arrays)
                    entry_metadata["all_layers_activations"] = activations
                else:
                    # Backward compatibility: load old JSON format
                    with open(activation_path, "r") as f:
                        activation_data = json.load(f)
                    activations = self._json_to_tensor(activation_data["all_layers_activations"])
                    entry_metadata["all_layers_activations"] = activations
            else:
                logger.warning(f"Activation file not found: {activation_path}")
                entry_metadata["has_activations"] = False

        return entry_metadata

    def get_entry_by_key(self, key: str, metadata_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the JSON store by key.
        This method provides compatibility with the LMDB logger interface.

        Args:
            key: Unique identifier for the entry to retrieve
            metadata_only: If True, only retrieve metadata without activations

        Returns:
            Dictionary containing the entry data if found, None otherwise
        """
        try:
            return self.get_entry(key, metadata_only)
        except KeyError:
            return None

    def list_entries(self) -> List[str]:
        """
        List all entry keys in the JSON store.

        Returns:
            List of entry keys as strings
        """
        return list(self.metadata["entries"].keys())

    def search_metadata(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search through metadata entries using a filter function.

        Args:
            filter_fn: Function that takes a metadata entry and returns True if it matches the search criteria

        Returns:
            List of tuples containing (key, metadata_entry) for matching entries
        """
        results = []
        for key, metadata in self.metadata["entries"].items():
            if filter_fn(metadata):
                results.append((key, metadata))
        return results

    def close(self):
        """Close the JSON logger (no-op for JSON files)."""
        pass

    def fix_cuda_tensors(self):
        """
        Fix existing entries by ensuring all tensors are on CPU.
        This method provides compatibility with the LMDB logger interface.
        """
        if self.read_only:
            raise ValueError("Cannot fix CUDA tensors in read-only mode")

        logger.info("Starting to fix CUDA tensors in existing JSON entries...")
        fixed_count = 0
        skipped_count = 0
        error_count = 0

        keys = self.list_entries()
        total_entries = len(keys)
        logger.info(f"Found {total_entries} entries to process")

        for key in keys:
            try:
                # Check if this entry has activations
                metadata = self.metadata["entries"][key]
                if not metadata.get("has_activations", False):
                    skipped_count += 1
                    continue

                activation_file = metadata.get("activation_file")
                if not activation_file:
                    skipped_count += 1
                    continue

                activation_path = self.output_dir / activation_file
                if not activation_path.exists():
                    error_count += 1
                    continue

                # Load activation data
                with open(activation_path, "r") as f:
                    activation_data = json.load(f)

                # Convert to tensors to check device
                activations = self._json_to_tensor(activation_data["all_layers_activations"])

                # Check if any tensors need to be moved to CPU
                needs_fixing = False
                fixed_tensors = []
                for tensor in activations:
                    if isinstance(tensor, torch.Tensor):
                        if tensor.device.type != 'cpu':
                            needs_fixing = True
                            fixed_tensors.append(tensor.cpu())
                        else:
                            fixed_tensors.append(tensor)
                    else:
                        fixed_tensors.append(tensor)

                # Only write back if tensors needed fixing
                if needs_fixing:
                    # Convert back to JSON-serializable format
                    fixed_activation_data = {
                        "all_layers_activations": self._tensor_to_json_serializable(fixed_tensors)
                    }

                    # Save back to file
                    with open(activation_path, "w") as f:
                        json.dump(fixed_activation_data, f, indent=2)

                    fixed_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                logger.error(f"Error fixing entry {key[:8]}: {e}")
                error_count += 1

        logger.info(f"Finished fixing CUDA tensors. Fixed {fixed_count} entries, skipped {skipped_count} entries (already on CPU), encountered {error_count} errors")

    def _tensors_to_numpy_arrays(self, obj):
        """Convert tensors to numpy arrays for NPY storage."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, list):
            return [self._tensors_to_numpy_arrays(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._tensors_to_numpy_arrays(value) for key, value in obj.items()}
        else:
            return obj

    def _numpy_arrays_to_tensors(self, obj):
        """Convert numpy arrays back to tensors."""
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, list):
            return [self._numpy_arrays_to_tensors(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._numpy_arrays_to_tensors(value) for key, value in obj.items()}
        else:
            return obj

    def _get_activation_shape_info(self, activation_arrays):
        """Get shape information for activation arrays for debugging purposes."""
        if isinstance(activation_arrays, np.ndarray):
            return {"shape": activation_arrays.shape, "dtype": str(activation_arrays.dtype)}
        elif isinstance(activation_arrays, list):
            return [self._get_activation_shape_info(item) for item in activation_arrays]
        elif isinstance(activation_arrays, dict):
            return {key: self._get_activation_shape_info(value) for key, value in activation_arrays.items()}
        else:
            return {"type": type(activation_arrays).__name__}
