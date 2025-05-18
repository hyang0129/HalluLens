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
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard module not found. Install with 'pip install zstandard' to enable compression.")


class BaseCompressor:
    """Base class for compression implementations."""
    
    @staticmethod
    def compress(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compress data.
        
        Args:
            data: Dictionary containing the data to be compressed
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        raise NotImplementedError("Subclasses must implement compress method")
    
    @staticmethod
    def decompress(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress data.
        
        Args:
            data: Dictionary containing the compressed data
            metadata: Dictionary with metadata about how the data was compressed
            
        Returns:
            Dictionary with decompressed data
        """
        raise NotImplementedError("Subclasses must implement decompress method")


class NoCompression(BaseCompressor):
    """No compression, just passes data through."""
    
    @staticmethod
    def compress(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Default compression function that doesn't actually compress.
        
        Args:
            data: Dictionary containing the data to be compressed
            
        Returns:
            Tuple of (uncompressed_data, metadata)
            - uncompressed_data: The original data unchanged
            - metadata: Dictionary with metadata about the compression (empty for default)
        """
        # No compression, just return the original data and empty metadata
        return data, {"compression": "none"}
    
    @staticmethod
    def decompress(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default decompression function that doesn't do anything.
        
        Args:
            data: Dictionary containing the data to be decompressed
            metadata: Dictionary with metadata about how the data was compressed
            
        Returns:
            The original data unchanged
        """
        # No decompression needed, just return the original data
        return data


class ZstdCompression(BaseCompressor):
    """Zstandard compression with configurable level."""
    
    def __init__(self, level: int = 19):
        """
        Initialize ZstdCompression with the specified compression level.
        
        Args:
            level: Compression level (1-22, higher = better compression but slower)
        """
        self.level = level
        if not ZSTD_AVAILABLE:
            logger.warning("zstandard module not available, compression will fail")
    
    def compress(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compress data using zstandard with the configured level.
        Only compresses the 'all_layers_activations' field to save space.
        
        Args:
            data: Dictionary containing the data to be compressed
            
        Returns:
            Tuple of (compressed_data, metadata)
            - compressed_data: Dictionary with compressed activations
            - metadata: Dictionary with metadata about the compression
        """
        if not ZSTD_AVAILABLE:
            logger.warning("zstandard module not available, skipping compression")
            return data, {"compression": "none", "error": "zstandard not available"}
            
        # Create a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Only compress the activations if present
        if "all_layers_activations" in result:
            try:
                # Create a compressor with the specified level
                compressor = zstd.ZstdCompressor(level=self.level)
                
                # Pickle and compress the activations
                pickled_data = pickle.dumps(result["all_layers_activations"])
                compressed_data = compressor.compress(pickled_data)
                
                # Replace the activations with the compressed version
                result["all_layers_activations"] = compressed_data
                
                # Return the compressed data and metadata
                return result, {
                    "compression": "zstd",
                    "level": self.level,
                    "original_size": len(pickled_data),
                    "compressed_size": len(compressed_data),
                    "ratio": len(pickled_data) / max(1, len(compressed_data))
                }
            except Exception as e:
                logger.error(f"Error during zstd compression: {e}")
                # Return the original data if compression fails
                return data, {"compression": "none", "error": str(e)}
        else:
            # No activations to compress
            return result, {"compression": "none", "reason": "no activations found"}
    
    @staticmethod
    def decompress(data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress data that was compressed with zstandard.
        
        Args:
            data: Dictionary containing the compressed data
            metadata: Dictionary with metadata about how the data was compressed
            
        Returns:
            Dictionary with decompressed data
        """
        if not ZSTD_AVAILABLE:
            logger.error("zstandard module not available, cannot decompress")
            raise ImportError("zstandard module not available, cannot decompress")
            
        # Create a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Check if this data was compressed with zstd
        compression = metadata.get("compression", "none")
        if compression != "zstd":
            # Data wasn't compressed with zstd, return as is
            return result
            
        # Decompress the activations if present
        if "all_layers_activations" in result and isinstance(result["all_layers_activations"], bytes):
            try:
                # Create a decompressor
                decompressor = zstd.ZstdDecompressor()
                
                # Decompress and unpickle the activations
                compressed_data = result["all_layers_activations"]
                decompressed_data = decompressor.decompress(compressed_data)
                activations = pickle.loads(decompressed_data)
                
                # Replace the compressed activations with the decompressed version
                result["all_layers_activations"] = activations
                
                return result
            except Exception as e:
                logger.error(f"Error during zstd decompression: {e}")
                # Return the original data if decompression fails
                return data
        else:
            # No compressed activations found
            return result


class ActivationsLogger:
    def __init__(self, lmdb_path: str = "lmdb_data/activations.lmdb", map_size: int = 16 << 30,
                 compression: Union[str, BaseCompressor, None] = 'zstd', 
                 read_only: bool = False):
        """
        Initialize the LMDB-based activations logger.
        
        Args:
            lmdb_path: Path to the LMDB file to store activations
            map_size: Maximum size of the LMDB file in bytes (default: 16GB)
            compression: Compression method ('zstd', None) or a BaseCompressor instance
            read_only: if True, the LMDB will be opened in read-only mode
        """
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
            
        # Open main LMDB for activations (large map size)
        self.env = lmdb.open(safe_lmdb_path, map_size=map_size, subdir=True, create=True, readonly=read_only, lock=True)
        
        # Open separate LMDB for metadata (smaller map size)
        metadata_path = os.path.join(os.path.dirname(safe_lmdb_path), f"{os.path.basename(safe_lmdb_path)}_metadata")
        os.makedirs(metadata_path, exist_ok=True)
        self.metadata_env = lmdb.open(metadata_path, map_size, subdir=True, create=True, readonly=read_only, lock=True)
        logger.info(f"Opened metadata store at {metadata_path}")

        self.read_only = True

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
            Dictionary containing activations from all layers
        """
        # If model_outputs is None or doesn't have hidden_states, return None
        if model_outputs is None or not hasattr(model_outputs, 'hidden_states'):
            logger.info("No hidden states found in model outputs")
            return None
            
        # Get the generated tokens (excluding prompt)
        gen_sequence = model_outputs.sequences[0]
        gen_ids = gen_sequence[input_length:]
        
        all_hidden_states = model_outputs.hidden_states
        prompt_hidden = all_hidden_states[0]
        gen_hiddens = all_hidden_states[1:]

        gen_hidden_per_layer = [
            torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
            for layer_idx in range(len(prompt_hidden))
        ]

        full_hidden_states = [
            torch.cat([prompt_hidden[layer_idx], gen_hidden_per_layer[layer_idx]], dim=1)
            for layer_idx in range(len(prompt_hidden))
        ]

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
            
            # Extract activations from model outputs
            all_layers_activations = self.extract_activations(model_outputs, input_length)
            
            entry.pop("model_outputs", None)
            metadata_entry = entry.copy()
            metadata_entry.pop("all_layers_activations", None)

            if all_layers_activations:
                # Replace model_outputs with extracted activations to save space
                entry["all_layers_activations"] = all_layers_activations
                

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
        if self.env is None:
            raise ValueError("LMDB is not initialized")
            
        # If only metadata is requested and metadata store is available
        if metadata_only and self.metadata_env is not None:
            with self.metadata_env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
                if value is not None:
                    return pickle.loads(value)
                else:
                    raise ValueError("No metadata found for key")
        
        # Otherwise retrieve full entry including activations
        with self.env.begin(write=False) as txn:
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
                    logger.debug(f"Retrieved legacy entry with key {key[:8]} (no compression)")
                    return stored_entry
            else:
                raise ValueError("No entry found for key")

    def list_entries(self) -> List[str]:
        """
        List all entry keys in the metadata store.
        
        Returns:
            List of entry keys as strings
        """
        if self.metadata_env is None:
            return []
            
        keys = []
        with self.metadata_env.begin(write=False) as txn:
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
        if self.env is not None:
            self.env.close()
            self.env = None
            
        if self.metadata_env is not None:
            self.metadata_env.close()
            self.metadata_env = None
