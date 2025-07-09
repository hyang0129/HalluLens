"""
compression.py
Handles compression and decompression of activation data.
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple
from loguru import logger
import pickle

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
                
                # Convert tensors to numpy arrays and collect metadata
                tensor_metadata = []
                all_hidden_np = []
                
                for i, tensor in enumerate(result["all_layers_activations"]):
                    # Ensure tensor is on CPU
                    tensor = tensor.cpu()
                    
                    # Store tensor metadata
                    tensor_metadata.append({
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": "cpu"  # Always store as CPU
                    })
                    # Convert to numpy and store
                    all_hidden_np.append(tensor.numpy())
                
                # Join all numpy arrays into a single bytes object
                data_bytes = b"".join(t.tobytes() for t in all_hidden_np)
                
                # Compress the bytes
                compressed_data = compressor.compress(data_bytes)
                
                # Replace the activations with the compressed version
                result["all_layers_activations"] = compressed_data
                
                # Return the compressed data and metadata
                return result, {
                    "compression": "zstd",
                    "level": self.level,
                    "original_size": len(data_bytes),
                    "compressed_size": len(compressed_data),
                    "ratio": len(data_bytes) / max(1, len(compressed_data)),
                    "tensor_metadata": tensor_metadata
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
                
                # Decompress the data
                compressed_data = result["all_layers_activations"]
                decompressed_data = decompressor.decompress(compressed_data)
                
                # Get tensor metadata
                tensor_metadata = metadata.get("tensor_metadata", [])
                if not tensor_metadata:
                    raise ValueError("No tensor metadata found for decompression")
                
                # Reconstruct tensors from decompressed data
                reconstructed_tensors = []
                offset = 0
                
                for meta in tensor_metadata:
                    # Calculate size of this tensor in bytes
                    shape = tuple(meta["shape"])
                    dtype = np.dtype(meta["dtype"].split(".")[-1])  # Convert torch dtype string to numpy dtype
                    tensor_size = int(np.prod(shape) * dtype.itemsize)
                    
                    # Extract bytes for this tensor
                    tensor_bytes = decompressed_data[offset:offset + tensor_size]
                    offset += tensor_size
                    
                    # Convert bytes to numpy array
                    tensor_np = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
                    
                    # Convert to torch tensor and always place on CPU
                    tensor = torch.from_numpy(tensor_np)
                    
                    reconstructed_tensors.append(tensor)
                
                # Replace the compressed activations with the reconstructed tensors
                result["all_layers_activations"] = reconstructed_tensors
                
                return result
            except Exception as e:
                logger.error(f"Error during zstd decompression: {e}")
                # Return the original data if decompression fails
                return data
        else:
            # No compressed activations found
            raise ValueError("No compressed activations found in data")


class BytesCompressor(BaseCompressor):
    """
    Compressor that simply serializes data to bytes using pickle (no actual compression).
    """
    def compress(self, data):
        return pickle.dumps(data), {"compression": "bytes"}

    def decompress(self, data, metadata):
        return pickle.loads(data) 