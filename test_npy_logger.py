#!/usr/bin/env python3
"""
Test script to verify NPY logging functionality in JsonActivationsLogger
"""

import sys
import os
import tempfile
import torch
import numpy as np
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

from activation_logging.activations_logger import JsonActivationsLogger

def test_npy_logger():
    """Test the NPY logging functionality"""
    print("🧪 Testing NPY Logger Functionality")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger_path = os.path.join(tmpdir, 'test_npy.json')
        print(f"📁 Creating logger at: {logger_path}")
        
        # Create logger
        logger = JsonActivationsLogger(logger_path)
        print(f"✅ Logger created successfully")
        print(f"📊 Logger version: {logger.metadata.get('logger_config', {}).get('version')}")
        print(f"💾 Storage format: {logger.metadata.get('logger_config', {}).get('storage_format')}")
        
        # Check if NPY methods exist
        has_tensors_to_numpy = hasattr(logger, '_tensors_to_numpy_arrays')
        has_numpy_to_tensors = hasattr(logger, '_numpy_arrays_to_tensors')
        has_shape_info = hasattr(logger, '_get_activation_shape_info')
        
        print(f"🔧 Has _tensors_to_numpy_arrays: {has_tensors_to_numpy}")
        print(f"🔧 Has _numpy_arrays_to_tensors: {has_numpy_to_tensors}")
        print(f"🔧 Has _get_activation_shape_info: {has_shape_info}")
        
        if not all([has_tensors_to_numpy, has_numpy_to_tensors, has_shape_info]):
            print("❌ Missing required NPY methods!")
            return False
        
        # Create test data with fake activations
        print("\n🎯 Creating test entry with fake activations...")
        
        # Create fake activation tensors (simulating model outputs)
        fake_activations = [
            torch.randn(2, 10, 768),  # Layer 0: batch_size=2, seq_len=10, hidden_size=768
            torch.randn(2, 10, 768),  # Layer 1
            torch.randn(2, 10, 768),  # Layer 2
        ]
        
        test_entry = {
            "prompt": "Test prompt for NPY logging",
            "response": "Test response",
            "input_length": 5,
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test prompt for NPY logging"}],
            "trim_position": None,
            "all_layers_activations": fake_activations
        }
        
        # Log the entry
        test_key = "test_key_12345"
        logger.log_entry(test_key, test_entry)
        print(f"✅ Entry logged with key: {test_key}")
        
        # Check if NPY file was created
        activations_dir = Path(logger_path) / "activations"
        npy_file = activations_dir / f"{test_key}.npy"
        
        print(f"📂 Activations directory exists: {activations_dir.exists()}")
        print(f"📄 NPY file exists: {npy_file.exists()}")
        
        if npy_file.exists():
            file_size = npy_file.stat().st_size
            print(f"📏 NPY file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Check metadata file
        metadata_file = Path(logger_path) / "metadata.json"
        print(f"📄 Metadata file exists: {metadata_file.exists()}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if test_key in metadata.get('entries', {}):
                entry_meta = metadata['entries'][test_key]
                print(f"✅ Entry found in metadata")
                print(f"📊 Has activations: {entry_meta.get('has_activations', False)}")
                print(f"📄 Activation file: {entry_meta.get('activation_file', 'None')}")
                
                # Check shape info
                shape_info = entry_meta.get('activation_shape_info', [])
                if shape_info:
                    print(f"📐 Activation shapes: {len(shape_info)} layers")
                    for i, info in enumerate(shape_info):
                        if isinstance(info, dict) and 'shape' in info:
                            print(f"   Layer {i}: {info['shape']} ({info.get('dtype', 'unknown')})")
        
        # Test retrieval
        print(f"\n🔄 Testing entry retrieval...")
        try:
            retrieved_entry = logger.get_entry(test_key)
            print(f"✅ Entry retrieved successfully")
            
            # Check if activations were loaded correctly
            if "all_layers_activations" in retrieved_entry:
                retrieved_activations = retrieved_entry["all_layers_activations"]
                print(f"📊 Retrieved {len(retrieved_activations)} activation layers")
                
                # Verify tensor shapes match
                for i, (original, retrieved) in enumerate(zip(fake_activations, retrieved_activations)):
                    if torch.allclose(original, retrieved, atol=1e-6):
                        print(f"✅ Layer {i}: Tensors match perfectly")
                    else:
                        print(f"❌ Layer {i}: Tensors don't match!")
                        return False
            else:
                print("❌ No activations found in retrieved entry!")
                return False
                
        except Exception as e:
            print(f"❌ Error retrieving entry: {e}")
            return False
        
        print(f"\n🎉 NPY Logger Test PASSED!")
        return True

if __name__ == "__main__":
    success = test_npy_logger()
    if success:
        print("\n✅ All tests passed! NPY logging is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! NPY logging has issues.")
        sys.exit(1)
