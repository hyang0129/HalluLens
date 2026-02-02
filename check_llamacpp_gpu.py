#!/usr/bin/env python3
"""
Check if llama-cpp-python has GPU support enabled.
"""

import sys

print("=" * 80)
print("llama-cpp-python GPU Support Check")
print("=" * 80)

# Check if llama-cpp-python is installed
print("\n1. Checking llama-cpp-python installation...")
try:
    import llama_cpp
    print(f"   ✓ llama-cpp-python is installed")
    print(f"   Version: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"   ❌ llama-cpp-python not installed: {e}")
    sys.exit(1)

# Check for CUDA support
print("\n2. Checking for CUDA support...")
try:
    from llama_cpp import llama_cpp
    
    # Check if CUDA functions are available
    has_cuda = hasattr(llama_cpp, 'llama_supports_gpu_offload') and llama_cpp.llama_supports_gpu_offload()
    
    if has_cuda:
        print(f"   ✓ CUDA support is ENABLED")
    else:
        print(f"   ❌ CUDA support is NOT available")
        print(f"\n   To enable GPU support, reinstall llama-cpp-python with CUDA:")
        print(f"   ")
        print(f"   # Uninstall current version")
        print(f"   pip uninstall llama-cpp-python -y")
        print(f"   ")
        print(f"   # Install with CUDA support (for CUDA 12.x)")
        print(f"   CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python --force-reinstall --no-cache-dir")
        print(f"   ")
        print(f"   # Or use pre-built wheel (faster)")
        print(f"   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
except Exception as e:
    print(f"   ⚠️  Could not check CUDA support: {e}")

# Try to detect available GPUs
print("\n3. Checking for available GPUs...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ PyTorch detects {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"     - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"       Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print(f"   ⚠️  PyTorch does not detect any GPUs")
except ImportError:
    print(f"   ⚠️  PyTorch not installed, cannot check GPU availability")
except Exception as e:
    print(f"   ⚠️  Error checking GPUs: {e}")

# Check environment variables
print("\n4. Checking environment variables...")
import os
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible:
    print(f"   CUDA_VISIBLE_DEVICES = {cuda_visible}")
else:
    print(f"   CUDA_VISIBLE_DEVICES not set (all GPUs should be visible)")

print("\n" + "=" * 80)
