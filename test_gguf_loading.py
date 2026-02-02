#!/usr/bin/env python3
"""
Test script to verify GGUF model loading without server infrastructure.
This helps diagnose model loading issues by testing llama-cpp-python directly.

Usage:
    python test_gguf_loading.py
    python test_gguf_loading.py --model path/to/model.gguf
    python test_gguf_loading.py --model models/Llama-3.3-70B-Instruct-Q6_K_L
"""

import argparse
import os
import sys
from pathlib import Path

def find_gguf_files(directory):
    """Recursively find all .gguf files in a directory."""
    gguf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gguf'):
                gguf_files.append(os.path.join(root, file))
    return gguf_files

def test_gguf_loading(model_path, run_inference=False):
    """Test loading a GGUF model and optionally run inference."""
    
    print("=" * 80)
    print("GGUF Model Loading Test")
    print("=" * 80)
    
    # Step 1: Check if path exists
    print(f"\n1. Checking model path: {model_path}")
    print(f"   Absolute path: {os.path.abspath(model_path)}")
    print(f"   Exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        if os.path.isdir(model_path):
            print(f"   Type: Directory")
            print(f"\n   Searching for .gguf files in directory...")
            gguf_files = find_gguf_files(model_path)
            if gguf_files:
                print(f"   Found {len(gguf_files)} .gguf file(s):")
                for f in gguf_files:
                    size_gb = os.path.getsize(f) / (1024**3)
                    print(f"     - {f} ({size_gb:.2f} GB)")
                
                # Check if this is a split model (e.g., -00001-of-00002.gguf)
                split_files = [f for f in gguf_files if '-00001-of-' in f or '00001-of-' in f]
                
                if len(gguf_files) == 1:
                    model_path = gguf_files[0]
                    print(f"\n   Using: {model_path}")
                elif split_files:
                    # Split model detected - use the first part
                    model_path = split_files[0]
                    print(f"\n   ✓ Split model detected, using first part: {os.path.basename(model_path)}")
                    print(f"     (llama.cpp will automatically load all parts)")
                else:
                    print(f"\n   ⚠️  Multiple .gguf files found. Please specify which one to use.")
                    print(f"     Example: python test_gguf_loading.py --model \"{gguf_files[0]}\"")
                    return False
            else:
                print(f"   ❌ No .gguf files found in directory!")
                return False
        else:
            size_gb = os.path.getsize(model_path) / (1024**3)
            print(f"   Type: File")
            print(f"   Size: {size_gb:.2f} GB")
    else:
        print(f"   ❌ Path does not exist!")
        
        # Try to find similar paths
        parent_dir = os.path.dirname(model_path)
        basename = os.path.basename(model_path)
        
        if os.path.exists(parent_dir):
            print(f"\n   Parent directory exists: {parent_dir}")
            print(f"   Contents:")
            try:
                items = os.listdir(parent_dir)
                for item in items[:10]:  # Show first 10 items
                    print(f"     - {item}")
                if len(items) > 10:
                    print(f"     ... and {len(items) - 10} more")
            except Exception as e:
                print(f"     Error listing directory: {e}")
        
        return False
    
    # Step 2: Try to import llama-cpp-python
    print(f"\n2. Importing llama-cpp-python...")
    try:
        from llama_cpp import Llama
        print(f"   ✓ Successfully imported llama_cpp.Llama")
    except ImportError as e:
        print(f"   ❌ Failed to import llama-cpp-python: {e}")
        print(f"   Install with: pip install llama-cpp-python")
        return False
    
    # Step 3: Try to load the model
    print(f"\n3. Loading model...")
    print(f"   This may take a few minutes for large models...")
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all GPU layers available
            verbose=True,
            n_ctx=4096,
        )
        print(f"   ✓ Successfully loaded model!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Run a simple inference test (optional)
    if run_inference:
        print(f"\n4. Running inference test...")
        test_prompt = "Hello, how are you?"
        print(f"   Prompt: {test_prompt}")
        try:
            output = llm(
                test_prompt,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9,
                echo=False
            )
            response = output['choices'][0]['text']
            print(f"   Response: {response}")
            print(f"   ✓ Inference successful!")
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'=' * 80}")
    print(f"✅ All tests passed!")
    print(f"{'=' * 80}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test GGUF model loading")
    parser.add_argument(
        "--model",
        default="models/Llama-3.3-70B-Instruct-Q6_K_L",
        help="Path to GGUF model file or directory containing .gguf files"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run a simple inference test after loading"
    )
    parser.add_argument(
        "--search",
        help="Search for .gguf files in the specified directory"
    )
    
    args = parser.parse_args()
    
    if args.search:
        print(f"Searching for .gguf files in: {args.search}")
        gguf_files = find_gguf_files(args.search)
        if gguf_files:
            print(f"\nFound {len(gguf_files)} .gguf file(s):")
            for f in gguf_files:
                size_gb = os.path.getsize(f) / (1024**3)
                print(f"  {f} ({size_gb:.2f} GB)")
        else:
            print("No .gguf files found.")
        return
    
    success = test_gguf_loading(args.model, run_inference=args.inference)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
