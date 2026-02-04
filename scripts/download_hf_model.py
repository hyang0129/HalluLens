#!/usr/bin/env python3
"""
Download and cache Hugging Face models for use with vLLM.

Prerequisites:
1. Accept the model license on Hugging Face
2. Authenticate: huggingface-cli login

Usage:
    python scripts/download_hf_model.py --model meta-llama/Llama-3.3-70B-Instruct
    python scripts/download_hf_model.py --model meta-llama/Llama-3.3-70B-Instruct --cache-dir /path/to/cache
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub is not installed!")
    print("Install it with: pip install huggingface-hub")
    sys.exit(1)


def check_authentication():
    """Check if user is authenticated with Hugging Face."""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated with Hugging Face!")
        print(f"   Error: {e}")
        print(f"\nPlease authenticate with: huggingface-cli login")
        return False


def download_model(model_id, cache_dir=None, allow_patterns=None, ignore_patterns=None):
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        cache_dir: Custom cache directory (optional)
        allow_patterns: List of file patterns to download (optional)
        ignore_patterns: List of file patterns to ignore (optional)
    
    Returns:
        Path to downloaded model directory
    """
    print(f"📦 Downloading model: {model_id}")
    print(f"   This may take a while depending on model size and network speed...")
    
    if cache_dir:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Cache directory: {cache_dir}")
    else:
        print(f"   Using default HuggingFace cache directory")
    
    try:
        # Download the model
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True,  # Resume interrupted downloads
            local_files_only=False
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Location: {model_path}")
        return model_path
        
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"❌ Authentication error!")
            print(f"   Make sure you:")
            print(f"   1. Accepted the license at https://huggingface.co/{model_id}")
            print(f"   2. Authenticated with: huggingface-cli login")
        else:
            print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for vLLM")
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID from Hugging Face Hub (e.g., meta-llama/Llama-3.3-70B-Instruct)"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory (default: HuggingFace default cache)"
    )
    
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip authentication check (not recommended)"
    )
    
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Download only model weights (skip tokenizer configs, etc.)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Hugging Face Model Downloader")
    print("=" * 80)
    print()
    
    # Check authentication
    if not args.skip_auth_check:
        if not check_authentication():
            sys.exit(1)
        print()
    
    # Determine file patterns if weights-only mode
    allow_patterns = None
    ignore_patterns = None
    
    if args.weights_only:
        print("📋 Weights-only mode: downloading model weights and essential configs")
        allow_patterns = [
            "*.safetensors",
            "*.bin",
            "*.json",
            "*.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
            "generation_config.json"
        ]
        ignore_patterns = ["*.msgpack", "*.h5", "*.ot"]
    
    # Download the model
    model_path = download_model(
        model_id=args.model,
        cache_dir=args.cache_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns
    )
    
    print()
    print("=" * 80)
    print("✅ Download Complete!")
    print("=" * 80)
    print()
    print(f"Model path: {model_path}")
    print()
    print("You can now use this model with vLLM:")
    print(f"  python scripts/run_with_server.py --step generate --task precisewikiqa --model {args.model} --N 100")
    print()


if __name__ == "__main__":
    main()
