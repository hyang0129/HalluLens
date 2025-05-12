#!/usr/bin/env python
"""
Command-line script to run the vLLM server with activation logging.
This script serves as a wrapper around vllm serve to ensure the activations
are properly logged.

Usage:
  python -m activation_logging.vllm_serve [--model MODEL] [--host HOST] [--port PORT] [--lmdb_path LMDB_PATH] [--auth_token AUTH_TOKEN]
"""

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run vLLM server with activation logging")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model ID or path (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run server on (default: 8000)")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/activations.lmdb",
                        help="Path to LMDB for storing activations (default: lmdb_data/activations.lmdb)")
    parser.add_argument("--auth_token", type=str, default=None,
                        help="HuggingFace authentication token for accessing gated models")
    
    args = parser.parse_args()
    
    # Set environment variables for server configuration
    os.environ["ACTIVATION_LMDB_PATH"] = args.lmdb_path
    if args.auth_token:
        os.environ["HF_TOKEN"] = args.auth_token
        print(f"Using provided HuggingFace token for model access")
    
    print(f"Starting vLLM server with activation logging")
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"LMDB Path: {args.lmdb_path}")
    
    # Create LMDB directory if it doesn't exist
    lmdb_dir = os.path.dirname(args.lmdb_path)
    if lmdb_dir and not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)
        print(f"Created LMDB directory: {lmdb_dir}")
    
    # Two server options:
    # 1. Using uvicorn directly (easier for debugging)
    uvicorn_cmd = [
        "uvicorn", "activation_logging.server:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    # 2. Using vllm serve (recommended for production)
    vllm_cmd = [
        "vllm", "serve",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", "1",  # Adjust based on GPU count
    ]
    
    # Add auth token to vllm command if provided
    if args.auth_token:
        vllm_cmd.extend(["--use-auth-token", args.auth_token])
    
    try:
        # Use uvicorn in development (comment this out to use vllm serve instead)
        subprocess.run(uvicorn_cmd)
        
        # Use vllm serve in production (uncomment to use)
        # subprocess.run(vllm_cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 