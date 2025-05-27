#!/usr/bin/env python
"""
Command-line script to run the vLLM server with activation logging.
This script serves as a wrapper around vllm serve to ensure the activations
are properly logged.

Usage:
  python -m activation_logging.vllm_serve [--model MODEL] [--host HOST] [--port PORT] [--lmdb_path LMDB_PATH] [--auth_token AUTH_TOKEN] [--trim-output-at TRIM_SEQUENCE] [--map-size-gb MAP_SIZE_GB]
"""

import argparse
import os
import subprocess
import sys
from loguru import logger

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
    parser.add_argument("--trim-output-at", type=str, default=None,
                        help="Sequence at which to trim model output (e.g. '\\n'). Note that you need to escape this in linux so it's --trim-output-at $'\\n'")
    parser.add_argument("--map-size-gb", type=int, default=64,
                        help="Size of LMDB map in gigabytes (default: 64)")
    parser.add_argument("--log-file", type=str, default="server.log",
                        help="Path to log file (default: server.log)")
    
    args = parser.parse_args()

    # Configure loguru logger
    logger.remove()  # Remove default handler
    logger.add(
        args.log_file,
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Set environment variables for server configuration
    os.environ["ACTIVATION_LMDB_PATH"] = args.lmdb_path
    os.environ["ACTIVATION_LMDB_MAP_SIZE"] = str(args.map_size_gb * (1 << 30))  # Convert GB to bytes
    if args.auth_token:
        os.environ["HF_TOKEN"] = args.auth_token
        logger.info("Using provided HuggingFace token for model access")
    
    # Set trim sequence if provided
    if args.trim_output_at:
        os.environ["TRIM_OUTPUT_AT"] = args.trim_output_at
        logger.info(f"Will trim output at sequence: {repr(args.trim_output_at)}")
    
    logger.info(f"Starting vLLM server with activation logging")
    logger.info(f"Model: {args.model}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"LMDB Path: {args.lmdb_path}")
    logger.info(f"LMDB Map Size: {args.map_size_gb} GB")
    
    # Create LMDB directory if it doesn't exist
    lmdb_dir = os.path.dirname(args.lmdb_path)
    if lmdb_dir and not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)
        logger.info(f"Created LMDB directory: {lmdb_dir}")
    
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
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 