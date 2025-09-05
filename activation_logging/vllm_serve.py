#!/usr/bin/env python
"""
Command-line script to run the vLLM server with activation logging.
This script serves as a wrapper around vllm serve to ensure the activations
are properly logged using either LMDB or JSON storage.

Usage:
  python -m activation_logging.vllm_serve [--model MODEL] [--host HOST] [--port PORT]
    [--logger-type {lmdb,json}] [--activations-path PATH] [--target-layers {all,first_half,second_half}]
    [--sequence-mode {all,prompt,response}] [--auth_token AUTH_TOKEN] [--trim-output-at TRIM_SEQUENCE]
    [--map-size-gb MAP_SIZE_GB]

Examples:
  # Use JSON logging
  python -m activation_logging.vllm_serve --logger-type json --activations-path json_data/activations

  # Use LMDB logging (default)
  python -m activation_logging.vllm_serve --logger-type lmdb --activations-path lmdb_data/activations.lmdb
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
    parser.add_argument("--activations-path", type=str, default=None,
                        help="Path for storing activations (LMDB file or JSON directory)")
    parser.add_argument("--logger-type", type=str, default="lmdb",
                        choices=["lmdb", "json"],
                        help="Type of activation logger to use (default: lmdb)")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/activations.lmdb",
                        help="Path to LMDB for storing activations (default: lmdb_data/activations.lmdb) - deprecated, use --activations-path")
    parser.add_argument("--auth_token", type=str, default=None,
                        help="HuggingFace authentication token for accessing gated models")
    parser.add_argument("--trim-output-at", type=str, default=None,
                        help="Sequence at which to trim model output (e.g. '\\n'). Note that you need to escape this in linux so it's --trim-output-at $'\\n'")
    parser.add_argument("--map-size-gb", type=int, default=64,
                        help="Size of LMDB map in gigabytes (default: 64)")
    parser.add_argument("--log-file", type=str, default="server.log",
                        help="Path to log file (default: server.log)")
    parser.add_argument("--target-layers", type=str, default="all",
                        choices=["all", "first_half", "second_half"],
                        help="Which layers to extract activations from (default: all)")
    parser.add_argument("--sequence-mode", type=str, default="all",
                        choices=["all", "prompt", "response"],
                        help="Which tokens to extract activations for (default: all)")
    
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
    
    # Determine the activations path
    activations_path = args.activations_path if args.activations_path else args.lmdb_path

    # Set default path based on logger type if not specified
    if not args.activations_path:
        if args.logger_type == "json":
            activations_path = "json_data/activations"
        else:
            activations_path = args.lmdb_path

    # Set environment variables for server configuration
    os.environ["ACTIVATION_STORAGE_PATH"] = activations_path
    os.environ["ACTIVATION_LOGGER_TYPE"] = args.logger_type
    os.environ["ACTIVATION_LMDB_PATH"] = args.lmdb_path  # Keep for backward compatibility
    os.environ["ACTIVATION_LMDB_MAP_SIZE"] = str(args.map_size_gb * (1 << 30))  # Convert GB to bytes
    os.environ["SERVER_LOG_FILE"] = args.log_file  # Add log file path to environment
    os.environ["ACTIVATION_TARGET_LAYERS"] = args.target_layers  # Add target layers setting
    os.environ["ACTIVATION_SEQUENCE_MODE"] = args.sequence_mode  # Add sequence mode setting
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
    logger.info(f"Logger Type: {args.logger_type}")
    logger.info(f"Activations Path: {activations_path}")
    logger.info(f"Target Layers: {args.target_layers}")
    logger.info(f"Sequence Mode: {args.sequence_mode}")
    if args.logger_type == "lmdb":
        logger.info(f"LMDB Map Size: {args.map_size_gb} GB")
    
    # Create activations directory if it doesn't exist
    if args.logger_type == "json":
        # For JSON logger, create the output directory
        if not os.path.exists(activations_path):
            os.makedirs(activations_path, exist_ok=True)
            logger.info(f"Created JSON activations directory: {activations_path}")
    else:
        # For LMDB logger, create the parent directory
        activations_dir = os.path.dirname(activations_path)
        if activations_dir and not os.path.exists(activations_dir):
            os.makedirs(activations_dir, exist_ok=True)
            logger.info(f"Created LMDB directory: {activations_dir}")
    
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