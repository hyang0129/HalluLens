#!/usr/bin/env python
"""
Command-line script to run the vLLM server with activation logging.
This script serves as a wrapper around vllm serve to ensure the activations
are properly logged using Zarr storage.

Usage:
  python -m activation_logging.vllm_serve [--model MODEL] [--host HOST] [--port PORT]
        [--logger-type {zarr}] [--activations-path PATH] [--target-layers {all,first_half,second_half}]
    [--sequence-mode {all,prompt,response}] [--auth_token AUTH_TOKEN] [--trim-output-at TRIM_SEQUENCE]
        [--map-size-gb MAP_SIZE_GB]

Examples:
    # Use Zarr logging
    python -m activation_logging.vllm_serve --logger-type zarr --activations-path zarr_data/activations.zarr
"""

import argparse
import os
import subprocess
import sys
from loguru import logger

# Respect the centralized log-level env var for stderr output.
# The file sink always logs at INFO so nothing is lost.
from utils.log_config import configure_logging  # noqa: E402

configure_logging()  # picks up HALLULENS_LOG_LEVEL from parent process

def main():
    parser = argparse.ArgumentParser(description="Run vLLM server with activation logging")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model ID or path (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run server on (default: 8000)")
    parser.add_argument("--activations-path", type=str, default=None,
                        help="Path for storing activations (Zarr store path ending in .zarr)")
    parser.add_argument("--logger-type", type=str, default="zarr",
                        choices=["zarr"],
                        help="Type of activation logger to use (zarr only)")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/activations.lmdb",
                        help="Deprecated and ignored (zarr-only activation logging).")
    parser.add_argument("--auth_token", type=str, default=None,
                        help="HuggingFace authentication token for accessing gated models")
    parser.add_argument("--trim-output-at", type=str, default=None,
                        help="Sequence at which to trim model output (e.g. '\\n'). Note that you need to escape this in linux so it's --trim-output-at $'\\n'")
    parser.add_argument("--map-size-gb", type=int, default=64,
                        help="Deprecated and ignored (zarr-only activation logging).")
    parser.add_argument("--log-file", type=str, default="server.log",
                        help="Path to log file (default: server.log)")
    parser.add_argument("--target-layers", type=str, default="all",
                        choices=["all", "first_half", "second_half"],
                        help="Which layers to extract activations from (default: all)")
    parser.add_argument("--sequence-mode", type=str, default="all",
                        choices=["all", "prompt", "response"],
                        help="Which tokens to extract activations for (default: all)")
    parser.add_argument("--logprobs-top-k", type=int, default=20,
                        help="Number of top logprobs to persist per generated token (default: 20)")
    parser.add_argument("--disable-logprobs", action="store_true",
                        help="Disable response token logprob logging (enabled by default)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum sequence length (prompt+response). Lower values reduce KV cache memory. "
                             "Overrides VLLM_MAX_MODEL_LEN env var.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Fraction of GPU memory reserved for vLLM (0.0–1.0). "
                             "Overrides VLLM_GPU_MEMORY_UTILIZATION env var.")

    args = parser.parse_args()

    # Add file sink for server logs (always at INFO level).
    # The stderr sink is already configured by configure_logging() above.
    logger.add(
        args.log_file,
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Determine the activations path (zarr-only)
    activations_path = args.activations_path if args.activations_path else "zarr_data/activations.zarr"
    if not str(activations_path).strip().lower().endswith(".zarr"):
        raise ValueError(
            "Unsupported activations path for zarr-only mode. "
            "Please use a .zarr path, e.g. shared/run/activations.zarr"
        )

    # Set environment variables for server configuration
    os.environ["ACTIVATION_STORAGE_PATH"] = activations_path
    os.environ["ACTIVATION_LOGGER_TYPE"] = "zarr"
    os.environ["ACTIVATION_LMDB_PATH"] = args.lmdb_path  # Keep for backward compatibility
    os.environ["ACTIVATION_LMDB_MAP_SIZE"] = str(args.map_size_gb * (1 << 30))  # Convert GB to bytes
    os.environ["SERVER_LOG_FILE"] = args.log_file  # Add log file path to environment
    os.environ["ACTIVATION_TARGET_LAYERS"] = args.target_layers  # Add target layers setting
    os.environ["ACTIVATION_SEQUENCE_MODE"] = args.sequence_mode  # Add sequence mode setting
    os.environ["ACTIVATION_LOGPROBS_ENABLED"] = "0" if args.disable_logprobs else "1"
    os.environ["ACTIVATION_LOGPROBS_TOPK"] = str(max(1, int(args.logprobs_top_k)))
    os.environ["DEFAULT_MODEL"] = args.model
    if args.auth_token:
        os.environ["HF_TOKEN"] = args.auth_token
        logger.info("Using provided HuggingFace token for model access")
    if args.max_model_len is not None:
        os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
        logger.info(f"Max model length: {args.max_model_len}")
    if args.gpu_memory_utilization is not None:
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
        logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
    
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
    logger.info(f"Response Logprobs Enabled: {not args.disable_logprobs}")
    logger.info(f"Response Logprobs Top-K: {max(1, int(args.logprobs_top_k))}")

    # Create zarr parent directory if it doesn't exist
    activations_dir = os.path.dirname(activations_path)
    if activations_dir and not os.path.exists(activations_dir):
        os.makedirs(activations_dir, exist_ok=True)
        logger.info(f"Created Zarr directory: {activations_dir}")
    
    # Two server options:
    # 1. Using uvicorn directly (easier for debugging)
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn", "activation_logging.server:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "warning",
        "--no-access-log",
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
        # IMPORTANT: ServerManager starts this process with stdout/stderr piped;
        # if we don't drain those pipes, the server can deadlock once buffers fill.
        # Redirect all child stdout/stderr into the log file to prevent backpressure hangs.
        with open(args.log_file, "a", encoding="utf-8") as log_f:
            subprocess.run(uvicorn_cmd, stdout=log_f, stderr=log_f)
        
        # Use vllm serve in production (uncomment to use)
        # subprocess.run(vllm_cmd)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 