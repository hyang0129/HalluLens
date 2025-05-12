#!/usr/bin/env python
"""
Script to run the nonsense_mixed_entities.py benchmark with activation logging.
This script ensures that:
1. The server is running with activation logging enabled
2. The benchmark is configured to use the right model and LMDB path
3. The results are properly stored and accessible for analysis
"""
import os
import argparse
import subprocess
import sys
import time
import requests
import hashlib

def check_server(host="0.0.0.0", port=8000, retries=5, delay=2):
    """Check if the server is running and responsive."""
    for i in range(retries):
        try:
            response = requests.get(f"http://{host}:{port}/health")
            if response.status_code == 200:
                print(f"Server is running at http://{host}:{port}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        if i < retries - 1:
            print(f"Server not responding. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print(f"Could not connect to server at http://{host}:{port}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Run nonsense_mixed_entities.py benchmark with activation logging")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                      help="Model to test (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Server port (default: 8000)")
    parser.add_argument("--exp", type=str, default="nonsense_all",
                      help="Experiment name (default: nonsense_all)")
    parser.add_argument("--seed", type=int, default=1,
                      help="Random seed (default: 1)")
    parser.add_argument("--N", type=int, default=100,
                      help="Number of samples (default: 100)")
    parser.add_argument("--output_base_dir", type=str, default="output",
                      help="Base directory for output (default: output)")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/benchmark_activations.lmdb",
                      help="Path to LMDB file for storing activations (default: lmdb_data/benchmark_activations.lmdb)")
    
    args = parser.parse_args()
    
    # Ensure server is running
    if not check_server(args.host, args.port):
        print("Make sure the server is running. You can start it with:")
        print(f"python -m activation_logging.vllm_serve --model {args.model} --host {args.host} --port {args.port}")
        sys.exit(1)
    
    # Create LMDB directory if needed
    lmdb_dir = os.path.dirname(args.lmdb_path)
    if lmdb_dir and not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)
        print(f"Created LMDB directory: {lmdb_dir}")
    
    # Run the benchmark
    exp_hash = hashlib.md5(f"{args.exp}_{args.model}_{args.seed}_{args.N}".encode()).hexdigest()[:8]
    lmdb_path = f"{args.lmdb_path.rstrip('.lmdb')}_{exp_hash}.lmdb"
    
    print(f"Running benchmark with the following configuration:")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.exp}")
    print(f"Seed: {args.seed}")
    print(f"Samples: {args.N}")
    print(f"Output directory: {args.output_base_dir}")
    print(f"LMDB path: {lmdb_path}")
    
    # Set environment variables for the benchmark to use our server
    env = os.environ.copy()
    env["OPENAI_KEY"] = "dummy_key_not_used"  # Required by utils/lm.py
    env["ACTIVATION_LMDB_PATH"] = lmdb_path
    
    # Run the benchmark
    benchmark_cmd = [
        sys.executable,
        "tasks/refusal_test/nonsense_mixed_entities.py",
        "--do_generate_prompt",
        "--do_inference",
        "--do_eval",
        "--exp", args.exp,
        "--tested_model", args.model,
        "--seed", str(args.seed),
        "--N", str(args.N),
        "--output_base_dir", args.output_base_dir,
        "--inference_method", "vllm"
    ]
    
    try:
        print("\nStarting benchmark...")
        subprocess.run(benchmark_cmd, env=env)
        print(f"\nBenchmark completed. Results saved to {args.output_base_dir}")
        print(f"Activations logged to {lmdb_path}")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 