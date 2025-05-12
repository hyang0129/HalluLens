"""
Test script for the activation logging feature.
Sends a request to the local inference server and checks LMDB for the logged activations.

Can be run as a command-line script or imported and called programmatically.
"""
import os
import hashlib
import requests
import sys
import subprocess
import argparse
from typing import Optional, Dict, Any, Union


def run_test(
    model: str = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    prompt: str = "Hello, world!",
    lmdb_path: str = "lmdb_data/test_activations.lmdb",
    auth_token: Optional[str] = None,
    server_url: str = "http://localhost:8000/v1/completions"
) -> Dict[str, Any]:
    """
    Run a test of the activation logging system.
    
    Args:
        model: HuggingFace model ID to test
        prompt: Text prompt to send to the model
        lmdb_path: Path to store activations in LMDB
        auth_token: HuggingFace authentication token (for gated models)
        server_url: URL of the inference server
        
    Returns:
        Dictionary containing test results with keys:
        - success: Whether the test was successful
        - response: Server response
        - prompt_hash: Hash of the prompt (used as LMDB key)
        - lmdb_result: Result of checking LMDB
        - message: Summary message
    """
    # Create the LMDB directory if it doesn't exist
    lmdb_dir = os.path.dirname(lmdb_path)
    if lmdb_dir and not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)
        print(f"Created LMDB directory: {lmdb_dir}")
    
    # Calculate prompt hash (used as LMDB key)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    
    # Prepare request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 5,
        "lmdb_path": lmdb_path
    }
    
    # Add auth token if provided
    if auth_token:
        payload["auth_token"] = auth_token
    
    # Send request and collect response
    print(f"Sending request to {server_url} ...")
    print(f"Testing with model: {model}")
    print(f"Using prompt: '{prompt}'")
    
    try:
        response = requests.post(server_url, json=payload)
        response_data = response.json()
        print(f"Status: {response.status_code}")
        print("Response:", response_data)
    except Exception as e:
        return {
            "success": False,
            "prompt_hash": prompt_hash,
            "error": str(e),
            "message": f"Error connecting to server: {e}"
        }
    
    # Check LMDB for the prompt hash
    print(f"\nChecking LMDB for key: {prompt_hash}")
    result = subprocess.run(
        [sys.executable, "activation_logging/test_check_lmdb.py", prompt_hash, lmdb_path], 
        capture_output=True, 
        text=True
    )
    
    print(result.stdout)
    
    # Determine test success
    success = result.returncode == 0
    message = (
        "Test completed successfully. Activations are being properly logged." 
        if success 
        else "Test failed: No activation found for the prompt."
    )
    print(message)
    
    return {
        "success": success,
        "response": response_data,
        "prompt_hash": prompt_hash,
        "lmdb_result": result.stdout,
        "message": message
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the activation logging feature")
    parser.add_argument("--model", type=str, default="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                      help="Model to test with (default: NousResearch/Nous-Hermes-2-Mistral-7B-DPO)")
    parser.add_argument("--prompt", type=str, default="Hello, world!",
                      help="Prompt to test with (default: 'Hello, world!')")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/test_activations.lmdb",
                      help="Path to LMDB for storing activations (default: lmdb_data/test_activations.lmdb)")
    parser.add_argument("--auth_token", type=str, default=None,
                      help="HuggingFace authentication token for accessing gated models")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/v1/completions",
                      help="URL of the inference server (default: http://localhost:8000/v1/completions)")
    
    return parser.parse_args()


def main():
    """Main entry point when script is run from command line"""
    args = parse_args()
    
    # Run the test with command line arguments
    result = run_test(
        model=args.model,
        prompt=args.prompt,
        lmdb_path=args.lmdb_path,
        auth_token=args.auth_token,
        server_url=args.server_url
    )
    
    # Exit with appropriate status code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main() 