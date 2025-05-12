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
import time
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


def test_default_lmdb_path_change(
    host: str = "localhost",
    port: int = 8000,
    model: str = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    prompt: str = "Testing default LMDB path functionality"
) -> Dict[str, Any]:
    """
    Test changing the default LMDB path and verify activations are logged properly.
    This test confirms we can change the default path without specifying a path in each request.

    Args:
        host: Server host
        port: Server port
        model: Model to use for testing
        prompt: Prompt to test with

    Returns:
        Dictionary containing test results
    """
    base_url = f"http://{host}:{port}"
    default_path_url = f"{base_url}/set_default_lmdb_path"
    completions_url = f"{base_url}/v1/completions"
    
    # Create a unique test LMDB path
    test_lmdb_path = f"lmdb_data/test_default_path_{int(time.time())}.lmdb"
    
    print(f"Testing default LMDB path change functionality")
    print(f"Setting default LMDB path to: {test_lmdb_path}")
    
    # 1. Change the default LMDB path
    try:
        response = requests.post(
            default_path_url, 
            json={"lmdb_path": test_lmdb_path}
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "message": f"Failed to change default LMDB path. Status code: {response.status_code}",
                "response": response.json() if response.headers.get("content-type") == "application/json" else None
            }
        
        path_change_result = response.json()
        print(f"Default LMDB path change result: {path_change_result}")
        
        if not path_change_result.get("success", False):
            return {
                "success": False,
                "message": f"Server reported failure changing default LMDB path: {path_change_result.get('message')}",
                "response": path_change_result
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error changing default LMDB path: {e}",
            "error": str(e)
        }
    
    # 2. Now send a completion request WITHOUT specifying an LMDB path
    # It should use the default path we just set
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    
    try:
        # Send request without lmdb_path in payload
        response = requests.post(
            completions_url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 5
            }
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "message": f"Failed to get completions. Status code: {response.status_code}",
                "response": response.json() if response.headers.get("content-type") == "application/json" else None
            }
        
        completions_result = response.json()
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting completions: {e}",
            "error": str(e)
        }
    
    # 3. Check that the activations were logged to the new default path
    print(f"\nChecking for activations in new default LMDB path: {test_lmdb_path}")
    result = subprocess.run(
        [sys.executable, "activation_logging/test_check_lmdb.py", prompt_hash, test_lmdb_path], 
        capture_output=True, 
        text=True
    )
    
    print(result.stdout)
    
    # Determine test success
    success = result.returncode == 0
    message = (
        "Default LMDB path change test successful. Activations are being properly logged to the new path." 
        if success 
        else "Default LMDB path change test failed: No activation found in the new path."
    )
    print(message)
    
    return {
        "success": success,
        "path_change_response": path_change_result,
        "completions_response": completions_result,
        "prompt_hash": prompt_hash,
        "lmdb_result": result.stdout,
        "message": message
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the activation logging feature")
    
    # Add subparsers for different test types
    subparsers = parser.add_subparsers(dest="test_type", help="Type of test to run")
    
    # Basic test
    basic_parser = subparsers.add_parser("basic", help="Run basic activation logging test")
    basic_parser.add_argument("--model", type=str, default="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                      help="Model to test with (default: NousResearch/Nous-Hermes-2-Mistral-7B-DPO)")
    basic_parser.add_argument("--prompt", type=str, default="Hello, world!",
                      help="Prompt to test with (default: 'Hello, world!')")
    basic_parser.add_argument("--lmdb_path", type=str, default="lmdb_data/test_activations.lmdb",
                      help="Path to LMDB for storing activations (default: lmdb_data/test_activations.lmdb)")
    basic_parser.add_argument("--auth_token", type=str, default=None,
                      help="HuggingFace authentication token for accessing gated models")
    basic_parser.add_argument("--server_url", type=str, default="http://localhost:8000/v1/completions",
                      help="URL of the inference server (default: http://localhost:8000/v1/completions)")
    
    # Default path change test
    path_parser = subparsers.add_parser("path", help="Test changing default LMDB path")
    path_parser.add_argument("--host", type=str, default="localhost",
                     help="Server host (default: localhost)")
    path_parser.add_argument("--port", type=int, default=8000,
                     help="Server port (default: 8000)")
    path_parser.add_argument("--model", type=str, default="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                     help="Model to test with (default: NousResearch/Nous-Hermes-2-Mistral-7B-DPO)")
    path_parser.add_argument("--prompt", type=str, default="Testing default LMDB path functionality",
                     help="Prompt to test with (default: 'Testing default LMDB path functionality')")
    
    return parser.parse_args()


def main():
    """Main entry point when script is run from command line"""
    args = parse_args()
    
    # Default to basic test if not specified
    if not args.test_type or args.test_type == "basic":
        result = run_test(
            model=getattr(args, "model", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"),
            prompt=getattr(args, "prompt", "Hello, world!"),
            lmdb_path=getattr(args, "lmdb_path", "lmdb_data/test_activations.lmdb"),
            auth_token=getattr(args, "auth_token", None),
            server_url=getattr(args, "server_url", "http://localhost:8000/v1/completions")
        )
    elif args.test_type == "path":
        result = test_default_lmdb_path_change(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt=args.prompt
        )
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)
    
    # Exit with appropriate status code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main() 