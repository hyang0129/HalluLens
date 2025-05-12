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
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    prompt: str = "Hello, world!",
    lmdb_path: str = "lmdb_data/test_activations.lmdb",
    auth_token: Optional[str] = None,
    server_url: str = "http://localhost:8000/v1/completions",
    max_tokens: int = 5
) -> Dict[str, Any]:
    """
    Run a test of the activation logging system.
    
    Args:
        model: HuggingFace model ID to test
        prompt: Text prompt to send to the model
        lmdb_path: Path to store activations in LMDB
        auth_token: HuggingFace authentication token (for gated models)
        server_url: URL of the inference server
        max_tokens: Maximum number of tokens to generate
        
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
        "max_tokens": max_tokens,
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
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    prompt: str = "Testing default LMDB path functionality",
    max_tokens: int = 5
) -> Dict[str, Any]:
    """
    Test changing the default LMDB path and verify activations are logged properly.
    This test confirms we can change the default path without specifying a path in each request.

    Args:
        host: Server host
        port: Server port
        model: Model to use for testing
        prompt: Prompt to test with
        max_tokens: Maximum number of tokens to generate

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
                "max_tokens": max_tokens
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


def test_overwrite_generation_params(
    host: str = "localhost",
    port: int = 8000,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    prompt: str = "Continue the sentence: 'The sky cracked open and out fell…'",
    max_tokens: int = 20
) -> Dict[str, Any]:
    """
    Test overwrite functionality for temperature and top_p parameters.
    First checks deterministic output with default settings, then tests creative output with high temperature/low top_p.
    Also tests handling of None values for temperature and top_p.
    
    Args:
        host: Server host
        port: Server port
        model: Model to use for testing
        prompt: Prompt to test with
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing test results
    """
    base_url = f"http://{host}:{port}"
    completions_url = f"{base_url}/v1/completions"
    set_temp_url = f"{base_url}/set_overwrite_temperature"
    set_top_p_url = f"{base_url}/set_overwrite_top_p"
    
    results = {
        "success": True,
        "deterministic_results": [],
        "creative_results": [],
        "none_param_results": [],
        "message": "",
        "details": {}
    }
    
    print(f"Testing generation parameter overwrites")
    print(f"Using prompt: '{prompt}'")
    
    # Step 1: Clear any existing overwrites to ensure deterministic results
    try:
        # Reset temperature to None (use default)
        temp_response = requests.post(
            set_temp_url,
            json={"temperature": None}
        )
        if temp_response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to reset temperature overwrite. Status code: {temp_response.status_code}"
            return results
        
        # Reset top_p to None (use default)
        top_p_response = requests.post(
            set_top_p_url,
            json={"top_p": None}
        )
        if top_p_response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to reset top_p overwrite. Status code: {top_p_response.status_code}"
            return results
        
        print("Reset generation parameters to defaults")
    except Exception as e:
        results["success"] = False
        results["message"] = f"Error resetting generation parameters: {e}"
        return results
    
    # Step 2: Send two requests with default settings to verify deterministic output
    print("\nTesting deterministic output with default parameters...")
    deterministic_responses = []
    
    for i in range(2):
        try:
            response = requests.post(
                completions_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,  # Deterministic
                    "top_p": 1.0  # No nucleus sampling restriction
                }
            )
            
            if response.status_code != 200:
                results["success"] = False
                results["message"] = f"Failed to get completions. Status code: {response.status_code}"
                return results
            
            completion_result = response.json()
            generated_text = completion_result["choices"][0]["text"]
            deterministic_responses.append(generated_text)
            print(f"Deterministic response {i+1}: {generated_text}")
            
        except Exception as e:
            results["success"] = False
            results["message"] = f"Error getting deterministic completions: {e}"
            return results
    
    # Check if both deterministic responses are the same (they should be)
    if deterministic_responses[0] == deterministic_responses[1]:
        print("✓ Deterministic test passed: Both responses are identical")
        deterministic_identical = True
    else:
        print("✗ Deterministic test failed: Responses differ despite deterministic settings")
        deterministic_identical = False
    
    results["deterministic_results"] = deterministic_responses
    results["details"]["deterministic_identical"] = deterministic_identical
    
    # Step 3: Test handling of None values for temperature and top_p
    print("\nTesting handling of None values for temperature and top_p...")
    try:
        response = requests.post(
            completions_url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": None,  # Server should use default 0.0
                "top_p": None  # Server should use default 1.0
            }
        )
        
        if response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to get completions with None parameters. Status code: {response.status_code}"
            return results
        
        completion_result = response.json()
        none_param_text = completion_result["choices"][0]["text"]
        results["none_param_results"].append(none_param_text)
        print(f"None parameters response: {none_param_text}")
        
        # Check if the response with None parameters matches the deterministic response
        # (they should match since default values should be used)
        if none_param_text == deterministic_responses[0]:
            print("✓ None parameters test passed: Response matches deterministic output")
            none_params_match = True
        else:
            print("✗ None parameters test failed: Response differs from deterministic output")
            none_params_match = False
        
        results["details"]["none_params_match"] = none_params_match
        
    except Exception as e:
        results["success"] = False
        results["message"] = f"Error testing None parameters: {e}"
        return results
    
    # Step 4: Set overwrite parameters for creative output
    try:
        # Set high temperature (more randomness)
        temp_response = requests.post(
            set_temp_url,
            json={"temperature": 1.5}
        )
        if temp_response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to set temperature overwrite. Status code: {temp_response.status_code}"
            return results
        
        # Set low top_p (more focused on fewer tokens)
        top_p_response = requests.post(
            set_top_p_url,
            json={"top_p": 0.5}
        )
        if top_p_response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to set top_p overwrite. Status code: {top_p_response.status_code}"
            return results
        
        print("\nSet creative parameters: temperature=1.5, top_p=0.5")
    except Exception as e:
        results["success"] = False
        results["message"] = f"Error setting creative parameters: {e}"
        return results
    
    # Step 5: Send request with creative settings (should override the request settings)
    print("\nTesting creative output with overwritten parameters...")
    try:
        # Note: We're still sending temperature=0.0 and top_p=1.0 in the request,
        # but the server should use the overwrite values instead
        response = requests.post(
            completions_url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,  # This should be ignored due to global overwrite
                "top_p": 1.0  # This should be ignored due to global overwrite
            }
        )
        
        if response.status_code != 200:
            results["success"] = False
            results["message"] = f"Failed to get creative completion. Status code: {response.status_code}"
            return results
        
        completion_result = response.json()
        creative_text = completion_result["choices"][0]["text"]
        results["creative_results"].append(creative_text)
        print(f"Creative response: {creative_text}")
        
    except Exception as e:
        results["success"] = False
        results["message"] = f"Error getting creative completion: {e}"
        return results
    
    # Check if creative response differs from deterministic responses
    if creative_text != deterministic_responses[0]:
        print("✓ Creative test passed: Response differs from deterministic output")
        creative_different = True
    else:
        print("✗ Creative test failed: Response is identical to deterministic output")
        creative_different = False
    
    results["details"]["creative_different"] = creative_different
    
    # Step 6: Clean up - reset parameters to default
    try:
        # Reset temperature to None (use default)
        temp_response = requests.post(
            set_temp_url,
            json={"temperature": None}
        )
        
        # Reset top_p to None (use default)
        top_p_response = requests.post(
            set_top_p_url,
            json={"top_p": None}
        )
        
        print("\nReset generation parameters to defaults")
    except Exception as e:
        print(f"Warning: Error resetting parameters: {e}")
    
    # Final success determination
    if deterministic_identical and creative_different and none_params_match:
        results["success"] = True
        results["message"] = "Generation parameter overwrite test successful: deterministic output consistent, None parameters handled correctly, and creative output differs."
    else:
        results["success"] = False
        if not deterministic_identical:
            results["message"] = "Generation parameter test failed: deterministic output inconsistent."
        elif not none_params_match:
            results["message"] = "Generation parameter test failed: None parameters not handled correctly."
        elif not creative_different:
            results["message"] = "Generation parameter test failed: creative output did not differ from deterministic output."
    
    print(f"\nTest result: {results['message']}")
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the activation logging feature")
    
    # Add subparsers for different test types
    subparsers = parser.add_subparsers(dest="test_type", help="Type of test to run")
    
    # Basic test
    basic_parser = subparsers.add_parser("basic", help="Run basic activation logging test")
    basic_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help="Model to test with (default: meta-llama/Llama-3.1-8B-Instruct)")
    basic_parser.add_argument("--prompt", type=str, default="Hello, world!",
                      help="Prompt to test with (default: 'Hello, world!')")
    basic_parser.add_argument("--lmdb_path", type=str, default="lmdb_data/test_activations.lmdb",
                      help="Path to LMDB for storing activations (default: lmdb_data/test_activations.lmdb)")
    basic_parser.add_argument("--auth_token", type=str, default=None,
                      help="HuggingFace authentication token for accessing gated models")
    basic_parser.add_argument("--server_url", type=str, default="http://localhost:8000/v1/completions",
                      help="URL of the inference server (default: http://localhost:8000/v1/completions)")
    basic_parser.add_argument("--max_tokens", type=int, default=5,
                      help="Maximum number of tokens to generate (default: 5)")
    
    # Default path change test
    path_parser = subparsers.add_parser("path", help="Test changing default LMDB path")
    path_parser.add_argument("--host", type=str, default="localhost",
                     help="Server host (default: localhost)")
    path_parser.add_argument("--port", type=int, default=8000,
                     help="Server port (default: 8000)")
    path_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                     help="Model to test with (default: meta-llama/Llama-3.1-8B-Instruct)")
    path_parser.add_argument("--prompt", type=str, default="Testing default LMDB path functionality",
                     help="Prompt to test with (default: 'Testing default LMDB path functionality')")
    path_parser.add_argument("--max_tokens", type=int, default=5,
                     help="Maximum number of tokens to generate (default: 5)")
    
    # Generation parameters test
    params_parser = subparsers.add_parser("params", help="Test temperature and top_p overwrites")
    params_parser.add_argument("--host", type=str, default="localhost",
                     help="Server host (default: localhost)")
    params_parser.add_argument("--port", type=int, default=8000,
                     help="Server port (default: 8000)")
    params_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                     help="Model to test with (default: meta-llama/Llama-3.1-8B-Instruct)")
    params_parser.add_argument("--prompt", type=str, 
                     default="Continue the sentence: 'The sky cracked open and out fell…'",
                     help="Prompt to test with")
    params_parser.add_argument("--max_tokens", type=int, default=20,
                     help="Maximum number of tokens to generate (default: 20)")
    
    return parser.parse_args()


def main():
    """Main entry point when script is run from command line"""
    args = parse_args()
    
    # Default to basic test if not specified
    if not args.test_type or args.test_type == "basic":
        result = run_test(
            model=getattr(args, "model", "meta-llama/Llama-3.1-8B-Instruct"),
            prompt=getattr(args, "prompt", "Hello, world!"),
            lmdb_path=getattr(args, "lmdb_path", "lmdb_data/test_activations.lmdb"),
            auth_token=getattr(args, "auth_token", None),
            server_url=getattr(args, "server_url", "http://localhost:8000/v1/completions"),
            max_tokens=args.max_tokens
        )
    elif args.test_type == "path":
        result = test_default_lmdb_path_change(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    elif args.test_type == "params":
        result = test_overwrite_generation_params(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)
    
    # Exit with appropriate status code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main() 