"""
Test script for the activation logging feature.
Sends a request to the local inference server and checks LMDB for the logged activations.
"""
import os
import hashlib
import requests
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test the activation logging feature")
    parser.add_argument("--model", type=str, default="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                      help="Model to test with (default: NousResearch/Nous-Hermes-2-Mistral-7B-DPO)")
    parser.add_argument("--prompt", type=str, default="Hello, world!",
                      help="Prompt to test with (default: 'Hello, world!')")
    parser.add_argument("--lmdb_path", type=str, default="lmdb_data/test_activations.lmdb",
                      help="Path to LMDB for storing activations (default: lmdb_data/test_activations.lmdb)")
    parser.add_argument("--auth_token", type=str, default=None,
                      help="HuggingFace authentication token for accessing gated models")
    
    args = parser.parse_args()
    
    # Specify LMDB path directly in the request payload
    lmdb_path = args.lmdb_path
    prompt = args.prompt
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Send OpenAI API request to FastAPI server
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 5,
        "lmdb_path": lmdb_path
    }
    
    # Add auth token if provided
    if args.auth_token:
        payload["auth_token"] = args.auth_token
    
    print(f"Sending request to {url} ...")
    print(f"Testing with model: {args.model}")
    print(f"Using prompt: '{prompt}'")
    
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print("Response:", response.json())

    # Check LMDB for the prompt hash
    print(f"\nChecking LMDB for key: {prompt_hash}")
    result = subprocess.run([sys.executable, "activation_logging/test_check_lmdb.py", prompt_hash, lmdb_path], 
                           capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("Test completed. If you see activations printed above, logging works!")
    else:
        print("Test failed: No activation found for the prompt.")

if __name__ == "__main__":
    main() 