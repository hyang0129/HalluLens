import os
import hashlib
import requests
import sys
import subprocess

def main():
    # Specify LMDB path directly in the request payload
    lmdb_path = "lmdb_data/test_activations.lmdb"
    prompt = "Hello, world!"
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Send OpenAI API request to FastAPI server
    url = "http://localhost:8000/v1/completions"
    payload = {
        "model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",  # Use public default model for test
        "prompt": prompt,
        "max_tokens": 5,
        "lmdb_path": lmdb_path  # Specify custom LMDB path for logging
    }
    print(f"Sending request to {url} ...")
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print("Response:", response.json())

    # Check LMDB for the prompt hash
    print(f"\nChecking LMDB for key: {prompt_hash}")
    result = subprocess.run([sys.executable, "hallu_llama_vllm/test_check_lmdb.py", prompt_hash], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("Test completed. If you see activations printed above, logging works!")
    else:
        print("Test failed: No activation found for the prompt.")

if __name__ == "__main__":
    main()
