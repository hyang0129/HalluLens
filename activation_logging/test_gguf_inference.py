#!/usr/bin/env python
"""
Test script for GGUF model inference on the activation logging server.
This script sends requests to the API server to test inference with a GGUF model.
"""

import os
import requests
import json
import argparse
from typing import Dict, Any, Optional
import time


def test_completion(
    server_url: str,
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> Dict[str, Any]:
    """
    Test text completion with a GGUF model.
    
    Args:
        server_url: URL of the server
        model_path: Path to the GGUF model file
        prompt: The prompt to generate text from
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Response JSON from the server
    """
    endpoint = f"{server_url.rstrip('/')}/v1/completions"
    
    payload = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {endpoint}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    response = requests.post(endpoint, headers=headers, json=payload)
    end_time = time.time()
    
    print(f"Request took {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def test_chat_completion(
    server_url: str,
    model_path: str,
    messages: list,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> Dict[str, Any]:
    """
    Test chat completion with a GGUF model.
    
    Args:
        server_url: URL of the server
        model_path: Path to the GGUF model file
        messages: List of message objects with role and content
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Response JSON from the server
    """
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    
    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {endpoint}")
    print(f"Model: {model_path}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    
    start_time = time.time()
    response = requests.post(endpoint, headers=headers, json=payload)
    end_time = time.time()
    
    print(f"Request took {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {}


def check_server_health(server_url: str) -> bool:
    """
    Check if the server is running and healthy.
    
    Args:
        server_url: URL of the server
        
    Returns:
        True if server is healthy, False otherwise
    """
    health_endpoint = f"{server_url.rstrip('/')}/health"
    
    try:
        response = requests.get(health_endpoint)
        if response.status_code == 200:
            print("Server is healthy!")
            return True
        else:
            print(f"Server returned status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False


def run_test(
    server_url: str = "http://localhost:8000",
    model_path: str = "Llama-3.3-70B-Instruct-IQ3_M.gguf",
    test_type: str = "chat",
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> Dict[str, Any]:
    """
    Run a test of the GGUF model inference.
    This function can be imported and called from other Python scripts.
    
    Args:
        server_url: URL of the server
        model_path: Path to the GGUF model file
        test_type: Type of test to run ("completion" or "chat")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Response JSON from the server or empty dict if error
    """
    # Check if server is running
    if not check_server_health(server_url):
        print("Server is not running or not healthy. Exiting.")
        return {}
    
    if test_type == "completion":
        # Test text completion
        prompt = "Explain the concept of machine learning in simple terms:"
        result = test_completion(
            server_url=server_url,
            model_path=model_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        if result:
            print("\nResponse:")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Model: {result.get('model', 'N/A')}")
            print(f"Generated text: {result.get('choices', [{}])[0].get('text', '')}")
        
        return result
    
    else:
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What are the main differences between supervised and unsupervised learning?"}
        ]
        
        result = test_chat_completion(
            server_url=server_url,
            model_path=model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        if result:
            print("\nResponse:")
            print(f"ID: {result.get('id', 'N/A')}")
            print(f"Model: {result.get('model', 'N/A')}")
            if result.get('choices'):
                message = result['choices'][0].get('message', {})
                print(f"Role: {message.get('role', 'N/A')}")
                print(f"Content: {message.get('content', '')}")
        
        return result


def main():
    """Parse command line arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test GGUF model inference on the activation logging server")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="URL of the server")
    parser.add_argument("--model-path", type=str, default="Llama-3.3-70B-Instruct-IQ3_M.gguf", help="Path to the GGUF model file")
    parser.add_argument("--test-type", type=str, choices=["completion", "chat"], default="chat", help="Type of test to run")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    run_test(
        server_url=args.server_url,
        model_path=args.model_path,
        test_type=args.test_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main() 