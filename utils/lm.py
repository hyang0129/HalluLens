# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import openai
from openai import APITimeoutError
from loguru import logger
import hashlib
import json

class SampleSkippedException(Exception):
    """Exception raised when a sample is skipped due to repeated failures."""
    def __init__(self, request_id, reason, attempts):
        self.request_id = request_id
        self.reason = reason
        self.attempts = attempts
        super().__init__(f"Sample {request_id} skipped after {attempts} attempts: {reason}")

# Global tracking of skipped samples
skipped_samples = set()
skip_stats = {"total_skipped": 0, "timeout_skipped": 0, "error_skipped": 0}

'''
NOTE:
    Available functions:
        - call_vllm_api: using vllm self-served models
        - openai_generate: using openai models
'''

def setup_client_logging():
    """Setup client logging to match server log format and location."""
    # Check if we should log to the same location as server
    server_log_file = os.environ.get("SERVER_LOG_FILE")
    if server_log_file:
        # Use the same log file as server but with client prefix
        client_log_file = server_log_file.replace(".log", "_client.log")
        logger.add(client_log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
        logger.info(f"Client logging configured to: {client_log_file}")
    else:
        # Default client log location
        client_log_file = "goodwiki_json/client.log"
        logger.add(client_log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
        logger.info(f"Client logging configured to: {client_log_file}")

def generate_request_id(prompt):
    """Generate a unique request ID based on prompt content."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]

def log_request_payload(request_id, prompt, model, max_tokens, temperature, top_p, reason="timeout"):
    """Log the full request payload for debugging purposes."""
    payload = {
        "request_id": request_id,
        "model": model,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reason": reason,
        "timestamp": time.time()
    }

    logger.error(f"[CLIENT {request_id}] Request payload logged due to {reason}:")
    logger.error(f"[CLIENT {request_id}] Model: {model}")
    logger.error(f"[CLIENT {request_id}] Prompt length: {len(prompt)} chars")
    logger.error(f"[CLIENT {request_id}] Prompt preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    logger.error(f"[CLIENT {request_id}] Max tokens: {max_tokens}, Temperature: {temperature}, Top-p: {top_p}")

    # Also save full payload to separate file for detailed analysis
    payload_file = f"goodwiki_json/failed_request_{request_id}.json"
    try:
        with open(payload_file, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.error(f"[CLIENT {request_id}] Full payload saved to: {payload_file}")
    except Exception as e:
        logger.error(f"[CLIENT {request_id}] Failed to save payload file: {e}")

def track_skipped_sample(request_id, reason, attempts):
    """Track a skipped sample and update statistics."""
    global skipped_samples, skip_stats

    if request_id not in skipped_samples:
        skipped_samples.add(request_id)
        skip_stats["total_skipped"] += 1

        if "timeout" in reason.lower():
            skip_stats["timeout_skipped"] += 1
        else:
            skip_stats["error_skipped"] += 1

        logger.warning(f"[SKIP TRACKER] Sample {request_id} skipped due to {reason} after {attempts} attempts")
        logger.warning(f"[SKIP TRACKER] Total skipped: {skip_stats['total_skipped']} (timeouts: {skip_stats['timeout_skipped']}, errors: {skip_stats['error_skipped']})")

        # Save skipped samples list
        try:
            skipped_file = "goodwiki_json/skipped_samples.json"
            with open(skipped_file, 'w') as f:
                json.dump({
                    "skipped_samples": list(skipped_samples),
                    "statistics": skip_stats,
                    "last_updated": time.time()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[SKIP TRACKER] Failed to save skipped samples file: {e}")

def get_skip_statistics():
    """Get current skip statistics."""
    return skip_stats.copy()

########################################################################################################
def custom_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):

    raise NotImplementedError()

def generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0, max_retries=3, base_delay=1.0):

    # TODO: You need to use your own inference method
    # return custom_api(prompt, model, temperature, top_p, max_tokens, port)
    return call_vllm_api(prompt, model, temperature, top_p, max_tokens, port, i, max_retries, base_delay)

CUSTOM_SERVER = "0.0.0.0" # you may need to change the port

model_map = {   'meta-llama/Llama-3.1-405B-Instruct-FP8': {'name': 'llama3.1_405B',
                                                            'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
                'meta-llama/Llama-3.3-70B-Instruct': {'name': 'llama3.3_70B',
                                                    'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
                'meta-llama/Llama-3.1-70B-Instruct': {'name': 'llama3.1_70B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                'meta-llama/Llama-3.1-8B-Instruct': {'name': 'llama3.1_8B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                'mistralai/Mistral-7B-Instruct-v0.2': {'name': 'mistral7B',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                "mistralai/Mistral-Nemo-Instruct-2407": {'name': 'Mistral-Nemo-Instruct-2407',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },
                "Llama-3.3-70B-Instruct-IQ3_M.gguf": {'name': 'Llama-3.3-70B-Instruct-IQ3_M.gguf',
                                                        'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"],
                                                    },

                                                    
            }
########################################################################################################

def call_vllm_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0, max_retries=3, base_delay=1.0):
    """
    Call vLLM API with retry mechanism for timeout errors.

    Args:
        prompt: The input prompt
        model: Model name
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        port: API port/URL
        i: Server index for model_map
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)

    Returns:
        Generated text content

    Raises:
        APITimeoutError: If all retry attempts fail due to timeout
        Exception: For other non-timeout errors
    """
    # Setup client logging on first call
    setup_client_logging()

    # Generate unique request ID for tracking
    request_id = generate_request_id(prompt)

    if port == None:
        port = model_map[model]["server_urls"][i]

    logger.info(f"[CLIENT {request_id}] Starting API call - Model: {model}, Prompt length: {len(prompt)} chars")

    client = openai.OpenAI(
        base_url=f"{port}",
        api_key="NOT A REAL KEY",
        timeout=300.0  # 5 minute timeout instead of default 30 minutes
    )

    for attempt in range(max_retries + 1):  # +1 to include the initial attempt
        try:
            import time
            start_time = time.time()
            logger.info(f"[CLIENT {request_id}] Sending request to {port} (attempt {attempt + 1}/{max_retries + 1})")

            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant. Please answer questions directly and clearly without adding any extra punctuation or filler characters and without multiple choice options."},
                    {"role": "user","content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            request_time = time.time() - start_time
            response_content = chat_completion.choices[0].message.content
            logger.info(f"[CLIENT {request_id}] Request completed successfully in {request_time:.2f}s - Response length: {len(response_content)} chars")
            return response_content

        except APITimeoutError as e:
            request_time = time.time() - start_time
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[CLIENT {request_id}] API timeout on attempt {attempt + 1}/{max_retries + 1} after {request_time:.1f}s. Retrying in {delay:.1f} seconds...")

                # Log payload on first timeout to help debug
                if attempt == 0:
                    log_request_payload(request_id, prompt, model, max_tokens, temperature, top_p, "first_timeout")

                time.sleep(delay)
            else:
                logger.error(f"[CLIENT {request_id}] API timeout after {max_retries + 1} attempts (final attempt took {request_time:.1f}s). Skipping sample.")

                # Log full payload on final failure
                log_request_payload(request_id, prompt, model, max_tokens, temperature, top_p, "final_timeout")

                # Track the skipped sample
                track_skipped_sample(request_id, "timeout", max_retries + 1)

                # Return a placeholder response instead of raising exception
                logger.warning(f"[CLIENT {request_id}] Returning placeholder response to continue with next sample")
                return f"[TIMEOUT_SKIPPED] Request {request_id} timed out after {max_retries + 1} attempts. Total skipped: {skip_stats['total_skipped']}"
        except Exception as e:
            request_time = time.time() - start_time
            # For non-timeout errors, don't retry but skip the sample
            logger.error(f"[CLIENT {request_id}] Non-timeout error occurred after {request_time:.1f}s: {type(e).__name__}: {e}")

            # Log payload for non-timeout errors too
            error_reason = f"error_{type(e).__name__}"
            log_request_payload(request_id, prompt, model, max_tokens, temperature, top_p, error_reason)

            # Track the skipped sample
            track_skipped_sample(request_id, error_reason, 1)

            # Return placeholder instead of raising exception
            logger.warning(f"[CLIENT {request_id}] Returning placeholder response due to error: {type(e).__name__}")
            return f"[ERROR_SKIPPED] Request {request_id} failed with {type(e).__name__}: {str(e)[:100]}. Total skipped: {skip_stats['total_skipped']}"

def openai_generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    # Create a client object
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_KEY"],
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content
