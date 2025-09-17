# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import openai
from openai import APITimeoutError
from loguru import logger

'''
NOTE: 
    Available functions:
        - call_vllm_api: using vllm self-served models
        - openai_generate: using openai models
'''
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
    if port == None:
        port = model_map[model]["server_urls"][i]

    client = openai.OpenAI(
        base_url=f"{port}",
        api_key="NOT A REAL KEY",
    )

    for attempt in range(max_retries + 1):  # +1 to include the initial attempt
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant. Please answer questions directly and clearly without adding any extra punctuation or filler characters and without multiple choice options."},
                    {"role": "user","content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return chat_completion.choices[0].message.content

        except APITimeoutError as e:
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API timeout on attempt {attempt + 1}/{max_retries + 1}. Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"API timeout after {max_retries + 1} attempts. Giving up.")
                raise e
        except Exception as e:
            # For non-timeout errors, don't retry
            logger.error(f"Non-timeout error occurred: {type(e).__name__}: {e}")
            raise e

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
