# Activation Logging for LLM Hallucination Analysis

This project provides a vLLM-based inference server with activation logging capabilities, capturing last-layer per-token activations, prompts, responses, and evaluation results to Zarr. Designed specifically for evaluating the nature of activations when an LLM is hallucinating vs not hallucinating.

## Backend Support Policy

- ✅ Supported: Zarr activation storage (`.zarr`)
- ❌ Deprecated and no longer supported: LMDB, JSON directory logging, WebDataset runtime backend

The runtime now expects `activations_path` to end with `.zarr` and `logger_type` to be `zarr`.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r activation_logging/requirements.txt
   ```

3. (Optional) For gated HuggingFace models that require approval:
   - Create a HuggingFace account and get an authentication token from https://huggingface.co/settings/tokens
   - Accept the model license agreement on the HuggingFace website for the model you want to use
   - Set your HuggingFace token as an environment variable:
     ```bash
     # On Linux/macOS
     export HF_TOKEN=your_huggingface_token
     # On Windows
     set HF_TOKEN=your_huggingface_token
     ```
   - Or you can pass the token directly to the server (see below)

## Usage

### Starting the Server

- Start the FastAPI OpenAI-compatible server with activation logging:
    1. (Optional) Set the Zarr path for experiment separation:
     ```bash
     # On Linux/macOS
         export ACTIVATION_STORAGE_PATH=zarr_data/exp1_activations.zarr
     # On Windows
         set ACTIVATION_STORAGE_PATH=zarr_data/exp1_activations.zarr
     ```
  2. Launch the server using vLLM serve:
     ```bash
     vllm serve --model mistralai/Mistral-7B-Instruct-v0.2 --host 0.0.0.0 --port 8000
     ```
     
     Or using our wrapper script:
     ```bash
     # For open models
     python -m activation_logging.vllm_serve --model mistralai/Mistral-7B-Instruct-v0.2
     
     # For gated models requiring authentication
     python -m activation_logging.vllm_serve --model mistralai/Mistral-7B-Instruct-v0.2 --auth_token your_huggingface_token
     ```
     
     Or using uvicorn directly:
     ```bash
     uvicorn activation_logging.server:app --host 0.0.0.0 --port 8000
     ```
     
  - This will launch a server at `http://localhost:8000/v1/completions` (OpenAI API protocol).
  - All activation logging will be handled transparently for each request.

### Running the Benchmark

This repo’s primary use case is **hallucination detection on fact-seeking QA tasks** (where correctness can be evaluated against references). Use one of the QA tasks below to generate (prompt, response) pairs while the server logs activations.

**PreciseWikiQA (recommended working example):**

```bash
python -m tasks.shortform.precise_wikiqa \
    --do_inference \
    --do_eval \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --wiki_src goodwiki \
    --mode dynamic \
    --inference_method vllm \
    --N 100
```

**TriviaQA (also supported):**

```bash
python -m tasks.triviaqa.triviaqa \
    --do_inference \
    --do_eval \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --inference_method vllm \
    --N 100
```

## Zarr Output

- Activations, prompts, and responses are stored in Zarr under `zarr_data/`.
- Each entry key is a SHA256 hash of the prompt.
- The data structure stored includes:
  - `prompt`: The original input prompt
  - `response`: The model's response text
  - `activations`: NumPy array of activation values
  - `model`: The model name/ID used for generation

## Testing the Setup

### Command Line Testing

Use Zarr-based benchmark/testing flows. Legacy `test_lmdb_logging.py` is deprecated and not supported.

### Programmatic Testing

You can also use the testing module programmatically in your own Python scripts:

```python
from activation_logging.test_lmdb_logging import run_test, test_overwrite_generation_params

# Basic test with default parameters
result = run_test()
print(f"Test success: {result['success']}")

# Test with custom parameters
custom_result = run_test(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Explain the concept of hallucination in LLMs.",
    lmdb_path="lmdb_data/my_custom_test.lmdb",
    auth_token="your_huggingface_token",
    server_url="http://localhost:8000/v1/completions",
    max_tokens=20
)

if custom_result["success"]:
    print("Activation logging is working correctly!")
else:
    print(f"Test failed: {custom_result['message']}")

# Test overwriting generation parameters (temperature and top_p)
params_result = test_overwrite_generation_params()
if params_result["success"]:
    print("Generation parameter overwrites are working correctly!")
    print(f"Deterministic responses: {params_result['deterministic_results']}")
    print(f"Creative response: {params_result['creative_results'][0]}")
else:
    print(f"Test failed: {params_result['message']}")
```

## API Client Usage

When using the API directly, you can include the HuggingFace authentication token in your request:

```python
import requests

# 1. Standard completion request with custom Zarr path
url = "http://localhost:8000/v1/completions"
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "auth_token": "your_huggingface_token",  # Include this for gated models
    "activations_path": "zarr_data/custom_path.zarr",  # Optional: specify custom Zarr path
    "logger_type": "zarr"
}

response = requests.post(url, json=payload)
print(response.json())

# 2. Set overwrite activations path for all requests
# (legacy endpoint name retained: set_overwrite_lmdb_path; value must be .zarr)
url = "http://localhost:8000/set_overwrite_lmdb_path"
payload = {
    "lmdb_path": "zarr_data/overwrite_path.zarr"
}

response = requests.post(url, json=payload)
print(response.json())

# 3. Now requests will use the overwrite path regardless of what's specified in the request
url = "http://localhost:8000/v1/completions"
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Using the overwrite path",
    "max_tokens": 100,
    "activations_path": "zarr_data/this_will_be_ignored.zarr"  # This will be ignored
}

response = requests.post(url, json=payload)
print(response.json())

# 4. Set overwrite temperature for all requests
url = "http://localhost:8000/set_overwrite_temperature"
payload = {
    "temperature": 1.2  # Higher temperature means more randomness
}
response = requests.post(url, json=payload)
print(response.json())

# 5. Set overwrite top_p for all requests
url = "http://localhost:8000/set_overwrite_top_p"
payload = {
    "top_p": 0.7  # Lower top_p focuses on more likely tokens
}
response = requests.post(url, json=payload)
print(response.json())

# 6. Send a request with different temperature/top_p values, which will be ignored
# due to the global overwrites
url = "http://localhost:8000/v1/completions"
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The sky cracked open and out fell...",
    "max_tokens": 100,
    "temperature": 0.5,  # This will be ignored, using 1.2 instead
    "top_p": 0.9  # This will be ignored, using 0.7 instead
}
response = requests.post(url, json=payload)
print(response.json())

# 7. Disable overwrites to go back to per-request values
url = "http://localhost:8000/set_overwrite_temperature"
payload = {
    "temperature": null  # null/None disables the overwrite
}
response = requests.post(url, json=payload)

url = "http://localhost:8000/set_overwrite_top_p"
payload = {
    "top_p": null  # null/None disables the overwrite
}
response = requests.post(url, json=payload)

url = "http://localhost:8000/set_overwrite_lmdb_path"
payload = {
    "lmdb_path": null  # null/None disables the overwrite
}
response = requests.post(url, json=payload)

# 8. Send a request with None values for temperature and top_p
# The server will use defaults: temperature=0.0 (deterministic) and top_p=1.0 (no filtering)
url = "http://localhost:8000/v1/completions"
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Explain what hallucination means in the context of LLMs",
    "max_tokens": 100,
    "temperature": None,  # Server will use default 0.0
    "top_p": None  # Server will use default 1.0
}
response = requests.post(url, json=payload)
print(response.json())
```

## Notes
- Python 3.10+ recommended
- Always run the server before attempting to use the inference utils
- Do not commit large activation stores or model weights to version control
- Never commit your HuggingFace authentication token to version control
- When temperature is set to None, a default value of 0.0 (deterministic) is used
- When top_p is set to None, a default value of 1.0 (no filtering) is used 