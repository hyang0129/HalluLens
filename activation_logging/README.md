# Activation Logging for LLM Hallucination Analysis

This project provides a vLLM-based inference server with activation logging capabilities, capturing last-layer per-token activations, prompts, responses, and evaluation results to LMDB. Designed specifically for evaluating the nature of activations when an LLM is hallucinating vs not hallucinating.

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
  1. (Optional) Set the LMDB path for experiment separation:
     ```bash
     # On Linux/macOS
     export ACTIVATION_LMDB_PATH=lmdb_data/exp1_activations.lmdb
     # On Windows
     set ACTIVATION_LMDB_PATH=lmdb_data/exp1_activations.lmdb
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

To run the nonsense_mixed_entities.py benchmark with activation logging:

```bash
python tasks/refusal_test/nonsense_mixed_entities.py --do_inference --do_eval --tested_model mistralai/Mistral-7B-Instruct-v0.2
```

## LMDB Output

- Activations, prompts, and responses are stored in LMDB under `lmdb_data/`.
- Each entry key is a SHA256 hash of the prompt.
- The data structure stored includes:
  - `prompt`: The original input prompt
  - `response`: The model's response text
  - `activations`: NumPy array of activation values
  - `model`: The model name/ID used for generation

## Testing the Setup

To verify that activation logging is working correctly:

```bash
# For open models
python activation_logging/test_lmdb_logging.py

# For gated models requiring authentication
python activation_logging/test_lmdb_logging.py --model mistralai/Mistral-7B-Instruct-v0.2 --auth_token your_huggingface_token
```

This will send a test request to the server and check if activations were properly logged to LMDB.

## API Client Usage

When using the API directly, you can include the HuggingFace authentication token in your request:

```python
import requests

url = "http://localhost:8000/v1/completions"
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "auth_token": "your_huggingface_token"  # Include this for gated models
}

response = requests.post(url, json=payload)
print(response.json())
```

## Notes
- Python 3.10+ recommended
- Always run the server before attempting to use the inference utils
- Do not commit large LMDB files or model weights to version control
- Never commit your HuggingFace authentication token to version control 