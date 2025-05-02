# vLLM Llama 3.3 Instruct Activation Logger

This project provides a vLLM-based inference server for Llama 3.3 Instruct, logging last-layer per-token activations, prompt, response, and evaluation results to LMDB. Designed for use with the `nonsense_mixed_entities.py` benchmark and compatible with Meta's evaluation methods.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Start the vLLM OpenAI-compatible server with activation logging (vLLM 0.4.x):
  1. (Optional) Set the LMDB path for experiment separation:
     ```bash
     set ACTIVATION_LMDB_PATH=lmdb_data/exp1_activations.lmdb
     ```
  2. Launch the OpenAI-compatible server:
     ```bash
     vllm serve meta-llama/Meta-Llama-3-8B-Instruct --openai-api --trust-remote-code --hook hallu_llama_vllm/logging_hook.py:ActivationLoggingHook
     ```
  - This will launch a server at `http://localhost:8000/v1/completions` (OpenAI API protocol).
  - All activation logging will be handled transparently for each request.

## LMDB Output

- Activations, prompts, and responses are stored in LMDB under `lmdb_data/`.
- Each entry key is a SHA256 hash of the prompt (see `logging_hook.py`).

## Notes
- Python 3.10+ recommended.
- Do not commit large LMDB or model files.

## License
See LICENSE (if applicable).
