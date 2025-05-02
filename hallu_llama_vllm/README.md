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

- Start the server:
  ```bash
  python server.py
  ```
- Use the `/generate` endpoint for inference.

## LMDB Output

- Activations, prompts, responses, and evaluation results are stored in LMDB under `lmdb_data/`.

## Notes
- Python 3.10+ recommended.
- Do not commit large LMDB or model files.

## License
See LICENSE (if applicable).
