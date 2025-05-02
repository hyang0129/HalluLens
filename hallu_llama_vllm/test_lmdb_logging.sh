#!/usr/bin/env bash
# Test that vLLM OpenAI API server logs activations to LMDB

set -e

export ACTIVATION_LMDB_PATH=lmdb_data/test_activations.lmdb
PROMPT='Hello, world!'
PROMPT_HASH=$(python -c "import hashlib; print(hashlib.sha256('$PROMPT'.encode('utf-8')).hexdigest())")

# Start vLLM server in background (assumes model and vllm are installed)
# The user should manually start the server in another terminal for real test, or use tmux/screen
# vllm serve --model meta-llama/Meta-Llama-3-8B-Instruct --openai-api --trust-remote-code --hook hallu_llama_vllm/logging_hook.py:ActivationLoggingHook &
# sleep 20  # Wait for server to be ready

# Send OpenAI API request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt": "Hello, world!", "max_tokens": 5}'

# Check LMDB for the prompt hash
python hallu_llama_vllm/test_check_lmdb.py "$PROMPT_HASH"

echo "Test completed. If you see activations printed above, logging works!"
