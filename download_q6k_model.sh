#!/bin/bash
# Download Llama-3.3-70B-Instruct Q6_K_L GGUF model from HuggingFace
# Run this from the HalluLens repository root

set -e  # Exit on error

echo "📥 Downloading Llama-3.3-70B-Instruct Q6_K_L GGUF model..."

# Create models directory if it doesn't exist
mkdir -p models

cd models

# Download using huggingface-cli (recommended)
echo "Using huggingface-cli to download..."
echo "Note: Q6_K_L is split into 2 files, downloading both parts..."
huggingface-cli download \
    bartowski/Llama-3.3-70B-Instruct-GGUF \
    --include "Llama-3.3-70B-Instruct-Q6_K_L/*" \
    --local-dir . \
    --local-dir-use-symlinks False

echo "✅ Model downloaded to: models/Llama-3.3-70B-Instruct-Q6_K_L/"
echo "   Files:"
echo "   - Llama-3.3-70B-Instruct-Q6_K_L-00001-of-00002.gguf"
echo "   - Llama-3.3-70B-Instruct-Q6_K_L-00002-of-00002.gguf"
echo ""
echo "To use this model, run:"
echo "  python scripts/run_with_server.py --step generate --task precisewikiqa \\"
echo "    --model 'models/Llama-3.3-70B-Instruct-Q6_K_L' \\"
echo "    --N 100"
