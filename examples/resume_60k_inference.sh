#!/bin/bash
# Example: Running 60k inference with automatic resume support
# 
# This script demonstrates how to run a large-scale inference job (60k samples)
# that can be safely interrupted and resumed on a shared research cluster.
#
# Usage:
#   1. Run this script to start the inference
#   2. If interrupted (cluster time limit, crash, etc.), simply run it again
#   3. It will automatically resume from where it left off
#   4. Repeat until all 60k samples are processed

# Configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
WIKI_SRC="goodwiki"
MODE="gguf_big"
N=60000
OUTPUT_DIR="goodwiki_json_60k"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run inference with automatic resume
# Note: Running the EXACT SAME command will resume from where it left off
python scripts/run_with_server.py \
  --step inference \
  --task precisewikiqa \
  --model "$MODEL" \
  --wiki_src "$WIKI_SRC" \
  --mode "$MODE" \
  --inference_method vllm \
  --logger-type json \
  --activations-path "$OUTPUT_DIR/activations.json" \
  --max_inference_tokens 64 \
  --N $N \
  --qa_output_path "data/precise_qa/save/qa_${WIKI_SRC}_Llama-3.1-8B-Instruct_${MODE}.jsonl" \
  --generations_file_path "$OUTPUT_DIR/generation.jsonl"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Inference completed successfully!"
    echo "📊 Results saved to: $OUTPUT_DIR/generation.jsonl"
    echo "📊 Activations saved to: $OUTPUT_DIR/activations.json"
else
    echo "⚠️  Inference was interrupted or failed"
    echo "💡 You can resume by running this script again"
    echo "   It will automatically continue from where it left off"
fi

