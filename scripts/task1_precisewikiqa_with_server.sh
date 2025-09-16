#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Enhanced version of task1_precisewikiqa.sh that automatically manages the activation logging server

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "claude-3-sonnet"
    # "claude-3-haiku"
    # "gpt-4o"
)

MODE=dynamic
N=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --N)
            N="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Default to running all steps if not specified
if [ -z "$STEP" ]; then
    STEP="all"
fi

echo "Running PreciseWikiQA with automatic server management"
echo "Step: $STEP"
echo "Mode: $MODE"
echo "N: $N"
echo "Models: ${MODELS[@]}"

for MODEL in "${MODELS[@]}"
do
    echo "Processing model: $MODEL"
    
    # Use the unified script with server management
    python scripts/run_with_server.py \
        --step "$STEP" \
        --task precisewikiqa \
        --model "$MODEL" \
        --wiki_src goodwiki \
        --mode "$MODE" \
        --inference_method vllm \
        --N "$N"
    
    if [ $? -ne 0 ]; then
        echo "Error processing model $MODEL"
        exit 1
    fi
    
    echo "Completed processing model: $MODEL"
done

echo "All models processed successfully!"
