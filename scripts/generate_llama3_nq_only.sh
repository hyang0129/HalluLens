#!/bin/bash
# Regenerate NQ inference + activations for Llama-3.1-8B-Instruct only.
# Hotpotqa splits are already complete; this handles just the NQ splits.
#
# Usage: bash scripts/generate_llama3_nq_only.sh

set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

GPU_PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
MODEL=meta-llama/Llama-3.1-8B-Instruct
BASE=output
SHARED=shared

run_inference_eval() {
    local task=$1 split=$2 gen_file=$3 zarr=$4
    echo ""
    echo "========================================"
    echo "  $task | split=$split"
    echo "========================================"
    mkdir -p "$(dirname "$gen_file")"
    echo "-- inference --"
    $GPU_PYTHON scripts/run_with_server.py \
        --step inference \
        --task "$task" \
        --model "$MODEL" \
        --split "$split" \
        --generations_file_path "$gen_file" \
        --activations-path "$zarr"
    echo "-- eval --"
    $GPU_PYTHON scripts/run_with_server.py \
        --step eval \
        --task "$task" \
        --model "$MODEL" \
        --split "$split" \
        --generations_file_path "$gen_file"
}

echo "=== Llama-3.1-8B-Instruct: NQ generation ==="
echo "Started: $(date)"

# test split: ~4,155 samples (80/20 split, seed=42)
run_inference_eval naturalquestions test \
    "$BASE/natural_questions/Llama-3.1-8B-Instruct/generation.jsonl" \
    "$SHARED/natural_questions_llama_3_1_8b_instruct/activations.zarr"

# train split: ~16,617 samples (80/20 split, seed=42)
run_inference_eval naturalquestions train \
    "$BASE/natural_questions_train/Llama-3.1-8B-Instruct/generation.jsonl" \
    "$SHARED/natural_questions_train_llama_3_1_8b_instruct/activations.zarr"

echo ""
echo "=== ALL DONE: $(date) ==="
