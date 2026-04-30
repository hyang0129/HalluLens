#!/bin/bash
# Regenerate NQ and hotpotQA inference + activations for Llama-3.1-8B-Instruct
# using the standard batch pipeline (HFTransformersAdapter + zarr logprobs).
#
# Produces canonical zarr paths under shared/{dataset}_llama_3_1_8b_instruct/
# with full logprob arrays (response_token_logprobs, response_topk_logprobs, etc.)
# needed for token entropy baseline.
#
# Estimated runtime: ~2 hrs on H200 (118k total samples @ ~16 samp/s)
#   hotpotqa test:          7,405  ~  8 min
#   hotpotqa train:        90,440  ~ 94 min
#   natural_questions test: 4,155  ~  4 min
#   natural_questions train:16,617 ~ 17 min
#
# Usage: bash scripts/generate_llama3_nq_hotpotqa.sh

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

echo "=== Llama-3.1-8B-Instruct: NQ + hotpotQA generation ==="
echo "Started: $(date)"

# ── hotpotqa ──────────────────────────────────────────────────────────────────
# test split: 7,405 samples (HF "validation")
run_inference_eval hotpotqa validation \
    "$BASE/hotpotqa/Llama-3.1-8B-Instruct/generation.jsonl" \
    "$SHARED/hotpotqa_llama_3_1_8b_instruct/activations.zarr"

# train split: 90,440 samples (HF "train")
run_inference_eval hotpotqa train \
    "$BASE/hotpotqa_train/Llama-3.1-8B-Instruct/generation.jsonl" \
    "$SHARED/hotpotqa_train_llama_3_1_8b_instruct/activations.zarr"

# ── natural_questions ─────────────────────────────────────────────────────────
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
