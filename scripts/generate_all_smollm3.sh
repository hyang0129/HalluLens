#!/bin/bash
# Generate all LLMsKnow datasets for SmolLM3-3B (all splits, inference + eval).
# Excludes movies (no train split). NQ test inference is already complete.
#
# Usage: bash scripts/generate_all_smollm3.sh

set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

GPU_PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
MODEL=HuggingFaceTB/SmolLM3-3B
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

echo "=== SmolLM3-3B Dataset Generation: all LLMsKnow tasks ==="
echo "Started: $(date)"

# ── natural questions ─────────────────────────────────────────────────────────
# test split: 4,155 samples (inference already complete — resume will skip)
run_inference_eval naturalquestions test \
    "$BASE/natural_questions/SmolLM3-3B/generation.jsonl" \
    "$SHARED/natural_questions_smollm3_3b/activations.zarr"

# train split: ~79,168 samples
run_inference_eval naturalquestions train \
    "$BASE/natural_questions_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/natural_questions_train_smollm3_3b/activations.zarr"

# ── hotpotqa ──────────────────────────────────────────────────────────────────
# test split: 7,405 samples (HF "validation")
run_inference_eval hotpotqa validation \
    "$BASE/hotpotqa/SmolLM3-3B/generation.jsonl" \
    "$SHARED/hotpotqa_smollm3_3b/activations.zarr"

# train split: 90,447 samples (HF "train")
run_inference_eval hotpotqa train \
    "$BASE/hotpotqa_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/hotpotqa_train_smollm3_3b/activations.zarr"

# ── mmlu ──────────────────────────────────────────────────────────────────────
# test split: 14,079 samples
run_inference_eval mmlu test \
    "$BASE/mmlu/SmolLM3-3B/generation.jsonl" \
    "$SHARED/mmlu_smollm3_3b/activations.zarr"

# train split: ~99,800 samples
run_inference_eval mmlu auxiliary_train \
    "$BASE/mmlu_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/mmlu_train_smollm3_3b/activations.zarr"

# ── popqa ─────────────────────────────────────────────────────────────────────
# test split: 2,854 samples
run_inference_eval popqa test \
    "$BASE/popqa/SmolLM3-3B/generation.jsonl" \
    "$SHARED/popqa_smollm3_3b/activations.zarr"

# train split: 11,413 samples
run_inference_eval popqa train \
    "$BASE/popqa_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/popqa_train_smollm3_3b/activations.zarr"

# ── sciq ──────────────────────────────────────────────────────────────────────
# test split: 1,000 samples
run_inference_eval sciq test \
    "$BASE/sciq/SmolLM3-3B/generation.jsonl" \
    "$SHARED/sciq_smollm3_3b/activations.zarr"

# train split: 11,679 samples
run_inference_eval sciq train \
    "$BASE/sciq_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/sciq_train_smollm3_3b/activations.zarr"

# ── searchqa ──────────────────────────────────────────────────────────────────
# test split: ~43,228 samples (HF "test")
run_inference_eval searchqa test \
    "$BASE/searchqa/SmolLM3-3B/generation.jsonl" \
    "$SHARED/searchqa_smollm3_3b/activations.zarr"

# train split: ~151,295 samples (HF "train")
run_inference_eval searchqa train \
    "$BASE/searchqa_train/SmolLM3-3B/generation.jsonl" \
    "$SHARED/searchqa_train_smollm3_3b/activations.zarr"

echo ""
echo "=== ALL DONE: $(date) ==="
