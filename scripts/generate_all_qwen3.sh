#!/bin/bash
# Generate all LLMsKnow datasets for Qwen3-8B (all splits, inference + eval).
# Runs serially. Natural Questions is handled separately (already running).
# Intended to run after NQ completes on alphagpu19.
#
# Usage: bash scripts/generate_all_qwen3.sh

set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

GPU_PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
MODEL=Qwen/Qwen3-8B
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

echo "=== Qwen3-8B Dataset Generation: all remaining LLMsKnow tasks ==="
echo "Started: $(date)"

# ── hotpotqa ──────────────────────────────────────────────────────────────────
# test split: 7,405 samples (HF "validation")
run_inference_eval hotpotqa validation \
    "$BASE/hotpotqa/Qwen3-8B/generation.jsonl" \
    "$SHARED/hotpotqa_qwen3_8b/activations.zarr"

# train split: 90,447 samples (HF "train")
run_inference_eval hotpotqa train \
    "$BASE/hotpotqa_train/Qwen3-8B/generation.jsonl" \
    "$SHARED/hotpotqa_train_qwen3_8b/activations.zarr"

# ── mmlu ──────────────────────────────────────────────────────────────────────
# test split: 14,079 samples
run_inference_eval mmlu test \
    "$BASE/mmlu/Qwen3-8B/generation.jsonl" \
    "$SHARED/mmlu_qwen3_8b/activations.zarr"

# train split: ~99,800 samples
run_inference_eval mmlu auxiliary_train \
    "$BASE/mmlu_train/Qwen3-8B/generation.jsonl" \
    "$SHARED/mmlu_train_qwen3_8b/activations.zarr"

# ── movies ────────────────────────────────────────────────────────────────────
# test only: 7,657 samples
run_inference_eval movies test \
    "$BASE/movies/Qwen3-8B/generation.jsonl" \
    "$SHARED/movies_qwen3_8b/activations.zarr"

# ── popqa ─────────────────────────────────────────────────────────────────────
# test split: 2,854 samples
run_inference_eval popqa test \
    "$BASE/popqa/Qwen3-8B/generation.jsonl" \
    "$SHARED/popqa_qwen3_8b/activations.zarr"

# train split: 11,413 samples
run_inference_eval popqa train \
    "$BASE/popqa_train/Qwen3-8B/generation.jsonl" \
    "$SHARED/popqa_train_qwen3_8b/activations.zarr"

# ── sciq ──────────────────────────────────────────────────────────────────────
# test split: 1,000 samples
run_inference_eval sciq test \
    "$BASE/sciq/Qwen3-8B/generation.jsonl" \
    "$SHARED/sciq_qwen3_8b/activations.zarr"

# train split: 11,679 samples
run_inference_eval sciq train \
    "$BASE/sciq_train/Qwen3-8B/generation.jsonl" \
    "$SHARED/sciq_train_qwen3_8b/activations.zarr"

# ── searchqa ──────────────────────────────────────────────────────────────────
# eval set: HF "test" split (~43,228 samples)
run_inference_eval searchqa test \
    "$BASE/searchqa/Qwen3-8B/generation.jsonl" \
    "$SHARED/searchqa_qwen3_8b/activations.zarr"

# train split: HF "train" split (~151,295 samples)
run_inference_eval searchqa train \
    "$BASE/searchqa_train/Qwen3-8B/generation.jsonl" \
    "$SHARED/searchqa_train_qwen3_8b/activations.zarr"

echo ""
echo "=== ALL DONE: $(date) ==="
