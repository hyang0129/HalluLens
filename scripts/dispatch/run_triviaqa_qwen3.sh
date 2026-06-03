#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== TriviaQA train — Qwen3-8B (n_samples=11000) ==="
$PYTHON scripts/capture_inference.py \
    --task triviaqa \
    --split train \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/triviaqa_train_Qwen3-8B \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20 \
    --n-samples 11000

echo "=== TriviaQA validation (test) — Qwen3-8B (n_samples=14000) ==="
$PYTHON scripts/capture_inference.py \
    --task triviaqa \
    --split validation \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/triviaqa_test_Qwen3-8B \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20 \
    --n-samples 14000
echo "=== DONE ==="
