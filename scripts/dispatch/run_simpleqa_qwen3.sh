#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== SimpleQA train — Qwen3-8B ==="
$PYTHON scripts/capture_inference.py \
    --task simpleqa --split train \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/simpleqa_train_Qwen3-8B \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20

echo "=== SimpleQA test — Qwen3-8B ==="
$PYTHON scripts/capture_inference.py \
    --task simpleqa --split test \
    --model Qwen/Qwen3-8B \
    --out-dir shared/icr_capture/simpleqa_test_Qwen3-8B \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20
echo "=== DONE ==="
