#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== SimpleQA train — Llama-3.1-8B-Instruct ==="
$PYTHON scripts/capture_inference.py \
    --task simpleqa --split train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir shared/icr_capture/simpleqa_train_Llama-3.1-8B-Instruct \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20

echo "=== SimpleQA test — Llama-3.1-8B-Instruct ==="
$PYTHON scripts/capture_inference.py \
    --task simpleqa --split test \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir shared/icr_capture/simpleqa_test_Llama-3.1-8B-Instruct \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20
echo "=== DONE ==="
