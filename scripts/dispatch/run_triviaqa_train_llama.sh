#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== TriviaQA train — Llama-3.1-8B-Instruct (n_samples=11000) ==="
$PYTHON scripts/capture_inference.py \
    --task triviaqa \
    --split train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir shared/icr_capture/triviaqa_train_Llama-3.1-8B-Instruct \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20 \
    --n-samples 11000
echo "=== DONE ==="
