#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
echo "=== HotpotQA train DELTA 50000-90447 — meta-llama/Llama-3.1-8B-Instruct (B=4, shuffle-seed 0) ==="
$PYTHON scripts/capture_inference.py \
    --task hotpotqa --split train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447 \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20 \
    --batch-size 4 \
    --shuffle-seed 0 --index-start 50000 --index-end 90447
echo "=== DONE ==="
