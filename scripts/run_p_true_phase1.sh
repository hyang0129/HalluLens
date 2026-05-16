#!/bin/bash
# P(true) Phase 1 — full 12-cell grid, both models, test split, b=256.
# Resumable (hotpotqa/Llama already done from calibration → will be a no-op).
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "P(true) Phase 1 — Llama"
echo "Started: $(date)"
echo "============================================"
$PYTHON scripts/run_p_true_for_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 64

echo ""
echo "============================================"
echo "P(true) Phase 1 — Qwen3"
echo "Started: $(date)"
echo "============================================"
$PYTHON scripts/run_p_true_for_model.py \
    --model Qwen/Qwen3-8B \
    --batch-size 64

echo ""
echo "============================================"
echo "P(true) Phase 1 DONE: $(date)"
echo "============================================"
