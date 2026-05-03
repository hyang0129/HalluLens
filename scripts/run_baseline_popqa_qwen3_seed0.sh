#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "baseline_comparison_popqa_qwen3 — seed 0"
echo "Started: $(date)"
echo "============================================"

$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_popqa_qwen3.json \
    --seeds 0 \
    --device cuda

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
