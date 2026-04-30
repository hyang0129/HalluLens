#!/bin/bash
# Seed 0 baseline comparison run for NQ with Llama-3.1-8B-Instruct activations.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "baseline_comparison_nq — seed 0 (Llama3)"
echo "Started: $(date)"
echo "============================================"

$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_nq.json \
    --seeds 0 \
    --device cuda

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
