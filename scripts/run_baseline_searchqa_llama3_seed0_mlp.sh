#!/bin/bash
# Re-run seed 0 for searchqa — only multi_layer_linear_probe is pending.
# Completed runs are skipped automatically by run_experiment.py.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "baseline_comparison_searchqa — seed 0 (Llama3, multi_layer_linear_probe)"
echo "Started: $(date)"
echo "============================================"

$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_searchqa.json \
    --seeds 0 \
    --device cuda

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
