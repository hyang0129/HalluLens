#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

for TASK in hotpotqa mmlu nq popqa sciq searchqa; do
    echo "============================================"
    echo "baseline_comparison_${TASK} — seed 2 (Llama3)"
    echo "Started: $(date)"
    echo "============================================"

    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${TASK}.json \
        --seeds 2 \
        --device cuda

    echo ""
done

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
