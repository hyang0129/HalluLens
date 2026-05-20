#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

for DATASET in nq sciq; do
    echo "============================================"
    echo "channel_dropout_ablation_${DATASET}_memmap — Llama-3.1-8B"
    echo "Started: $(date)"
    echo "============================================"

    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/channel_dropout_ablation_${DATASET}_memmap.json \
        --device cuda

    echo ""
done

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
