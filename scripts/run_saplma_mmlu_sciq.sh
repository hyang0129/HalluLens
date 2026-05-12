#!/bin/bash
# SAPLMA fanout — script B (intended for port 8888)
# Datasets: mmlu (99k train) + sciq (12k train)
# Models: Llama-3.1-8B-Instruct + Qwen3-8B
# Seeds: 0,1,2,3,4 -> 20 cells total
# Outputs land under runs/baseline_comparison_<task>[_qwen3]/ so audit_datasets.py picks them up.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

for TASK in mmlu sciq; do
    for VARIANT in "" "_qwen3"; do
        EXP_CFG="configs/experiments/baseline_comparison_${TASK}${VARIANT}.json"
        echo "============================================"
        echo "saplma | experiment=baseline_comparison_${TASK}${VARIANT} | seeds=0..4"
        echo "Started: $(date)"
        echo "============================================"

        $PYTHON scripts/run_experiment.py \
            --experiment "$EXP_CFG" \
            --methods saplma \
            --seeds 0,1,2,3,4 \
            --device cuda
    done
done

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
