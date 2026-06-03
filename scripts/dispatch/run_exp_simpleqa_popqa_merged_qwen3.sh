#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== Experiment: baseline_comparison_simpleqa_popqa_merged_qwen3_memmap (Qwen3, SimpleQA+PopQA merged ~54/46) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_simpleqa_popqa_merged_qwen3_memmap.json \
    2>&1 | tee /tmp/exp_simpleqa_popqa_merged_qwen3.log

echo "=== DONE ==="
