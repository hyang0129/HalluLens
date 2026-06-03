#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== Experiment: baseline_comparison_simpleqa_popqa_merged_memmap (issue #124) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_simpleqa_popqa_merged_memmap.json \
    2>&1 | tee /tmp/exp_simpleqa_popqa_merged.log

echo "=== DONE ==="
