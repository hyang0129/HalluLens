#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== Experiment: baseline_comparison_simpleqa_flipped_memmap (Llama, halluc-as-inlier) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_simpleqa_flipped_memmap.json \
    2>&1 | tee /tmp/exp_simpleqa_flipped_llama.log

echo "=== Experiment: baseline_comparison_simpleqa_qwen3_flipped_memmap (Qwen3, halluc-as-inlier) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_simpleqa_qwen3_flipped_memmap.json \
    2>&1 | tee /tmp/exp_simpleqa_flipped_qwen3.log

echo "=== DONE ==="
