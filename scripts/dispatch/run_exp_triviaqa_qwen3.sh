#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== Experiment: baseline_comparison_triviaqa_qwen3_memmap (Qwen/Qwen3-8B) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_triviaqa_qwen3_memmap.json \
    2>&1 | tee /tmp/exp_triviaqa_qwen3.log

echo "=== DONE ==="
