#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== Experiment: baseline_comparison_triviaqa_memmap (Llama-3.1-8B-Instruct) ==="
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_triviaqa_memmap.json \
    2>&1 | tee /tmp/exp_triviaqa_llama.log

echo "=== DONE ==="
