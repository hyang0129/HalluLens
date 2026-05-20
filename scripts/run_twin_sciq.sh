#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== twin-grid SciQ (Llama) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/twin_grid_sciq_memmap.json
echo "=== twin-grid SciQ (Llama) done: $(date) ==="

echo "=== twin-grid SciQ (Qwen3) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/twin_grid_sciq_qwen3_memmap.json
echo "=== twin-grid SciQ (Qwen3) done: $(date) ==="

echo "=== unlabeled-ablation SciQ (Llama) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/unlabeled_ablation_sciq_memmap.json
echo "=== unlabeled-ablation SciQ (Llama) done: $(date) ==="

echo "=== unlabeled-ablation SciQ (Qwen3) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/unlabeled_ablation_sciq_qwen3_memmap.json
echo "=== unlabeled-ablation SciQ (Qwen3) done: $(date) ==="
