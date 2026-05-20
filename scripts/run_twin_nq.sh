#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== twin-grid NQ (Llama) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/twin_grid_nq_memmap.json
echo "=== twin-grid NQ (Llama) done: $(date) ==="

echo "=== twin-grid NQ (Qwen3) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/twin_grid_nq_qwen3_memmap.json
echo "=== twin-grid NQ (Qwen3) done: $(date) ==="

echo "=== unlabeled-ablation NQ (Llama) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/unlabeled_ablation_nq_memmap.json
echo "=== unlabeled-ablation NQ (Llama) done: $(date) ==="

echo "=== unlabeled-ablation NQ (Qwen3) started: $(date) ==="
$PYTHON scripts/run_experiment.py --experiment configs/experiments/unlabeled_ablation_nq_qwen3_memmap.json
echo "=== unlabeled-ablation NQ (Qwen3) done: $(date) ==="
