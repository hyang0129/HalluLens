#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "=== PopQA contrastive (unified config, train+test) ==="
$PYTHON scripts/run_experiment.py \
    --dataset configs/datasets/popqa.json \
    --method configs/methods/contrastive.json \
    --seed 42 \
    --device cuda \
    --force
echo "=== PopQA DONE ==="

echo ""
echo "=== SciQ contrastive (unified config, train+test) ==="
$PYTHON scripts/run_experiment.py \
    --dataset configs/datasets/sciq.json \
    --method configs/methods/contrastive.json \
    --seed 42 \
    --device cuda \
    --force
echo "=== SciQ DONE ==="
