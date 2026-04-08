#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "Training runs: PopQA + SciQ (remaining)"
echo "============================================"

# --- PopQA: remaining methods, seed 42 ---
for method in token_entropy logprob_baseline; do
  echo ""
  echo ">>> PopQA / $method / seed 42"
  $PYTHON scripts/run_experiment.py \
    --dataset configs/datasets/popqa.json \
    --method configs/methods/${method}.json \
    --seed 42 \
    --device cuda
  echo ">>> PopQA / $method / seed 42 DONE"
done
echo "=== PopQA complete ==="

# --- SciQ: all methods, seed 42 ---
for method in contrastive linear_probe token_entropy logprob_baseline; do
  echo ""
  echo ">>> SciQ / $method / seed 42"
  $PYTHON scripts/run_experiment.py \
    --dataset configs/datasets/sciq.json \
    --method configs/methods/${method}.json \
    --seed 42 \
    --device cuda
  echo ">>> SciQ / $method / seed 42 DONE"
done
echo "=== SciQ complete ==="

echo ""
echo "============================================"
echo "ALL TRAINING RUNS COMPLETE"
echo "============================================"
