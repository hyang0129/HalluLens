#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "SciQ: all methods, all seeds"
echo "============================================"

for method in contrastive linear_probe token_entropy logprob_baseline; do
  for seed in 0 5 26 42 63; do
    echo ""
    echo ">>> SciQ / $method / seed $seed"
    $PYTHON scripts/run_experiment.py \
        --dataset configs/datasets/sciq.json \
        --method configs/methods/${method}.json \
        --seed $seed \
        --device cuda
    echo ">>> SciQ / $method / seed $seed DONE"
  done
done

echo ""
echo "============================================"
echo "ALL SCIQ RUNS COMPLETE"
echo "============================================"
