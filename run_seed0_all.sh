#!/bin/bash
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

# Pull latest code
git pull origin main

echo "============================================"
echo "Seed 0 runs for all datasets (updated configs)"
echo "Started: $(date)"
echo "============================================"

for exp in hotpotqa nq mmlu movies popqa sciq searchqa; do
    echo ""
    echo "============================================"
    echo "Experiment: $exp — seed 0 only"
    echo "============================================"
    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${exp}.json \
        --seeds 0 \
        --device cuda || echo ">>> WARNING: $exp had failures (continuing)"
    echo ">>> $exp seed 0 DONE"
done

echo ""
echo "============================================"
echo "ALL SEED 0 RUNS COMPLETE"
echo "Finished: $(date)"
echo "============================================"
