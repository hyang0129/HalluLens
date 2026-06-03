#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

for ds in hotpotqa nq popqa sciq searchqa; do
    echo "=== Experiment: baseline_comparison_${ds}_flipped_memmap_seed1 (Llama, halu-as-inlier, seed 1 only) ==="
    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${ds}_flipped_memmap_seed1.json \
        2>&1 | tee /tmp/exp_${ds}_flipped_llama_seed1.log
done

echo "=== ALL DONE ==="
