#!/bin/bash
# Issue #66/#67 ablations on Qwen3-8B activations.
#   #66 contrastive             — ProgressiveCompressor, use_labels=true, no recon aux
#   #67 saplma_logprob_recon    — SAPLMA with logprob recon auxiliary
# Datasets: hotpotqa_qwen3, popqa_qwen3 | Seeds: 0..4 | 20 runs total (resumable).
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
METHODS=contrastive,saplma_logprob_recon
SEEDS=0,1,2,3,4

echo "============================================"
echo "Issue #66/#67 ablations - Qwen3-8B"
echo "Methods: $METHODS"
echo "Seeds:   $SEEDS"
echo "Started: $(date) | Host: $(hostname)"
echo "============================================"

for CFG in baseline_comparison_hotpotqa_qwen3.json baseline_comparison_popqa_qwen3.json; do
    echo ""
    echo "--- $CFG ---"
    $PYTHON scripts/run_experiment.py \
        --experiment "configs/experiments/$CFG" \
        --methods "$METHODS" \
        --seeds "$SEEDS" \
        --device cuda
done

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"
