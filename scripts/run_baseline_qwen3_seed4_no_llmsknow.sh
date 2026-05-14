#!/bin/bash
# Resume wrapper around run_baseline_qwen3_seed4.sh that excludes llmsknow_probe.
# Why: my llmsknow_probe fanout is already running on alphagpu01 and writes to
# the same runs/<exp>/<dataset>/llmsknow_probe/ path. Letting two hosts race
# the same non-learned cell would corrupt eval_metrics.json.
# Behaviour: the runner skips any (dataset, method, seed) cell whose
# eval_metrics.json exists, so completed cells are no-ops; only the partial
# contrastive_logprob_recon cell that was killed on alphagpu01 will actually
# retrain.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
METHODS="contrastive_logprob_recon,linear_probe,token_entropy,logprob_baseline,saplma"

for TASK in hotpotqa mmlu nq popqa sciq searchqa; do
    echo "============================================"
    echo "baseline_comparison_${TASK}_qwen3 — seed 4 (Qwen3) — no llmsknow_probe"
    echo "Started: $(date)"
    echo "============================================"

    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${TASK}_qwen3.json \
        --methods "$METHODS" \
        --seeds 4 \
        --device cuda

    echo ""
done

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
