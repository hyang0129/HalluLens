#!/bin/bash
# SearchQA-only rerun, seed 1. Learned methods only — non-learned baselines
# llmsknow_probe is excluded: handled by gpu03 fanout (fanout_20260514_122011).
# (token_entropy, logprob_baseline) are handled by rerun_searchqa_seed0.sh.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
SEED=1
METHODS="contrastive_logprob_recon,linear_probe,saplma"

echo "============================================"
echo "rerun_searchqa_seed${SEED} — Llama-3.1-8B-Instruct"
echo "Started: $(date)"
echo "============================================"
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_searchqa.json \
    --methods "$METHODS" \
    --seeds $SEED \
    --device cuda

echo ""
echo "============================================"
echo "rerun_searchqa_seed${SEED} — Qwen3-8B"
echo "Started: $(date)"
echo "============================================"
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_searchqa_qwen3.json \
    --methods "$METHODS" \
    --seeds $SEED \
    --device cuda

echo ""
echo "============================================"
echo "DONE seed ${SEED}: $(date)"
echo "============================================"
