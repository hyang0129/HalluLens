#!/bin/bash
# SearchQA-only rerun after the #60 train/test flip (151K → train, 43K → test).
# Pre-flip runs were archived to runs_archive/baseline_comparison_searchqa*_pre_issue60_2026-05-14/.
#
# Seed 0 — includes non-learned baselines (token_entropy, logprob_baseline).
# Other seed scripts exclude these to avoid racing on the shared artifacts/ dir.
# llmsknow_probe is excluded: the gpu03 fanout (fanout_20260514_122011) handles
# SearchQA llmsknow_probe for both models and will pick up the flipped configs
# automatically when its queue reaches the SearchQA units.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
SEED=0
METHODS="contrastive_logprob_recon,linear_probe,saplma,token_entropy,logprob_baseline"

echo "============================================"
echo "rerun_searchqa_seed${SEED} — Llama-3.1-8B-Instruct (with non-learned)"
echo "Started: $(date)"
echo "============================================"
$PYTHON scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_searchqa.json \
    --methods "$METHODS" \
    --seeds $SEED \
    --device cuda

echo ""
echo "============================================"
echo "rerun_searchqa_seed${SEED} — Qwen3-8B (with non-learned)"
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
