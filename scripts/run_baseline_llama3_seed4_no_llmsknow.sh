#!/bin/bash
# Resume wrapper around run_baseline_llama3_seed4.sh that excludes llmsknow_probe.
# See run_baseline_qwen3_seed4_no_llmsknow.sh for rationale.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
METHODS="contrastive_logprob_recon,linear_probe,token_entropy,logprob_baseline,saplma"

for TASK in hotpotqa mmlu nq popqa sciq searchqa; do
    echo "============================================"
    echo "baseline_comparison_${TASK} — seed 4 (Llama-3.1-8B-Instruct) — no llmsknow_probe"
    echo "Started: $(date)"
    echo "============================================"

    $PYTHON scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_${TASK}.json \
        --methods "$METHODS" \
        --seeds 4 \
        --device cuda

    echo ""
done

echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
