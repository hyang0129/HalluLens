#!/bin/bash
set -euo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
LOG=/tmp/simclr_gate0_nq_qwen3_seed0.log
PY=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "[$(date)] Starting simclr_cotrained_gate0 smoke-test on NQ Qwen3 seed 0" | tee "$LOG"

$PY scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_nq_qwen3.json \
    --method simclr_cotrained_gate0 \
    --seeds 0 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done. Log: $LOG"
