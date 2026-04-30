#!/bin/bash
set -euo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
LOG=/tmp/simclr_gate_sweep_nq_qwen3_seed0.log
PY=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "[$(date)] Starting simclr gate sweep (B=0.5, B=1.0) on NQ Qwen3 seed 0" | tee "$LOG"

for METHOD in simclr_cotrained_gate05 simclr_cotrained; do
    echo "[$(date)] --- Running $METHOD ---" | tee -a "$LOG"
    $PY scripts/run_experiment.py \
        --experiment configs/experiments/baseline_comparison_nq_qwen3.json \
        --method "$METHOD" \
        --seed 0 \
        2>&1 | tee -a "$LOG"
done

echo "[$(date)] Done. Log: $LOG"
