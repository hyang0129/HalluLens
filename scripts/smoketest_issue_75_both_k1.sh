#!/bin/bash
# Smoketest for Issue #75 — variant: attn_direction='both', attn_offset_k=1.
# Validates the forward decoder code path + both-direction loss assembly +
# k=1 indexing on real captured data before launching the lambda sweep.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
LOG=/tmp/smoketest_issue_75_both_k1.log

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "============================================"     | tee "$LOG"
echo "smoketest_issue_75_both_k1: started $(date)"      | tee -a "$LOG"
echo "host=$(hostname)  python=$PYTHON"                 | tee -a "$LOG"
echo "============================================"     | tee -a "$LOG"

"$PYTHON" scripts/smoketest_issue_75.py \
    --capture-dir shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \
    --output-dir  /tmp/smoketest_issue_75_both_k1 \
    --epochs 2 \
    --steps-per-epoch 20 \
    --batch-size 64 \
    --sub-batch-size 64 \
    --num-workers 2 \
    --relevant-layers 14-29 \
    --attn-direction both \
    --attn-offset-k 1 \
    --seed 0 \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "============================================"     | tee -a "$LOG"
echo "smoketest_issue_75_both_k1: exit=$rc $(date)"     | tee -a "$LOG"
echo "============================================"     | tee -a "$LOG"
exit "$rc"
