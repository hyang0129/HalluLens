#!/bin/bash
# Smoketest for Issue #75 (combined logprob + attention reconstruction).
# Runs a 2-epoch x 20-step pass on hotpotqa_test_Llama-3.1-8B-Instruct
# capture and writes a checkpoint to /tmp/smoketest_issue_75/.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
LOG=/tmp/smoketest_issue_75.log

# Cap BLAS / OpenMP thread fan-out so we don't fight cgroup limits.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "============================================"   | tee "$LOG"
echo "smoketest_issue_75: started $(date)"             | tee -a "$LOG"
echo "host=$(hostname)  python=$PYTHON"                | tee -a "$LOG"
echo "============================================"   | tee -a "$LOG"

# Run the smoketest, tee'ing combined stdout+stderr to the log.
"$PYTHON" scripts/smoketest_issue_75.py \
    --capture-dir shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \
    --output-dir  /tmp/smoketest_issue_75 \
    --epochs 2 \
    --steps-per-epoch 20 \
    --batch-size 64 \
    --sub-batch-size 64 \
    --num-workers 2 \
    --relevant-layers 14-29 \
    --attn-direction backward \
    --attn-offset-k 4 \
    --seed 0 \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "============================================"   | tee -a "$LOG"
echo "smoketest_issue_75: exit=$rc finished $(date)"   | tee -a "$LOG"
echo "============================================"   | tee -a "$LOG"
exit "$rc"
