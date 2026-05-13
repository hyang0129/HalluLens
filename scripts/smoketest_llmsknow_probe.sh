#!/bin/bash
# Smoketest for LLMsKnow probe baseline (PR #54).
# Verifies the SAPLMA-style fanout assumptions:
#   1. The method runs end-to-end via run_experiment.py
#   2. Wall time and peak RAM for one cell on the smallest dataset (sciq, 11k train)
#   3. Whether the (layer, token) sweep is the bottleneck vs. cache load
#   4. Whether the work is CPU-bound (i.e. GPU-idle once cache is loaded), which is
#      what makes --force-concurrent co-tenancy on a busy SAPLMA node viable.
#
# Run on a Jupyter GPU node (alphagpu01-8887). Single seed, single dataset.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
EXP_CFG=configs/experiments/baseline_comparison_sciq.json
LOG_DIR=reports/smoketest_llmsknow_probe
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/smoketest_${STAMP}.log"
GPU_LOG="$LOG_DIR/gpu_${STAMP}.log"
TIME_FILE="$LOG_DIR/time_${STAMP}.txt"

echo "============================================" | tee "$LOG_FILE"
echo "llmsknow_probe smoketest" | tee -a "$LOG_FILE"
echo "Started:    $(date)" | tee -a "$LOG_FILE"
echo "Host:       $(hostname)" | tee -a "$LOG_FILE"
echo "CPUs:       $(nproc)" | tee -a "$LOG_FILE"
echo "Experiment: $EXP_CFG" | tee -a "$LOG_FILE"
echo "Methods:    llmsknow_probe" | tee -a "$LOG_FILE"
echo "Seeds:      0" | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Background GPU sampler — verifies that sweep is CPU-bound (low GPU util after cache load)
(
    while true; do
        ts=$(date +%H:%M:%S)
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
            2>/dev/null | sed "s/^/$ts /" >> "$GPU_LOG" || true
        sleep 5
    done
) &
SAMPLER_PID=$!
trap "kill $SAMPLER_PID 2>/dev/null || true" EXIT

# Force unbuffered Python so phase log lines stream into the file in real time;
# /usr/bin/time -v reports peak RSS and wall clock.
export PYTHONUNBUFFERED=1
/usr/bin/time -v -o "$TIME_FILE" \
    $PYTHON scripts/run_experiment.py \
        --experiment "$EXP_CFG" \
        --methods llmsknow_probe \
        --seeds 0 \
        --device cuda \
    2>&1 | tee -a "$LOG_FILE"

EXIT=$?

echo "============================================" | tee -a "$LOG_FILE"
echo "DONE: $(date) exit=$EXIT" | tee -a "$LOG_FILE"
echo "--- time -v output ---" | tee -a "$LOG_FILE"
cat "$TIME_FILE" | tee -a "$LOG_FILE"
echo "--- gpu samples (head/tail) ---" | tee -a "$LOG_FILE"
head -n 5 "$GPU_LOG" 2>/dev/null | tee -a "$LOG_FILE" || true
echo "..." | tee -a "$LOG_FILE"
tail -n 10 "$GPU_LOG" 2>/dev/null | tee -a "$LOG_FILE" || true
echo "============================================" | tee -a "$LOG_FILE"

exit $EXIT
