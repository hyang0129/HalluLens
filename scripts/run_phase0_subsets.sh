#!/bin/bash
# Phase 0: build stratified 5k train subsets + SearchQA 10k test cap.
# CPU-only but dispatched to a GPU node per the no-local-workloads policy.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
LOG_DIR=reports/sampling_baselines_runs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase0_${STAMP}.log"

echo "============================================" | tee "$LOG_FILE"
echo "Phase 0: subset + cap index files"            | tee -a "$LOG_FILE"
echo "Started: $(date) | Host: $(hostname)"         | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
$PYTHON scripts/compute_subsets.py 2>&1 | tee -a "$LOG_FILE"

echo                                                | tee -a "$LOG_FILE"
echo "--- Resulting index files ---"                | tee -a "$LOG_FILE"
ls -1 output/sep_subset_*.json output/searchqa_test_cap_*.json 2>&1 | tee -a "$LOG_FILE"
echo "DONE: $(date)"                                | tee -a "$LOG_FILE"
