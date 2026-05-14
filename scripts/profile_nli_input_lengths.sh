#!/bin/bash
# CPU-only NLI input-length profile. Safe to --force-concurrent alongside a
# GPU job since this only loads the DeBERTa tokenizer (~2 MB, no model weights)
# and tokenizes greedy generations on CPU.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
OUT_DIR=reports/profile_nli_lengths
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUT_DIR/profile_${STAMP}.log"
JSON_OUT="$OUT_DIR/profile_${STAMP}.json"

echo "============================================" | tee "$LOG_FILE"
echo "NLI input-length profile (CPU-only)"          | tee -a "$LOG_FILE"
echo "Started:    $(date)"                           | tee -a "$LOG_FILE"
echo "Host:       $(hostname)"                       | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                         | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
$PYTHON scripts/profile_nli_input_lengths.py --out "$JSON_OUT" 2>&1 | tee -a "$LOG_FILE"

echo                                                 | tee -a "$LOG_FILE"
echo "DONE: $(date)"                                 | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
