#!/bin/bash
set -euo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
LOG=/tmp/split_llama3_nq.log
PY=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
echo "=== split_llama3_nq_from_qwen3.py starting on $(hostname) at $(date) ===" | tee "$LOG"
$PY scripts/split_llama3_nq_from_qwen3.py 2>&1 | tee -a "$LOG"
echo "=== finished at $(date) ===" | tee -a "$LOG"
