#!/bin/bash
# Smoke test for k-fold seed split correctness on gpu22.
# Runs all tests including the integration test that reads real NQ Qwen3 zarr metadata.
# No GPU training is performed — the test only reads metadata and applies sklearn splits.
set -euo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

LOG=/tmp/seed_split_smoke_test.log
PY=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "[$(date)] Starting seed split smoke tests" | tee "$LOG"

$PY -m pytest tests/test_seed_split_smoke.py -v 2>&1 | tee -a "$LOG"

echo "[$(date)] Done. Log: $LOG"
