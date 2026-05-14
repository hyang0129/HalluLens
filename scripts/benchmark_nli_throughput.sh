#!/bin/bash
# NLI throughput batch-size sweep (fp32, paper-faithful).
#
# Dispatch:
#   python scripts/gpu_dispatch.py run --jupyter --node alphagpu04-8884 -- \
#       bash scripts/benchmark_nli_throughput.sh
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
BATCH_SIZES="${BATCH_SIZES:-256,512,1024,2048}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DATASET=hotpotqa
SPLIT=test
GEN_MODEL=meta-llama/Llama-3.1-8B-Instruct

OUT_DIR=reports/benchmark_nli_throughput
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUT_DIR/bench_${STAMP}.log"
JSON_OUT="$OUT_DIR/bench_${STAMP}.json"

echo "============================================" | tee "$LOG_FILE"
echo "NLI batch-size throughput sweep"              | tee -a "$LOG_FILE"
echo "Started:    $(date)"                           | tee -a "$LOG_FILE"
echo "Host:       $(hostname)"                       | tee -a "$LOG_FILE"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)" | tee -a "$LOG_FILE"
echo "Batches:    $BATCH_SIZES"                      | tee -a "$LOG_FILE"
echo "max_length: $MAX_LENGTH"                       | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                         | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
$PYTHON scripts/benchmark_nli_throughput.py \
    --batch-sizes "$BATCH_SIZES" \
    --max-length "$MAX_LENGTH" \
    --gen-model "$GEN_MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --out "$JSON_OUT" 2>&1 | tee -a "$LOG_FILE"

echo                                                | tee -a "$LOG_FILE"
echo "DONE: $(date)"                                | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
