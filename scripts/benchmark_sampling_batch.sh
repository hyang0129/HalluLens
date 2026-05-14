#!/bin/bash
# Run scripts/benchmark_sampling_batch.py on a Jupyter GPU node.
# Reports rows/sec + peak VRAM at each batch size for both supported models.
#
# Dispatch:
#   python scripts/gpu_dispatch.py run --jupyter --node alphagpu04-8884 -- \
#       bash scripts/benchmark_sampling_batch.sh
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
BATCH_SIZES="${BATCH_SIZES:-16,32,64,128,256}"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen3-8B")
DATASET=hotpotqa
SPLIT=test

OUT_DIR=reports/benchmark_sampling_batch
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUT_DIR/bench_${STAMP}.log"

echo "============================================" | tee "$LOG_FILE"
echo "sampling batch-size benchmark"               | tee -a "$LOG_FILE"
echo "Started:    $(date)"                          | tee -a "$LOG_FILE"
echo "Host:       $(hostname)"                      | tee -a "$LOG_FILE"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)" | tee -a "$LOG_FILE"
echo "Models:     ${MODELS[*]}"                     | tee -a "$LOG_FILE"
echo "Batches:    $BATCH_SIZES"                     | tee -a "$LOG_FILE"
echo "Dataset:    $DATASET/$SPLIT"                  | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                        | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
for MODEL in "${MODELS[@]}"; do
    MNAME=$(basename "$MODEL")
    JSON_OUT="$OUT_DIR/bench_${MNAME}_${STAMP}.json"
    echo                                                       | tee -a "$LOG_FILE"
    echo "--- $MODEL ---"                                      | tee -a "$LOG_FILE"
    $PYTHON scripts/benchmark_sampling_batch.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --batch-sizes "$BATCH_SIZES" \
        --out "$JSON_OUT" 2>&1 | tee -a "$LOG_FILE"
done

echo                                                            | tee -a "$LOG_FILE"
echo "DONE: $(date)"                                            | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
