#!/bin/bash
# Run Phase 1-4 of the sampling baselines pipeline for one (dataset, split, model)
# cell. Each phase is resumable — re-running an already-complete cell is a no-op.
#
# Usage (via gpu_dispatch):
#   python scripts/gpu_dispatch.py run --jupyter --node <NODE> -- \
#       env DATASET=hotpotqa SPLIT=test MODEL=meta-llama/Llama-3.1-8B-Instruct \
#       bash scripts/run_sampling_baselines_cell.sh
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

: "${DATASET:?DATASET env var required (hotpotqa|nq|popqa|sciq|searchqa)}"
: "${SPLIT:?SPLIT env var required (test|train)}"
: "${MODEL:?MODEL env var required (meta-llama/Llama-3.1-8B-Instruct|Qwen/Qwen3-8B)}"

MODEL_NAME=$(basename "$MODEL")
LOG_DIR=reports/sampling_baselines_runs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cell_${DATASET}_${SPLIT}_${MODEL_NAME}_${STAMP}.log"

echo "============================================" | tee "$LOG_FILE"
echo "sampling baselines cell"                       | tee -a "$LOG_FILE"
echo "Cell:    $DATASET / $SPLIT / $MODEL_NAME"       | tee -a "$LOG_FILE"
echo "Started: $(date) | Host: $(hostname)"           | tee -a "$LOG_FILE"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" | tee -a "$LOG_FILE"
echo "Log:     $LOG_FILE"                             | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
TIME_FILE="$LOG_DIR/cell_${DATASET}_${SPLIT}_${MODEL_NAME}_${STAMP}.time"

phase () {
    local label="$1"; shift
    echo                                              | tee -a "$LOG_FILE"
    echo "--- $label ---"                             | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

/usr/bin/time -v -o "$TIME_FILE" bash -c "
    set -eo pipefail
    cd '$(pwd)'
    phase () {
        local label=\"\$1\"; shift
        echo; echo \"--- \$label ---\"
        \"\$@\"
    }
    phase 'Phase 1 sampling' $PYTHON scripts/run_sampling_pass.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL'
    phase 'Phase 2 NLI matrix' $PYTHON scripts/compute_nli_matrix.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL'
    phase 'Phase 3 SE' $PYTHON scripts/compute_se.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL'
    phase 'Phase 4 SelfCheck' $PYTHON scripts/compute_selfcheck.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL' --no-bertscore
" 2>&1 | tee -a "$LOG_FILE"

EXIT=$?
echo                                                  | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "DONE: $(date) exit=$EXIT"                      | tee -a "$LOG_FILE"
echo "--- time -v ---"                                | tee -a "$LOG_FILE"
cat "$TIME_FILE"                                      | tee -a "$LOG_FILE" || true
echo "============================================" | tee -a "$LOG_FILE"
exit $EXIT
