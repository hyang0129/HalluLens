#!/bin/bash
# Run a shard of sampling-baseline cells sequentially on one GPU.
# Cells are read from $CELLS_FILE (TSV: dataset \t split \t model per line).
# Each cell is fully resumable, so failures don't lose progress within a cell.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

: "${CELLS_FILE:?CELLS_FILE env var required (TSV: dataset\\tsplit\\tmodel)}"
if [[ ! -f "$CELLS_FILE" ]]; then
    echo "ERROR: cells file not found: $CELLS_FILE" >&2
    exit 1
fi

LOG_DIR=reports/sampling_baselines_runs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SHARD_LOG="$LOG_DIR/shard_${STAMP}.log"

N_CELLS=$(grep -cv '^\s*\(#.*\)\?$' "$CELLS_FILE" || echo 0)
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================" | tee "$SHARD_LOG"
echo "sampling baselines shard"                      | tee -a "$SHARD_LOG"
echo "Started: $(date) | Host: $(hostname)"          | tee -a "$SHARD_LOG"
echo "Cells:   $N_CELLS (from $CELLS_FILE)"          | tee -a "$SHARD_LOG"
echo "Log:     $SHARD_LOG"                            | tee -a "$SHARD_LOG"
echo "============================================" | tee -a "$SHARD_LOG"

# Phase 0: ensure stratified subset + SearchQA cap index files exist.
# Idempotent — compute_subsets.py skips cells whose output already exists.
echo                                                  | tee -a "$SHARD_LOG"
echo "--- Phase 0 (subsets + caps) ---"               | tee -a "$SHARD_LOG"
$PYTHON scripts/compute_subsets.py 2>&1 | tee -a "$SHARD_LOG"

FAILED_CELLS=()
i=0
while IFS=$'\t' read -r ds split model; do
    # Skip blank/comment lines
    case "$ds" in ''|\#*) continue ;; esac
    i=$((i+1))
    echo                                              | tee -a "$SHARD_LOG"
    echo ">>> [$i/$N_CELLS] $ds / $split / $(basename "$model")" | tee -a "$SHARD_LOG"
    echo "    started: $(date)"                       | tee -a "$SHARD_LOG"
    if DATASET="$ds" SPLIT="$split" MODEL="$model" \
       bash scripts/run_sampling_baselines_cell.sh; then
        echo "    OK"                                 | tee -a "$SHARD_LOG"
    else
        rc=$?
        echo "    FAILED (exit=$rc) — continuing with next cell" | tee -a "$SHARD_LOG"
        FAILED_CELLS+=("$ds/$split/$(basename "$model")")
    fi
done < "$CELLS_FILE"

echo                                                  | tee -a "$SHARD_LOG"
echo "============================================" | tee -a "$SHARD_LOG"
echo "Shard DONE: $(date)"                            | tee -a "$SHARD_LOG"
if [[ ${#FAILED_CELLS[@]} -gt 0 ]]; then
    echo "Failed cells (${#FAILED_CELLS[@]}):"        | tee -a "$SHARD_LOG"
    printf '  %s\n' "${FAILED_CELLS[@]}"              | tee -a "$SHARD_LOG"
    exit 1
fi
echo "All $i cells succeeded."                        | tee -a "$SHARD_LOG"
