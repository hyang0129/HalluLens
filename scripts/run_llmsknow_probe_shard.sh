#!/bin/bash
# Run a shard of llmsknow_probe re-runs sequentially on one GPU node.
#
# Each "unit" in the shard is one (dataset, model) cache unit — i.e. one
# baseline_comparison_*.json experiment config. We invoke run_experiment.py
# once per unit with --methods llmsknow_probe --seeds 0,1,2,3,4 so all five
# seeds share the in-memory activation cache (the dev-subset materialization
# is the heavy I/O step; reloading per seed wastes ~10-30 min/dataset).
#
# Units are read from $UNITS_FILE (TSV: experiment_config_path per line).
# Each unit is fully resumable via run_experiment.py's own seed_N/eval_metrics
# resume logic; failures within a unit don't lose progress on other seeds.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

: "${UNITS_FILE:?UNITS_FILE env var required (TSV: experiment_config_path)}"
if [[ ! -f "$UNITS_FILE" ]]; then
    echo "ERROR: units file not found: $UNITS_FILE" >&2
    exit 1
fi

SEEDS="${SEEDS:-0,1,2,3,4}"
METHODS="${METHODS:-llmsknow_probe}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR=reports/llmsknow_probe_runs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SHARD_LOG="$LOG_DIR/shard_${STAMP}.log"

N_UNITS=$(grep -cv '^\s*\(#.*\)\?$' "$UNITS_FILE" || echo 0)
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================" | tee "$SHARD_LOG"
echo "llmsknow_probe shard"                          | tee -a "$SHARD_LOG"
echo "Started: $(date) | Host: $(hostname)"          | tee -a "$SHARD_LOG"
echo "Units:   $N_UNITS (from $UNITS_FILE)"          | tee -a "$SHARD_LOG"
echo "Seeds:   $SEEDS"                               | tee -a "$SHARD_LOG"
echo "Methods: $METHODS"                             | tee -a "$SHARD_LOG"
echo "Device:  $DEVICE"                              | tee -a "$SHARD_LOG"
echo "Log:     $SHARD_LOG"                           | tee -a "$SHARD_LOG"
echo "============================================" | tee -a "$SHARD_LOG"

FAILED_UNITS=()
i=0
# TSV format (written by scripts/fanout_llmsknow_probe.py):
#   <experiment_config_path>\t# <human-readable label>
# Split on TAB so the trailing label comment goes into the (ignored) second field.
while IFS=$'\t' read -r experiment_cfg _label_comment; do
    case "$experiment_cfg" in ''|\#*) continue ;; esac
    i=$((i+1))
    unit_name=$(basename "$experiment_cfg" .json)
    echo                                              | tee -a "$SHARD_LOG"
    echo ">>> [$i/$N_UNITS] $unit_name"               | tee -a "$SHARD_LOG"
    echo "    started: $(date)"                       | tee -a "$SHARD_LOG"
    if $PYTHON scripts/run_experiment.py \
            --experiment "$experiment_cfg" \
            --methods "$METHODS" \
            --seeds "$SEEDS" \
            --device "$DEVICE" 2>&1 | tee -a "$SHARD_LOG"; then
        echo "    OK"                                 | tee -a "$SHARD_LOG"
    else
        rc=${PIPESTATUS[0]}
        echo "    FAILED (exit=$rc) — continuing with next unit" | tee -a "$SHARD_LOG"
        FAILED_UNITS+=("$unit_name")
    fi
done < "$UNITS_FILE"

echo                                                  | tee -a "$SHARD_LOG"
echo "============================================" | tee -a "$SHARD_LOG"
echo "Shard DONE: $(date)"                            | tee -a "$SHARD_LOG"
if [[ ${#FAILED_UNITS[@]} -gt 0 ]]; then
    echo "Failed units (${#FAILED_UNITS[@]}):"        | tee -a "$SHARD_LOG"
    printf '  %s\n' "${FAILED_UNITS[@]}"              | tee -a "$SHARD_LOG"
    exit 1
fi
echo "All $i units succeeded."                        | tee -a "$SHARD_LOG"
