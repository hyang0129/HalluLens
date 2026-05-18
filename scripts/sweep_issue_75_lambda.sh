#!/bin/bash
# Per-slice wrapper for the Issue #75 lambda sweep.
# Takes one positional argument: the method config name (no path, no .json).
#
# Examples:
#   bash scripts/sweep_issue_75_lambda.sh contrastive_logprob_attn_recon_l10_a00
#   bash scripts/sweep_issue_75_lambda.sh contrastive_logprob_attn_recon_l00_a10
#
# Defaults to all 6 Llama datasets at seed 0; override via env vars:
#   SWEEP_DATASETS=hotpotqa,popqa  bash scripts/sweep_issue_75_lambda.sh <method>
#   SWEEP_SEED=1                   bash scripts/sweep_issue_75_lambda.sh <method>
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

if [ $# -lt 1 ]; then
    echo "usage: $0 <method_config_name>" >&2
    echo "  method_config_name = filename under configs/methods/ without .json" >&2
    exit 64
fi
METHOD="$1"
CONFIG="configs/methods/${METHOD}.json"
if [ ! -f "$CONFIG" ]; then
    echo "method config not found: $CONFIG" >&2
    exit 66
fi

DATASETS="${SWEEP_DATASETS:-hotpotqa,popqa,nq,sciq,searchqa,mmlu}"
SEED="${SWEEP_SEED:-0}"
NUM_WORKERS="${SWEEP_NUM_WORKERS:-4}"
# Smoketest overrides — leave unset for production runs.
MAX_EPOCHS="${SWEEP_MAX_EPOCHS:-}"
STEPS_PER_EPOCH="${SWEEP_STEPS_PER_EPOCH:-}"
SUB_BATCH_SIZE="${SWEEP_SUB_BATCH_SIZE:-}"

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
LOG="/tmp/sweep_issue_75_${METHOD}.log"
OUTPUT_DIR="runs/issue_75_lambda_sweep/${METHOD}/seed_${SEED}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "===================================================="     | tee "$LOG"
echo "sweep_issue_75_lambda: $METHOD"                            | tee -a "$LOG"
echo "  host=$(hostname)"                                        | tee -a "$LOG"
echo "  python=$PYTHON"                                          | tee -a "$LOG"
echo "  config=$CONFIG"                                          | tee -a "$LOG"
echo "  datasets=$DATASETS"                                      | tee -a "$LOG"
echo "  seed=$SEED  num_workers=$NUM_WORKERS"                    | tee -a "$LOG"
echo "  output_dir=$OUTPUT_DIR"                                  | tee -a "$LOG"
echo "  started=$(date)"                                         | tee -a "$LOG"
echo "===================================================="     | tee -a "$LOG"

EXTRA_ARGS=()
if [ -n "$MAX_EPOCHS" ]; then
    EXTRA_ARGS+=(--max-epochs "$MAX_EPOCHS")
fi
if [ -n "$STEPS_PER_EPOCH" ]; then
    EXTRA_ARGS+=(--steps-per-epoch "$STEPS_PER_EPOCH")
fi
if [ -n "$SUB_BATCH_SIZE" ]; then
    EXTRA_ARGS+=(--sub-batch-size "$SUB_BATCH_SIZE")
fi

"$PYTHON" scripts/sweep_issue_75_lambda.py \
    --method-config "$CONFIG" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "===================================================="     | tee -a "$LOG"
echo "sweep_issue_75_lambda: $METHOD exit=$rc $(date)"           | tee -a "$LOG"
echo "===================================================="     | tee -a "$LOG"
exit "$rc"
