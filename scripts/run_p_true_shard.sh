#!/usr/bin/env bash
# Run a shard of P(true) cells defined in a TSV file.
#
# TSV format (one header line, then one cell per line):
#   # dataset\tmodel
#   hotpotqa\tmeta-llama/Llama-3.1-8B-Instruct
#   ...
#
# Usage:
#   CELLS_FILE=reports/p_true_runs/fanout_XYZ/cells_nodeA.tsv bash scripts/run_p_true_shard.sh
#
# Within a node, cells are grouped by model to avoid reloading the model
# between datasets with the same model. Uses run_p_true_for_model.py per
# model group so the model is loaded exactly once per model.

set -euo pipefail

PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CELLS_FILE="${CELLS_FILE:?CELLS_FILE env var must be set}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$(dirname "$0")/.."

if [[ ! -f "$CELLS_FILE" ]]; then
    echo "ERROR: CELLS_FILE not found: $CELLS_FILE" >&2
    exit 1
fi

echo "=== P(true) shard: $CELLS_FILE ==="
echo "PYTHON=$PYTHON | SPLIT=$SPLIT | BATCH_SIZE=$BATCH_SIZE"
echo

# Parse TSV into (dataset, model) pairs, skipping comments/header
declare -A model_datasets  # model -> comma-separated datasets

while IFS=$'\t' read -r dataset model; do
    [[ "$dataset" =~ ^#.*$ ]] && continue
    [[ -z "$dataset" || -z "$model" ]] && continue
    if [[ -v model_datasets["$model"] ]]; then
        model_datasets["$model"]="${model_datasets[$model]},${dataset}"
    else
        model_datasets["$model"]="$dataset"
    fi
done < "$CELLS_FILE"

for model in "${!model_datasets[@]}"; do
    datasets="${model_datasets[$model]}"
    echo "--- Model: $model | Datasets: $datasets ---"
    "$PYTHON" scripts/run_p_true_for_model.py \
        --model "$model" \
        --datasets "$datasets" \
        --split "$SPLIT" \
        --batch-size "$BATCH_SIZE"
    echo
done

echo "=== Shard complete ==="
