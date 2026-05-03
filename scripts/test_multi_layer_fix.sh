#!/bin/bash
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "Test: MultiLayerLinearProbe LayerNorm fix — sciq_qwen3 seed 0"
echo "Started: $(date)"
echo "============================================"

$PYTHON scripts/run_experiment.py \
    --dataset configs/datasets/sciq_qwen3.json \
    --method configs/methods/multi_layer_linear_probe.json \
    --seed 0 \
    --device cuda 2>&1 | tee /tmp/test_multi_layer_fix.log

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "============================================"

# Print result summary
RESULT_DIR="runs/sciq_qwen3_multi_layer_linear_probe/sciq_qwen3/multi_layer_linear_probe/seed_0"
if [ -f "$RESULT_DIR/eval_metrics.json" ]; then
    echo "eval_metrics.json:"
    cat "$RESULT_DIR/eval_metrics.json"
fi
