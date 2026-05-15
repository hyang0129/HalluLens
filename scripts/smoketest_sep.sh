#!/bin/bash
# Smoke-test the SEP-SE pipeline end-to-end on the smallest dataset (sciq, 1000
# test rows). Sweeps 2 layers for Qwen3, then fits one probe via auto-layer.
# Verifies outputs exist, contain expected keys, and produce a sane AUROC.
#
# Expect wall-clock ~3-5 min on alphacpu.
set -euo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
MODEL="Qwen/Qwen3-8B"
DATASET="sciq"
LAYERS="22 24"   # 2 layers — full default is 6
MODEL_NAME=$(basename "$MODEL")

LOG_DIR=reports/sampling_baselines_runs/sep_logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/smoketest_${DATASET}_${MODEL_NAME}_${STAMP}.log"

SWEEP_FILE="output/sampling_baselines/sep/${MODEL_NAME}/layer_sweep_${DATASET}.json"
RESULT_FILE="output/sampling_baselines/sep/${MODEL_NAME}/${DATASET}_sep_results.json"

echo "============================================" | tee "$LOG"
echo "SEP-SE smoketest"                              | tee -a "$LOG"
echo "Model:     $MODEL"                             | tee -a "$LOG"
echo "Dataset:   $DATASET"                           | tee -a "$LOG"
echo "Layers:    $LAYERS"                            | tee -a "$LOG"
echo "Started:   $(date) | Host: $(hostname)"        | tee -a "$LOG"
echo "Log:       $LOG"                               | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Phase 5a: layer sweep
# ---------------------------------------------------------------------------
echo                                                  | tee -a "$LOG"
echo "--- Phase 5a: layer sweep ---"                  | tee -a "$LOG"
$PYTHON scripts/compute_sep_layer_sweep.py \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --layers  $LAYERS \
    2>&1 | tee -a "$LOG"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "FAIL: sweep file not written → $SWEEP_FILE"  | tee -a "$LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 5: probe fit (auto layer from sweep)
# ---------------------------------------------------------------------------
echo                                                  | tee -a "$LOG"
echo "--- Phase 5: probe fit (auto layer) ---"        | tee -a "$LOG"
$PYTHON scripts/compute_sep.py \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --force \
    2>&1 | tee -a "$LOG"

if [[ ! -f "$RESULT_FILE" ]]; then
    echo "FAIL: result file not written → $RESULT_FILE" | tee -a "$LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo                                                  | tee -a "$LOG"
echo "--- Verifying outputs ---"                      | tee -a "$LOG"
$PYTHON - <<PY 2>&1 | tee -a "$LOG"
import json, sys
sweep = json.load(open("$SWEEP_FILE"))
res   = json.load(open("$RESULT_FILE"))

problems = []

# Sweep
for k in ("dataset", "model", "layers_swept", "auroc_by_layer", "best_layer", "seed", "val_frac"):
    if k not in sweep:
        problems.append(f"sweep missing key: {k}")
if sweep.get("dataset") != "$DATASET":
    problems.append(f"sweep dataset mismatch: {sweep.get('dataset')}")
sw_layers = set(int(k) for k in sweep.get("auroc_by_layer", {}).keys())
exp_layers = set(int(x) for x in "$LAYERS".split())
if sw_layers != exp_layers:
    problems.append(f"sweep layers mismatch: got {sw_layers}, expected {exp_layers}")
for layer, auroc in sweep.get("auroc_by_layer", {}).items():
    if not (0.0 <= auroc <= 1.0):
        problems.append(f"sweep AUROC out of range for layer {layer}: {auroc}")

# Result
for k in ("dataset", "model", "layer", "alpha", "sep_se_auroc",
          "train_size_sep_se", "test_size", "layer_source"):
    if k not in res:
        problems.append(f"result missing key: {k}")
if res.get("layer") != sweep.get("best_layer"):
    problems.append(
        f"result layer={res.get('layer')} != sweep best_layer={sweep.get('best_layer')}"
    )
if not (0.0 <= res.get("sep_se_auroc", -1) <= 1.0):
    problems.append(f"result AUROC out of range: {res.get('sep_se_auroc')}")
if res.get("test_size", 0) < 100:
    problems.append(f"result test_size suspiciously small: {res.get('test_size')}")

print()
print("SWEEP:")
print(f"  best_layer = {sweep.get('best_layer')}")
for layer in sorted(sweep.get("auroc_by_layer", {}), key=int):
    print(f"  layer {layer}: val AUROC = {sweep['auroc_by_layer'][layer]:.4f}")
print()
print("RESULT:")
print(f"  layer = {res.get('layer')}  (source: {res.get('layer_source')})")
print(f"  test AUROC = {res.get('sep_se_auroc'):.4f}")
print(f"  n_train = {res.get('train_size_sep_se')}, n_test = {res.get('test_size')}")
print()
if problems:
    print("FAIL — issues found:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("PASS — all checks ok.")
PY

echo                                                  | tee -a "$LOG"
echo "DONE: $(date)"                                  | tee -a "$LOG"
