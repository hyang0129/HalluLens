#!/bin/bash
# GPU smoketest for the #72 capture rewrite — sciq × <model>, N=50.
# Parametrized by env var SMOKETEST_MODEL (defaults to Llama-3.1-8B-Instruct).
# Dispatched via scripts/gpu_dispatch.py; runs on whichever GPU node it lands on.
set -euo pipefail

REPO=/mnt/home/hyang1/LLM_research/HalluLens
PY=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

MODEL="${SMOKETEST_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_SLUG="${MODEL##*/}"

LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/issue72_smoketest_sciq_${MODEL_SLUG}_$(date +%Y%m%d_%H%M%S).log"

OUT_DIR=$REPO/shared/icr_capture/sciq_${MODEL_SLUG}_smoketest

cd "$REPO"

echo "=== node: $(hostname)" | tee -a "$LOG"
echo "=== cuda: $(nvidia-smi -L 2>/dev/null | head -2)" | tee -a "$LOG"
echo "=== branch: $(git rev-parse --abbrev-ref HEAD)" | tee -a "$LOG"
echo "=== commit: $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "=== model: $MODEL" | tee -a "$LOG"
echo "=== out_dir: $OUT_DIR" | tee -a "$LOG"
echo "=== log: $LOG" | tee -a "$LOG"

"$PY" scripts/capture_inference.py \
    --task sciq \
    --split test \
    --model "$MODEL" \
    --out-dir "$OUT_DIR" \
    --max-prompt-len 256 \
    --max-response-len 64 \
    --r-max 64 \
    --top-k 20 \
    --n-samples 50 2>&1 | tee -a "$LOG"

EXIT=${PIPESTATUS[0]}
echo "=== exit: $EXIT" | tee -a "$LOG"

if [ "$EXIT" -eq 0 ]; then
    echo "=== readback check ===" | tee -a "$LOG"
    OUT_DIR="$OUT_DIR" "$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json
import os
import numpy as np
from pathlib import Path
out_dir = Path(os.environ["OUT_DIR"])

with open(out_dir / "config.json") as f:
    cfg = json.load(f)
print("config.json:", json.dumps(cfg, indent=2))

with open(out_dir / "meta.jsonl") as f:
    meta_lines = [json.loads(l) for l in f if l.strip()]
print(f"meta.jsonl: {len(meta_lines)} entries; first={meta_lines[0] if meta_lines else None}")

icr = np.load(out_dir / "icr_scores.npy")
print(f"icr_scores.npy: shape={icr.shape}, dtype={icr.dtype}, "
      f"mean={icr.mean():.4f}, std={icr.std():.4f}, nan_count={np.isnan(icr).sum()}")

with open(out_dir / "eval_results.json") as f:
    er = json.load(f)
n_halu = sum(er["halu_test_res"])
print(f"eval_results.json: {len(er['halu_test_res'])} samples, "
      f"{n_halu} hallucinated, {len(er['halu_test_res']) - n_halu} correct")

with open(out_dir / "generation.jsonl") as f:
    gen_lines = [json.loads(l) for l in f if l.strip()]
print(f"generation.jsonl: {len(gen_lines)} entries; "
      f"first generation = {gen_lines[0]['generation'][:200]!r}")
print("readback OK")
PYEOF
fi

echo "=== done ===" | tee -a "$LOG"
exit $EXIT
