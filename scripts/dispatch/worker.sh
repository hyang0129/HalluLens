#!/usr/bin/env bash
# worker.sh — coordinator-free GPU worker for ICR capture dispatch.
#
# Dispatched via:
#   gpu_dispatch.py run --jupyter --node alphagpu17-8883 -- bash scripts/dispatch/worker.sh
#
# Env vars (override defaults):
#   DISPATCH_ROOT   — path to <root>/_dispatch/ (default: shared/icr_capture/_dispatch)
#   PROJECT_ROOT    — repo root (default: directory of this script's parent-parent)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/icr_capture/_dispatch}"

# Why: the Jupyter kernel's default `python` is the `ml` miniconda env which
# lacks our project deps (jsonlines, datasets pinned versions, etc.). Pin to
# the p311 mamba env unless overridden, matching scripts/smoketest_capture_72.sh.
PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CLI="$SCRIPT_DIR/_claim_cli.py"

WORKER_ID="${HOSTNAME}_$$_$RANDOM"
CAPTURE_LOG="/tmp/capture_${WORKER_ID}.log"

trap 'kill "$HB_PID" 2>/dev/null; echo "worker $WORKER_ID exiting (trap)."' EXIT

echo "worker $WORKER_ID starting — dispatch_root=$DISPATCH_ROOT"

# One-shot GC on startup to reclaim any cells from crashed prior workers.
"$PYTHON" "$CLI" gc --root "$DISPATCH_ROOT"

# Heartbeat loop in background.
(
  while true; do
    "$PYTHON" "$CLI" heartbeat --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" 2>/dev/null
    sleep 60
  done
) &
HB_PID=$!

N_DONE=0
N_FAIL=0

while true; do
  CELL_PATH=$("$PYTHON" "$CLI" claim --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID")

  if [ -z "$CELL_PATH" ]; then
    echo "worker $WORKER_ID: no cells remaining, exiting loop."
    break
  fi

  echo "worker $WORKER_ID: claimed $CELL_PATH"

  # Parse cell JSON fields.
  TASK=$(         "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['task'])")
  SPLIT=$(        "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['split'])")
  MODEL=$(        "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['model'])")
  OUT_DIR=$(      "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['out_dir'])")
  MAX_PROMPT=$(   "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['max_prompt_len'])")
  MAX_RESP=$(     "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['max_response_len'])")
  R_MAX=$(        "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['r_max'])")
  TOP_K=$(        "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d['top_k'])")
  N_SAMPLES=$(    "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); v=d['n_samples']; print('' if v is None else str(v))")
  BATCH_SIZE=$(   "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d.get('batch_size', 1))")

  CAPTURE_ARGS=(
    --task        "$TASK"
    --split       "$SPLIT"
    --model       "$MODEL"
    --out-dir     "$PROJECT_ROOT/$OUT_DIR"
    --max-prompt-len   "$MAX_PROMPT"
    --max-response-len "$MAX_RESP"
    --r-max       "$R_MAX"
    --top-k       "$TOP_K"
    --batch-size  "$BATCH_SIZE"
  )
  if [ -n "$N_SAMPLES" ]; then
    CAPTURE_ARGS+=(--n-samples "$N_SAMPLES")
  fi

  echo "worker $WORKER_ID: running capture_inference.py task=$TASK split=$SPLIT model=$MODEL"
  set +e
  "$PYTHON" "$PROJECT_ROOT/scripts/capture_inference.py" "${CAPTURE_ARGS[@]}" \
    2>&1 | tee "$CAPTURE_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [ "$EXIT_CODE" -eq 0 ]; then
    "$PYTHON" "$CLI" complete \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH"
    N_DONE=$(( N_DONE + 1 ))
    echo "worker $WORKER_ID: completed $CELL_PATH"
  else
    "$PYTHON" "$CLI" fail \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH" \
      --err-file "$CAPTURE_LOG"
    N_FAIL=$(( N_FAIL + 1 ))
    echo "worker $WORKER_ID: FAILED $CELL_PATH (exit $EXIT_CODE)"
  fi
done

kill "$HB_PID" 2>/dev/null
trap - EXIT
echo "worker $WORKER_ID processed $N_DONE cells, $N_FAIL failed, exiting."
