#!/usr/bin/env bash
# worker_aug_combo.sh — coordinator-free GPU worker for the mixup + channel_drop
# combo sweep (x1, x2, Llama + Qwen3, 6 datasets, seed 0).
#
# Dispatched via:
#   gpu_dispatch.py run [--force-concurrent] --node NODE bash scripts/dispatch/worker_aug_combo.sh
#
# Env vars (override defaults):
#   DISPATCH_ROOT — (default: shared/issue_122_dispatch)
#   PROJECT_ROOT  — repo root
#   PYTHON        — interpreter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_122_dispatch}"
PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CLI="$SCRIPT_DIR/_claim_cli.py"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_MPS_PIPE_DIRECTORY=/no/such/path
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

WORKER_ID="${DISPATCH_NODE:-$HOSTNAME}_$$_$RANDOM"
RUN_LOG="/tmp/aug_combo_${WORKER_ID}.log"

_cleanup() {
  [ -n "${RUN_PID:-}" ] && kill -TERM "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -TERM "$HB_PID"  2>/dev/null
  pkill -TERM -P $$ 2>/dev/null
  sleep 2
  [ -n "${RUN_PID:-}" ] && kill -KILL "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -KILL "$HB_PID"  2>/dev/null
  pkill -KILL -P $$ 2>/dev/null
  echo "worker_aug_combo $WORKER_ID exiting (trap)."
}
_signal_cleanup_and_exit() { _cleanup; exit 130; }
trap _cleanup EXIT
trap _signal_cleanup_and_exit INT TERM

echo "worker_aug_combo $WORKER_ID starting — dispatch_root=$DISPATCH_ROOT"
"$PYTHON" "$CLI" gc --root "$DISPATCH_ROOT"

(
  while true; do
    "$PYTHON" "$CLI" heartbeat --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" 2>/dev/null
    sleep 60
  done
) &
HB_PID=$!

N_DONE=0; N_SKIP=0; N_FAIL=0

while true; do
  CELL_PATH=$("$PYTHON" "$CLI" claim --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID")
  if [ -z "$CELL_PATH" ]; then
    echo "worker_aug_combo $WORKER_ID: no cells remaining, exiting."
    break
  fi

  EXPERIMENT=$("$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['experiment_config'])")
  METHOD=$(    "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['method'])")
  SEED=$(      "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['seed'])")
  OUTPUT_CHECK=$("$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['output_check'])")
  ABS_OUTPUT="$PROJECT_ROOT/$OUTPUT_CHECK"

  if [ -f "$ABS_OUTPUT" ]; then
    echo "worker_aug_combo $WORKER_ID: output exists — marking done."
    "$PYTHON" "$CLI" complete --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH"
    N_SKIP=$(( N_SKIP + 1 ))
    continue
  fi

  echo "worker_aug_combo $WORKER_ID: running experiment=$EXPERIMENT method=$METHOD seed=$SEED"
  set +e
  "$PYTHON" "$PROJECT_ROOT/scripts/run_experiment.py" \
      --experiment "$PROJECT_ROOT/$EXPERIMENT" \
      --methods    "$METHOD" \
      --seeds      "$SEED" \
      > "$RUN_LOG" 2>&1 &
  RUN_PID=$!; wait "$RUN_PID"; EXIT_CODE=$?; RUN_PID=""
  set -e

  if [ "$EXIT_CODE" -eq 0 ] && [ -f "$ABS_OUTPUT" ]; then
    "$PYTHON" "$CLI" complete --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH"
    N_DONE=$(( N_DONE + 1 ))
    echo "worker_aug_combo $WORKER_ID: completed $CELL_PATH"
  elif [ "$EXIT_CODE" -eq 0 ]; then
    printf '%s\n' "exit_code=0 but output_check missing: $OUTPUT_CHECK" >> "$RUN_LOG"
    "$PYTHON" "$CLI" fail --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH" --err-file "$RUN_LOG"
    N_FAIL=$(( N_FAIL + 1 ))
  else
    "$PYTHON" "$CLI" fail --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH" --err-file "$RUN_LOG"
    N_FAIL=$(( N_FAIL + 1 ))
    echo "worker_aug_combo $WORKER_ID: FAILED $CELL_PATH (exit $EXIT_CODE)"
  fi
done

kill "$HB_PID" 2>/dev/null
trap - EXIT
echo "worker_aug_combo $WORKER_ID processed $N_DONE done, $N_SKIP skipped, $N_FAIL failed."
