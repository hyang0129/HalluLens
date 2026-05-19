#!/usr/bin/env bash
# worker_89.sh — coordinator-free CPU worker for the issue #89 transfer
# matrix sweep.
#
# Dispatched via (CPU nodes, no --min-vram needed):
#   gpu_dispatch.py run --jupyter --node alphagpu23-8881 -- bash scripts/dispatch/worker_89.sh
#
# Multiple workers across nodes (or on the same node) drain the shared
# shared/issue_89_dispatch/pending/ queue via atomic os.rename(2).
# Each worker:
#   1. Claims one cell
#   2. If output_check already exists → marks done (no re-run)
#   3. Runs eval_transfer_matrix_memmap.py --cell-json <cell>
#   4. Marks done/failed and loops
#
# Env vars (override defaults):
#   DISPATCH_ROOT — path to dispatch root  (default: shared/issue_89_dispatch)
#   PROJECT_ROOT  — repo root              (default: parent of this script's directory)
#   PYTHON        — interpreter to invoke  (default: p311 mamba env)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_89_dispatch}"

# Why: the Jupyter kernel's default `python` is the `ml` miniconda env which
# lacks our project deps. Pin to the p311 mamba env unless overridden — same
# pattern as worker.sh and worker_79.sh.
PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CLI="$SCRIPT_DIR/_claim_cli.py"

# Why: transfer eval is CPU-only but still imports torch; skip the HF hub
# check to avoid network latency and failures on offline nodes.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Why: $HOSTNAME only names the physical node; use the DISPATCH_NODE var
# injected by gpu_dispatch.py on the --jupyter path (full slice name like
# alphagpu17-8888) to avoid WORKER_ID collisions when two slices of the same
# node both run workers.
WORKER_ID="${DISPATCH_NODE:-$HOSTNAME}_$$_$RANDOM"
RUN_LOG="/tmp/issue_89_${WORKER_ID}.log"

_cleanup() {
  [ -n "${RUN_PID:-}" ] && kill -TERM "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -TERM "$HB_PID"  2>/dev/null
  pkill -TERM -P $$ 2>/dev/null
  sleep 2
  [ -n "${RUN_PID:-}" ] && kill -KILL "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -KILL "$HB_PID"  2>/dev/null
  pkill -KILL -P $$ 2>/dev/null
  echo "worker_89 $WORKER_ID exiting (trap)."
}
_signal_cleanup_and_exit() {
  _cleanup
  exit 130
}
trap _cleanup EXIT
trap _signal_cleanup_and_exit INT TERM

echo "worker_89 $WORKER_ID starting — dispatch_root=$DISPATCH_ROOT"

"$PYTHON" "$CLI" gc --root "$DISPATCH_ROOT"

# Heartbeat loop in background — 60s cadence, GC threshold is 5 min.
(
  while true; do
    "$PYTHON" "$CLI" heartbeat --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" 2>/dev/null
    sleep 60
  done
) &
HB_PID=$!

N_DONE=0
N_SKIP=0
N_FAIL=0

while true; do
  CELL_PATH=$("$PYTHON" "$CLI" claim --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID")

  if [ -z "$CELL_PATH" ]; then
    echo "worker_89 $WORKER_ID: no cells remaining, exiting loop."
    break
  fi

  echo "worker_89 $WORKER_ID: claimed $CELL_PATH"

  OUTPUT_CHECK=$("$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['output_check'])")
  ABS_OUTPUT="$PROJECT_ROOT/$OUTPUT_CHECK"

  # Resume semantics — if the result JSON already exists, mark complete without re-running.
  if [ -f "$ABS_OUTPUT" ]; then
    echo "worker_89 $WORKER_ID: output exists at $OUTPUT_CHECK — marking done."
    "$PYTHON" "$CLI" complete \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH"
    N_SKIP=$(( N_SKIP + 1 ))
    continue
  fi

  CELL_ID=$("$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['cell_id'])")
  echo "worker_89 $WORKER_ID: evaluating cell=$CELL_ID"

  set +e
  "$PYTHON" "$PROJECT_ROOT/scripts/eval_transfer_matrix_memmap.py" \
      --cell-json "$CELL_PATH" \
      > "$RUN_LOG" 2>&1 &
  RUN_PID=$!
  wait "$RUN_PID"
  EXIT_CODE=$?
  RUN_PID=""
  set -e

  # Guard against exit 0 with missing output — eval_transfer_matrix_memmap.py
  # can return 0 on "single_class" but still writes the result JSON.
  # Treat missing output as failure regardless of exit code.
  if [ "$EXIT_CODE" -eq 0 ] && [ -f "$ABS_OUTPUT" ]; then
    "$PYTHON" "$CLI" complete \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH"
    N_DONE=$(( N_DONE + 1 ))
    echo "worker_89 $WORKER_ID: completed $CELL_ID"
  elif [ "$EXIT_CODE" -eq 0 ]; then
    echo "worker_89 $WORKER_ID: exit=0 but $OUTPUT_CHECK missing — treating as FAIL"
    printf '%s\n' "exit_code=0 but output_check missing: $OUTPUT_CHECK" >> "$RUN_LOG"
    "$PYTHON" "$CLI" fail \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH" \
      --err-file "$RUN_LOG"
    N_FAIL=$(( N_FAIL + 1 ))
  else
    "$PYTHON" "$CLI" fail \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH" \
      --err-file "$RUN_LOG"
    N_FAIL=$(( N_FAIL + 1 ))
    echo "worker_89 $WORKER_ID: FAILED $CELL_ID (exit $EXIT_CODE)"
  fi
done

kill "$HB_PID" 2>/dev/null
trap - EXIT
echo "worker_89 $WORKER_ID processed $N_DONE cells, $N_SKIP skipped, $N_FAIL failed, exiting."
