#!/usr/bin/env bash
# worker_79.sh — coordinator-free GPU worker for the Issue #79 baseline
# re-train sweep.
#
# Dispatched via:
#   gpu_dispatch.py run --jupyter --node alphagpu23-8881 -- bash scripts/dispatch/worker_79.sh
#
# Multiple workers across nodes drain the same shared/issue_79_dispatch/pending/
# queue via atomic os.rename(2). Each worker:
#   1. claims one cell
#   2. if its output_check already exists → marks done (no re-run)
#   3. else invokes run_experiment.py for that (config, method, seed)
#   4. marks done/failed and loops
#
# Env vars (override defaults):
#   DISPATCH_ROOT — path to the dispatch root  (default: shared/issue_79_dispatch)
#   PROJECT_ROOT  — repo root                  (default: parent of this script)
#   PYTHON        — interpreter to invoke      (default: p311 mamba env)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_79_dispatch}"

# Why: the Jupyter kernel's default `python` is the `ml` miniconda env which
# lacks our project deps (loguru, transformers pin, etc.). Pin to the p311
# mamba env unless overridden — same pattern as worker.sh.
PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CLI="$SCRIPT_DIR/_claim_cli.py"

# Why: from_pretrained() makes a HEAD request to huggingface.co even when
# weights are cached locally. Skip the check — all target models are NFS-cached.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Why: on nodes where another user has started a CUDA MPS daemon, the runtime
# blocks on the MPS control socket. Point CUDA_MPS_PIPE_DIRECTORY at a
# non-existent path so the runtime falls back to direct GPU access.
export CUDA_MPS_PIPE_DIRECTORY=/no/such/path

# BLAS threads — match worker.sh defaults; downstream torch DataLoader workers
# spawn their own processes.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Why: $HOSTNAME only names the physical node (alphagpu17), which collides
# when two Jupyter slices of the same node both run workers. gpu_dispatch.py
# injects $DISPATCH_NODE with the full slice name (alphagpu17-8888) on the
# --jupyter path; fall back to $HOSTNAME for ssh dispatches without it.
WORKER_ID="${DISPATCH_NODE:-$HOSTNAME}_$$_$RANDOM"
RUN_LOG="/tmp/issue_79_${WORKER_ID}.log"

# Signal handling — mirror worker.sh. Without forwarding SIGTERM to the
# Python child it can survive as an orphan and race the next worker on the
# same output_dir.
_cleanup() {
  [ -n "${RUN_PID:-}" ] && kill -TERM "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -TERM "$HB_PID"  2>/dev/null
  pkill -TERM -P $$ 2>/dev/null
  sleep 2
  [ -n "${RUN_PID:-}" ] && kill -KILL "$RUN_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]  && kill -KILL "$HB_PID"  2>/dev/null
  pkill -KILL -P $$ 2>/dev/null
  echo "worker_79 $WORKER_ID exiting (trap)."
}
_signal_cleanup_and_exit() {
  _cleanup
  exit 130
}
trap _cleanup EXIT
trap _signal_cleanup_and_exit INT TERM

echo "worker_79 $WORKER_ID starting — dispatch_root=$DISPATCH_ROOT"

# One-shot GC on startup to reclaim cells from crashed prior workers.
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
    echo "worker_79 $WORKER_ID: no cells remaining, exiting loop."
    break
  fi

  echo "worker_79 $WORKER_ID: claimed $CELL_PATH"

  TASK_TYPE=$(  "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d.get('task_type','train'))")
  OUTPUT_CHECK=$("$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['output_check'])")

  ABS_OUTPUT="$PROJECT_ROOT/$OUTPUT_CHECK"

  # Resume semantics — if output already exists, mark complete without re-running.
  if [ -f "$ABS_OUTPUT" ]; then
    echo "worker_79 $WORKER_ID: output exists at $OUTPUT_CHECK — marking done."
    "$PYTHON" "$CLI" complete \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH"
    N_SKIP=$(( N_SKIP + 1 ))
    continue
  fi

  set +e
  if [ "$TASK_TYPE" = "transfer_eval" ]; then
    echo "worker_79 $WORKER_ID: transfer_eval cell=$(basename $CELL_PATH .json)"
    "$PYTHON" "$PROJECT_ROOT/scripts/eval_transfer_matrix_memmap.py" \
        --cell-json "$CELL_PATH" \
        > "$RUN_LOG" 2>&1 &
  else
    EXPERIMENT=$( "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['experiment_config'])")
    METHOD=$(     "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['method'])")
    SEED=$(       "$PYTHON" -c "import json; d=json.load(open('$CELL_PATH')); print(d['seed'])")
    echo "worker_79 $WORKER_ID: running experiment=$EXPERIMENT method=$METHOD seed=$SEED"
    "$PYTHON" "$PROJECT_ROOT/scripts/run_experiment.py" \
        --experiment "$PROJECT_ROOT/$EXPERIMENT" \
        --methods    "$METHOD" \
        --seeds      "$SEED" \
        > "$RUN_LOG" 2>&1 &
  fi
  RUN_PID=$!
  wait "$RUN_PID"
  EXIT_CODE=$?
  RUN_PID=""
  set -e

  # Why: both run_experiment.py and eval_transfer_matrix_memmap.py have exited 0
  # on internal failure paths without writing output. Guard with post-run
  # existence check so a silent skip becomes a failed cell, not a phantom completion.
  if [ "$EXIT_CODE" -eq 0 ] && [ -f "$ABS_OUTPUT" ]; then
    "$PYTHON" "$CLI" complete \
      --root "$DISPATCH_ROOT" \
      --worker-id "$WORKER_ID" \
      --cell "$CELL_PATH"
    N_DONE=$(( N_DONE + 1 ))
    echo "worker_79 $WORKER_ID: completed $CELL_PATH"
  elif [ "$EXIT_CODE" -eq 0 ]; then
    echo "worker_79 $WORKER_ID: exit=0 but $OUTPUT_CHECK missing — treating as FAIL"
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
    echo "worker_79 $WORKER_ID: FAILED $CELL_PATH (exit $EXIT_CODE)"
  fi
done

kill "$HB_PID" 2>/dev/null
trap - EXIT
echo "worker_79 $WORKER_ID processed $N_DONE cells, $N_SKIP skipped, $N_FAIL failed, exiting."
