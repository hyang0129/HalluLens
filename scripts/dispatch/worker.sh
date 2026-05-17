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

# Why: from_pretrained() makes a HEAD request to huggingface.co even when the
# weights are cached locally — to validate the cache freshness. All target
# models are in NFS-shared ~/.cache/huggingface/, so we disable the check.
# Set both env vars: HF_HUB_OFFLINE governs huggingface_hub, TRANSFORMERS_OFFLINE
# governs transformers, and transformers consults both depending on version.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Why: on nodes where another user has started a CUDA MPS daemon
# (/tmp/nvidia-mps/ owned by their UID), the CUDA runtime tries to connect
# to that MPS control socket during torch.cuda.is_available() and blocks
# indefinitely (we saw alphagpu04 hang here for 15+ min with wchan
# skb_wait_for_more_packets — the socket was the MPS control pipe, not HF
# or NFS). Point CUDA_MPS_PIPE_DIRECTORY at a non-existent path so the
# runtime falls back to direct GPU access. Verified on alphagpu04 GPU 6:
# torch.cuda.is_available() returned True in 6.2s with this set, vs.
# timing out at 60s without.
export CUDA_MPS_PIPE_DIRECTORY=/no/such/path

WORKER_ID="${HOSTNAME}_$$_$RANDOM"
CAPTURE_LOG="/tmp/capture_${WORKER_ID}.log"

# Why: the previous trap only killed $HB_PID, so a SIGTERM to this script left
# capture_inference.py running as an orphan (PPID=1) — orphans then wrote to
# the same memmap out_dir as the next dispatched worker, racing on every
# sample. We now track $CAPTURE_PID explicitly and propagate signals to it.
# pkill -P $$ is a safety net for any other direct children.
_cleanup() {
  [ -n "${CAPTURE_PID:-}" ] && kill -TERM "$CAPTURE_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]      && kill -TERM "$HB_PID"      2>/dev/null
  pkill -TERM -P $$ 2>/dev/null
  sleep 2
  [ -n "${CAPTURE_PID:-}" ] && kill -KILL "$CAPTURE_PID" 2>/dev/null
  [ -n "${HB_PID:-}" ]      && kill -KILL "$HB_PID"      2>/dev/null
  pkill -KILL -P $$ 2>/dev/null
  echo "worker $WORKER_ID exiting (trap)."
}
# Why: after trap runs for SIGTERM/SIGINT, bash would otherwise resume from
# the interrupted `wait`, see EXIT_CODE != 0, mark the current cell as failed,
# and claim the NEXT pending cell. Killing a worker then double-fails: the
# active cell + one more from pending. Set a flag and exit explicitly.
_signal_cleanup_and_exit() {
  _cleanup
  exit 130
}
trap _cleanup EXIT
trap _signal_cleanup_and_exit INT TERM

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
  IDX_START=$(    "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); v=d.get('index_start'); print('' if v is None else str(v))")
  IDX_END=$(      "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); v=d.get('index_end'); print('' if v is None else str(v))")
  SHUFFLE_SEED=$( "$PYTHON" -c "import json,sys; d=json.load(open('$CELL_PATH')); print(d.get('shuffle_seed', 0))")

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
  if [ -n "$IDX_START" ]; then
    CAPTURE_ARGS+=(--index-start "$IDX_START")
  fi
  if [ -n "$IDX_END" ]; then
    CAPTURE_ARGS+=(--index-end "$IDX_END")
  fi
  CAPTURE_ARGS+=(--shuffle-seed "$SHUFFLE_SEED")

  echo "worker $WORKER_ID: running capture_inference.py task=$TASK split=$SPLIT model=$MODEL"
  set +e
  # Why: launch in background and track $CAPTURE_PID so the EXIT trap can
  # forward SIGTERM to it. A previous pipeline + ${PIPESTATUS[0]} pattern
  # left the python child unkillable from the trap.
  "$PYTHON" "$PROJECT_ROOT/scripts/capture_inference.py" "${CAPTURE_ARGS[@]}" \
    > "$CAPTURE_LOG" 2>&1 &
  CAPTURE_PID=$!
  wait "$CAPTURE_PID"
  EXIT_CODE=$?
  CAPTURE_PID=""
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
