#!/usr/bin/env bash
# GPU worker for issue #149 prefix-specific ICR cache prerequisites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_149_icr_cache_dispatch}"
PYTHON="${PYTHON:-/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python}"
CLI="$SCRIPT_DIR/_claim_cli.py"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

WORKER_ID="${DISPATCH_NODE:-$HOSTNAME}_$$_$RANDOM"
RUN_LOG="/tmp/issue149_icr_cache_${WORKER_ID}.log"

_cleanup() {
  [ -n "${RUN_PID:-}" ] && kill -TERM "$RUN_PID" 2>/dev/null || true
  [ -n "${HB_PID:-}" ] && kill -TERM "$HB_PID" 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

"$PYTHON" "$CLI" gc --root "$DISPATCH_ROOT"
(
  while true; do
    "$PYTHON" "$CLI" heartbeat --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" 2>/dev/null
    sleep 60
  done
) &
HB_PID=$!

while true; do
  CELL_PATH=$("$PYTHON" "$CLI" claim --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID")
  [ -n "$CELL_PATH" ] || break

  CAPTURE_DIR=$("$PYTHON" -c "import json; print(json.load(open('$CELL_PATH'))['capture_dir'])")
  OUTPUT_DIR=$("$PYTHON" -c "import json; print(json.load(open('$CELL_PATH'))['output_dir'])")
  PREFIXES=$("$PYTHON" -c "import json; print(json.load(open('$CELL_PATH'))['prefixes'])")
  BATCH_SIZE=$("$PYTHON" -c "import json; print(json.load(open('$CELL_PATH'))['batch_size'])")
  OUTPUT_CHECK=$("$PYTHON" -c "import json; print(json.load(open('$CELL_PATH'))['output_check'])")

  if [ -f "$PROJECT_ROOT/$OUTPUT_CHECK" ]; then
    "$PYTHON" "$CLI" complete --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH"
    continue
  fi

  set +e
  "$PYTHON" "$PROJECT_ROOT/scripts/build_prefix_icr_cache.py" \
    "$PROJECT_ROOT/$CAPTURE_DIR" \
    --output-dir "$PROJECT_ROOT/$OUTPUT_DIR" \
    --prefixes "$PREFIXES" \
    --batch-size "$BATCH_SIZE" \
    --device cuda >"$RUN_LOG" 2>&1 &
  RUN_PID=$!
  wait "$RUN_PID"
  EXIT_CODE=$?
  RUN_PID=""
  set -e

  if [ "$EXIT_CODE" -eq 0 ] && [ -f "$PROJECT_ROOT/$OUTPUT_CHECK" ]; then
    "$PYTHON" "$CLI" complete --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH"
  else
    "$PYTHON" "$CLI" fail --root "$DISPATCH_ROOT" --worker-id "$WORKER_ID" --cell "$CELL_PATH" --err-file "$RUN_LOG"
  fi
done

kill "$HB_PID" 2>/dev/null || true
trap - EXIT
