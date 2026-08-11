#!/usr/bin/env bash
# Reuse the tested issue #149 experiment worker with an isolated queue root.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PROJECT_ROOT
export DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_149_matched_dispatch}"

# Fail before claiming any experiment cell if the four ICR prerequisites are
# incomplete. This keeps an accidental early launch recoverable and leaves the
# matched queue untouched.
REQUIRED_CACHES=(
  "hotpotqa_train_Llama-3.1-8B-Instruct_0-50000"
  "hotpotqa_test_Llama-3.1-8B-Instruct"
  "sciq_train_Llama-3.1-8B-Instruct"
  "sciq_test_Llama-3.1-8B-Instruct"
)
for CACHE_NAME in "${REQUIRED_CACHES[@]}"; do
  CACHE_PATH="$PROJECT_ROOT/shared/prefix149_icr/$CACHE_NAME/icr_scores_k64.npy"
  if [ ! -f "$CACHE_PATH" ]; then
    echo "missing prerequisite ICR cache: $CACHE_PATH" >&2
    echo "run worker_149_icr_cache.sh to completion first" >&2
    exit 2
  fi
done

exec "$SCRIPT_DIR/worker_149.sh"
