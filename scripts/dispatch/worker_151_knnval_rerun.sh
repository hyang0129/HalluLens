#!/usr/bin/env bash
# Drain the issue #151 KNN-validation rerun/sweep queue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PROJECT_ROOT
export DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_151_knnval_rerun_dispatch}"

exec "$SCRIPT_DIR/worker_experiment.sh"
