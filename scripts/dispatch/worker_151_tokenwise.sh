#!/usr/bin/env bash
# Issue #151 worker: reuse the tested experiment-cell loop with an isolated
# high-priority queue. Building cells does not invoke this script.
# Pre-submit check (does not claim cells):
#   python scripts/dispatch/validate_issue_151_tokenwise.py
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PROJECT_ROOT
export DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_151_tokenwise_dispatch}"

exec "$SCRIPT_DIR/worker_149.sh"
