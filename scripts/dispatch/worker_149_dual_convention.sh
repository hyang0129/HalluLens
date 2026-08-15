#!/usr/bin/env bash
# Drain the isolated issue-#149 dual-convention classifier queue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PROJECT_ROOT
export DISPATCH_ROOT="${DISPATCH_ROOT:-$PROJECT_ROOT/shared/issue_149_dual_convention_dispatch}"

exec "$SCRIPT_DIR/worker_149.sh"
