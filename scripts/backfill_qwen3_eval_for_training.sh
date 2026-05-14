#!/bin/bash
# Backfill missing eval_results_for_training.json for Qwen3 train splits, then
# re-run compute_subsets.py to generate the 3 missing subset index files.
#
# Safe to --force-concurrent alongside a running GPU job — this is pure CPU.
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
LOG_DIR=reports/sampling_baselines_runs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/backfill_qwen3_${STAMP}.log"

echo "============================================" | tee "$LOG_FILE"
echo "Backfill Qwen3 eval_results_for_training.json" | tee -a "$LOG_FILE"
echo "Started: $(date) | Host: $(hostname)"         | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

$PYTHON - 2>&1 <<'PY' | tee -a "$LOG_FILE"
import json
from pathlib import Path

# Slim schema (matches Llama eval_results_for_training.json shape).
KEEP_KEYS = [
    "evaluator_abstantion", "evaluator_hallucination",
    "abstantion", "halu_test_res",
    "total_count", "accurate_count", "hallu_count", "refusal_count",
    "correct_rate", "halu_Rate", "refusal_Rate",
]

for ds in ("popqa", "sciq", "searchqa"):
    src = Path(f"output/{ds}_train/Qwen3-8B/eval_results.json")
    dst = Path(f"output/{ds}_train/Qwen3-8B/eval_results_for_training.json")
    if dst.exists():
        print(f"  {dst}: already exists, skipping.")
        continue
    if not src.exists():
        print(f"  WARN {src}: missing — cannot backfill {dst}")
        continue
    d = json.load(open(src))
    out = {k: d[k] for k in KEEP_KEYS if k in d}
    # Sanity: halu_test_res is what compute_subsets reads.
    n = len(out.get("halu_test_res", []))
    print(f"  {dst}: writing slim schema ({len(out)} keys, {n} rows)")
    with open(dst, "w") as f:
        json.dump(out, f)
PY

echo                                                 | tee -a "$LOG_FILE"
echo "--- Re-running Phase 0 (compute_subsets.py) ---" | tee -a "$LOG_FILE"
$PYTHON scripts/compute_subsets.py 2>&1 | tee -a "$LOG_FILE"

echo                                                 | tee -a "$LOG_FILE"
echo "--- Verifying subset files exist ---"          | tee -a "$LOG_FILE"
MISSING=0
for ds in popqa sciq searchqa; do
    f="output/sep_subset_${ds}_Qwen3-8B_seed42.json"
    if [[ -f "$f" ]]; then
        echo "  OK: $f"                              | tee -a "$LOG_FILE"
    else
        echo "  MISSING: $f"                         | tee -a "$LOG_FILE"
        MISSING=$((MISSING+1))
    fi
done

echo                                                 | tee -a "$LOG_FILE"
echo "DONE: $(date) missing=$MISSING"                | tee -a "$LOG_FILE"
exit $MISSING
