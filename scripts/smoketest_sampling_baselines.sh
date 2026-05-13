#!/bin/bash
# Smoketest for the sampling-based baselines pipeline (PR #49, #56).
# Chains all four phases on a single (dataset, model, split) cell at 50 rows:
#
#   Phase 1 — run_sampling_pass.py --smoketest      (GPU; K=10 stochastic samples)
#   Phase 2 — compute_nli_matrix.py                  (GPU; DeBERTa-v2-xlarge-mnli)
#   Phase 3 — compute_se.py                          (CPU; cluster + entropy)
#   Phase 4 — compute_selfcheck.py --no-bertscore    (CPU; NLI + n-gram variants)
#
# Each phase output is asserted to have exactly 50 lines.
# Logs + timing + GPU samples land in reports/smoketest_sampling_baselines/.
#
# Run on a Jupyter GPU node (e.g. alphagpu04-8884):
#   python scripts/gpu_dispatch.py run --jupyter --node alphagpu04-8884 -- \
#       bash scripts/smoketest_sampling_baselines.sh
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
DATASET="${DATASET:-hotpotqa}"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_NAME=$(basename "$MODEL")
EXPECTED_ROWS=50

OUT_DIR="output/sampling_baselines/${DATASET}/${MODEL_NAME}"
SAMPLES="${OUT_DIR}/selfcheck_samples.jsonl"
NLI="${OUT_DIR}/nli_matrix.jsonl"
SE="${OUT_DIR}/se_labels.jsonl"
SELFCHECK="${OUT_DIR}/selfcheck_scores.jsonl"

LOG_DIR=reports/smoketest_sampling_baselines
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/smoketest_${STAMP}.log"
GPU_LOG="$LOG_DIR/gpu_${STAMP}.log"
TIME_FILE="$LOG_DIR/time_${STAMP}.txt"

# Clean previous smoke artifacts so resume logic doesn't pre-fill the smoke output.
# (Phase scripts append on resume; for a fresh smoke we want known empty state.)
rm -f "$SAMPLES" "$NLI" "$SE" "$SELFCHECK"

echo "============================================" | tee "$LOG_FILE"
echo "sampling_baselines smoketest"                  | tee -a "$LOG_FILE"
echo "Started:    $(date)"                            | tee -a "$LOG_FILE"
echo "Host:       $(hostname)"                        | tee -a "$LOG_FILE"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)" | tee -a "$LOG_FILE"
echo "Dataset:    $DATASET | Split: $SPLIT"           | tee -a "$LOG_FILE"
echo "Model:      $MODEL"                             | tee -a "$LOG_FILE"
echo "Rows:       $EXPECTED_ROWS"                     | tee -a "$LOG_FILE"
echo "Out dir:    $OUT_DIR"                           | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                          | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Background GPU sampler
(
    while true; do
        ts=$(date +%H:%M:%S)
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
            2>/dev/null | sed "s/^/$ts /" >> "$GPU_LOG" || true
        sleep 5
    done
) &
SAMPLER_PID=$!
trap "kill $SAMPLER_PID 2>/dev/null || true" EXIT

export PYTHONUNBUFFERED=1
PHASE_FAIL=0

run_phase () {
    local label="$1"; shift
    echo                                                        | tee -a "$LOG_FILE"
    echo "--- $label ---"                                       | tee -a "$LOG_FILE"
    echo "+ $*"                                                 | tee -a "$LOG_FILE"
    if ! "$@" 2>&1 | tee -a "$LOG_FILE"; then
        echo "FAIL: $label exited non-zero"                     | tee -a "$LOG_FILE"
        PHASE_FAIL=1
        return 1
    fi
}

assert_rows () {
    local label="$1"; local path="$2"; local expected="$3"
    if [[ ! -f "$path" ]]; then
        echo "FAIL: $label output missing: $path"               | tee -a "$LOG_FILE"
        PHASE_FAIL=1
        return 1
    fi
    local actual
    actual=$(wc -l < "$path")
    if [[ "$actual" -ne "$expected" ]]; then
        echo "FAIL: $label expected $expected lines, got $actual ($path)" | tee -a "$LOG_FILE"
        PHASE_FAIL=1
        return 1
    fi
    echo "OK:   $label has $actual lines ($path)"               | tee -a "$LOG_FILE"
}

/usr/bin/time -v -o "$TIME_FILE" bash -c "
    set -eo pipefail
    cd '$(pwd)'

    $PYTHON scripts/run_sampling_pass.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL' --smoketest 2>&1 | tee -a '$LOG_FILE'

    $PYTHON scripts/compute_nli_matrix.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL' 2>&1 | tee -a '$LOG_FILE'

    $PYTHON scripts/compute_se.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL' 2>&1 | tee -a '$LOG_FILE'

    $PYTHON scripts/compute_selfcheck.py \
        --dataset '$DATASET' --split '$SPLIT' --model '$MODEL' --no-bertscore 2>&1 | tee -a '$LOG_FILE'
" || PHASE_FAIL=1

echo                                                            | tee -a "$LOG_FILE"
echo "--- Row-count assertions ---"                             | tee -a "$LOG_FILE"
assert_rows "Phase 1 sampling"  "$SAMPLES"   "$EXPECTED_ROWS" || true
assert_rows "Phase 2 NLI"       "$NLI"       "$EXPECTED_ROWS" || true
assert_rows "Phase 3 SE"        "$SE"        "$EXPECTED_ROWS" || true
assert_rows "Phase 4 SelfCheck" "$SELFCHECK" "$EXPECTED_ROWS" || true

echo                                                            | tee -a "$LOG_FILE"
echo "--- SE label sanity (first row) ---"                      | tee -a "$LOG_FILE"
$PYTHON - <<PY | tee -a "$LOG_FILE"
import json
with open("$SE") as f:
    rec = json.loads(f.readline())
print("Keys:", sorted(rec.keys()))
for k in ("semantic_entropy", "length_normalized_se", "discrete_se", "n_clusters", "strict_entailment"):
    print(f"  {k}: {rec.get(k)}")
PY

echo                                                            | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "DONE: $(date) phase_fail=$PHASE_FAIL"        | tee -a "$LOG_FILE"
echo "--- time -v output ---"                       | tee -a "$LOG_FILE"
cat "$TIME_FILE"                                    | tee -a "$LOG_FILE"
echo "--- gpu samples (head/tail) ---"              | tee -a "$LOG_FILE"
head -n 5 "$GPU_LOG" 2>/dev/null                    | tee -a "$LOG_FILE" || true
echo "..."                                          | tee -a "$LOG_FILE"
tail -n 10 "$GPU_LOG" 2>/dev/null                   | tee -a "$LOG_FILE" || true
echo "============================================" | tee -a "$LOG_FILE"

exit $PHASE_FAIL
