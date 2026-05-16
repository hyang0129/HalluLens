#!/bin/bash
# Smoketest for the ICR-Probe attention recomputation pipeline (PR #71, Issue #69).
# Chains the two validation paths exposed by scripts/recompute_attention.py:
#
#   Phase 1 — --validate-first        (GPU; 4-sample diff vs. full-model forward)
#   Phase 2 — --max-samples N         (GPU; write small attention store, N=20)
#   Phase 3 — Attention store readback (CPU; verify shapes + write-evidence)
#   Phase 4 — icr_scores.npy readback  (CPU; verify shape + no all-zero rows)
#
# Phase 1 asserts max|A_recomp - A_full| < 1e-3 per block (fp16 tolerance,
# enforced by the script itself — exits 1 on fail).  Phase 2 writes a 20-sample
# attention store under reports/smoketest_attention_recompute/.  Phase 3 reads it
# back via np.memmap and asserts the expected number of entries and that at least
# the first written row is non-zero.  Phase 4 loads icr_scores.npy and asserts
# shape (N, num_blocks) and that no written row is all-zero.
#
# Logs + timing + GPU samples land in reports/smoketest_attention_recompute/.
#
# Run on a Jupyter GPU node (the recompute path needs the full HF model loaded
# with attn_implementation='eager'):
#   python scripts/gpu_dispatch.py run --jupyter --node <NODE> -- \
#       bash scripts/smoketest_attention_recompute.sh
#
# Defaults run Llama-3.1-8B on HotpotQA.  Override via env vars to smoke-test
# the Qwen3-8B path or a different dataset:
#   MODEL="Qwen/Qwen3-8B" DATASET="hotpotqa" bash scripts/smoketest_attention_recompute.sh
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
DATASET="${DATASET:-hotpotqa}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SAMPLES="${SAMPLES:-20}"
R_MAX="${R_MAX:-64}"

MODEL_BASENAME=$(basename "$MODEL")
# Slug matches scripts/audit_datasets.py:model_slug — lowercased, '-'/'.' → '_'.
MODEL_SLUG=$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]' | tr -- '-.' '__')

SRC_ZARR="shared/${DATASET}_${MODEL_SLUG}/activations.zarr"

LOG_DIR=reports/smoketest_attention_recompute
SMOKE_OUT_DIR="${LOG_DIR}/${DATASET}_${MODEL_SLUG}"
ATTN_DIR="${SMOKE_OUT_DIR}/attention"
ICR_SCORES="${SMOKE_OUT_DIR}/icr_scores.npy"
mkdir -p "$LOG_DIR" "$SMOKE_OUT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/smoketest_${MODEL_SLUG}_${STAMP}.log"
GPU_LOG="$LOG_DIR/gpu_${MODEL_SLUG}_${STAMP}.log"
TIME_FILE="$LOG_DIR/time_${MODEL_SLUG}_${STAMP}.txt"

echo "============================================" | tee "$LOG_FILE"
echo "attention_recompute smoketest"                | tee -a "$LOG_FILE"
echo "Started:    $(date)"                           | tee -a "$LOG_FILE"
echo "Host:       $(hostname)"                       | tee -a "$LOG_FILE"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)" | tee -a "$LOG_FILE"
echo "Dataset:    $DATASET"                          | tee -a "$LOG_FILE"
echo "Model:      $MODEL  (slug: $MODEL_SLUG)"       | tee -a "$LOG_FILE"
echo "Source:     $SRC_ZARR"                         | tee -a "$LOG_FILE"
echo "Attn out:   $ATTN_DIR"                         | tee -a "$LOG_FILE"
echo "ICR out:    $ICR_SCORES"                       | tee -a "$LOG_FILE"
echo "Samples:    $SAMPLES (Phase 2)"                | tee -a "$LOG_FILE"
echo "R_max:      $R_MAX"                            | tee -a "$LOG_FILE"
echo "Log:        $LOG_FILE"                         | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Source activations.zarr must exist — otherwise no point continuing.
if [[ ! -d "$SRC_ZARR" ]]; then
    echo "FAIL: source activations.zarr missing at $SRC_ZARR"   | tee -a "$LOG_FILE"
    echo "Run inference first (e.g. python scripts/run_with_server.py --task $DATASET --model $MODEL --step inference)" | tee -a "$LOG_FILE"
    exit 1
fi

# Clean prior smoke output so the writer starts from a known empty state
# (recompute_attention.py opens AttentionMemmapWriter with mode='w' on a fresh
# path; we want to exercise that fresh-write path on every smoketest).
rm -rf "$ATTN_DIR" "$ICR_SCORES" "${SMOKE_OUT_DIR}/icr_scores_meta.jsonl"

# Background GPU sampler — useful for spotting GPU starvation during the
# per-sample loop (the recompute pass is CPU-light, GPU-heavy on attn).
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
    echo                                            | tee -a "$LOG_FILE"
    echo "--- $label ---"                           | tee -a "$LOG_FILE"
    echo "+ $*"                                     | tee -a "$LOG_FILE"
    if ! "$@" 2>&1 | tee -a "$LOG_FILE"; then
        echo "FAIL: $label exited non-zero"         | tee -a "$LOG_FILE"
        PHASE_FAIL=1
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Phase 1 — --validate-first
#   4-sample numerical equivalence vs. full-model forward.  recompute_attention.py
#   prints a per-block (sample × block → max_abs_diff) table and exits 1 if any
#   diff >= 1e-3.  We don't need to scrape the table — the exit code is the
#   pass/fail signal.
# ---------------------------------------------------------------------------
/usr/bin/time -v -o "${TIME_FILE}.phase1" \
    $PYTHON scripts/recompute_attention.py \
        --activations-zarr "$SRC_ZARR" \
        --attention-dir    "$ATTN_DIR" \
        --icr-scores-path  "$ICR_SCORES" \
        --model            "$MODEL" \
        --r-max            "$R_MAX" \
        --validate-first 2>&1 | tee -a "$LOG_FILE"
PHASE1_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE1_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 1 (--validate-first) exited $PHASE1_EXIT — some block diff >= 1e-3" | tee -a "$LOG_FILE"
    PHASE_FAIL=1
else
    echo "OK:   Phase 1 (--validate-first) passed (all blocks < 1e-3)"                   | tee -a "$LOG_FILE"
fi

# ---------------------------------------------------------------------------
# Phase 2 — --max-samples N
#   Write a small attention store.  --validate-first does NOT write the store
#   (it short-circuits after the check), so this is the first phase that
#   exercises the AttentionMemmapWriter path end-to-end.
# ---------------------------------------------------------------------------
rm -rf "$ATTN_DIR" "$ICR_SCORES" "${SMOKE_OUT_DIR}/icr_scores_meta.jsonl"
/usr/bin/time -v -o "${TIME_FILE}.phase2" \
    $PYTHON scripts/recompute_attention.py \
        --activations-zarr "$SRC_ZARR" \
        --attention-dir    "$ATTN_DIR" \
        --icr-scores-path  "$ICR_SCORES" \
        --model            "$MODEL" \
        --r-max            "$R_MAX" \
        --max-samples      "$SAMPLES" 2>&1 | tee -a "$LOG_FILE"
PHASE2_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE2_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 2 (--max-samples) exited $PHASE2_EXIT"  | tee -a "$LOG_FILE"
    PHASE_FAIL=1
fi

# ---------------------------------------------------------------------------
# Phase 3 — Attention store readback (CPU-only)
#   Confirms that what we wrote in Phase 2 can be read back via np.memmap,
#   and that shapes / counts match expectations.
# ---------------------------------------------------------------------------
echo                                                          | tee -a "$LOG_FILE"
echo "--- Phase 3: Attention store readback ---"              | tee -a "$LOG_FILE"
$PYTHON - "$ATTN_DIR" "$SAMPLES" "$R_MAX" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from pathlib import Path
import numpy as np

attn_dir = Path(sys.argv[1])
expected_n = int(sys.argv[2])
expected_rmax = int(sys.argv[3])

# config.json at the root of attn_dir (not meta/config.json)
cfg = json.loads((attn_dir / "config.json").read_text())
print("config.json keys:", sorted(cfg.keys()))
for k in ("source_activations_zarr", "model_name", "num_layers", "num_heads",
          "head_dim", "attention_region", "r_max", "dtype", "storage_format"):
    print(f"  {k}: {cfg.get(k)}")
assert cfg["r_max"] == expected_rmax, (cfg["r_max"], expected_rmax)
assert cfg["storage_format"] == "numpy_memmap_v1", cfg["storage_format"]

# meta.jsonl — authoritative record of written samples
meta_lines = [json.loads(line) for line in (attn_dir / "meta.jsonl").read_text().strip().splitlines()]
print(f"\nmeta.jsonl: {len(meta_lines)} sample entries")
assert len(meta_lines) == expected_n, (len(meta_lines), expected_n)

# response_attn.npy as np.memmap (raw binary, no .npy header — shape from config)
n_samples = cfg["n_samples"]
num_layers = cfg["num_layers"]
r_max = cfg["r_max"]
dtype = cfg["dtype"]
mm = np.memmap(attn_dir / "response_attn.npy", dtype=dtype, mode="r",
               shape=(n_samples, num_layers, r_max, r_max))
print(f"response_attn.npy: shape={mm.shape} dtype={mm.dtype}")
# Spot check: the row indexed by the first meta entry should be non-zero
first_idx = meta_lines[0]["sample_index"]
first_row = np.asarray(mm[first_idx])
assert first_row.shape == (num_layers, r_max, r_max)
nonzero_fraction = (first_row != 0).mean()
print(f"  first written row[{first_idx}]: nonzero fraction = {nonzero_fraction:.3f}")
assert nonzero_fraction > 0.0, f"first written row is all zero — write failed"
print("\nOK: Phase 3 — readback shapes + dtype + write-evidence all consistent.")
PY
PHASE3_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE3_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 3 (readback) exited $PHASE3_EXIT"       | tee -a "$LOG_FILE"
    PHASE_FAIL=1
else
    echo "OK:   Phase 3 (readback) passed"                    | tee -a "$LOG_FILE"
fi

# ---------------------------------------------------------------------------
# Phase 4 — icr_scores.npy readback (CPU-only)
#   Confirms shape (N, num_blocks), dtype float32, and that every written
#   sample has at least one non-zero score (all-zero row indicates a
#   score-compute failure).
# ---------------------------------------------------------------------------
echo                                                          | tee -a "$LOG_FILE"
echo "--- Phase 4: icr_scores.npy readback ---"               | tee -a "$LOG_FILE"
$PYTHON - "$ICR_SCORES" "$SAMPLES" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from pathlib import Path
import numpy as np

icr_path = Path(sys.argv[1])
expected_n = int(sys.argv[2])

arr = np.load(icr_path)
print(f"icr_scores.npy: shape={arr.shape} dtype={arr.dtype}")
assert arr.dtype == np.float32, arr.dtype
assert arr.ndim == 2, arr.shape

meta_path = icr_path.parent / (icr_path.stem + "_meta.jsonl")
meta_lines = [json.loads(line) for line in meta_path.read_text().strip().splitlines()]
written_indices = [m["sample_index"] for m in meta_lines]
print(f"icr_scores_meta.jsonl: {len(written_indices)} entries")
assert len(written_indices) == expected_n, (len(written_indices), expected_n)

# Every written sample must have at least one non-zero score
all_zero_count = 0
for idx in written_indices:
    if np.all(arr[idx] == 0):
        all_zero_count += 1
        print(f"  WARN: row {idx} is all-zero")
print(f"\n{all_zero_count}/{len(written_indices)} written samples are all-zero")
assert all_zero_count == 0, f"{all_zero_count} written samples have all-zero ICR scores — compute failed"
print("OK: Phase 4 — icr_scores.npy populated correctly.")
PY
PHASE4_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE4_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 4 (icr_scores readback) exited $PHASE4_EXIT" | tee -a "$LOG_FILE"
    PHASE_FAIL=1
else
    echo "OK:   Phase 4 (icr_scores readback) passed"              | tee -a "$LOG_FILE"
fi

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------
echo                                                            | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "DONE: $(date) phase_fail=$PHASE_FAIL"        | tee -a "$LOG_FILE"

for f in "${TIME_FILE}.phase1" "${TIME_FILE}.phase2"; do
    if [[ -f "$f" ]]; then
        echo "--- $(basename $f) ---"                | tee -a "$LOG_FILE"
        cat "$f"                                     | tee -a "$LOG_FILE"
    fi
done

echo "--- gpu samples (head/tail) ---"              | tee -a "$LOG_FILE"
head -n 5 "$GPU_LOG" 2>/dev/null                    | tee -a "$LOG_FILE" || true
echo "..."                                          | tee -a "$LOG_FILE"
tail -n 10 "$GPU_LOG" 2>/dev/null                   | tee -a "$LOG_FILE" || true
echo "============================================" | tee -a "$LOG_FILE"

exit $PHASE_FAIL
