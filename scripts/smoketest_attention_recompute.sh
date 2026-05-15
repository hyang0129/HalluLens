#!/bin/bash
# Smoketest for the ICR-Probe attention recomputation pipeline (PR #71, Issue #69).
# Chains the two validation paths exposed by scripts/recompute_attention.py:
#
#   Phase 1 — --validate-first        (GPU; 4-sample diff vs. full-model forward)
#   Phase 2 — --max-samples N         (GPU; write small attention.zarr, N=20)
#   Phase 3 — AttentionParser readback (CPU; verify shapes + key alignment)
#
# Phase 1 asserts max|A_recomp - A_full| < 1e-3 per block (fp16 tolerance,
# enforced by the script itself — exits 1 on fail).  Phase 2 writes a 20-sample
# attention.zarr under reports/smoketest_attention_recompute/.  Phase 3 reads it
# back via AttentionParser and asserts the expected number of keys and the
# layer-axis size matches the model's num_hidden_layers.
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
MODEL_SLUG=$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]' | tr '-.' '__')

SRC_ZARR="shared/${DATASET}_${MODEL_SLUG}/activations.zarr"

LOG_DIR=reports/smoketest_attention_recompute
SMOKE_OUT_DIR="${LOG_DIR}/${DATASET}_${MODEL_SLUG}"
ATTN_ZARR="${SMOKE_OUT_DIR}/attention.zarr"
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
echo "Attn out:   $ATTN_ZARR"                        | tee -a "$LOG_FILE"
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
# (recompute_attention.py opens AttentionZarrLogger with mode='w' on a fresh
# path; we want to exercise that fresh-write path on every smoketest).
rm -rf "$ATTN_ZARR"

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
        --attention-zarr   "$ATTN_ZARR" \
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
#   Write a small attention.zarr.  --validate-first does NOT write the store
#   (it short-circuits after the check), so this is the first phase that
#   exercises the AttentionZarrLogger path end-to-end.
# ---------------------------------------------------------------------------
rm -rf "$ATTN_ZARR"
/usr/bin/time -v -o "${TIME_FILE}.phase2" \
    $PYTHON scripts/recompute_attention.py \
        --activations-zarr "$SRC_ZARR" \
        --attention-zarr   "$ATTN_ZARR" \
        --model            "$MODEL" \
        --r-max            "$R_MAX" \
        --max-samples      "$SAMPLES" 2>&1 | tee -a "$LOG_FILE"
PHASE2_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE2_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 2 (--max-samples) exited $PHASE2_EXIT"  | tee -a "$LOG_FILE"
    PHASE_FAIL=1
fi

# ---------------------------------------------------------------------------
# Phase 3 — AttentionParser readback (CPU-only)
#   Confirms that what we wrote in Phase 2 can be read back via the public
#   reader API, and that shapes / counts match expectations.
# ---------------------------------------------------------------------------
echo                                                          | tee -a "$LOG_FILE"
echo "--- Phase 3: AttentionParser readback ---"              | tee -a "$LOG_FILE"
$PYTHON - "$ATTN_ZARR" "$SAMPLES" "$R_MAX" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
from pathlib import Path
import torch

attn_path = Path(sys.argv[1])
expected_n = int(sys.argv[2])
expected_rmax = int(sys.argv[3])

# Sanity: config.json present and well-formed
cfg = json.loads((attn_path / "meta" / "config.json").read_text())
print("config.json keys:", sorted(cfg.keys()))
for k in ("source_activations_zarr", "model_name", "num_layers", "num_heads",
          "head_dim", "attention_region", "r_max", "dtype"):
    print(f"  {k}: {cfg.get(k)}")
assert cfg["attention_region"] == "response_to_response", cfg["attention_region"]
assert cfg["r_max"] == expected_rmax, (cfg["r_max"], expected_rmax)

# Reader API
from activation_logging.attention_parser import AttentionParser
ap = AttentionParser(str(attn_path))
keys = ap.list_keys()
print(f"\nAttentionParser: {len(keys)} keys, len()={len(ap)}")
assert len(keys) == expected_n, (len(keys), expected_n)
assert len(ap) == expected_n, (len(ap), expected_n)

# Round-trip the first key through get_attention()
first = ap.get_attention(keys[0])
print(f"\nget_attention({keys[0]!r}):")
for k, v in first.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")
ra = first["response_attn"]
L = cfg["num_layers"]
assert ra.shape == (L, expected_rmax, expected_rmax), (ra.shape, L, expected_rmax)
assert ra.dtype == torch.float32, ra.dtype
assert 1 <= first["response_len"] <= expected_rmax, first["response_len"]
assert first["prompt_len"] >= 0, first["prompt_len"]
print("\nOK: Phase 3 — readback shapes + dtype + bounds all consistent.")
PY
PHASE3_EXIT=${PIPESTATUS[0]}

if [[ "$PHASE3_EXIT" -ne 0 ]]; then
    echo "FAIL: Phase 3 (readback) exited $PHASE3_EXIT"       | tee -a "$LOG_FILE"
    PHASE_FAIL=1
else
    echo "OK:   Phase 3 (readback) passed"                    | tee -a "$LOG_FILE"
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
