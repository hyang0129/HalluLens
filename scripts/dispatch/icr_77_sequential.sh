#!/bin/bash
# Sequential dispatch of ICR probe across all 12 (dataset x model) cells.
# Priority order: largest first so we get a runtime signal early.
#
# Note: run_experiment.py --seeds takes a comma-separated string,
# NOT space-separated. The earlier wrapper passed `0 1 2 3 4` and
# argparse rejected the trailing positionals; see PR adding this file.
set -uo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

CELLS=(
  searchqa
  searchqa_qwen3
  hotpotqa
  hotpotqa_qwen3
  mmlu
  mmlu_qwen3
  nq
  nq_qwen3
  popqa
  popqa_qwen3
  sciq
  sciq_qwen3
)

ok=0
fail=0
echo "icr_77 start: $(date)  host=$(hostname)"
for cell in "${CELLS[@]}"; do
  echo "===================================================="
  echo "icr_77: starting $cell at $(date)"
  echo "===================================================="
  cfg="configs/experiments/baseline_comparison_${cell}.json"
  if [ ! -f "$cfg" ]; then
    echo "icr_77: MISSING CONFIG $cfg"
    fail=$((fail+1))
    continue
  fi
  if "$PYTHON" scripts/run_experiment.py --experiment "$cfg" --methods icr_probe --seeds 0,1,2,3,4; then
    echo "icr_77: $cell OK at $(date)"
    ok=$((ok+1))
  else
    rc=$?
    echo "icr_77: $cell FAIL rc=$rc at $(date)"
    fail=$((fail+1))
  fi
done
echo "===================================================="
echo "icr_77 end: $(date)  ok=$ok  fail=$fail  total=${#CELLS[@]}"
echo "===================================================="
