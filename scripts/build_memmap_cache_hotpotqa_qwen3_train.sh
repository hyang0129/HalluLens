#!/bin/bash
# Build the canonical memmap cache for hotpotqa_train + Qwen3-8B.
# Designed to run co-resident with a training job on the same SLURM allocation.
# Caps thread fan-out via OMP/MKL/etc and pins the process to 8 CPUs (the
# cgroup-allowed range is intersected; passing cores outside the allocation
# is a no-op for taskset, so we use --cpu-list with `taskset -ac` to be safe).
set -eo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

# Cap BLAS / OpenMP / numexpr to 8 threads each. The Python process itself is
# GIL-bound, so this keeps the underlying linalg libs from fanning out.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Pin to 8 CPUs chosen from the cgroup-allowed set at runtime. Hard-coding a
# range would fail when the SLURM cgroup mask doesn't include those cores.
PINNED=$($PYTHON -c "import os; cores = sorted(os.sched_getaffinity(0))[:8]; print(','.join(map(str, cores)))")

echo "============================================"
echo "build_memmap_cache: hotpotqa_train (Qwen3-8B)"
echo "Started: $(date)"
echo "Allowed affinity: $(taskset -p $$ | awk '{print $NF}')"
echo "Pinning to 8 cores: $PINNED"
echo "============================================"

exec taskset -ac "$PINNED" $PYTHON scripts/build_memmap_cache.py \
    --activations-path shared/hotpotqa_train_qwen3_8b/activations.zarr \
    --inference-json   output/hotpotqa_train/Qwen3-8B/generation.jsonl \
    --eval-json        output/hotpotqa_train/Qwen3-8B/eval_results_for_training.json \
    --split-seed 42 \
    --split-strategy two_way \
    --relevant-layers 14-29 \
    --pad-length 63 \
    --include-logprobs \
    --response-logprobs-top-k 20 \
    --chunk-size 512
