#!/bin/bash
#SBATCH --job-name=pregen_profile
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=pregen_profile_%j.out
#SBATCH --error=pregen_profile_%j.err
#
# Issue #161 -- profiling run for the streaming pre-generation baseline pass
# (baseline 2 difference-of-means, baseline 7 first-token entropy).
#
# CPU-ONLY. No GPU requested, so this does not consume GPU allocation and is
# outside the guarded Jupyter caps.
#
# This job has TWO jobs:
#   1. Answer whether `grace` (betagg*) nodes can see /mnt/home at all. The
#      alpha cluster mounts it over NFS from 10.0.4.75:/home; whether the beta
#      segment does was never confirmed. preflight() exits non-zero if not.
#   2. Measure real streaming throughput so the full run can be sized.
#
# Resources are deliberately small: the accumulators are ~2 MB, so this is
# I/O bound, not memory bound. 64G is generous. RAM only matters for the
# logistic probe's design matrix (~27 GB for all 33 layers of the largest
# dataset) -- and note SLURM does NOT pool memory across nodes, so asking for
# two nodes gives two separate pools, not one large address space.
#
# Submit (needs explicit approval per CLAUDE.md -- this is job submission):
#   ssh empire-ai 'cd ~/LLM_research/HalluLens && sbatch scripts/dispatch/profile_pregen_grace.sbatch'
#
# To test the cpu partition instead, change --partition=grace to --partition=cpu
# (alphacpu01: 96 cores, 931 GB, same cluster as the login node, so filesystem
# visibility is near-certain there).

set -euo pipefail

PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
REPO=/mnt/home/hyang1/LLM_research/HalluLens
CAPTURE="$REPO/shared/icr_capture/sciq_test_Llama-3.1-8B-Instruct"

# Run from wherever this script lives (scratch copy or repo) so the remote
# checkout is never made dirty by a profiling run. Only PYTHONPATH points at
# the repo, for activation_research.memmap_activation_parser.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$HERE"

echo "=== node ==="
hostname; nproc; free -g | head -2
echo "=== filesystem visibility ==="
ls -d "$CAPTURE" && echo "capture dir READABLE" || echo "capture dir NOT VISIBLE"
df -hT "$REPO" 2>/dev/null | tail -2
ls -d "$REPO/activation_research" >/dev/null && echo "repo READABLE"

echo "=== profile: 2000 rows (SciQ test, Llama) ==="
PYTHONPATH="$REPO" "$PYTHON" "$HERE/profile_pregen_baselines.py" \
    --capture-dir "$CAPTURE" \
    --split test --limit 2000 \
    --out-dir "$OUT"

echo "=== profile: full split, same capture ==="
PYTHONPATH="$REPO" "$PYTHON" "$HERE/profile_pregen_baselines.py" \
    --capture-dir "$CAPTURE" \
    --split test --limit -1 \
    --out-dir "$OUT"

echo "=== done ==="
