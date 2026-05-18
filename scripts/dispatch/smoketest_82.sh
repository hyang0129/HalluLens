#!/bin/bash
# Smoketest the full-attention recon path on NQ at seed 0.
# Limits to 1 epoch x 5 steps so the run completes in ~2-5 min — verifies
# the new attn_target='full' code path works end-to-end on real data
# before launching the multi-hour direction sweep.
set -uo pipefail
cd /mnt/home/hyang1/LLM_research/HalluLens
echo "smoketest_82 start: $(date)"
SWEEP_DATASETS=nq SWEEP_SEED=0 SWEEP_MAX_EPOCHS=1 SWEEP_STEPS_PER_EPOCH=5 \
    bash scripts/sweep_issue_75_lambda.sh contrastive_logprob_attn_recon_l10_a10_full
rc=$?
echo "smoketest_82 end: $(date)  rc=$rc"
exit $rc
