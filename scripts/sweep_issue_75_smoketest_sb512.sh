#!/bin/bash
# Timing smoketest A: sub_batch_size=512 (= batch_size), num_workers=2.
# Compares per-step throughput against the default sub_batch=64 baseline.
# Runs 1 epoch x 20 steps on hotpotqa.
set -eo pipefail

export SWEEP_DATASETS=hotpotqa
export SWEEP_MAX_EPOCHS=1
export SWEEP_STEPS_PER_EPOCH=20
export SWEEP_SEED=0
export SWEEP_NUM_WORKERS=2
export SWEEP_SUB_BATCH_SIZE=512

exec bash scripts/sweep_issue_75_lambda.sh contrastive_logprob_attn_recon_l10_a00
