#!/bin/bash
# Driver-level smoketest for the Issue #75 lambda sweep.
# Runs sweep_issue_75_lambda.sh against one dataset (hotpotqa) for 1 epoch
# x 20 steps using the l10_a00 method config. Validates the full path:
# config load -> dataset construction -> train -> eval -> summary.csv write.
#
# Wraps the generic sweep wrapper in fixed env-var overrides so the
# dispatch command can be a plain `bash scripts/sweep_issue_75_smoketest.sh`
# without env-var quoting headaches inside gpu_dispatch.py.
set -eo pipefail

export SWEEP_DATASETS=hotpotqa
export SWEEP_MAX_EPOCHS=1
export SWEEP_STEPS_PER_EPOCH=20
export SWEEP_SEED=0
export SWEEP_NUM_WORKERS=2

exec bash scripts/sweep_issue_75_lambda.sh contrastive_logprob_attn_recon_l10_a00
