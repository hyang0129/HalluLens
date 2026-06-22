#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
echo "=== merge hotpotqa Qwen3-8B: 0-50000 + 50000-90440 ==="
$PYTHON scripts/merge_icr_captures.py \
  --a shared/icr_capture/hotpotqa_train_Qwen3-8B_0-50000 \
  --b shared/icr_capture/hotpotqa_train_Qwen3-8B_50000-90440 \
  --out shared/icr_capture/hotpotqa_train_Qwen3-8B_0-90440_merged \
  --skip prompt_activations.npy icr_scores.npy prompt_token_ids.npy
echo "=== DONE ==="
