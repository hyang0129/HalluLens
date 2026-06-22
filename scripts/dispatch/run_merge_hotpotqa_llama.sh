#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python
echo "=== merge hotpotqa Llama-3.1-8B-Instruct: 0-50000 + 50000-90447 ==="
$PYTHON scripts/merge_icr_captures.py \
  --a shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000 \
  --b shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_50000-90447 \
  --out shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-90447_merged \
  --skip prompt_activations.npy icr_scores.npy prompt_token_ids.npy
echo "=== DONE ==="
