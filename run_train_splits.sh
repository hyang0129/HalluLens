#!/bin/bash
set -e
cd /mnt/home/hyang1/LLM_research/HalluLens
PYTHON=/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python

echo "============================================"
echo "Finishing SearchQA train + MMLU train"
echo "============================================"

# --- SearchQA train (155 remaining) ---
echo ""
echo "=== SearchQA train (finishing last ~155) ==="
mkdir -p output/searchqa_train/Llama-3.1-8B-Instruct
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from tasks.llmsknow.searchqa import run_step
run_step(step='inference', model='meta-llama/Llama-3.1-8B-Instruct',
         split='train', batch_size=8, max_tokens=128, temperature=0.0,
         activations_path='shared/searchqa_train/activations.zarr', resume=True,
         generations_file_path='output/searchqa_train/Llama-3.1-8B-Instruct/generation.jsonl')
run_step(step='eval', model='meta-llama/Llama-3.1-8B-Instruct',
         generations_file_path='output/searchqa_train/Llama-3.1-8B-Instruct/generation.jsonl',
         eval_results_path='output/searchqa_train/Llama-3.1-8B-Instruct/eval_results.json')
"
echo "SearchQA train DONE"

# --- MMLU aux_train (99,842 unfiltered, resuming from 4,248) ---
echo ""
echo "=== MMLU aux_train (99,842 unfiltered, resuming) ==="
mkdir -p output/mmlu_train/Llama-3.1-8B-Instruct
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from tasks.llmsknow.mmlu import run_step
# Pass subjects=[''] to match the empty subject field in auxiliary_train
run_step(step='inference', model='meta-llama/Llama-3.1-8B-Instruct',
         split='auxiliary_train', batch_size=8, max_tokens=128, temperature=0.0,
         activations_path='shared/mmlu_train/activations.zarr', resume=True,
         generations_file_path='output/mmlu_train/Llama-3.1-8B-Instruct/generation.jsonl',
         subjects=[''])
run_step(step='eval', model='meta-llama/Llama-3.1-8B-Instruct',
         generations_file_path='output/mmlu_train/Llama-3.1-8B-Instruct/generation.jsonl',
         eval_results_path='output/mmlu_train/Llama-3.1-8B-Instruct/eval_results.json')
"
echo "MMLU train DONE"

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
