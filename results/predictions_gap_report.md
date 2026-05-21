# Predictions Gap Report

Generated: 2026-05-21 03:11 UTC

Checks per-sample prediction files on Empire AI across all memmap baseline experiments.
Note: `contrastive_logprob_recon` uses distance-based scores (cosine/mahalanobis/knn) stored in `eval_metrics.json` — it does not write `predictions.csv`. This is tracked as a gap.

## Training Baselines (memmap)

### Complete (16 method × dataset × model cells)

| Dataset | Model | Method | Seeds |
|---------|-------|--------|-------|
| hotpotqa | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| popqa | Qwen3-8B | act_vit | 5/5 ✓ |
| popqa | Qwen3-8B | contrastive_logprob_recon | 5/5 ✓ |
| popqa | Qwen3-8B | icr_probe | 5/5 ✓ |
| popqa | Qwen3-8B | linear_probe | 5/5 ✓ |
| popqa | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| popqa | Qwen3-8B | saplma | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |

### Incomplete — partial seed coverage (9 cells)

| Dataset | Model | Method | Gap | Seeds with predictions.csv |
|---------|-------|--------|-----|---------------------------|
| hotpotqa | Qwen3-8B | *(experiment directory missing)* | — | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | *(experiment directory missing)* | — | 0/5 |
| natural_questions | Qwen3-8B | *(experiment directory missing)* | — | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | *(experiment directory missing)* | — | 0/5 |
| mmlu | Qwen3-8B | *(experiment directory missing)* | — | 0/5 |
| popqa | Llama-3.1-8B-Instruct | *(experiment directory missing)* | — | 0/5 |
| sciq | Qwen3-8B | *(experiment directory missing)* | — | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | *(experiment directory missing)* | — | 0/5 |
| searchqa | Qwen3-8B | *(experiment directory missing)* | — | 0/5 |

### Methods absent from specific datasets

| Dataset | Model | Method | Note |
|---------|-------|--------|------|
| hotpotqa | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| hotpotqa | Qwen3-8B | icr_probe | not present in experiment dir |
| mmlu | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| mmlu | Qwen3-8B | icr_probe | not present in experiment dir |
| natural_questions | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| natural_questions | Qwen3-8B | icr_probe | not present in experiment dir |
| popqa | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| sciq | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| sciq | Qwen3-8B | icr_probe | not present in experiment dir |
| searchqa | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| searchqa | Qwen3-8B | icr_probe | not present in experiment dir |

## §6 Transfer Matrix

Expected grid: 2 models × 4 methods × 6 src × 6 tgt × 5 seeds = **1440 cells**

Sources are gated on baseline checkpoints existing — missing source-dataset cells may reflect incomplete training rather than a missing transfer evaluation.

### Coverage by method × model

| Method | Llama (180 cells) | Qwen3 (180 cells) |
|--------|-------------------|-------------------|
| contrastive_logprob_recon | 180/180 ✓ | 180/180 ✓ |
| saplma | 180/180 ✓ | 180/180 ✓ |
| llmsknow_probe | 180/180 ✓ | 180/180 ✓ |
| act_vit | 30/180 | 1/180 |

### Missing cells (329 of 1440)

| Model | Method | Source | Target | Seed |
|-------|--------|--------|--------|------|
| Llama-3.1-8B-Instruct | act_vit | mmlu | hotpotqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | mmlu | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | mmlu | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | mmlu | 4 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | nq | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | nq | 4 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | popqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | sciq | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | mmlu | searchqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | hotpotqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | mmlu | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | mmlu | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | mmlu | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | nq | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | nq | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | popqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | sciq | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | nq | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | nq | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | nq | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | nq | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | nq | searchqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | hotpotqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | mmlu | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | mmlu | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | mmlu | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | nq | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | nq | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | popqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | sciq | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | popqa | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | popqa | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | popqa | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | popqa | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | popqa | searchqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | hotpotqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | hotpotqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | mmlu | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | mmlu | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | mmlu | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | nq | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | nq | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | popqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | sciq | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | searchqa | searchqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 1 |
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 2 |
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 3 |
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 0 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 1 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 2 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 3 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 4 |
| Qwen3-8B | act_vit | hotpotqa | nq | 0 |
| Qwen3-8B | act_vit | hotpotqa | nq | 1 |
| Qwen3-8B | act_vit | hotpotqa | nq | 2 |
| Qwen3-8B | act_vit | hotpotqa | nq | 3 |
| Qwen3-8B | act_vit | hotpotqa | nq | 4 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 0 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 1 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 2 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 3 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 0 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 1 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 2 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 3 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 4 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 0 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 1 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 2 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 3 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 4 |
| Qwen3-8B | act_vit | mmlu | hotpotqa | 0 |
| Qwen3-8B | act_vit | mmlu | hotpotqa | 1 |
| Qwen3-8B | act_vit | mmlu | hotpotqa | 2 |
| Qwen3-8B | act_vit | mmlu | hotpotqa | 3 |
| Qwen3-8B | act_vit | mmlu | hotpotqa | 4 |
| Qwen3-8B | act_vit | mmlu | mmlu | 0 |
| Qwen3-8B | act_vit | mmlu | mmlu | 1 |
| Qwen3-8B | act_vit | mmlu | mmlu | 2 |
| Qwen3-8B | act_vit | mmlu | mmlu | 3 |
| Qwen3-8B | act_vit | mmlu | mmlu | 4 |
| Qwen3-8B | act_vit | mmlu | nq | 0 |
| Qwen3-8B | act_vit | mmlu | nq | 1 |
| Qwen3-8B | act_vit | mmlu | nq | 2 |
| Qwen3-8B | act_vit | mmlu | nq | 3 |
| Qwen3-8B | act_vit | mmlu | nq | 4 |
| Qwen3-8B | act_vit | mmlu | popqa | 0 |
| Qwen3-8B | act_vit | mmlu | popqa | 1 |
| Qwen3-8B | act_vit | mmlu | popqa | 2 |
| Qwen3-8B | act_vit | mmlu | popqa | 3 |
| Qwen3-8B | act_vit | mmlu | popqa | 4 |
| Qwen3-8B | act_vit | mmlu | sciq | 0 |
| Qwen3-8B | act_vit | mmlu | sciq | 1 |
| Qwen3-8B | act_vit | mmlu | sciq | 2 |
| Qwen3-8B | act_vit | mmlu | sciq | 3 |
| Qwen3-8B | act_vit | mmlu | sciq | 4 |
| Qwen3-8B | act_vit | mmlu | searchqa | 0 |
| Qwen3-8B | act_vit | mmlu | searchqa | 1 |
| Qwen3-8B | act_vit | mmlu | searchqa | 2 |
| Qwen3-8B | act_vit | mmlu | searchqa | 3 |
| Qwen3-8B | act_vit | mmlu | searchqa | 4 |
| Qwen3-8B | act_vit | nq | hotpotqa | 0 |
| Qwen3-8B | act_vit | nq | hotpotqa | 1 |
| Qwen3-8B | act_vit | nq | hotpotqa | 2 |
| Qwen3-8B | act_vit | nq | hotpotqa | 3 |
| Qwen3-8B | act_vit | nq | hotpotqa | 4 |
| Qwen3-8B | act_vit | nq | mmlu | 0 |
| Qwen3-8B | act_vit | nq | mmlu | 1 |
| Qwen3-8B | act_vit | nq | mmlu | 2 |
| Qwen3-8B | act_vit | nq | mmlu | 3 |
| Qwen3-8B | act_vit | nq | mmlu | 4 |
| Qwen3-8B | act_vit | nq | nq | 0 |
| Qwen3-8B | act_vit | nq | nq | 1 |
| Qwen3-8B | act_vit | nq | nq | 2 |
| Qwen3-8B | act_vit | nq | nq | 3 |
| Qwen3-8B | act_vit | nq | nq | 4 |
| Qwen3-8B | act_vit | nq | popqa | 0 |
| Qwen3-8B | act_vit | nq | popqa | 1 |
| Qwen3-8B | act_vit | nq | popqa | 2 |
| Qwen3-8B | act_vit | nq | popqa | 3 |
| Qwen3-8B | act_vit | nq | popqa | 4 |
| Qwen3-8B | act_vit | nq | sciq | 0 |
| Qwen3-8B | act_vit | nq | sciq | 1 |
| Qwen3-8B | act_vit | nq | sciq | 2 |
| Qwen3-8B | act_vit | nq | sciq | 3 |
| Qwen3-8B | act_vit | nq | sciq | 4 |
| Qwen3-8B | act_vit | nq | searchqa | 0 |
| Qwen3-8B | act_vit | nq | searchqa | 1 |
| Qwen3-8B | act_vit | nq | searchqa | 2 |
| Qwen3-8B | act_vit | nq | searchqa | 3 |
| Qwen3-8B | act_vit | nq | searchqa | 4 |
| Qwen3-8B | act_vit | popqa | hotpotqa | 0 |
| Qwen3-8B | act_vit | popqa | hotpotqa | 1 |
| Qwen3-8B | act_vit | popqa | hotpotqa | 2 |
| Qwen3-8B | act_vit | popqa | hotpotqa | 3 |
| Qwen3-8B | act_vit | popqa | hotpotqa | 4 |
| Qwen3-8B | act_vit | popqa | mmlu | 0 |
| Qwen3-8B | act_vit | popqa | mmlu | 1 |
| Qwen3-8B | act_vit | popqa | mmlu | 2 |
| Qwen3-8B | act_vit | popqa | mmlu | 3 |
| Qwen3-8B | act_vit | popqa | mmlu | 4 |
| Qwen3-8B | act_vit | popqa | nq | 0 |
| Qwen3-8B | act_vit | popqa | nq | 1 |
| Qwen3-8B | act_vit | popqa | nq | 2 |
| Qwen3-8B | act_vit | popqa | nq | 3 |
| Qwen3-8B | act_vit | popqa | nq | 4 |
| Qwen3-8B | act_vit | popqa | popqa | 0 |
| Qwen3-8B | act_vit | popqa | popqa | 1 |
| Qwen3-8B | act_vit | popqa | popqa | 2 |
| Qwen3-8B | act_vit | popqa | popqa | 3 |
| Qwen3-8B | act_vit | popqa | popqa | 4 |
| Qwen3-8B | act_vit | popqa | sciq | 0 |
| Qwen3-8B | act_vit | popqa | sciq | 1 |
| Qwen3-8B | act_vit | popqa | sciq | 2 |
| Qwen3-8B | act_vit | popqa | sciq | 3 |
| Qwen3-8B | act_vit | popqa | sciq | 4 |
| Qwen3-8B | act_vit | popqa | searchqa | 0 |
| Qwen3-8B | act_vit | popqa | searchqa | 1 |
| Qwen3-8B | act_vit | popqa | searchqa | 2 |
| Qwen3-8B | act_vit | popqa | searchqa | 3 |
| Qwen3-8B | act_vit | popqa | searchqa | 4 |
| Qwen3-8B | act_vit | sciq | hotpotqa | 0 |
| Qwen3-8B | act_vit | sciq | hotpotqa | 1 |
| Qwen3-8B | act_vit | sciq | hotpotqa | 2 |
| Qwen3-8B | act_vit | sciq | hotpotqa | 3 |
| Qwen3-8B | act_vit | sciq | hotpotqa | 4 |
| Qwen3-8B | act_vit | sciq | mmlu | 0 |
| Qwen3-8B | act_vit | sciq | mmlu | 1 |
| Qwen3-8B | act_vit | sciq | mmlu | 2 |
| Qwen3-8B | act_vit | sciq | mmlu | 3 |
| Qwen3-8B | act_vit | sciq | mmlu | 4 |
| Qwen3-8B | act_vit | sciq | nq | 0 |
| Qwen3-8B | act_vit | sciq | nq | 1 |
| Qwen3-8B | act_vit | sciq | nq | 2 |
| Qwen3-8B | act_vit | sciq | nq | 3 |
| Qwen3-8B | act_vit | sciq | nq | 4 |
| Qwen3-8B | act_vit | sciq | popqa | 0 |
| Qwen3-8B | act_vit | sciq | popqa | 1 |
| Qwen3-8B | act_vit | sciq | popqa | 2 |
| Qwen3-8B | act_vit | sciq | popqa | 3 |
| Qwen3-8B | act_vit | sciq | popqa | 4 |
| Qwen3-8B | act_vit | sciq | sciq | 0 |
| Qwen3-8B | act_vit | sciq | sciq | 1 |
| Qwen3-8B | act_vit | sciq | sciq | 2 |
| Qwen3-8B | act_vit | sciq | sciq | 3 |
| Qwen3-8B | act_vit | sciq | sciq | 4 |
| Qwen3-8B | act_vit | sciq | searchqa | 0 |
| Qwen3-8B | act_vit | sciq | searchqa | 1 |
| Qwen3-8B | act_vit | sciq | searchqa | 2 |
| Qwen3-8B | act_vit | sciq | searchqa | 3 |
| Qwen3-8B | act_vit | sciq | searchqa | 4 |
| Qwen3-8B | act_vit | searchqa | hotpotqa | 0 |
| Qwen3-8B | act_vit | searchqa | hotpotqa | 1 |
| Qwen3-8B | act_vit | searchqa | hotpotqa | 2 |
| Qwen3-8B | act_vit | searchqa | hotpotqa | 3 |
| Qwen3-8B | act_vit | searchqa | hotpotqa | 4 |
| Qwen3-8B | act_vit | searchqa | mmlu | 0 |
| Qwen3-8B | act_vit | searchqa | mmlu | 1 |
| Qwen3-8B | act_vit | searchqa | mmlu | 2 |
| Qwen3-8B | act_vit | searchqa | mmlu | 3 |
| Qwen3-8B | act_vit | searchqa | mmlu | 4 |
| Qwen3-8B | act_vit | searchqa | nq | 0 |
| Qwen3-8B | act_vit | searchqa | nq | 1 |
| Qwen3-8B | act_vit | searchqa | nq | 2 |
| Qwen3-8B | act_vit | searchqa | nq | 3 |
| Qwen3-8B | act_vit | searchqa | nq | 4 |
| Qwen3-8B | act_vit | searchqa | popqa | 0 |
| Qwen3-8B | act_vit | searchqa | popqa | 1 |
| Qwen3-8B | act_vit | searchqa | popqa | 2 |
| Qwen3-8B | act_vit | searchqa | popqa | 3 |
| Qwen3-8B | act_vit | searchqa | popqa | 4 |
| Qwen3-8B | act_vit | searchqa | sciq | 0 |
| Qwen3-8B | act_vit | searchqa | sciq | 1 |
| Qwen3-8B | act_vit | searchqa | sciq | 2 |
| Qwen3-8B | act_vit | searchqa | sciq | 3 |
| Qwen3-8B | act_vit | searchqa | sciq | 4 |
| Qwen3-8B | act_vit | searchqa | searchqa | 0 |
| Qwen3-8B | act_vit | searchqa | searchqa | 1 |
| Qwen3-8B | act_vit | searchqa | searchqa | 2 |
| Qwen3-8B | act_vit | searchqa | searchqa | 3 |
| Qwen3-8B | act_vit | searchqa | searchqa | 4 |

## Sampling Baselines

Complete: 0/30 files present.

| Dataset | Model | File | Status |
|---------|-------|------|--------|
| hotpotqa | Llama-3.1-8B-Instruct | se_labels.jsonl | missing |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_scores.jsonl | missing |
| hotpotqa | Llama-3.1-8B-Instruct | ptrue.jsonl | missing |
| hotpotqa | Qwen3-8B | se_labels.jsonl | missing |
| hotpotqa | Qwen3-8B | selfcheck_scores.jsonl | missing |
| hotpotqa | Qwen3-8B | ptrue.jsonl | missing |
| natural_questions | Llama-3.1-8B-Instruct | se_labels.jsonl | missing |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_scores.jsonl | missing |
| natural_questions | Llama-3.1-8B-Instruct | ptrue.jsonl | missing |
| natural_questions | Qwen3-8B | se_labels.jsonl | missing |
| natural_questions | Qwen3-8B | selfcheck_scores.jsonl | missing |
| natural_questions | Qwen3-8B | ptrue.jsonl | missing |
| popqa | Llama-3.1-8B-Instruct | se_labels.jsonl | missing |
| popqa | Llama-3.1-8B-Instruct | selfcheck_scores.jsonl | missing |
| popqa | Llama-3.1-8B-Instruct | ptrue.jsonl | missing |
| popqa | Qwen3-8B | se_labels.jsonl | missing |
| popqa | Qwen3-8B | selfcheck_scores.jsonl | missing |
| popqa | Qwen3-8B | ptrue.jsonl | missing |
| sciq | Llama-3.1-8B-Instruct | se_labels.jsonl | missing |
| sciq | Llama-3.1-8B-Instruct | selfcheck_scores.jsonl | missing |
| sciq | Llama-3.1-8B-Instruct | ptrue.jsonl | missing |
| sciq | Qwen3-8B | se_labels.jsonl | missing |
| sciq | Qwen3-8B | selfcheck_scores.jsonl | missing |
| sciq | Qwen3-8B | ptrue.jsonl | missing |
| searchqa | Llama-3.1-8B-Instruct | se_labels.jsonl | missing |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_scores.jsonl | missing |
| searchqa | Llama-3.1-8B-Instruct | ptrue.jsonl | missing |
| searchqa | Qwen3-8B | se_labels.jsonl | missing |
| searchqa | Qwen3-8B | selfcheck_scores.jsonl | missing |
| searchqa | Qwen3-8B | ptrue.jsonl | missing |

## Summary

| Category | Gaps |
|----------|------|
| Training: partial seeds | 9 |
| Training: stub runs (no seeds) | 0 |
| Training: method absent from dataset | 11 |
| Sampling: missing files | 30 |
| Transfer matrix: missing cells | 329 |
| **Total gaps** | **379** |
