# Predictions Gap Report

Generated: 2026-05-20 15:32 UTC

Checks per-sample prediction files on Empire AI across all memmap baseline experiments.
Note: `contrastive_logprob_recon` uses distance-based scores (cosine/mahalanobis/knn) stored in `eval_metrics.json` — it does not write `predictions.csv`. This is tracked as a gap.

## Training Baselines (memmap)

### Complete (49 method × dataset × model cells)

| Dataset | Model | Method | Seeds |
|---------|-------|--------|-------|
| hotpotqa | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| hotpotqa | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| hotpotqa | Qwen3-8B | linear_probe | 5/5 ✓ |
| hotpotqa | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| hotpotqa | Qwen3-8B | saplma | 5/5 ✓ |
| mmlu | Llama-3.1-8B-Instruct | icr_probe | 5/5 ✓ |
| mmlu | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| mmlu | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| mmlu | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| mmlu | Qwen3-8B | icr_probe | 5/5 ✓ |
| mmlu | Qwen3-8B | linear_probe | 5/5 ✓ |
| mmlu | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| mmlu | Qwen3-8B | saplma | 5/5 ✓ |
| natural_questions | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| natural_questions | Llama-3.1-8B-Instruct | icr_probe | 5/5 ✓ |
| natural_questions | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| natural_questions | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| natural_questions | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| natural_questions | Qwen3-8B | act_vit | 5/5 ✓ |
| natural_questions | Qwen3-8B | icr_probe | 5/5 ✓ |
| natural_questions | Qwen3-8B | linear_probe | 5/5 ✓ |
| natural_questions | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| natural_questions | Qwen3-8B | saplma | 5/5 ✓ |
| popqa | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| popqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 ✓ |
| popqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| popqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| popqa | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| popqa | Qwen3-8B | act_vit | 5/5 ✓ |
| popqa | Qwen3-8B | icr_probe | 5/5 ✓ |
| popqa | Qwen3-8B | linear_probe | 5/5 ✓ |
| popqa | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| popqa | Qwen3-8B | saplma | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | act_vit | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| sciq | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| sciq | Qwen3-8B | act_vit | 5/5 ✓ |
| sciq | Qwen3-8B | linear_probe | 5/5 ✓ |
| sciq | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| sciq | Qwen3-8B | saplma | 5/5 ✓ |
| searchqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 ✓ |
| searchqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 ✓ |
| searchqa | Llama-3.1-8B-Instruct | saplma | 5/5 ✓ |
| searchqa | Qwen3-8B | linear_probe | 5/5 ✓ |
| searchqa | Qwen3-8B | llmsknow_probe | 5/5 ✓ |
| searchqa | Qwen3-8B | saplma | 5/5 ✓ |

### Stub runs — method dir exists but no seeds executed (2 cells)

| Dataset | Model | Method | Status | Seeds |
|---------|-------|--------|--------|-------|
| mmlu | Llama-3.1-8B-Instruct | logprob_baseline | stub (no seeds) | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | token_entropy | stub (no seeds) | 0/5 |

### Incomplete — partial seed coverage (17 cells)

| Dataset | Model | Method | Gap | Seeds with predictions.csv |
|---------|-------|--------|-----|---------------------------|
| hotpotqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| hotpotqa | Qwen3-8B | act_vit | 3/5 seeds have predictions.csv | 3/5 |
| hotpotqa | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| natural_questions | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | act_vit | 2/5 seeds have predictions.csv | 2/5 |
| mmlu | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| mmlu | Qwen3-8B | act_vit | 3/5 seeds have predictions.csv | 3/5 |
| mmlu | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| popqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| popqa | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| sciq | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | act_vit | 3/5 seeds have predictions.csv | 3/5 |
| searchqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |
| searchqa | Qwen3-8B | act_vit | 3/5 seeds have predictions.csv | 3/5 |
| searchqa | Qwen3-8B | contrastive_logprob_recon | 0/5 seeds have predictions.csv | 0/5 |

### Methods absent from specific datasets

| Dataset | Model | Method | Note |
|---------|-------|--------|------|
| hotpotqa | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| hotpotqa | Qwen3-8B | icr_probe | not present in experiment dir |
| sciq | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| sciq | Qwen3-8B | icr_probe | not present in experiment dir |
| searchqa | Llama-3.1-8B-Instruct | icr_probe | not present in experiment dir |
| searchqa | Qwen3-8B | icr_probe | not present in experiment dir |

## Sampling Baselines

Complete: 30/30 files present.

All sampling files present. No gaps.

## Summary

| Category | Gaps |
|----------|------|
| Training: partial seeds | 17 |
| Training: stub runs (no seeds) | 2 |
| Training: method absent from dataset | 6 |
| Sampling: missing files | 0 |
| **Total gaps** | **25** |
