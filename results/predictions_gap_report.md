# Predictions Gap Report

Generated: 2026-05-21 10:30 UTC

Source of truth: `results/results_table.csv` and `results/transfer_matrix_table.csv` (produced by `scripts/results_table.py` — re-run that first if numbers feel stale). Local cache: `results/preds` (synced by `scripts/pull_predictions.py`).

Each cell is rated on two axes:

- **Summary AUROC** — the headline metric is present in `results_table.csv` (i.e. upstream training + eval finished as of the last `results_table.py` run).
- **Predictions** — the per-sample prediction file is in `results/preds/`, available for offline downstream analysis.

`contrastive_logprob_recon` (and its ablation variants) produces only distance-based scores in `eval_metrics.json` — no `predictions.csv` — so its Predictions column is marked `n/a (distance)`.

## §1 Training Baselines (kind=training)

### Complete (12)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|
| hotpotqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| hotpotqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |
| mmlu | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| mmlu | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |
| popqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| popqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |
| searchqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | n/a (distance) |
| searchqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | n/a (distance) |

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions — run `scripts/pull_predictions.py` (68)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|
| hotpotqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | contrastive | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | saplma_logprob_recon | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | contrastive | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | saplma | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | saplma_logprob_recon | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| mmlu | Qwen3-8B | act_vit | 5/5 | 0/5 |
| mmlu | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | saplma | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | act_vit | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | saplma | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | contrastive | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | saplma_logprob_recon | 5/5 | 0/5 |
| popqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| popqa | Qwen3-8B | contrastive | 5/5 | 0/5 |
| popqa | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | saplma | 5/5 | 0/5 |
| popqa | Qwen3-8B | saplma_logprob_recon | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| sciq | Qwen3-8B | act_vit | 5/5 | 0/5 |
| sciq | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| sciq | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| sciq | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| sciq | Qwen3-8B | saplma | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| searchqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| searchqa | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| searchqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| searchqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| searchqa | Qwen3-8B | saplma | 5/5 | 0/5 |

## §2 Ablations (kind=ablation)

### Complete (0)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|

### Missing Summary AUROC — need more training/eval (76)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b0 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b1 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b2 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b3 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b4 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b5 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b6 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b7 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b8 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b9 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c0 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c1 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c2 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c3 | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd1 | 0/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd2 | 0/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd3 | 0/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_d2a | 1/5 | n/a (distance) |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon_d2b | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b0 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b1 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b2 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b3 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b4 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b5 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b6 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b7 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b8 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_b9 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_c0 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_c1 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_c2 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_c3 | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_cd1 | 0/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_cd2 | 0/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_cd3 | 0/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_d2a | 1/5 | n/a (distance) |
| natural_questions | Qwen3-8B | contrastive_logprob_recon_d2b | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b0 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b1 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b2 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b3 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b4 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b5 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b6 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b7 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b8 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_b9 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c0 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c1 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c2 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_c3 | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd1 | 0/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd2 | 0/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_cd3 | 0/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_d2a | 1/5 | n/a (distance) |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon_d2b | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b0 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b1 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b2 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b3 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b4 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b5 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b6 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b7 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b8 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_b9 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_c0 | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_c1 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_c2 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_c3 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_cd1 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_cd2 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_cd3 | 0/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_d2a | 1/5 | n/a (distance) |
| sciq | Qwen3-8B | contrastive_logprob_recon_d2b | 1/5 | n/a (distance) |

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §3 Sampling Baselines (kind=sampling)

### Complete (0)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions — run `scripts/pull_predictions.py` (60)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | missing |
| hotpotqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | missing |
| hotpotqa | Llama-3.1-8B-Instruct | se_discrete | complete | missing |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | missing |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | missing |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | missing |
| hotpotqa | Qwen3-8B | se_semantic_entropy | complete | missing |
| hotpotqa | Qwen3-8B | se_length_normalized | complete | missing |
| hotpotqa | Qwen3-8B | se_discrete | complete | missing |
| hotpotqa | Qwen3-8B | selfcheck_nli | complete | missing |
| hotpotqa | Qwen3-8B | selfcheck_ngram | complete | missing |
| hotpotqa | Qwen3-8B | selfcheck_bertscore | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | se_length_normalized | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | se_discrete | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_nli | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | missing |
| natural_questions | Qwen3-8B | se_semantic_entropy | complete | missing |
| natural_questions | Qwen3-8B | se_length_normalized | complete | missing |
| natural_questions | Qwen3-8B | se_discrete | complete | missing |
| natural_questions | Qwen3-8B | selfcheck_nli | complete | missing |
| natural_questions | Qwen3-8B | selfcheck_ngram | complete | missing |
| natural_questions | Qwen3-8B | selfcheck_bertscore | complete | missing |
| popqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | missing |
| popqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | missing |
| popqa | Llama-3.1-8B-Instruct | se_discrete | complete | missing |
| popqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | missing |
| popqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | missing |
| popqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | missing |
| popqa | Qwen3-8B | se_semantic_entropy | complete | missing |
| popqa | Qwen3-8B | se_length_normalized | complete | missing |
| popqa | Qwen3-8B | se_discrete | complete | missing |
| popqa | Qwen3-8B | selfcheck_nli | complete | missing |
| popqa | Qwen3-8B | selfcheck_ngram | complete | missing |
| popqa | Qwen3-8B | selfcheck_bertscore | complete | missing |
| sciq | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | missing |
| sciq | Llama-3.1-8B-Instruct | se_length_normalized | complete | missing |
| sciq | Llama-3.1-8B-Instruct | se_discrete | complete | missing |
| sciq | Llama-3.1-8B-Instruct | selfcheck_nli | complete | missing |
| sciq | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | missing |
| sciq | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | missing |
| sciq | Qwen3-8B | se_semantic_entropy | complete | missing |
| sciq | Qwen3-8B | se_length_normalized | complete | missing |
| sciq | Qwen3-8B | se_discrete | complete | missing |
| sciq | Qwen3-8B | selfcheck_nli | complete | missing |
| sciq | Qwen3-8B | selfcheck_ngram | complete | missing |
| sciq | Qwen3-8B | selfcheck_bertscore | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | se_discrete | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | missing |
| searchqa | Qwen3-8B | se_semantic_entropy | complete | missing |
| searchqa | Qwen3-8B | se_length_normalized | complete | missing |
| searchqa | Qwen3-8B | se_discrete | complete | missing |
| searchqa | Qwen3-8B | selfcheck_nli | complete | missing |
| searchqa | Qwen3-8B | selfcheck_ngram | complete | missing |
| searchqa | Qwen3-8B | selfcheck_bertscore | complete | missing |

## §4 P(True) (kind=p_true)

### Complete (0)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions — run `scripts/pull_predictions.py` (12)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Llama-3.1-8B-Instruct | p_true | complete | missing |
| hotpotqa | Qwen3-8B | p_true | complete | missing |
| mmlu | Llama-3.1-8B-Instruct | p_true | complete | missing |
| mmlu | Qwen3-8B | p_true | complete | missing |
| natural_questions | Llama-3.1-8B-Instruct | p_true | complete | missing |
| natural_questions | Qwen3-8B | p_true | complete | missing |
| popqa | Llama-3.1-8B-Instruct | p_true | complete | missing |
| popqa | Qwen3-8B | p_true | complete | missing |
| sciq | Llama-3.1-8B-Instruct | p_true | complete | missing |
| sciq | Qwen3-8B | p_true | complete | missing |
| searchqa | Llama-3.1-8B-Instruct | p_true | complete | missing |
| searchqa | Qwen3-8B | p_true | complete | missing |

## §5 SEP (kind=sep)

### Complete (6)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Llama-3.1-8B-Instruct | sep | complete | n/a |
| natural_questions | Llama-3.1-8B-Instruct | sep | complete | n/a |
| popqa | Llama-3.1-8B-Instruct | sep | complete | n/a |
| sciq | Llama-3.1-8B-Instruct | sep | complete | n/a |
| sciq | Qwen3-8B | sep | complete | n/a |
| searchqa | Llama-3.1-8B-Instruct | sep | complete | n/a |

### Missing Summary AUROC — need more training/eval (6)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Qwen3-8B | sep | missing | n/a |
| mmlu | Llama-3.1-8B-Instruct | sep | missing | n/a |
| mmlu | Qwen3-8B | sep | missing | n/a |
| natural_questions | Qwen3-8B | sep | missing | n/a |
| popqa | Qwen3-8B | sep | missing | n/a |
| searchqa | Qwen3-8B | sep | missing | n/a |

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §6 Transfer Matrix

Expected grid: 2 models × 4 methods × 6 src × 6 tgt × 5 seeds = **1440 cells**.

Sources are gated on baseline checkpoints existing — missing source-dataset cells may reflect incomplete training rather than a missing transfer evaluation.

### Coverage by method × model

| Method | Llama (180) | Qwen3 (180) |
|--------|---------|---------|
| contrastive_logprob_recon | 174/180 | 162/180 |
| saplma | 156/180 | 150/180 |
| llmsknow_probe | 156/180 | 150/180 |
| act_vit | 30/180 | 0/180 |

### Missing cells (462 of 1440)

| Model | Method | Source | Target | Seed |
|-------|--------|--------|--------|------|
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | popqa | 3 |
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | sciq | 3 |
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | searchqa | 1 |
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | searchqa | 2 |
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | searchqa | 3 |
| Llama-3.1-8B-Instruct | contrastive_logprob_recon | searchqa | searchqa | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | mmlu | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | mmlu | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | mmlu | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | mmlu | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | nq | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | nq | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | nq | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | nq | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | popqa | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | popqa | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | popqa | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | popqa | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | sciq | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | sciq | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | sciq | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | sciq | 4 |
| Llama-3.1-8B-Instruct | saplma | searchqa | searchqa | 1 |
| Llama-3.1-8B-Instruct | saplma | searchqa | searchqa | 2 |
| Llama-3.1-8B-Instruct | saplma | searchqa | searchqa | 3 |
| Llama-3.1-8B-Instruct | saplma | searchqa | searchqa | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | hotpotqa | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | hotpotqa | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | hotpotqa | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | hotpotqa | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | mmlu | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | mmlu | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | mmlu | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | mmlu | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | nq | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | nq | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | nq | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | nq | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | popqa | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | popqa | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | popqa | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | popqa | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | sciq | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | sciq | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | sciq | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | sciq | 4 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | searchqa | 1 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | searchqa | 2 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | searchqa | 3 |
| Llama-3.1-8B-Instruct | llmsknow_probe | searchqa | searchqa | 4 |
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
| Qwen3-8B | contrastive_logprob_recon | searchqa | nq | 1 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | nq | 2 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | nq | 3 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | popqa | 0 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | popqa | 1 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | popqa | 2 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | popqa | 3 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | popqa | 4 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | sciq | 0 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | sciq | 1 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | sciq | 2 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | sciq | 3 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | sciq | 4 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | searchqa | 0 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | searchqa | 1 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | searchqa | 2 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | searchqa | 3 |
| Qwen3-8B | contrastive_logprob_recon | searchqa | searchqa | 4 |
| Qwen3-8B | saplma | searchqa | hotpotqa | 0 |
| Qwen3-8B | saplma | searchqa | hotpotqa | 1 |
| Qwen3-8B | saplma | searchqa | hotpotqa | 2 |
| Qwen3-8B | saplma | searchqa | hotpotqa | 3 |
| Qwen3-8B | saplma | searchqa | hotpotqa | 4 |
| Qwen3-8B | saplma | searchqa | mmlu | 0 |
| Qwen3-8B | saplma | searchqa | mmlu | 1 |
| Qwen3-8B | saplma | searchqa | mmlu | 2 |
| Qwen3-8B | saplma | searchqa | mmlu | 3 |
| Qwen3-8B | saplma | searchqa | mmlu | 4 |
| Qwen3-8B | saplma | searchqa | nq | 0 |
| Qwen3-8B | saplma | searchqa | nq | 1 |
| Qwen3-8B | saplma | searchqa | nq | 2 |
| Qwen3-8B | saplma | searchqa | nq | 3 |
| Qwen3-8B | saplma | searchqa | nq | 4 |
| Qwen3-8B | saplma | searchqa | popqa | 0 |
| Qwen3-8B | saplma | searchqa | popqa | 1 |
| Qwen3-8B | saplma | searchqa | popqa | 2 |
| Qwen3-8B | saplma | searchqa | popqa | 3 |
| Qwen3-8B | saplma | searchqa | popqa | 4 |
| Qwen3-8B | saplma | searchqa | sciq | 0 |
| Qwen3-8B | saplma | searchqa | sciq | 1 |
| Qwen3-8B | saplma | searchqa | sciq | 2 |
| Qwen3-8B | saplma | searchqa | sciq | 3 |
| Qwen3-8B | saplma | searchqa | sciq | 4 |
| Qwen3-8B | saplma | searchqa | searchqa | 0 |
| Qwen3-8B | saplma | searchqa | searchqa | 1 |
| Qwen3-8B | saplma | searchqa | searchqa | 2 |
| Qwen3-8B | saplma | searchqa | searchqa | 3 |
| Qwen3-8B | saplma | searchqa | searchqa | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | hotpotqa | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | hotpotqa | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | hotpotqa | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | hotpotqa | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | hotpotqa | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | mmlu | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | mmlu | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | mmlu | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | mmlu | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | mmlu | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | nq | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | nq | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | nq | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | nq | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | nq | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | popqa | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | popqa | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | popqa | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | popqa | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | popqa | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | sciq | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | sciq | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | sciq | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | sciq | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | sciq | 4 |
| Qwen3-8B | llmsknow_probe | searchqa | searchqa | 0 |
| Qwen3-8B | llmsknow_probe | searchqa | searchqa | 1 |
| Qwen3-8B | llmsknow_probe | searchqa | searchqa | 2 |
| Qwen3-8B | llmsknow_probe | searchqa | searchqa | 3 |
| Qwen3-8B | llmsknow_probe | searchqa | searchqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 0 |
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

## Summary

**Missing Summary AUROC** = `results_table.csv` had no `status=complete` row for this cell as of the last `results_table.py` run (more training/eval needed upstream, or re-run the table if it's stale).

**Missing Predictions** = the cell has a Summary AUROC but its per-sample prediction file isn't in `results/preds/` yet (run `scripts/pull_predictions.py`).

| Section | Missing Summary AUROC | Missing Predictions |
|---------|-----------------------|---------------------|
| §1 Training | 0 | 68 |
| §2 Ablations | 76 | 0 |
| §3 Sampling | 0 | 60 |
| §4 P(True) | 0 | 12 |
| §5 SEP | 6 | 0 |
| §6 Transfer matrix | 462 | — |
| **Total** | **544** | **140** |
