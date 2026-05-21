# Predictions Gap Report

Generated: 2026-05-21 11:01 UTC

Source of truth: `results/results_table.csv` and `results/transfer_matrix_table.csv` (produced by `scripts/results_table.py` — re-run that first if numbers feel stale). Local cache: `results/preds` (synced by `scripts/pull_predictions.py`).

Each cell is rated on two axes:

- **Summary AUROC** — the headline metric is present in `results_table.csv` (i.e. upstream training + eval finished as of the last `results_table.py` run).
- **Predictions** — the per-sample prediction file is in `results/preds/`, available for offline downstream analysis.

The headline `contrastive_logprob_recon` emits per-sample distance scores in `predictions.csv` (one column: `score_halu`, the kNN distance). Its ablation variants (`*_b0`..`*_b9`, `*_c0`..`*_c3`, `*_cd1`..`*_cd3`, `*_d2a`, `*_d2b`) skip per-sample output and report `eval_metrics.json` only — those cells are marked `n/a (distance)`.

## §1 Training Baselines (kind=training)

### Complete (66)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|
| hotpotqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| hotpotqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| hotpotqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| hotpotqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| hotpotqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| hotpotqa | Qwen3-8B | act_vit | 5/5 | 5/5 |
| hotpotqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| hotpotqa | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| hotpotqa | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| hotpotqa | Qwen3-8B | saplma | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| mmlu | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| mmlu | Qwen3-8B | act_vit | 5/5 | 5/5 |
| mmlu | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| mmlu | Qwen3-8B | icr_probe | 5/5 | 5/5 |
| mmlu | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| mmlu | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| mmlu | Qwen3-8B | saplma | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| natural_questions | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | act_vit | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | icr_probe | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| natural_questions | Qwen3-8B | saplma | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| popqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| popqa | Qwen3-8B | act_vit | 5/5 | 5/5 |
| popqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| popqa | Qwen3-8B | icr_probe | 5/5 | 5/5 |
| popqa | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| popqa | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| popqa | Qwen3-8B | saplma | 5/5 | 5/5 |
| sciq | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| sciq | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| sciq | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| sciq | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| sciq | Qwen3-8B | act_vit | 5/5 | 5/5 |
| sciq | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| sciq | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| sciq | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| sciq | Qwen3-8B | saplma | 5/5 | 5/5 |
| searchqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 5/5 |
| searchqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 5/5 |
| searchqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 5/5 |
| searchqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 5/5 |
| searchqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 5/5 |
| searchqa | Qwen3-8B | act_vit | 5/5 | 5/5 |
| searchqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 5/5 |
| searchqa | Qwen3-8B | linear_probe | 5/5 | 5/5 |
| searchqa | Qwen3-8B | llmsknow_probe | 5/5 | 5/5 |
| searchqa | Qwen3-8B | saplma | 5/5 | 5/5 |

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §2 Ablations (kind=ablation)

### Complete (0)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §3 Sampling Baselines (kind=sampling)

### Complete (60)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | present |
| hotpotqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | present |
| hotpotqa | Llama-3.1-8B-Instruct | se_discrete | complete | present |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | present |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | present |
| hotpotqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | present |
| hotpotqa | Qwen3-8B | se_semantic_entropy | complete | present |
| hotpotqa | Qwen3-8B | se_length_normalized | complete | present |
| hotpotqa | Qwen3-8B | se_discrete | complete | present |
| hotpotqa | Qwen3-8B | selfcheck_nli | complete | present |
| hotpotqa | Qwen3-8B | selfcheck_ngram | complete | present |
| hotpotqa | Qwen3-8B | selfcheck_bertscore | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | se_length_normalized | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | se_discrete | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_nli | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | present |
| natural_questions | Qwen3-8B | se_semantic_entropy | complete | present |
| natural_questions | Qwen3-8B | se_length_normalized | complete | present |
| natural_questions | Qwen3-8B | se_discrete | complete | present |
| natural_questions | Qwen3-8B | selfcheck_nli | complete | present |
| natural_questions | Qwen3-8B | selfcheck_ngram | complete | present |
| natural_questions | Qwen3-8B | selfcheck_bertscore | complete | present |
| popqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | present |
| popqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | present |
| popqa | Llama-3.1-8B-Instruct | se_discrete | complete | present |
| popqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | present |
| popqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | present |
| popqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | present |
| popqa | Qwen3-8B | se_semantic_entropy | complete | present |
| popqa | Qwen3-8B | se_length_normalized | complete | present |
| popqa | Qwen3-8B | se_discrete | complete | present |
| popqa | Qwen3-8B | selfcheck_nli | complete | present |
| popqa | Qwen3-8B | selfcheck_ngram | complete | present |
| popqa | Qwen3-8B | selfcheck_bertscore | complete | present |
| sciq | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | present |
| sciq | Llama-3.1-8B-Instruct | se_length_normalized | complete | present |
| sciq | Llama-3.1-8B-Instruct | se_discrete | complete | present |
| sciq | Llama-3.1-8B-Instruct | selfcheck_nli | complete | present |
| sciq | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | present |
| sciq | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | present |
| sciq | Qwen3-8B | se_semantic_entropy | complete | present |
| sciq | Qwen3-8B | se_length_normalized | complete | present |
| sciq | Qwen3-8B | se_discrete | complete | present |
| sciq | Qwen3-8B | selfcheck_nli | complete | present |
| sciq | Qwen3-8B | selfcheck_ngram | complete | present |
| sciq | Qwen3-8B | selfcheck_bertscore | complete | present |
| searchqa | Llama-3.1-8B-Instruct | se_semantic_entropy | complete | present |
| searchqa | Llama-3.1-8B-Instruct | se_length_normalized | complete | present |
| searchqa | Llama-3.1-8B-Instruct | se_discrete | complete | present |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_nli | complete | present |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_ngram | complete | present |
| searchqa | Llama-3.1-8B-Instruct | selfcheck_bertscore | complete | present |
| searchqa | Qwen3-8B | se_semantic_entropy | complete | present |
| searchqa | Qwen3-8B | se_length_normalized | complete | present |
| searchqa | Qwen3-8B | se_discrete | complete | present |
| searchqa | Qwen3-8B | selfcheck_nli | complete | present |
| searchqa | Qwen3-8B | selfcheck_ngram | complete | present |
| searchqa | Qwen3-8B | selfcheck_bertscore | complete | present |

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §4 P(True) (kind=p_true)

### Complete (12)

| Dataset | Model | Method | Summary AUROC | Predictions file |
|---------|-------|--------|---------------|------------------|
| hotpotqa | Llama-3.1-8B-Instruct | p_true | complete | present |
| hotpotqa | Qwen3-8B | p_true | complete | present |
| mmlu | Llama-3.1-8B-Instruct | p_true | complete | present |
| mmlu | Qwen3-8B | p_true | complete | present |
| natural_questions | Llama-3.1-8B-Instruct | p_true | complete | present |
| natural_questions | Qwen3-8B | p_true | complete | present |
| popqa | Llama-3.1-8B-Instruct | p_true | complete | present |
| popqa | Qwen3-8B | p_true | complete | present |
| sciq | Llama-3.1-8B-Instruct | p_true | complete | present |
| sciq | Qwen3-8B | p_true | complete | present |
| searchqa | Llama-3.1-8B-Instruct | p_true | complete | present |
| searchqa | Qwen3-8B | p_true | complete | present |

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions

None — per-sample preds pulled for every Summary-AUROC-complete cell.

## §6 Transfer Matrix

Expected grid: 2 models × 4 methods × 6 src × 6 tgt × 5 seeds = **1440 cells**.

Sources are gated on baseline checkpoints existing — missing source-dataset cells may reflect incomplete training rather than a missing transfer evaluation.

### Coverage by method × model

| Method | Llama (180) | Qwen3 (180) |
|--------|---------|---------|
| contrastive_logprob_recon | 180/180 ✓ | 180/180 ✓ |
| saplma | 180/180 ✓ | 180/180 ✓ |
| llmsknow_probe | 180/180 ✓ | 180/180 ✓ |
| act_vit | 134/180 | 119/180 |

### Missing cells (107 of 1440)

| Model | Method | Source | Target | Seed |
|-------|--------|--------|--------|------|
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | mmlu | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | nq | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | popqa | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | sciq | 4 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 0 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 1 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 2 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 3 |
| Llama-3.1-8B-Instruct | act_vit | sciq | searchqa | 4 |
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
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 4 |
| Qwen3-8B | act_vit | hotpotqa | nq | 4 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 4 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 4 |
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
| §1 Training | 0 | 0 |
| §2 Ablations | 0 | 0 |
| §3 Sampling | 0 | 0 |
| §4 P(True) | 0 | 0 |
| §6 Transfer matrix | 107 | — |
| **Total** | **107** | **0** |

Descoped (not counted): §5 SEP (kind=sep); legacy non-memmap training runs (`runs/baseline_comparison_{ds}/` without `_memmap` suffix).
