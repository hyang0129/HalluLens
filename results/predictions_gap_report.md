# Predictions Gap Report

Generated: 2026-05-21 14:11 UTC

Source of truth: `results/results_table.csv` and `results/transfer_matrix_table.csv` (produced by `scripts/results_table.py` — re-run that first if numbers feel stale). Local cache: `results/preds` (synced by `scripts/pull_predictions.py`).

Each cell is rated on two axes:

- **Summary AUROC** — the headline metric is present in `results_table.csv` (i.e. upstream training + eval finished as of the last `results_table.py` run).
- **Predictions** — the per-sample prediction file is in `results/preds/`, available for offline downstream analysis.

The headline `contrastive_logprob_recon` emits per-sample distance scores in `predictions.csv` (one column: `score_halu`, the kNN distance). Its ablation variants (`*_b0`..`*_b9`, `*_c0`..`*_c3`, `*_cd1`..`*_cd3`, `*_d2a`, `*_d2b`) skip per-sample output and report `eval_metrics.json` only — those cells are marked `n/a (distance)`.

## §1 Training Baselines (kind=training)

### Complete (0)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

### Missing Predictions — run `scripts/pull_predictions.py` (66)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|
| hotpotqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| hotpotqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| hotpotqa | Qwen3-8B | saplma | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| mmlu | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| mmlu | Qwen3-8B | act_vit | 5/5 | 0/5 |
| mmlu | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| mmlu | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| mmlu | Qwen3-8B | saplma | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| natural_questions | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | act_vit | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| natural_questions | Qwen3-8B | saplma | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | icr_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| popqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| popqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| popqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| popqa | Qwen3-8B | icr_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| popqa | Qwen3-8B | saplma | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| sciq | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| sciq | Qwen3-8B | act_vit | 5/5 | 0/5 |
| sciq | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| sciq | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| sciq | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| sciq | Qwen3-8B | saplma | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | act_vit | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | contrastive_logprob_recon | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | linear_probe | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | llmsknow_probe | 5/5 | 0/5 |
| searchqa | Llama-3.1-8B-Instruct | saplma | 5/5 | 0/5 |
| searchqa | Qwen3-8B | act_vit | 5/5 | 0/5 |
| searchqa | Qwen3-8B | contrastive_logprob_recon | 5/5 | 0/5 |
| searchqa | Qwen3-8B | linear_probe | 5/5 | 0/5 |
| searchqa | Qwen3-8B | llmsknow_probe | 5/5 | 0/5 |
| searchqa | Qwen3-8B | saplma | 5/5 | 0/5 |

## §2 Ablations (kind=ablation)

### Complete (0)

| Dataset | Model | Method | Summary AUROC seeds | Predictions seeds |
|---------|-------|--------|---------------------|-------------------|

### Missing Summary AUROC

None — every cell has a Summary AUROC in results_table.csv.

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

## §6 Transfer Matrix

Expected grid: 2 models × 4 methods × 6 src × 6 tgt × 5 seeds = **1440 cells**.

Sources are gated on baseline checkpoints existing — missing source-dataset cells may reflect incomplete training rather than a missing transfer evaluation.

### Coverage by method × model

| Method | Llama (180) | Qwen3 (180) |
|--------|---------|---------|
| contrastive_logprob_recon | 180/180 ✓ | 180/180 ✓ |
| saplma | 180/180 ✓ | 180/180 ✓ |
| llmsknow_probe | 180/180 ✓ | 180/180 ✓ |
| act_vit | 180/180 ✓ | 174/180 |

### Missing cells (6 of 1440)

| Model | Method | Source | Target | Seed |
|-------|--------|--------|--------|------|
| Qwen3-8B | act_vit | hotpotqa | hotpotqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | mmlu | 4 |
| Qwen3-8B | act_vit | hotpotqa | nq | 4 |
| Qwen3-8B | act_vit | hotpotqa | popqa | 4 |
| Qwen3-8B | act_vit | hotpotqa | sciq | 4 |
| Qwen3-8B | act_vit | hotpotqa | searchqa | 4 |

## Summary

**Missing Summary AUROC** = `results_table.csv` had no `status=complete` row for this cell as of the last `results_table.py` run (more training/eval needed upstream, or re-run the table if it's stale).

**Missing Predictions** = the cell has a Summary AUROC but its per-sample prediction file isn't in `results/preds/` yet (run `scripts/pull_predictions.py`).

| Section | Missing Summary AUROC | Missing Predictions |
|---------|-----------------------|---------------------|
| §1 Training | 0 | 66 |
| §2 Ablations | 0 | 0 |
| §3 Sampling | 0 | 60 |
| §4 P(True) | 0 | 12 |
| §6 Transfer matrix | 6 | — |
| **Total** | **6** | **138** |

Descoped (not counted): §5 SEP (kind=sep); legacy non-memmap training runs (`runs/baseline_comparison_{ds}/` without `_memmap` suffix).
