# Draft Headline AUROC Table

Canonical numbers for §5 main results. Source: `results_table.csv` (memmap + sampling categories only).

\* = incomplete: N/5 seeds complete. All other trained cells are 5/5 seeds.

MMLU has no sampling results (multiple-choice; sampling-based methods not applicable).

---

## Llama-3.1-8B-Instruct

### Baseline (memmap trained) — AUROC

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
|---|---|---|---|---|---|---|
| LogProb (seq) | — | — | 0.595\*(1/5) | — | — | — |
| Token Entropy | — | — | 0.589\*(1/5) | — | — | — |
| Linear Probe | 0.818 | 0.746 | 0.783 | 0.862 | 0.698 | 0.739 |
| SAPLMA | 0.689 | 0.611 | 0.640 | 0.721 | 0.578 | 0.609 |
| LLMsKnow Probe | 0.828 | 0.722 | 0.801 | 0.870 | 0.709 | 0.741 |
| ICR Probe | — | 0.494 | 0.561 | 0.617 | — | — |
| ACT-ViT | 0.849 | 0.768 | 0.664\*(2/5) | 0.791 | 0.763 | 0.815\*(3/5) |
| **Contrastive+Recon (ours)** | **0.851** | **0.755** | **0.811** | **0.877** | **0.769** | **0.806** |

### Sampling — AUROC

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
|---|---|---|---|---|---|---|
| SE (length-norm) | 0.648 | 0.552 | — | 0.655 | 0.561 | 0.573 |
| SE (semantic) | 0.496 | 0.519 | — | 0.490 | 0.514 | 0.501 |
| SelfCheckGPT-NLI | 0.605 | 0.563 | — | 0.682 | 0.573 | 0.594 |
| SelfCheckGPT-BERT | — | — | — | — | — | — |
| SelfCheckGPT-ngram | 0.622 | 0.577 | — | 0.644 | 0.520 | 0.522 |

---

## Qwen3-8B

### Baseline (memmap trained) — AUROC

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
|---|---|---|---|---|---|---|
| LogProb (seq) | — | — | — | — | — | — |
| Token Entropy | — | — | — | — | — | — |
| Linear Probe | 0.864 | 0.809 | 0.828 | 0.913 | 0.738 | 0.816 |
| SAPLMA | 0.783 | 0.710 | 0.684 | 0.812 | 0.627 | 0.706 |
| LLMsKnow Probe | 0.855 | 0.732 | 0.792 | 0.904 | 0.713 | 0.742 |
| ICR Probe | — | 0.467 | 0.529 | 0.756 | — | — |
| ACT-ViT | 0.881\*(3/5) | 0.842 | 0.675\*(2/5) | 0.830 | 0.787 | 0.838\*(2/5) |
| **Contrastive+Recon (ours)** | **0.875** | **0.824** | **0.833** | **0.921** | **0.816** | **0.840** |

### Sampling — AUROC

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
|---|---|---|---|---|---|---|
| SE (length-norm) | 0.735 | 0.659 | — | 0.780 | 0.579 | 0.622 |
| SE (semantic) | 0.573 | 0.551 | — | 0.556 | 0.523 | 0.504 |
| SelfCheckGPT-NLI | 0.681 | 0.681 | — | 0.818 | 0.595 | 0.688 |
| SelfCheckGPT-BERT | — | — | — | — | — | — |
| SelfCheckGPT-ngram | 0.690 | 0.601 | — | 0.791 | 0.567 | 0.642 |

---

### Scorer notes

Trained methods: `contrastive_logprob_recon` → `knn_auroc`; `logprob_baseline` → `seq_logprob_auroc`; `token_entropy` → `mean_entropy_auroc`; all probe methods → `auroc`. Means reported across 5 seeds; stdev available in `results_table.csv`.

*Generated from `results_table.csv` via `scripts/results_table.py`. Re-run to refresh.*  
*Last updated: 2026-05-20*
