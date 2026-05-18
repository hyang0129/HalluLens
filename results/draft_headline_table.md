# Draft headline table — mean AUROC across seeds

Source: `results\results_table.json`  
Snapshot generated at: 2026-05-18T08:52:41-0400  
Git: 141dd76e on `feat/issue-79-baseline-retrain-50k-memmap`  

Methods: family #1 (headline) per `results/README.md`. Metric is per-seed test AUROC, aggregated as `mean ± popstdev (n=seeds)` for trained methods; single-run methods (logprob, entropy, P(true)) report the scalar. Empty cells = no complete runs in the table.

### Llama-3.1-8B-Instruct

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
| --- | --- | --- | --- | --- | --- | --- |
| Logprob (mean) | 0.614 | 0.542 | 0.594 | 0.625 | 0.534 | 0.580 |
| Token entropy | — | — | — | — | — | — |
| P(true) | 0.580 | 0.622 | 0.660 | 0.672 | 0.581 | — |
| Linear probe | 0.822 ± 0.001 (n=5) | 0.705 ± 0.081 (n=5) | 0.804 ± 0.008 (n=5) | 0.844 ± 0.002 (n=5) | 0.692 ± 0.006 (n=5) | 0.749 ± 0.001 (n=5) |
| SAPLMA | 0.694 ± 0.003 (n=5) | 0.636 ± 0.007 (n=5) | 0.662 ± 0.006 (n=5) | 0.690 ± 0.003 (n=5) | 0.576 ± 0.015 (n=5) | 0.624 ± 0.003 (n=5) |
| LLMsKnow probe | 0.826 ± 0.002 (n=5) | 0.704 ± 0.010 (n=5) | 0.828 ± 0.019 (n=5) | 0.857 ± 0.004 (n=5) | 0.683 ± 0.021 (n=5) | 0.759 ± 0.002 (n=5) |
| ICR probe | — | — | — | — | — | — |
| **Ours (KNN)** | 0.853 ± 0.002 (n=5) | 0.728 ± 0.080 (n=5) | 0.821 ± 0.008 (n=5) | 0.856 ± 0.002 (n=5) | 0.757 ± 0.008 (n=5) | 0.807 ± 0.003 (n=5) |

### Qwen3-8B

| Method | HotpotQA | NQ | MMLU | PopQA | SciQ | SearchQA |
| --- | --- | --- | --- | --- | --- | --- |
| Logprob (mean) | 0.593 | 0.625 | 0.620 | 0.730 | 0.561 | 0.645 |
| Token entropy | — | — | — | — | — | — |
| P(true) | 0.641 | 0.618 | 0.682 | 0.709 | 0.612 | — |
| Linear probe | 0.753 ± 0.002 (n=5) | 0.766 ± 0.003 (n=5) | 0.774 ± 0.003 (n=5) | 0.908 ± 0.002 (n=5) | 0.722 ± 0.006 (n=5) | 0.822 ± 0.001 (n=5) |
| SAPLMA | 0.699 ± 0.004 (n=5) | 0.662 ± 0.002 (n=5) | 0.635 ± 0.015 (n=5) | 0.806 ± 0.005 (n=5) | 0.618 ± 0.009 (n=5) | 0.722 ± 0.003 (n=5) |
| LLMsKnow probe | 0.745 ± 0.002 (n=5) | 0.689 ± 0.016 (n=5) | 0.787 ± 0.019 (n=5) | 0.839 ± 0.009 (n=5) | 0.671 ± 0.026 (n=5) | 0.759 ± 0.006 (n=5) |
| ICR probe | — | — | — | — | — | — |
| **Ours (KNN)** | 0.753 ± 0.001 (n=5) | 0.789 ± 0.004 (n=5) | 0.825 ± 0.009 (n=5) | 0.917 ± 0.002 (n=5) | 0.807 ± 0.006 (n=5) | 0.842 ± 0.002 (n=5) |

Scorer choices: `contrastive_logprob_recon` → `knn_auroc` (headline per PAPER_ROADMAP §6); `logprob_baseline` → `mean_logprob_auroc`; `token_entropy` → `mean_entropy_auroc`; `p_true` → `p_true_auroc_best`. Other scorers available in `results/results_table.csv`.
