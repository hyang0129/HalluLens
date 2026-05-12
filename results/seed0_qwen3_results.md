# Seed 0 Results — Hallucination Detection AUROC

**Model:** Qwen3-8B  
**Seed:** 0 | **Split seed:** 42 | **Evaluation:** held-out test split

---

## Results

| Dataset | n_test | Logprob | Entropy | Lin. Probe | Multi-Layer | Contr. Cosine | Contr. Mahal. | Contr. KNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 7,405 | 0.593 | 0.573 | 0.752 | 0.500 | 0.720 | 0.733 | **0.753** |
| MMLU | 2,863 | 0.620 | 0.625 | 0.772 | 0.500 | 0.719 | 0.785 | **0.813** |
| NQ | 4,155 | 0.625 | 0.646 | 0.770 | 0.500 | 0.779 | 0.779 | **0.794** |
| PopQA | 2,750 | 0.730 | 0.784 | 0.906 | 0.500 | 0.891 | 0.892 | **0.916** |
| SciQ | 1,000 | 0.561 | 0.526 | 0.725 | 0.500 | 0.793 | **0.802** | 0.801 |
| SearchQA | 150,763 | 0.650 | 0.638 | 0.815 | 0.500 | 0.819 | 0.836 | **0.845** |

Bold = best per dataset. `—` = not yet complete or failed. Contr. = Contrastive+Logprob recon (SimCLR + logprob aux loss); scorer variant in parentheses. Entropy column reports `mean_entropy_auroc` when available, otherwise falls back to `min_logprob_auroc` (mean entropy is NaN for all Qwen3 runs).

---

## Method Details

| Method | Type | Key metric |
|--------|------|------------|
| Logprob Baseline | non-learned | `mean_logprob_auroc` |
| Token Entropy | non-learned | `mean_entropy_auroc` (best non-NaN) |
| Linear Probe | learned, layer 22 | `auroc` |
| Multi-Layer Linear Probe | learned, layers 14–29 | `auroc` |
| Contrastive+Logprob (Cosine) | learned | contrastive (SimCLR) + logprob recon aux loss; cosine distance scorer |
| Contrastive+Logprob (Mahal.) | learned | same model; Mahalanobis distance scorer |
| Contrastive+Logprob (KNN) | learned | same model; k-NN scorer (calibrated k) |
