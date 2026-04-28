# Seed 0 Results — Hallucination Detection AUROC

**Model:** Llama-3.1-8B-Instruct  
**Seed:** 0 | **Split seed:** 42 | **Evaluation:** held-out test split

---

## Results

| Dataset | n_test | Logprob | Entropy | Lin. Probe | Multi-Layer | Contr. Cosine | Contr. Mahal. | Contr. KNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 7,405 | 0.614 | — | 0.821 | 0.822 | 0.835 | 0.846 | **0.850** |
| NQ | 4,155 | 0.540 | — | 0.745 | 0.731 | 0.729 | 0.769 | **0.774** |
| MMLU | 9,940 | 0.594 | 0.554 | 0.808 | 0.760 | 0.762 | 0.789 | **0.820** |
| Movies | 1,502 | 0.612 | 0.618 | **0.772** | 0.738 | 0.704 | 0.714 | 0.733 |
| PopQA | 2,532 | 0.625 | 0.718 | 0.847 | 0.828 | 0.824 | 0.811 | **0.860** |
| SciQ | 777 | 0.534 | 0.527 | 0.694 | 0.673 | 0.723 | 0.751 | **0.754** |
| SearchQA | 42,761 | 0.580 | 0.595 | 0.751 | — | 0.779 | **0.808** | 0.807 |

Bold = best per dataset. `—` = not yet complete or failed. Contr. = Contrastive+Logprob recon (SimCLR + logprob aux loss); scorer variant in parentheses.

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
