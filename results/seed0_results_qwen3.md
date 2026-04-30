# Seed 0 Results — Hallucination Detection AUROC

**Model:** Qwen3-8B  
**Seed:** 0 | **Split seed:** 42 | **Evaluation:** held-out test split

---

## Results

| Dataset | n_test | Logprob | Entropy | Lin. Probe | Multi-Layer | Contr. Cosine | Contr. Mahal. | Contr. KNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NQ | 4,155 | 0.625 | 0.646 | 0.770 | 0.500 | 0.779 | 0.779 | **0.794** |

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
