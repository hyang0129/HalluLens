# Seed 1 Results — Hallucination Detection AUROC

**Model:** Llama-3.1-8B-Instruct  
**Seed:** 1 | **Split seed:** 42 | **Evaluation:** held-out test split

---

## Results

| Dataset | n_test | Logprob | Entropy | Lin. Probe | Multi-Layer | Contr. Cosine | Contr. Mahal. | Contr. KNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

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
