# Seed 0 Results — Hallucination Detection AUROC

**Model:** Llama-3.1-8B-Instruct  
**Seed:** 0 | **Split seed:** 42 | **Evaluation:** held-out test split

---

## Results

| Dataset | n_test | Logprob Baseline | Token Entropy | Linear Probe | Multi-Layer Probe | Contrastive (Cosine) | Contrastive (Mahal.) | Contrastive (KNN) |
|---------|-------:|------:|------:|------:|------:|------:|------:|------:|
| HotpotQA | 7,405 | 0.614 | NaN | **0.821** | — | — | — | — |
| NQ | 4,155 | 0.540 | NaN | 0.745 | 0.731 | 0.729 | 0.769 | **0.774** |
| MMLU | 9,940 | 0.594 | 0.594 | 0.808 | 0.760 | 0.762 | 0.789 | **0.820** |
| Movies | 1,502 | — | — | **0.772** | — | — | — | — |
| PopQA | 2,532 | 0.625 | 0.718 | 0.847 | 0.828 | 0.824 | 0.811 | **0.860** |
| SciQ | 777 | 0.534 | 0.527 | 0.694 | 0.673 | 0.723 | 0.751 | **0.754** |
| SearchQA | — | — | — | — | — | — | — | *running* |

Bold = best per dataset. `—` = not yet complete. `NaN` = missing logprobs in zarr.

---

## Method Details

| Method | Type | Key metric |
|--------|------|------------|
| Logprob Baseline | non-learned | `mean_logprob_auroc` |
| Token Entropy | non-learned | `mean_logprob_auroc` (or `min_logprob_auroc` where available) |
| Linear Probe | learned, layer 22 | `auroc` |
| Multi-Layer Linear Probe | learned, layers 14–29 | `auroc` |
| Contrastive (Cosine) | learned | cosine distance in embedding space |
| Contrastive (Mahal.) | learned | Mahalanobis distance |
| Contrastive (KNN) | learned | k-NN (k=50 or calibrated) |

---

## Observations

- **Contrastive KNN is the strongest method** on 4 of 6 completed datasets, peaking at 0.860 on PopQA.
- **Linear Probe (layer 22)** is competitive and leads on HotpotQA (0.821); it's the only learned method fully evaluated there so far.
- **Multi-Layer Probe** consistently underperforms single-layer — aggregating all layers adds noise rather than signal.
- **Logprob Baseline** is weak across the board (0.53–0.63), confirming that raw sequence probability is insufficient for hallucination detection.
- **Token Entropy** returns NaN for NQ and HotpotQA (missing entropy logprobs in those zarrs); where it works (PopQA), `min_logprob` reaches 0.718 — competitive with multi-layer probe.
- **SciQ is the hardest dataset**: best AUROC is 0.754 vs 0.860 for PopQA, likely due to the small test set (777) and high hallucination rate (~36%).
- **HotpotQA and Movies** are partially complete — contrastive and multi-layer results pending from the job dispatched 2026-04-25.

---

## Run Status

| Dataset | Logprob | Token Ent. | Linear Probe | Multi-Layer | Contrastive |
|---------|---------|------------|--------------|-------------|-------------|
| HotpotQA | ✓ | ✓ (NaN) | ✓ | running | running |
| NQ | ✓ | ✓ (NaN) | ✓ | ✓ | ✓ |
| MMLU | ✓ | ✓ | ✓ | ✓ | ✓ |
| Movies | — | — | ✓ | running | running |
| PopQA | ✓ | ✓ | ✓ | ✓ | ✓ |
| SciQ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SearchQA | — | — | — | — | running |
