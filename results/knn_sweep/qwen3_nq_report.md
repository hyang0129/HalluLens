# KNN Sweep Report — qwen3_nq

**Generated:** 2026-05-03T09:46:01
**Dump dir:** `/mnt/home/hyang1/LLM_research/HalluLens/runs/baseline_comparison_nq_qwen3/nq_qwen3/contrastive_logprob_recon/seed_0`
**Elapsed:** 571s

## Dataset
| Split | N | Halu | Clean | Halu rate |
|-------|---|------|-------|-----------|
| train | 13293 | 10321 | 2972 | 77.6% |
| val | 3324 | 2581 | 743 | 77.6% |
| test | 4155 | 3185 | 970 | 76.7% |

### Split overlap (prompt_hash)
- train/val: OK
- train/test: OK
- val/test: OK

**Layers dumped:** [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
**Original eval layers:** 22, 26

## Best overall
- **AUROC:** 0.7948
- Strategy: `concat`  Subset: `concat_original_22_26`
- Reference set: `val`  k=63  metric=`euclidean`

**Original config** (mean [22,26], train ref, euclidean):
- Best AUROC: 0.7941 at k=127

## Best AUROC by aggregation strategy
| Strategy | Best AUROC | Subset | Ref | k | Metric |
|----------|-----------|--------|-----|---|--------|
| concat | 0.7948 | concat_original_22_26 | val | 63 | euclidean |
| concat_normalized | 0.7947 | concat_normalized_original_22_26 | val | 63 | euclidean |
| mean | 0.7941 | mean_original_22_26 | val | 127 | euclidean |
| per_layer_auroc_mean | 0.7820 | per_layer_auroc_mean_even16_l14-29 | val | 127 | euclidean |
| per_layer_dist_mean | 0.7929 | per_layer_dist_mean_even16_l14-29 | train | 127 | euclidean |
| single | 0.7925 | single_l23 | val | 127 | euclidean |

## Single-layer AUROC (best k, val ref, euclidean)
| Layer | Best AUROC |
|-------|-----------|
| 14 | 0.7596 |
| 15 | 0.7691 |
| 16 | 0.7749 |
| 17 | 0.7805 |
| 18 | 0.7846 |
| 19 | 0.7874 |
| 20 | 0.7885 |
| 21 | 0.7889 |
| 22 | 0.7920 ← original |
| 23 | 0.7925 |
| 24 | 0.7909 |
| 25 | 0.7858 |
| 26 | 0.7874 ← original |
| 27 | 0.7786 |
| 28 | 0.7785 |
| 29 | 0.7723 |

## Effect of layer count (mean, evenly-spaced, val ref, euclidean)
| N layers | Best AUROC |
|---------|-----------|
| 2 | 0.7775 |
| 4 | 0.7888 |
| 8 | 0.7897 |

## Reference set effect (best strategy, euclidean, best k)
| Ref set | Best AUROC |
|---------|-----------|
| val | 0.7948 |
| train | 0.7946 |
| train_val | 0.7946 |

## Memorization check: train vs val reference (mean [22,26], euclidean)
| Ref set | Best AUROC |
|---------|-----------|
| val | 0.7941 |
| train | 0.7941 |
| train_val | 0.7940 |

Train − val AUROC gap: **-0.0000**

## k sensitivity (best strategy, val ref, euclidean)
Subset: `concat_original_22_26`
| k | AUROC |
|---|-------|
| 1 | 0.7798 |
| 3 | 0.7910 |
| 5 | 0.7930 |
| 9 | 0.7939 |
| 15 | 0.7942 |
| 31 | 0.7946 |
| 63 | 0.7948 |
| 127 | 0.7947 |

## Distance metric comparison (best strategy, val ref, best k)
| Metric | Best AUROC |
|--------|-----------|
| euclidean | 0.7948 |
| cosine | 0.7947 |

## Recommendations
1. **Best layer subset:** `concat_original_22_26` (2 layers)
2. **Best aggregation:** `concat`
3. **Best reference set:** `val`
4. **Best k:** 63
5. **Best distance metric:** `euclidean`
6. **Best AUROC:** 0.7948
