# Dataset Sizes Reference

## Summary Table

| Dataset | Project Split | HF Dataset | HF Split | Expected | Observed (Qwen3-8B) | Observed (Llama-3.1-8B-I) |
|---------|--------------|-----------|----------|----------|---------------------|--------------------------|
| hotpotqa | test | hotpot_qa (distractor) | validation | 7,405 | 7,405 ✓ | 7,405 ✓ |
| hotpotqa | train | hotpot_qa (distractor) | train | 90,447 | 90,440 | 77,673 (incomplete) |
| mmlu | test | cais/mmlu (all) | test | ~10,200* | 10,421 | 10,225 |
| mmlu | train | cais/mmlu (all) | auxiliary_train | ~99,800 | 99,842 ✓ | 94,200 |
| movies | test | external CSV | — | 7,657 | 7,856 (+199) | 7,657 ✓ |
| natural_questions | test | external CSV | 80/20 split | ~4,155 | 4,155 ✓ | 4,155 ✓ |
| natural_questions | train | external CSV | 80/20 split | ~16,617 | 16,617 ✓ | 16,617 ✓ |
| popqa | test | akariasai/PopQA | 80/20 split | ~2,854 | 2,854 ✓ | 2,854 ✓ |
| popqa | train | akariasai/PopQA | 80/20 split | ~11,413 | 11,413 ✓ | 11,413 ✓ |
| sciq | test | allenai/sciq | test | 1,000 | 1,000 ✓ | 1,000 ✓ |
| sciq | train | allenai/sciq | train | 11,679 | 11,679 ✓ | 11,679 ✓ |
| searchqa | test | kyunghyuncho/search_qa | **see note** | ? | 151,295 ⚠ | 43,227 ⚠ |
| searchqa | train | kyunghyuncho/search_qa | **see note** | ? | 43,228 | 151,140 ⚠ |

\* MMLU filters to factual subjects only, excluding ~26% of the full 14,079-sample test set.

---

## Notes

### MMLU — factual subjects filter
The task file filters to a `FACTUAL_SUBJECTS` list, dropping reasoning-heavy categories (algebra,
logic, formal reasoning, etc.). This reduces the nominal 14,079-sample test split to ~10,200
samples. The filtering is intentional — document it if quoting MMLU sample counts externally.

### NQ and PopQA — synthetic train/test split
Neither dataset ships with an official train split in the form used here. Both are partitioned
with an 80/20 random split (`split_seed=42`) at load time. Sizes are therefore deterministic
given the seed but not guaranteed stable if the upstream HF dataset changes.

### Movies — minor size drift
Qwen3 observed 7,856 vs the expected/Llama 7,657 (+199 samples, 2.6%). Likely the external CSV
was updated between the two runs. No train split exists for movies.

### SearchQA — split inversion between models ⚠
This is a known quirk of the searchqa task: the project's "test" output dir uses one HF split and
"train" uses the other, but which is which changed between the Llama and Qwen3 runs:

| Model | output/searchqa/ | output/searchqa_train/ |
|-------|-----------------|----------------------|
| Llama-3.1-8B-Instruct | 43,227 | 151,140 |
| Qwen3-8B | 151,295 | 43,228 |

This means the two models were run with **opposite split assignments**. Before training on both,
verify that the correct split (HF "test" ≈ 43k for eval, HF "train" ≈ 99–151k for training) is
used consistently. The `generate_all_qwen3.sh` comments indicate the intent:
- `searchqa` (project test) ← HF `train` split  
- `searchqa_train` (project train) ← HF `test` split  

Llama appears to have been run with the **opposite** convention.

---

## Status as of 2026-04-30

| Model | Complete | Needs attention |
|-------|---------|-----------------|
| Qwen3-8B | 13/13 ✓ | — |
| Llama-3.1-8B-Instruct | 11/13 | hotpotqa_train (incomplete gen), movies (no eval), searchqa split inversion |
