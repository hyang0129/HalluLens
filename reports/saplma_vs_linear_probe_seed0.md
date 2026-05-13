# SAPLMA vs Linear Probe — AUROC Comparison (Seed 0)

**Date:** 2026-05-13  
**Models:** Llama-3.1-8B-Instruct, Qwen3-8B  
**Metric:** AUROC on held-out test split  
**Probe layer:** 22 (same for both methods, selected by experiment config)

---

## Results

### Llama-3.1-8B-Instruct

| Dataset    | Linear Probe | SAPLMA | Δ (SAPLMA − LP) |
|------------|:------------:|:------:|:----------------:|
| mmlu       | 0.8084       | 0.6642 | **−0.1442**      |
| sciq       | 0.6941       | 0.5678 | **−0.1264**      |
| hotpotqa   | 0.8211       | 0.6936 | **−0.1275**      |
| nq         | 0.5431       | 0.6366 | **+0.0935** ✓   |
| searchqa   | 0.7506       | 0.6023 | **−0.1483**      |
| popqa      | 0.8471       | 0.6920 | **−0.1551**      |
| **Mean**   | **0.7441**   | **0.6428** | **−0.1013** |

### Qwen3-8B

| Dataset    | Linear Probe | SAPLMA | Δ (SAPLMA − LP) |
|------------|:------------:|:------:|:----------------:|
| mmlu       | 0.7717       | 0.6189 | **−0.1528**      |
| sciq       | 0.7247       | 0.6105 | **−0.1142**      |
| hotpotqa   | 0.7520       | 0.6984 | **−0.0536**      |
| nq         | 0.7702       | 0.6597 | **−0.1105**      |
| searchqa   | 0.8153       | 0.7029 | **−0.1125**      |
| popqa      | 0.9056       | 0.8037 | **−0.1019**      |
| **Mean**   | **0.7899**   | **0.6824** | **−0.1076** |

---

## Summary

Linear probe outperforms SAPLMA on **11 of 12** task-model pairs (seed 0).

- **Average gap:** ~10 AUROC points across both models.
- **Largest gap:** popqa/Llama (−0.155), mmlu/Qwen3 (−0.153), searchqa/Llama (−0.148).
- **Only exception:** nq/Llama, where SAPLMA edges ahead by +0.094. This may be an artifact of the linear probe's notably low AUROC (0.543 ≈ near-random) on that split rather than a genuine SAPLMA strength.
- **Closest matchup:** hotpotqa/Qwen3 (−0.054), suggesting SAPLMA is most competitive on multi-hop reasoning tasks with Qwen3.

### Interpretation

SAPLMA uses the model's final token logprob as a scalar hallucination signal, whereas the linear probe learns a hyperplane over a full hidden-state vector at layer 22. The consistent gap suggests the hidden-state geometry at that layer carries substantially more discriminative information about hallucination than the model's own output confidence alone.

The nq/Llama anomaly is worth investigating — the linear probe's 0.543 is suspiciously low and could indicate a label distribution or data alignment issue in that split rather than a real SAPLMA advantage.

---

*Single seed (seed=0); multi-seed averages available from the 5-run ensembles.*
