# ICR Probe Sanity Checks (Plan §E — Post-Capture)

Template to fill in after completing plan §E sanity checks.
These checks are deferred from PR #70 because the test-cell capture
directories on Empire AI were still draining when this PR was submitted.

---

## Gate: capture directories

Before running any check below, confirm both capture cells are complete:

```bash
# On Empire AI login node (alpha1.empire-ai.org via ssh empire-ai):
ls shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000/
ls shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct/
ls shared/icr_capture/hotpotqa_train_Qwen3-8B_0-50000/
ls shared/icr_capture/hotpotqa_test_Qwen3-8B/
# Each directory must contain: config.json meta.jsonl icr_scores.npy
# response_activations.npy response_attention.npy prompt_activations.npy
# response_len.npy prompt_len.npy
```

---

## Check 1 — Single-sample ICR score printout

Verify that `icr_scores.npy` contains finite, non-trivial values.

```python
import numpy as np
scores = np.load("shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000/icr_scores.npy")
print(scores.shape)      # expected: (N, L) e.g. (50000, 32)
print(scores[0])         # per-layer scores for sample 0 — should be non-zero floats
print(scores.mean(axis=0))  # layer-wise mean — should vary across layers
print(scores.std(axis=0))   # layer-wise std — should be > 0
```

Fill in below:
- `scores.shape`:
- `scores[0]` (first 5 values):
- `scores.mean(axis=0)` range (min, max):
- `scores.std(axis=0)` range (min, max):

---

## Check 2 — Train discriminability plot

Verify that ICR scores are predictive of hallucination labels before training.

```python
import numpy as np
import json
from pathlib import Path

cap = Path("shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000")
scores = np.load(cap / "icr_scores.npy")
meta = [json.loads(l) for l in (cap / "meta.jsonl").read_text().splitlines() if l]
labels = np.array([int(r["hallucinated"]) for r in meta])
sample_indices = np.array([r["sample_index"] for r in meta])
scores_valid = scores[sample_indices]

# Layer-wise AUROC
from sklearn.metrics import roc_auc_score
aurocs = [roc_auc_score(labels, scores_valid[:, l]) for l in range(scores_valid.shape[1])]
print(f"Max layer AUROC: {max(aurocs):.4f} at layer {aurocs.index(max(aurocs))}")
print(f"Mean layer AUROC: {sum(aurocs)/len(aurocs):.4f}")
```

Fill in below:
- Max single-layer AUROC:
- Layer with max AUROC:
- Mean across layers:

---

## Check 3 — 1K-sample probe smoke

Train a probe on 1000 train samples to verify the pipeline works end-to-end.

```bash
python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa.json \
    --methods icr_probe \
    --seeds 0 \
    --max-epochs 5
```

Fill in below:
- Test AUROC:
- Checkpoint written to `runs/.../artifacts/linear_probe_last.pt`: yes/no
- `predictions.csv` written: yes/no
- Any errors:

---

## Check 4 — top_p deviation gate (Plan §A.2)

If test AUROC from Check 3 is below 0.81 (paper reports ~0.84 on HotpotQA),
investigate whether the `top_p` deviation (plan §A.2) is the cause.

Remediation path:
1. Write `scripts/recompute_icr_scores.py` that reconstructs the (P+R)-length
   attention vector from `prompt_activations.npy` + `response_attention.npy`.
2. Re-run compute with upstream-equivalent effective k = int(0.1 × (P+R)).
3. Re-train probe and compare AUROC.

Status: [ ] Not triggered (AUROC >= 0.81) / [ ] Triggered — remediation pending

---

## Results summary (fill after Phase 1 launch)

| Model | Split | N_train | N_test | AUROC (seed 0) | AUROC (mean ± std, 5 seeds) |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | HotpotQA | | | | |
| Qwen3-8B | HotpotQA | | | | |

Paper baseline (HotpotQA, Llama): ~0.84 AUROC (Zhang et al. ACL 2025, Table 2).
