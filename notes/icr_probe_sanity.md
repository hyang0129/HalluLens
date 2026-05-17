# ICR Probe Sanity Checks (Plan §E — Post-Capture)

Template to fill in after the `shared/icr_capture/` cells for HotpotQA test
+ train (both Llama-3.1-8B and Qwen3-8B) complete on Empire AI.

## What we're checking — and what we're NOT

The paper's headline ~0.84 AUROC on HotpotQA is on **Gemma-2**. We are on
Llama-3.1-8B-Instruct and Qwen3-8B — different model family, different
tokenizer, different chat template, different residual stream geometry.
**There is no published number for our models to match.** Earlier drafts of
this spec (and the original Issue #70 body) framed the gate as "within ~0.03
of paper's 0.84"; that target was always cross-model and conceptually
wobbly. We've redefined the gate to what's actually meaningful.

**We are NOT trying to:**
- Reproduce the paper's 0.84 AUROC on Llama / Qwen (impossible; the paper
  reports Gemma-2 numbers).
- Match upstream `compute_icr_score` byte-for-byte (a known top_p deviation
  is documented in `.agent-work/issue_70_plan.md` §A.2 and intentionally
  deferred unless this gate flags it).

**We ARE checking that:**
1. The score formula is implemented correctly and produces finite, non-trivial
   values that vary across layers.
2. The score has predictive signal for hallucination on our data — single-layer
   AUROC clearly above chance.
3. The trained probe is competitive with other baselines on the same model +
   dataset (head-to-head SAPLMA / contrastive / linear_probe on the same
   Llama outputs is the comparison that matters for our paper).

If all three pass, ship Phase 1 and document as a faithful reimplementation
of the method, *not* a reproduction of the Gemma-2 result.

---

## Gate 0 — capture directories exist and are populated

```bash
# On Empire AI login node (alpha1.empire-ai.org via ssh empire-ai):
for d in \
  shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000 \
  shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \
  shared/icr_capture/hotpotqa_train_Qwen3-8B_0-50000 \
  shared/icr_capture/hotpotqa_test_Qwen3-8B
do
  echo "=== $d ==="
  ls "$d" 2>/dev/null
  wc -l "$d/meta.jsonl" 2>/dev/null
done
```

Each directory must contain `config.json`, `meta.jsonl`, `icr_scores.npy`,
`response_activations.npy`, `response_attention.npy`, `prompt_activations.npy`,
`response_len.npy`, `prompt_len.npy`. `meta.jsonl` line count tells you how
many samples actually committed.

Status: [ ] all four cells complete  /  [ ] still draining

---

## Check 1 — score sanity (per-layer distribution is reasonable)

Verify `icr_scores.npy` contains finite, non-degenerate values across layers.

```python
import numpy as np
scores = np.load("shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000/icr_scores.npy")
print("shape:        ", scores.shape)              # (N, L), e.g. (50000, 32)
print("dtype:        ", scores.dtype)              # float32
print("any nan:      ", np.isnan(scores).any())    # MUST be False
print("any inf:      ", np.isinf(scores).any())    # MUST be False
print("global range: ", scores.min(), scores.max())  # JSD-bounded: [0, ~0.7]
print("per-layer mean:", scores.mean(axis=0))      # should vary across layers, not all zero or all identical
print("per-layer std: ", scores.std(axis=0))       # should be > 0 at most layers
```

**Pass conditions:**
- [ ] No NaN, no Inf
- [ ] Global range fits inside `[0, ~1.0]` (JSD is bounded by ln(2) ≈ 0.693)
- [ ] Per-layer mean varies across layers (not a flat horizontal line)
- [ ] At least 80% of layers have std > 1e-4 (score actually varies sample-to-sample)

**Fail mode — investigate before training:** all scores ≈ 0 (probably a
top_p-collapsed-to-1 issue) or all scores ≈ some constant (probably a
z-score-then-softmax issue with degenerate inputs).

Fill in:
- shape:
- nan / inf:
- global range:
- per-layer mean range (min over layers, max over layers):
- per-layer std range (min over layers, max over layers):

---

## Check 2 — train-set discriminability (per-layer single-feature AUROC)

Verify each layer's ICR score has *some* predictive power on hallucination
labels, **without** training the probe yet. This is the "is there signal at all"
check.

```python
import numpy as np, json
from pathlib import Path
from sklearn.metrics import roc_auc_score

cap = Path("shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000")
scores = np.load(cap / "icr_scores.npy")
meta = [json.loads(l) for l in (cap / "meta.jsonl").read_text().splitlines() if l]
labels = np.array([int(r["hallucinated"]) for r in meta])
sample_indices = np.array([r["sample_index"] for r in meta])
scores_valid = scores[sample_indices]

aurocs = [roc_auc_score(labels, scores_valid[:, l]) for l in range(scores_valid.shape[1])]
best_layer = int(np.argmax(aurocs))
print(f"label balance: pos={labels.mean():.3f}")
print(f"best single-layer AUROC: {max(aurocs):.4f} at layer {best_layer}")
print(f"mean across layers:       {np.mean(aurocs):.4f}")
print(f"n layers above 0.55:      {sum(a > 0.55 for a in aurocs)}/{len(aurocs)}")
```

**Pass conditions:**
- [ ] Best single-layer AUROC > 0.55 (some layer has clear signal)
- [ ] At least 25% of layers have AUROC > 0.55 (signal spans a reasonable band, not a single fluke layer)

**Note on direction:** raw per-layer AUROC < 0.5 means the score is *anti*-correlated
with hallucination — that's still signal, just inverted, and the probe will learn
the sign during training. Report `max(auroc, 1 - auroc)` if you want a direction-free view.

Fill in (Llama):
- Label balance (pos fraction):
- Best single-layer AUROC + layer index:
- Mean across layers:
- Layers above 0.55:

Repeat for Qwen3 cell and fill in:
- Label balance:
- Best single-layer AUROC + layer index:
- Mean across layers:
- Layers above 0.55:

---

## Check 3 — 1K-sample probe smoke (end-to-end pipeline works)

Train the actual probe on a small slice of train data, evaluate on test.

```bash
# From repo root, on a GPU node (REMOTE_GPU):
python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa.json \
    --methods icr_probe \
    --seeds 0 \
    --max-epochs 5
```

(Cap to 1K train samples by either limiting the dataset or using `--max-epochs 5`
with the small batch count — choose what's easiest given current run_experiment.py
options; the goal is just "the pipeline runs end-to-end without erroring," not a
final number.)

**Pass conditions:**
- [ ] Run completes without raised exceptions
- [ ] `runs/.../artifacts/linear_probe_last.pt` is written
- [ ] `runs/.../predictions.csv` is written with `example_id, score_halu, label_halu`
- [ ] `runs/.../eval_metrics.json` has a finite `auroc` field
- [ ] Test AUROC > 0.55 (confirms training produces a probe that's at least
      doing something, even at this tiny budget)

Fill in:
- Test AUROC:
- Run wall time:
- Errors / warnings of note:

---

## Check 4 — competitive vs other baselines on same data (the actual gate that matters)

After all five seeds of Phase 1 complete, compare the probe's mean AUROC to
the other baselines on the same `(model, dataset)`. This is the comparison
that goes in the paper.

```bash
# After Phase 1 finishes, fold ICR Probe into the results table:
python scripts/results_table.py
```

Inspect `output/results_table/results_table.csv` for the HotpotQA × Llama row:
how does `icr_probe` mean AUROC compare to `contrastive`, `saplma`,
`saplma_logprob_recon`, `linear_probe`, `llmsknow_probe`?

**Pass conditions:**
- [ ] ICR probe mean AUROC > 0.55 (better than chance, robust across seeds)
- [ ] ICR probe ranks "reasonably" against the other baselines — defining
      "reasonably" qualitatively: it's not catastrophically worse than every
      other method, and we can write a defensible methods-section sentence
      about its behavior on our regime.

A result where ICR probe is mid-pack, or even one of the weaker baselines, is
fine and publishable. The paper's claim is about our contrastive method;
ICR probe is one comparison point among many.

Fill in:
- ICR probe mean AUROC (Llama HotpotQA, 5 seeds):
- ICR probe mean AUROC (Qwen3 HotpotQA, 5 seeds):
- Rank among baselines on Llama HotpotQA:
- Rank among baselines on Qwen3 HotpotQA:

---

## Check 5 — top_p remediation gate (conditional, only if Check 3 collapses)

If Check 3 test AUROC is ≤ 0.55 (probe at chance), the top_p deviation in
`.agent-work/issue_70_plan.md` §A.2 becomes suspect. **Do NOT trigger this
just because the number is below paper's 0.84 — that's not the right
target.** Trigger only if the probe shows no signal at all.

Remediation path (separate follow-up PR):
1. Write `scripts/recompute_icr_scores.py` that reconstructs the (P+R)-length
   attention vector from `prompt_activations.npy` + a zero-padded version of
   `response_attention.npy` (per upstream's set_other_attn_scores_to_zero).
2. Re-run `compute_icr_score` with upstream-equivalent effective k = int(0.1 × (P+R)).
3. Overwrite `icr_scores.npy` for the affected cells.
4. Re-run Phase 1 with the new scores and report whether AUROC moves.

Status:
- [ ] Not triggered (Check 3 AUROC > 0.55)
- [ ] Triggered — follow-up PR in flight: #___

---

## Phase 2 decision (after Phase 1 results land)

Per spec §14, decide whether to roll out to the remaining 5 datasets
(popqa, mmlu, natural_questions, sciq, searchqa). Decision criteria are
now framed against the redefined gates above, not against paper's 0.84:

- **Roll out to Phase 2 IF:** Phase 1 passes Check 4 on both models
  (probe shows clear signal and is reasonably positioned vs baselines).
- **Stop and diagnose IF:** Phase 1 fails Check 3 (probe at chance) — fix
  via top_p remediation or other diagnosis before scaling out.
- **Document and proceed IF:** Phase 1 passes Check 3 but is in the lower
  half of baselines on both models — that's a finding worth reporting; no
  reason to delay Phase 2 unless we suspect a bug.

Phase 2 decision: [ ] roll out  /  [ ] stop and diagnose  /  [ ] document + proceed

---

## Results summary (fill after Phase 1 launch)

| Model | Dataset | N_train | N_test | Test AUROC seed 0 | Test AUROC mean ± std (5 seeds) | Rank vs baselines |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | HotpotQA | | | | | |
| Qwen3-8B              | HotpotQA | | | | | |

Comparison baselines on same data (for context — fill from `results_table.csv`):

| Baseline | Llama HotpotQA mean AUROC | Qwen3 HotpotQA mean AUROC |
|---|---|---|
| contrastive            | | |
| contrastive_logprob_recon | | |
| linear_probe           | | |
| saplma                 | | |
| saplma_logprob_recon   | | |
| llmsknow_probe         | | |
| token_entropy          | | |
| logprob_baseline       | | |
| **icr_probe** (this PR) | | |

---

## Reference

- Paper: Zhang et al., ACL 2025. arXiv:2507.16488. Reports ~0.84 AUROC on
  HotpotQA with **Gemma-2** (not the models we use).
- Upstream code: github.com/XavierZhang2002/ICR_Probe
- Resolution notes: `notes/icr_probe_paper_notes.md`
- Implementation plan + top_p deviation note: `.agent-work/issue_70_plan.md`
  (gitignored, local-only)
