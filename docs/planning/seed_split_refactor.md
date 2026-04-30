# Seed / Split Refactor Plan

**Goal:** make each "seed" correspond to a distinct data split (k-fold style) rather than
a distinct weight initialisation on the same split.

---

## Background — what "seed" meant before

Two independent seeds exist in the codebase:

| Name | What it controls |
|------|-----------------|
| `split_seed` | Which samples go to train vs. val inside `ActivationParser` |
| `training_seed` | `seed_everything()` — PyTorch/NumPy/random weight init and batch order |

In all current experiment configs `split_seed` is fixed at **42** and
`training_seeds: [0, 5, 26, 42, 63]` only varies weight init. This means all five
"seeds" see the **same data split**, which is not a proper k-fold evaluation.

The intention was always that each seed = a different held-out partition. This plan
fixes the mismatch.

---

## Desired semantics

| Seed index | `split_seed` (data split) | `training_seed` (weight init) |
|-----------|--------------------------|-------------------------------|
| 0 | **42** (keep existing results) | 0 |
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 3 | 3 | 3 |
| 4 | 4 | 4 |

Seed 0 is intentionally mapped to `split_seed=42` so all already-completed seed-0
runs remain valid and comparable.

---

## What needs to change

### 1. Experiment configs — add `split_seeds`

Add a `split_seeds` list parallel to `training_seeds`. Position `i` in `split_seeds`
is the `split_seed` used when running `training_seeds[i]`.

```json
"training_seeds": [0, 1, 2, 3, 4],
"split_seeds":    [42, 1, 2, 3, 4]
```

**Backward compatibility:** if `split_seeds` is absent the runner falls back to the
single `split_seed` field for all seeds (existing behaviour unchanged).

Configs to update (all Llama + Qwen3 experiments except movies):

- `baseline_comparison_hotpotqa.json`
- `baseline_comparison_mmlu.json`
- `baseline_comparison_nq.json`
- `baseline_comparison_popqa.json`
- `baseline_comparison_sciq.json`
- `baseline_comparison_searchqa.json`
- `baseline_comparison_hotpotqa_qwen3.json`
- `baseline_comparison_nq_qwen3.json`

For Qwen3 experiments currently at seed 0 only: add the full 5-seed sweep once the
data is confirmed complete (pending question — see open items below).

### 2. `run_experiment.py` — restructure the dispatch loop

**Current structure:**
```
create AP(split_seed=42)          # fixed, outside both loops
for method in methods:
    for seed in training_seeds:
        seed_everything(seed)
        run(ap, seed)
```

**New structure:**
```
for i, seed in enumerate(training_seeds):
    actual_split_seed = split_seeds[i] if split_seeds else global_split_seed
    create AP(actual_split_seed)  # new split per seed
    for method in methods:
        seed_everything(seed)
        run(ap, seed)
```

Key points:
- `ActivationParser` is re-created once per seed with the corresponding `split_seed`
- Memmap cache makes re-creation cheap: different `split_seed` → different cache key,
  but the underlying zarr data is only read from disk once
- The test `ActivationParser` (unified-format datasets) always uses
  `split_strategy="none"` and is unaffected by `split_seed` — it is still
  created once (the test set is constant across all seeds)

### 3. Non-learned methods

`token_entropy` and `logprob_baseline` have no training seed — they write to a
seedless output directory and currently set `seeds = [None]`. In the new
seed-outer-loop structure they will naturally skip after the first run because
`eval_metrics.json` already exists (resume logic). No special handling needed.

### 4. Rename `training_seeds` values to `[0, 1, 2, 3, 4]`

The arbitrary values `[0, 5, 26, 42, 63]` were chosen when seeds only meant weight
init. Replace with `[0, 1, 2, 3, 4]` — cleaner and the index now matches the split
number, making it obvious which run corresponds to which fold.

---

## What does NOT change

- Seed 0 results already on disk are valid — `split_seed=42` is preserved for index 0
- The test set is always constant for unified-format datasets (separate test zarr,
  `split_strategy="none"`) — split_seed only affects the train/val partition within
  the training zarr
- Legacy flat datasets (`movies`) are excluded from the multi-seed sweep entirely
  since they have no separate train zarr

---

## Resolved design decisions

1. **Weight init coupling:** `seed_everything(seed)` is still called with the
   training seed, so weight init varies alongside the split. This adds a small amount
   of noise but produces a more realistic variance estimate — sampling over both data
   partitioning and initialisation is closer to real deployment variance.

2. **Qwen3 5-seed sweep:** existing seed-0 results are valid and slide into the new
   setup automatically (seed 0 → split_seed 42 is the preserved mapping). Expand
   Qwen3 experiments to the full `[0,1,2,3,4]` sweep; the runner will skip seed 0
   via the resume check and only run seeds 1–4.

---

## Order of operations

1. Answer the two open items above
2. Update `run_experiment.py` loop structure
3. Update all 8 experiment configs
4. Run smoke-test: one method, two seeds, verify different train/val sizes in log
5. Queue full sweep
