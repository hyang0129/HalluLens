# Implementation Spec: Issue #102 — Shared-Trunk Twin-Head Variants

## Context

Issue #99 trains **fully independent** twin compressors and concatenates their embeddings (see `TwinConcatModel` and `run_contrastive_logprob_recon_twin` in `scripts/run_experiment.py:437`). This spec adds two **shared-trunk** variants that test whether dual-loss training over a single trunk preserves the twin-head gain at half the compute.

The branch `feat/102-shared-trunk-twin-heads` is **stacked on `feat/99-twin-heads-simclr-ablation`** — all the #99 infrastructure (`use_labels`, `ignore_label`, `TwinConcatModel`, `run_contrastive_logprob_recon_twin`) is already present. **Reuse it. Do not reimplement it.**

Read [the full issue](https://github.com/hyang0129/HalluLens/issues/102) for the scientific motivation. This spec covers the implementation only.

---

## Architectures to implement

### D1 — Split-output, shared trunk

A single `LogprobReconProgressiveCompressor` with `final_dim=2D`. The output is sliced and each half is fed to its own SupCon loss with the opposite `ignore_label`. Eval uses the full (unsliced) output.

```
z = compressor(x)                       # (B, 2D)
z_A, z_B = z[:, :D], z[:, D:]           # each (B, D)
L_supcon_A = SupCon(z_A, labels, ignore_label=1)
L_supcon_B = SupCon(z_B, labels, ignore_label=0)
L_recon    = recon_loss(decoder(z), logprob_target)
loss       = L_supcon_A + L_supcon_B + λ · L_recon
```

**Eval surface:** full `z` (2D-dim). KNN/Maha only — no cosine.

### D2 — Shared trunk + projection heads, KNN on trunk

A trunk (the existing `LogprobReconProgressiveCompressor` encoder) plus two small projection MLPs. SupCon is computed on the head outputs; eval uses the **trunk** output (heads discarded at eval).

```
z = compressor(x)                       # (B, trunk_dim) -- the eval surface
zA, zB = head_A(z), head_B(z)           # each (B, D), training-only
L_supcon_A = SupCon(zA, labels, ignore_label=1)
L_supcon_B = SupCon(zB, labels, ignore_label=0)
L_recon    = recon_loss(decoder(z), logprob_target)   # recon on TRUNK z
loss       = L_supcon_A + L_supcon_B + λ · L_recon
```

**Eval surface:** trunk `z` (trunk_dim). KNN/Maha + cosine all apply (trunk is a single coordinate system).

---

## Files to modify / create

### 1. `activation_research/model.py` — add two model classes

#### `SharedTrunkSplitOutputCompressor` (D1)

Wraps a single `LogprobReconProgressiveCompressor` whose `final_dim = 2 * D`. Stores `D` (`half_dim`) so the trainer can slice. `forward(x)` returns the full `z` (B, 2D). Helper method `forward_slices(x)` returns `(z, z_A, z_B)` for the trainer; `forward_with_recon(x)` returns `(z, logprob_pred)` for the recon path. The decoder operates on full `z`.

Constructor signature:
```python
SharedTrunkSplitOutputCompressor(
    input_dim: int = 4096,
    half_dim: int = 256,                  # D; total output is 2 * half_dim
    dropout: float = 0.1,
    input_dropout: float = 0.2,
    normalize_input: bool = False,
    recon_seq_len: int = 64,
    recon_hidden_dim: int = 256,
    recon_lambda: float = 1.0,
    logprob_var_threshold: float = 1e-4,
)
```

The class is a thin wrapper around `LogprobReconProgressiveCompressor(final_dim=2*half_dim, ...)`. Expose `recon_loss` by delegating to the inner module so the training loop can reuse it unchanged.

#### `SharedTrunkProjectionHeadCompressor` (D2)

Wraps a `LogprobReconProgressiveCompressor` with `final_dim = trunk_dim`. Adds two `nn.Sequential` heads (each `nn.Linear(trunk_dim, head_hidden_dim) → GELU → nn.Linear(head_hidden_dim, head_dim)`). `forward(x)` returns the trunk `z` (B, trunk_dim) — this is the eval surface. Helper `forward_with_heads(x)` returns `(z, zA, zB)` for the trainer. `forward_with_recon(x)` returns `(z, logprob_pred)` from the inner module.

Constructor signature:
```python
SharedTrunkProjectionHeadCompressor(
    input_dim: int = 4096,
    trunk_dim: int = 512,
    head_dim: int = 256,
    head_hidden_dim: int = 256,
    dropout: float = 0.1,
    input_dropout: float = 0.2,
    normalize_input: bool = False,
    recon_seq_len: int = 64,
    recon_hidden_dim: int = 256,
    recon_lambda: float = 1.0,
    logprob_var_threshold: float = 1e-4,
)
```

Heads are **trained but discarded at eval**. `eval()`-time `forward(x)` must return the trunk `z`, not anything from the heads, so existing eval code (`MultiMetricHallucinationEvaluator`, `TwinConcatModel`'s pattern of calling `model(x)` to get an embedding) works without modification.

### 2. `activation_research/training.py` — add a dual-loss training function

Add `train_contrastive_logprob_recon_dualloss(...)` that mirrors the signature of `train_contrastive_logprob_recon` but accepts:
- `model`: an instance of `SharedTrunkSplitOutputCompressor` **or** `SharedTrunkProjectionHeadCompressor`
- `ignore_labels: tuple[int, int]` — defaults to `(1, 0)` (head A gets `ignore_label=1`, head B gets `ignore_label=0`)

Internally:
1. Call the model's helper (`forward_slices` for D1, `forward_with_heads` for D2) to get `(z, zA, zB)`.
2. Compute `SupCon(zA, labels, ignore_label=ignore_labels[0]) + SupCon(zB, labels, ignore_label=ignore_labels[1])`.
3. Compute `recon_loss(decoder(z), target)` on the **full z** (D1) or the **trunk z** (D2) — in both cases this is what `forward_with_recon` returns from the inner module.
4. Sum: `loss = L_A + L_B + recon_lambda * L_recon`.
5. Everything else (optimizer, scheduler, checkpointing, AMP, grad clip, infinite stream, etc.) must be identical to `train_contrastive_logprob_recon`. **Extract shared logic into a helper if duplication is heavy**; otherwise copy-and-modify is acceptable for this PR.

The trainer must NOT accept a per-loss `ignore_label` outside the `(1, 0)` default for this PR — out of scope.

Add a per-head diagnostic to the existing per-step log: `view_cos_A`, `view_cos_B` (cosine between view-1 and view-2 of `zA` / `zB` averaged over batch) for the first 100 steps. Reuse whatever per-step diagnostic infrastructure already exists in `train_contrastive_logprob_recon` — do not invent a new one.

### 3. `scripts/run_experiment.py` — add a dispatch routine

Add `run_contrastive_logprob_recon_shared_trunk(...)` (signature matches `run_contrastive_logprob_recon_twin` at `scripts/run_experiment.py:437`). It:

1. Reads `method_cfg["model_params"]["variant"]` (`"split_output"` or `"projection_head"`) to pick the model class.
2. Constructs the model with config-driven dims.
3. Calls `train_contrastive_logprob_recon_dualloss(...)`.
4. Saves checkpoint to `artifacts/final_weights.pt` (single weights file — no `head_a/head_b` split; the trunk is one module).
5. Evaluates: wraps `model.eval()` and passes directly into `MultiMetricHallucinationEvaluator` (no `TwinConcatModel` needed because both variants expose a single embedding via `forward(x)`).
6. For D1, omit cosine from eval metrics (concat space). For D2, include cosine (single coordinate system).

Wire the dispatch in the `if routine == ...` chain near `scripts/run_experiment.py:3340`:
```python
elif routine == "contrastive_logprob_recon_shared_trunk":
    eval_metrics, predictions = run_contrastive_logprob_recon_shared_trunk(...)
```

### 4. Method configs — 4 new files in `configs/methods/`

Model each after `configs/methods/contrastive_logprob_recon_c2.json`. Change `routine` to `"contrastive_logprob_recon_shared_trunk"`. Add `model_params.variant` and the appropriate dim fields.

- **`contrastive_logprob_recon_d1a.json`**: `variant: "split_output"`, `half_dim: 256` → 512-dim eval. Metrics: `["mds", "knn"]`.
- **`contrastive_logprob_recon_d1b.json`**: `variant: "split_output"`, `half_dim: 512` → 1024-dim eval. Metrics: `["mds", "knn"]`.
- **`contrastive_logprob_recon_d2a.json`**: `variant: "projection_head"`, `trunk_dim: 512`, `head_dim: 256`, `head_hidden_dim: 256` → 512-dim eval. Metrics: `["cosine", "mds", "knn"]`.
- **`contrastive_logprob_recon_d2b.json`**: `variant: "projection_head"`, `trunk_dim: 1024`, `head_dim: 512`, `head_hidden_dim: 512` → 1024-dim eval. Metrics: `["cosine", "mds", "knn"]`.

All other fields (training hyperparams, data block, knn_params) match `contrastive_logprob_recon_c2.json` exactly.

### 5. Experiment configs — 4 new files in `configs/experiments/`

Pattern after `configs/experiments/twin_grid_sciq_llama_memmap.json` (or the closest existing config from #99). Files:
- `sharedtrunk_grid_sciq_llama_memmap.json`
- `sharedtrunk_grid_sciq_qwen3_memmap.json`
- `sharedtrunk_grid_nq_llama_memmap.json`
- `sharedtrunk_grid_nq_qwen3_memmap.json`

Each lists the 4 D-configs and 1 seed (seed 0).

### 6. Tests — `tests/test_shared_trunk_twin.py` (new)

Minimum coverage (use `tests/test_twin_concat_model.py` as a style reference):

- `test_split_output_forward_shape`: `SharedTrunkSplitOutputCompressor(half_dim=8)` forward on `(B=4, L=10, 4096)` returns `(4, 16)`.
- `test_split_output_slice_helper`: `forward_slices` returns `z, z_A, z_B` with `z_A == z[:, :8]` and `z_B == z[:, 8:]`.
- `test_projection_head_forward_returns_trunk`: `SharedTrunkProjectionHeadCompressor(trunk_dim=16, head_dim=8)` forward returns `(B, 16)`, NOT `(B, 8)`. Heads must not be called in `forward`.
- `test_projection_head_with_heads_helper`: `forward_with_heads` returns `(z, zA, zB)` with shapes `(B, 16), (B, 8), (B, 8)`.
- `test_dualloss_step_runs`: instantiate either model on CPU, run one training step via `train_contrastive_logprob_recon_dualloss` on a tiny synthetic dataset, assert loss is finite and gradients flow to all parameters (trunk + both heads for D2; trunk + decoder for D1).
- `test_eval_surface_d2_is_trunk`: after one training step on D2 model, assert `model.eval(); model(x)` returns trunk-dim embedding and does NOT depend on head weights (e.g., zero out head weights and confirm output is unchanged).

Skip GPU tests; CPU only.

---

## Acceptance criteria

- [ ] Both new model classes (`SharedTrunkSplitOutputCompressor`, `SharedTrunkProjectionHeadCompressor`) exist in `activation_research/model.py` and pass forward/shape tests.
- [ ] `train_contrastive_logprob_recon_dualloss` exists in `activation_research/training.py` and runs one step on CPU without error.
- [ ] `run_contrastive_logprob_recon_shared_trunk` exists in `scripts/run_experiment.py` and is wired into the routine dispatch.
- [ ] The 4 method configs and 4 experiment configs exist and are valid JSON.
- [ ] `python -c "import json; [json.load(open(f'configs/methods/contrastive_logprob_recon_{c}.json')) for c in ['d1a','d1b','d2a','d2b']]"` succeeds.
- [ ] `pytest tests/test_shared_trunk_twin.py -x` passes on CPU.
- [ ] `python -c "from activation_research.model import SharedTrunkSplitOutputCompressor, SharedTrunkProjectionHeadCompressor; from activation_research.training import train_contrastive_logprob_recon_dualloss"` succeeds (import smoke test).
- [ ] Existing test suite still passes (specifically `pytest tests/test_twin_concat_model.py -x` to confirm #99 was not broken).
- [ ] No files outside the scope list below are modified.

---

## Scope — files you may edit or create

- `activation_research/model.py` (add classes; do not modify existing classes)
- `activation_research/training.py` (add `train_contrastive_logprob_recon_dualloss`; do not modify existing functions unless extracting a shared helper, in which case the existing function must remain behaviourally identical)
- `scripts/run_experiment.py` (add new dispatch routine + wire one `elif` branch; do not modify other routines)
- `configs/methods/contrastive_logprob_recon_d{1a,1b,2a,2b}.json` (4 new files)
- `configs/experiments/sharedtrunk_grid_{sciq,nq}_{llama,qwen3}_memmap.json` (4 new files)
- `tests/test_shared_trunk_twin.py` (1 new file)

---

## Out of scope (do not touch)

- Existing C0–C3 method configs, `run_contrastive_logprob_recon_twin`, `TwinConcatModel`, `LogprobReconProgressiveCompressor`, `train_contrastive_logprob_recon`. **Read-only.**
- Loss reweighting (non-1:1 SupCon weights between heads).
- Trunk:head ratio sweeps beyond the fixed `(512, 256)` and `(1024, 512)` defined above.
- 3+ heads.
- Asymmetric `ignore_label` values beyond `(1, 0)`.
- GPU smoketest (login-node-only environment).
- Running the actual experiments (dispatch is a follow-up; this PR is implementation only).
- Any post-hoc eval logic from #101 — those apply to artifacts after the fact.

---

## Implementation order (suggested)

1. Add `SharedTrunkSplitOutputCompressor` + tests for it.
2. Add `SharedTrunkProjectionHeadCompressor` + tests for it.
3. Add `train_contrastive_logprob_recon_dualloss` + one-step test.
4. Add `run_contrastive_logprob_recon_shared_trunk` dispatch.
5. Add the 4 method configs + 4 experiment configs.
6. Run full test suite. Verify acceptance criteria. Commit.

Commit at logical boundaries (one commit per step is fine). Conventional commit prefix: `feat(#102): ...`.

---

## Working directory

You are working in the git worktree at `/workspaces/hub_4/hallulens-issue-102` on branch `feat/102-shared-trunk-twin-heads`. **Do not touch `/workspaces/hub_4/hallulens` (the main checkout — that holds an independent feature branch).** All edits, tests, and commits happen inside the worktree.

Push the branch when done. Do not open a new PR — one already exists (draft PR will be referenced in your starting context).
