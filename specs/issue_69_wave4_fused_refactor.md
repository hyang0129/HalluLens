# Wave 4 Refactor: Fused Stage-1+2 Pass with Numpy Memmap Storage

**Parent docs:**
- [`specs/issue_69_icr_probe_attention_infra.md`](issue_69_icr_probe_attention_infra.md) (original design)
- [`specs/issue_69_task_queue.md`](issue_69_task_queue.md) (Waves 1–3 task breakdown)
- [`notes/icr_probe_paper_notes.md`](../notes/icr_probe_paper_notes.md) (paper / code reading)

**PR:** #71 (`feat/issue-69-icr-probe-attn-infra`)
**Status:** Spec — Waves 1–3 already landed in `659a957`, `51050f4`, `860cae3`. This Wave 4 refactors the pipeline before any GPU validation runs.

---

## 1. Motivation

After Waves 1–3 landed, three IO/architecture problems became visible:

1. **Per-block, per-sample zarr reads** (status quo): `recompute_attention.py` does `2 × num_blocks` zarr chunk reads per sample. At HotpotQA scale (~16k samples × 32 blocks × 2 arrays) that's **~1M NFS round-trips per dataset**. The training preload at `activation_logging/activation_parser.py:1598-1610` already shows the right pattern (batched `oindex` over 256 samples), and the recompute path doesn't use it.
2. **Two-pass IO is redundant.** Stage 1 (`recompute_attention.py`) reads `activations.zarr` once to produce `attention.zarr`. Stage 2 (the planned `compute_icr_scores.py`) would then read both `activations.zarr` (response slices) **and** `attention.zarr` to produce `icr_scores.npy`. The activations slice that Stage 2 needs is *already in RAM* inside Stage 1's per-sample loop — re-reading it is pure waste.
3. **Zarr was chosen for `attention.zarr` under the assumption of random-access training reads.** Once the probe trains on `icr_scores.npy` (a 2 MB array), `attention.zarr` is purely an inspectable intermediate, and zarr's chunked/compressed format adds complexity without buying anything. A pre-allocated numpy memmap is strictly simpler and faster for the access pattern that actually exists.

Wave 4 fixes all three.

---

## 2. New Architecture

```
                          ┌────────────────────────────┐
                          │   activations.zarr (NFS)   │   read-only, batched oindex
                          │   prompt + response, L+1   │
                          └──────────────┬─────────────┘
                                         │  one pass, batched (B=16)
                                         ▼
                          ┌────────────────────────────┐
                          │  scripts/                  │
                          │  recompute_attention.py    │   fused Stage 1+2
                          │  ─────────────────────     │
                          │  for each batch:           │
                          │    read activations slice  │
                          │    for each sample:        │
                          │      for each block:       │
                          │        compute attention   │   ← GPU
                          │        compute icr_score   │   ← CPU, on-the-fly
                          │      write attn row        │
                          │      append icr row        │
                          └──────────────┬─────────────┘
                                         │
                          ┌──────────────┴─────────────┐
                          ▼                            ▼
              ┌────────────────────┐    ┌────────────────────────┐
              │  attention/        │    │  icr_scores.npy        │
              │   response_attn.npy│    │  shape (N, num_blocks) │
              │   meta.jsonl       │    │  fp32                  │
              │   config.json      │    │  ~2 MB total           │
              │  (~4 GB, fp16)     │    │                        │
              └────────────────────┘    └────────────────────────┘
                  inspectable                 probe input
                  intermediate                (Issue #70)
```

Key properties:

- **Single pass over `activations.zarr`** — no Stage 2 reads required after Wave 4.
- **Batched `oindex` reads** — `prompt_activations.oindex[batch_idx, :, :, :]` and `response_activations.oindex[batch_idx, :, :, :]` once per batch (B≈16). Reduces NFS round-trips by ~`B × num_blocks` vs. status quo.
- **Numpy memmap** for the response-attention intermediate — pre-allocated, append-by-slot, no compression layer.
- **`icr_scores.npy`** is the probe's actual training input. Shape `(N, num_blocks)` fp32. Fits in RAM trivially.

Memory bound (Llama-8B, B=16, fp16, P_max≈400, R_max=64, H=4096):

| Item | Size |
|---|---|
| `prompt_h_batch` (B, L+1, P_max, H) | 1.7 GB |
| `response_h_batch` (B, L+1, R_max, H) | 0.27 GB |
| GPU model + activations | ~16 GB (fp16 weights) |

Total host RAM for IO buffers: ~2 GB / batch. Fits on every Empire AI node we use.

---

## 3. Storage layout

### 3.1 `attention/` (replaces `attention.zarr`)

```
<dataset>_<model_slug>/attention/
  response_attn.npy       # np.memmap, shape (N, num_blocks, R_max, R_max), dtype float16
  meta.jsonl              # one JSON line per *written* sample:
                          #   {"key": "...", "sample_index": int,
                          #    "response_len": int, "prompt_len": int}
  config.json             # same fields as in Wave 1 (model_name, num_layers, head_dim, ...)
                          # plus "storage_format": "numpy_memmap_v1"
```

- `N` is set at creation time from the source `activations.zarr` sample count. The memmap is pre-allocated zero-filled — unwritten rows are exactly zero.
- Writing sample at index `i`: `mm[i] = attn_array; meta.jsonl.append({key, i, lens})`.
- Resume: read `meta.jsonl` → set of written keys → skip those.
- `meta.jsonl` is append-only and authoritative for "what's written"; `response_attn.npy` is the bulk data.

### 3.2 `icr_scores.npy`

```
<dataset>_<model_slug>/icr_scores.npy   # shape (N, num_blocks), dtype float32
<dataset>_<model_slug>/icr_scores_meta.jsonl   # mirrors attention/meta.jsonl
```

- Token-mean ICR score per `(sample, block)`. Pre-allocated like `response_attn.npy`.
- Indexed identically to `response_attn.npy` — sample at index `i` in one is sample at index `i` in the other.
- `icr_scores_meta.jsonl` exists so the probe trainer doesn't need `attention/` at all to map keys to rows.

### 3.3 What gets deleted

`activation_logging/attention_zarr_logger.py` — replaced by `attention_memmap_writer.py` (below). The Wave 1 zarr writer code is preserved in git history; no need to keep it around.

The `AttentionZarrLogger` class is not exported anywhere else, so this is a contained deletion.

---

## 4. New / changed code

### 4.1 New: `activation_research/icr_score.py`

The CPU-side ICR Score primitive that Stage 2 would have computed. Pure-numpy, no torch dependency required (operates on numpy arrays produced by the fused pass).

```python
def compute_icr_score(
    response_attn: np.ndarray,           # (R, R) float32 — single-block, single-sample, response-to-response
    h_block_input: np.ndarray,           # (R, H) float32 — h^{ℓ-1} at response positions
    delta_h: np.ndarray,                 # (R, H) float32 — h^ℓ - h^{ℓ-1} at response positions
    response_len: int,                   # actual response length (<= R)
    top_p: float = 0.1,                  # per notes §3: top_p overrides top_k in upstream
) -> float:
    """Return token-mean ICR Score for one (sample, layer).

    Per icr_score.py:217-267 in the upstream repo, with the standardize-then-softmax
    JSD variant from notes §10. Top-k key positions are the top-p fraction of the
    attention row; projection direction is the l2-normalized h_block_input at those
    positions; projected quantity is delta_h at the query position.
    """
```

This is the function that Issue #70 was supposed to own. We pull it forward into Wave 4 because the fused pass needs it; **Issue #70's scope shrinks to the probe MLP + training loop only.**

Tests: `tests/test_icr_score.py` — at minimum a numerical regression test against a hand-computed example on a 4x4 attention block.

### 4.2 New: `activation_logging/attention_memmap_writer.py`

Replaces `attention_zarr_logger.py`. Same public interface (constructor, `write`, `is_written`, `finalize`) so `recompute_attention.py` only changes its import line.

```python
class AttentionMemmapWriter:
    def __init__(
        self,
        out_dir: str,                  # writes <out_dir>/response_attn.npy + meta.jsonl + config.json
        mode: Literal["w", "a"],
        n_samples: int,                # required at creation — pre-allocates memmap
        num_layers: int,
        r_max: int = 64,
        config_dict: dict | None = None,
        dtype: str = "float16",
    ) -> None: ...

    def is_written(self, sample_key: str) -> bool: ...
    def write(self, sample_index: int, sample_key: str,
              response_attn: np.ndarray, response_len: int, prompt_len: int) -> None: ...
    def finalize(self) -> None: ...
```

Note the API shift: `write()` now takes `sample_index` explicitly (the source-zarr row index) rather than auto-incrementing. This is what makes append-anywhere resume work and keeps row alignment with the source.

### 4.3 New: `activation_research/icr_scores_writer.py`

Thin sibling to `AttentionMemmapWriter` — pre-allocates `icr_scores.npy` shape `(N, num_blocks)` fp32 and appends one row per sample.

```python
class ICRScoresWriter:
    def __init__(self, out_path: str, mode: Literal["w", "a"], n_samples: int,
                 num_blocks: int) -> None: ...
    def is_written(self, sample_key: str) -> bool: ...
    def write(self, sample_index: int, sample_key: str, icr_vector: np.ndarray) -> None: ...
    def finalize(self) -> None: ...
```

### 4.4 Rewritten: `scripts/recompute_attention.py`

The processing loop becomes:

```python
attn_writer  = AttentionMemmapWriter(args.attention_dir,  mode=..., n_samples=N, ...)
icr_writer   = ICRScoresWriter      (args.icr_scores_path, mode=..., n_samples=N, num_blocks=num_blocks)

for batch_start in range(0, N, args.batch_size):
    batch_idx = source_indices[batch_start : batch_start + args.batch_size]
    # Filter out already-written samples on resume.
    batch_idx = [s for s in batch_idx if not (attn_writer.is_written(key_of[s])
                                              and icr_writer.is_written(key_of[s]))]
    if not batch_idx:
        continue

    # Two zarr calls per batch — all IO for this batch happens here.
    prompt_h_batch   = np.asarray(prompt_activations.oindex[batch_idx, :, :, :])
    response_h_batch = np.asarray(response_activations.oindex[batch_idx, :, :, :])
    prompt_lens      = prompt_len_arr[batch_idx]
    response_lens    = response_len_arr[batch_idx]

    for i, s in enumerate(batch_idx):
        P, R = int(prompt_lens[i]), int(response_lens[i])
        prompt_h_all   = prompt_h_batch[i, :, :P, :]      # (L+1, P, H)
        response_h_all = response_h_batch[i, :, :R, :]    # (L+1, R, H)

        attn_per_block = np.zeros((num_blocks, R_max, R_max), dtype=np.float16)
        icr_per_block  = np.zeros((num_blocks,), dtype=np.float32)

        for b in range(num_blocks):
            h_in_resp  = response_h_all[b]
            h_out_resp = response_h_all[b + 1]
            delta_h    = h_out_resp - h_in_resp                       # (R, H)
            h_prev     = np.concatenate([prompt_h_all[b], h_in_resp], axis=0)

            attn_resp = recompute_block_attention(                   # (R, R) float32, on GPU
                h_prev=torch.from_numpy(h_prev.astype(np.float32)),
                block=model.model.layers[b],
                prompt_len=P, response_len=R, device=args.device,
            ).cpu().numpy()

            attn_per_block[b, :R, :R] = attn_resp.astype(np.float16)
            icr_per_block[b]          = compute_icr_score(
                response_attn=attn_resp,
                h_block_input=h_in_resp.astype(np.float32),
                delta_h=delta_h.astype(np.float32),
                response_len=R,
                top_p=icr_top_p,
            )

        attn_writer.write(s, key_of[s], attn_per_block, R, P)
        icr_writer .write(s, key_of[s], icr_per_block)

attn_writer.finalize()
icr_writer .finalize()
```

CLI changes:

- Replace `--attention-zarr <path>` with `--attention-dir <path>` (writes a directory now, not a zarr store).
- Add `--icr-scores-path <path>` (default: alongside `--attention-dir`).
- Add `--icr-top-p` (default 0.1, per notes §3) and `--icr-top-k` (default None; if set, overrides top-p).
- `--batch-size` becomes meaningful — it's now the zarr-read batch size. Default 16.
- `--validate-first` still does the 4-sample diff vs. full-forward check (unchanged).

### 4.5 Rewritten: `activation_logging/attention_parser.py`

Becomes a thin reader over the new memmap layout. Public API stays the same (`get_attention`, `get_paired`, `list_keys`, `__len__`) so `ICRDataset` and any ablation code keep working with a single line change.

The `_config`-on-`ZarrActivationsLogger` dead-code bug noted in the PR description is fixed as a side effect: the parser now reads `meta/config.json` from disk on both sides for the model-name cross-check.

### 4.6 Simplified: `activation_research/icr_dataset.py`

After Wave 4, `ICRDataset` has two modes:

- **Fast path (default):** reads `icr_scores.npy` directly. `__getitem__` returns `{"hashkey": str, "halu": int, "icr_score": Tensor (num_blocks,)}`. RAM-trivial. This is what Issue #70's probe trainer uses.
- **Raw path (ablations only):** reads attention + activations via `AttentionParser.get_paired()` for ad-hoc ICR-formula ablations. Same shape as Wave 3.

Switched by a constructor flag `mode: Literal["icr", "raw"] = "icr"`.

### 4.7 Updated tests

- `tests/test_attention_recompute.py` — unchanged (tests `recompute_block_attention()`, not the pipeline).
- `tests/test_attention_parser.py` — fixture switches from `AttentionZarrLogger` to `AttentionMemmapWriter`. Synthetic activations.zarr stays.
- `tests/test_icr_score.py` — new; numerical regression against hand-computed examples.
- `tests/test_icr_dataset.py` — new (small); covers both `mode="icr"` and `mode="raw"`.

### 4.8 Updated smoke test

`scripts/smoketest_attention_recompute.sh` Phase 3 readback updates:

- Read `<smoke_dir>/attention/response_attn.npy` via `np.memmap` instead of `AttentionParser`.
- Add a Phase 4: load `icr_scores.npy`, assert shape `(N, num_blocks)` and that no row is all-zero (would indicate score-compute failure).

---

## 5. Task breakdown

| Task | File(s) | Owner | Wave | Depends on |
|---|---|---|---|---|
| H | `activation_research/icr_score.py` | Coder F | 4a | — |
| I | `tests/test_icr_score.py` | Tester 3 | 4a | — (spec-first) |
| J | `activation_logging/attention_memmap_writer.py` | Coder G | 4a | — |
| K | `activation_research/icr_scores_writer.py` | Coder G | 4a | — |
| L | `activation_logging/attention_parser.py` (rewrite) | Coder H | 4b | J |
| M | `scripts/recompute_attention.py` (rewrite) | Coder I | 4b | H, J, K |
| N | `activation_research/icr_dataset.py` (rewrite) | Coder J | 4c | L (raw mode) |
| O | `tests/test_attention_parser.py` (update fixture) | Tester 4 | 4c | L |
| P | `tests/test_icr_dataset.py` | Tester 5 | 4c | N |
| Q | `scripts/smoketest_attention_recompute.sh` (update Phase 3, add Phase 4) | Coder K | 4c | M, N |

Wave 4a tasks (H, I, J, K) are fully independent and can run in parallel. Wave 4b (L, M) waits on 4a. Wave 4c (N, O, P, Q) waits on 4b.

---

## 6. Acceptance criteria

- [ ] `pytest tests/test_attention_recompute.py tests/test_attention_parser.py tests/test_icr_score.py tests/test_icr_dataset.py` → all pass on CPU without real model weights.
- [ ] `scripts/recompute_attention.py --validate-first` still exits 0 on Llama-8B and Qwen3-8B HotpotQA (Wave 2 acceptance criterion preserved).
- [ ] `scripts/smoketest_attention_recompute.sh` passes end-to-end (all 4 phases) on a Jupyter GPU node.
- [ ] Wall-clock time for one full HotpotQA dataset (Llama-8B, ~7400 samples) under 4 hours on an H200 (measured during smoke test; document in PR).
- [ ] `attention/response_attn.npy` size matches `N × num_blocks × R_max × R_max × 2` bytes exactly (no compression).
- [ ] `icr_scores.npy` shape is `(N, num_blocks)` fp32; no row is all-zero (verified in Phase 4 of smoke test).
- [ ] No mention of `AttentionZarrLogger` anywhere in `activation_logging/`, `activation_research/`, `scripts/`, or `tests/` after the refactor.
- [ ] PR description updated to note the spec deviation (numpy memmap instead of zarr; reason: one-pass sequential access).

---

## 7. Out of scope

- The probe MLP itself and its training/eval loop — still Issue #70.
- Score-formula ablations (different top-p, JSD variants, projection rules) — Issue #70 or a follow-up.
- Re-running Qwen3 HotpotQA inference at higher `response_max_tokens` — separate ticket per parent spec §3.
- The four other-issue specs sitting untracked in `specs/` — they belong to their own PRs.

---

## 8. Risks and notes

1. **Pre-allocation requires knowing `N` up front.** This is fine: we read `prompt_activations.shape[0]` from the source zarr at script start. If a sample is later determined to be unusable (`response_len < 1`), its row stays zero — harmless, and `meta.jsonl` simply won't have an entry for that key, so readers skip it.
2. **`np.memmap` file growth.** Pre-allocating a 4 GB sparse file on first write should be near-instant on local SSD; on NFS this could be slow. Mitigation: write the memmap to the GPU node's local scratch (`/scratch` or `/tmp`) during the run and rsync to NFS after `finalize()`. The smoke test should measure this.
3. **fp16 storage rounding.** Wave 1's `attention.zarr` was fp16; Wave 4 keeps the same. The `--validate-first` 1e-3 tolerance is unchanged. Document.
4. **Score-formula uncertainty.** The upstream JSD-standardization formula (notes §10) is unusual. Land Wave 4 with the upstream formula exactly; flag any deviation in a PR comment.
5. **Issue #70's scope shrinks.** Pulling `compute_icr_score()` into Wave 4 means #70 is now just the probe MLP + training/eval loop. Update Issue #70's description before opening that ticket's PR.
