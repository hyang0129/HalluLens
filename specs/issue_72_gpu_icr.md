# GPU ICR scoring (additive optimization for #72)

**Status:** proposal
**Branch target:** `feat/issue-72-inference-capture-rewrite` (on top of the B=4 batched capture commit `e3eea22`)
**Motivation:** The B=4 batched capture path saves time on the GPU portion of each sample (~1.5 → ~0.5 sec/sample, ~3×) but the CPU portion — `compute_icr_per_layer` looping over 32 layers in numpy — stays at ~0.5 sec/sample regardless of batch size. That CPU portion becomes the bottleneck once B>1.

This spec ports `compute_icr_per_layer` to a batched GPU kernel that runs the full `(B, L)` score grid in one pass. Expected drop: ~0.5 sec/sample → ~0.05 sec/sample. **End-to-end B=4 throughput improves from ~1.3 s/sample to ~0.85 s/sample (~1.5× wall-time improvement on top of the B=4 win).**

Strictly additive: the numpy reference (`activation_research/icr_score.compute_icr_score`) stays as the source of truth. The new GPU kernel is gated behind the same `--batch-size > 1` flag and asserts numerical equivalence with the numpy path in CI.

## Goal

Add `compute_icr_per_layer_batched_gpu` that takes the same inputs as today's per-sample numpy loop but processes all B × L scores in parallel on GPU, returning a `(B, L) float32` array. Wire it into the batched orchestrator path so `scripts/capture_inference.py::_run_batch` calls it once per batch instead of `B × L` Python-level calls.

## What stays unchanged

| | |
|---|---|
| `activation_research/icr_score.compute_icr_score` (numpy) | **Reference implementation. Do not modify.** The new GPU kernel must produce bit-equivalent (within 1e-5 fp32 tolerance) output for any single sample. |
| `InferenceCaptureWriter` | Per-sample. After GPU ICR, orchestrator still does `writer.append(...)` B times per batch. |
| Cell JSON schema | No new fields. |
| B=1 path | Continues to call the numpy reference. |
| Capture file format | Unchanged. |

## What changes

### 1. New module: `activation_research/icr_score_gpu.py`

```python
import torch

def compute_icr_per_layer_batched_gpu(
    response_attn: torch.Tensor,   # (B, L, r_max, r_max) fp16 or fp32, on GPU
    h_block_input: torch.Tensor,   # (B, L, r_max, hidden_dim) fp32, on GPU
    delta_h: torch.Tensor,         # (B, L, r_max, hidden_dim) fp32, on GPU
    response_lens: torch.Tensor,   # (B,) int64, on GPU or CPU
    top_p: float = 0.1,
) -> torch.Tensor:                  # (B, L) float32, on GPU
    """Batched GPU equivalent of activation_research.icr_score.compute_icr_score
    looped over B samples × L layers.

    Implements the same formula as upstream's icr_score.py (z-score → softmax → JSD
    on top-p subsets), vectorized across (B, L). Falls back to CPU when CUDA is
    unavailable; correctness must hold either way.
    """
```

Algorithm (all batched over B × L):

1. **Cast inputs to fp32**, move to GPU if needed. fp16 input is OK for `response_attn` (matches storage); compute is fp32.
2. **Mask attention by response_lens**: for each `(b, l)`, zero rows/cols past `response_lens[b]`. Use a broadcast mask of shape `(B, 1, r_max, 1)` and its transpose for symmetry. **Critical — without this, padding-zero rows still pass top-k and contaminate scores.**
3. **Effective k from top_p**: `k = max(int(top_p * response_lens[b]), 1)`. Per-sample (different for each b). Use the smallest k across the batch as a unified `k_unified`, then pad per-sample to that k by zeroing the tail; or use a loop over the rare distinct k values. **Simpler: use a single global `k = max(1, int(top_p * r_max))` and rely on the masked-out positions never being picked. This matches upstream's behavior when `top_p * len == top_p * effective_seq_len`.**
4. **Top-k indices**: `torch.topk(response_attn, k=k_unified, dim=-1)` → `(B, L, r_max, k_unified)` indices and values.
5. **Gather projection targets**: `h_block_input.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim))` → `(B, L, r_max, k_unified, hidden_dim)`.
6. **Per-position projection**: `w_i = (h_block_input_topk @ delta_h.unsqueeze(-1)).squeeze(-1) / (h_block_input_topk.norm(dim=-1) + 1e-8)` → `(B, L, r_max, k_unified)`. Note: upstream divides by the norm of the *target* hidden state (paper notes §5), not by `||delta_h||`.
7. **JSD on z-score → softmax**:
   - `a_norm = softmax(zscore(attn_topk, dim=-1), dim=-1)` → `(B, L, r_max, k_unified)`
   - `w_norm = softmax(zscore(w_i, dim=-1), dim=-1)` → same shape
   - `m = 0.5 * (a_norm + w_norm)`
   - `jsd = 0.5 * (kl(a_norm, m) + kl(w_norm, m))` → `(B, L, r_max)` per-position scalar
8. **Per-token mean → per-layer ICR**: mask out positions past `response_lens[b]`, then `jsd.sum(dim=-1) / response_lens[b].clamp(min=1)` → `(B, L)`.

The numpy reference does this with per-sample, per-layer loops. The GPU kernel does it with broadcasted tensor ops. Same math, different layout.

### 2. Orchestrator wiring: `scripts/capture_inference.py`

In `_run_batch`, after the stitching primitives return their `(B, ...)` tensors:

```python
if args.batch_size > 1 and torch.cuda.is_available():
    # GPU path: one call, (B, L) result.
    icr_scores_batch = compute_icr_per_layer_batched_gpu(
        torch.from_numpy(resp_attn).to(device),
        torch.from_numpy(resp_hs[:, :-1, :r_max]).float().to(device),
        torch.from_numpy(resp_hs[:, 1:, :r_max] - resp_hs[:, :-1, :r_max]).float().to(device),
        torch.from_numpy(response_lens).to(device),
        top_p=0.1,
    ).cpu().numpy()  # (B, L)
else:
    # B=1 or no CUDA: numpy reference, per-sample loop.
    icr_scores_batch = np.stack([
        compute_icr_per_layer(resp_attn[b], resp_hs[b], response_lens[b])
        for b in range(B)
    ])
```

Note `r_max` slice on hidden states: `compute_icr_score` expects `h_block_input` shaped `(R, hidden_dim)` where `R = r_max`. The stitched `resp_hs` is shape `(B, L+1, max_response_len, hidden_dim)`. We already know `max_response_len == r_max` from the recent default change (issue #72 cap), but slice defensively.

### 3. Tests: `tests/test_icr_score_gpu.py`

The gating contract is numerical equivalence with the numpy reference. Mandatory tests:

1. **`test_gpu_matches_numpy_single_sample`** — Build a synthetic `(L=4, r_max=8, hidden_dim=32)` sample, run both paths, assert `max|gpu - numpy| < 1e-5` (fp32).
2. **`test_gpu_matches_numpy_batched`** — Batch of B=3 samples with **different** response_lens (e.g. 2, 5, 8), run GPU path once and numpy reference per-sample. Assert per-sample match within `1e-5`.
3. **`test_gpu_handles_response_len_zero`** — Sample with `response_lens[b] = 0`. Result should be `0` (or `NaN`-free) for that row. The numpy reference returns 0 in this case (no positions to score).
4. **`test_gpu_handles_response_len_equals_r_max`** — Sample with `response_lens[b] == r_max`. No padding to mask out. Same as numpy.
5. **`test_gpu_fp16_attention_input`** — Pass `response_attn` as fp16 (matches memmap storage); kernel should upcast internally. Assert match within `5e-4` (fp16 vs. fp32 tolerance).
6. **`test_cpu_fallback`** — Force `torch.cuda.is_available() == False` via monkeypatch; assert the GPU function still works on CPU with identical numerical output.

All tests must pass on CPU (use `device='cpu'`). The GPU vs. CPU output equivalence is also a test target — same kernel code, same result, just dispatched to a different device.

### 4. Equivalence gate update

`tests/test_capture_equivalence_batched.py` currently asserts the batched-capture outputs match the unbatched-capture outputs. **Add one more assertion**: when `--batch-size > 1`, the ICR scores in `icr_scores.npy` (written by the GPU path) must match the per-sample numpy reference within `1e-4` (relaxed from `1e-5` to allow some fp16/fp32 GPU/CPU drift).

If this assertion holds, the GPU path is safe for production.

## Edge cases

| Case | Handling |
|---|---|
| `response_lens[b] == 0` (immediate EOS) | Masked attention is all zero; top-k picks zeros; scores at those positions are 0; per-token mean over 0 positions returns 0. Same as numpy reference. |
| `response_lens[b] < k_unified` | Mask-zero-out padding then top-k still picks `k_unified` indices but the masked positions contribute zero — verify this matches numpy's per-sample `k = max(1, int(top_p * response_lens[b]))` behavior. **If they diverge, fall back to per-b loop on GPU for this case.** |
| Numerical drift in z-score with near-constant input | The numpy reference divides by `max(x.std(), 1e-8)`; GPU kernel must use the same clamp. |
| Mixed dtypes (fp16 attention, fp32 hidden states) | Upcast attention to fp32 before any math. Don't mix in fp16. |
| GPU OOM with very large `(B, L, r_max, k, hidden_dim)` | At B=4, L=32, r_max=64, k≈6 (top_p=0.1 × 64), hidden_dim=4096: 4 × 32 × 64 × 6 × 4096 × 4 bytes = ~770 MB intermediate. Fits comfortably. **Document the formula in the function docstring so we know when to chunk.** |

## Out of scope

- Porting `stitch_*` primitives to GPU. They're already cheap (~0.2 sec/sample at B=4) and require careful padding handling.
- Removing the numpy reference. It's the source of truth for correctness audits.
- Changing `compute_icr_score` (numpy) signature or behavior.
- Async overlap of GPU ICR with the next batch's `generate()`. Premium optimization, ~10% gain, not worth the complexity.

## Files

| File | Action | LOC |
|---|---|---|
| `activation_research/icr_score_gpu.py` | Create | ~120 |
| `scripts/capture_inference.py` | Modify `_run_batch` to call GPU kernel when B>1 | +15 |
| `tests/test_icr_score_gpu.py` | Create | ~150 |
| `tests/test_capture_equivalence_batched.py` | Add one numpy-equivalence assertion | +20 |

Total ~300 LOC; about half a sonnet-day.

## Risks

| Risk | Mitigation |
|---|---|
| GPU kernel diverges from numpy reference at fp32 epsilon | The 6 tests above explicitly enumerate divergence cases. Failing any of them blocks the merge. |
| `k_unified = max(1, int(top_p * r_max))` differs from numpy's per-sample `k = max(1, int(top_p * response_lens[b]))` | Run the gate at multiple `response_lens` values. If divergence is real, fall back to per-b loop on GPU — slower but still much faster than CPU loop. |
| GPU/CPU non-determinism causes flaky tests | Use `torch.use_deterministic_algorithms(True)` in tests; clamp tolerances to `1e-5` (CPU) and `1e-4` (GPU). |
| OOM at higher B values in production | Document the memory formula; smoketest at B=4 on H100 to verify before kicking off Phase 1. |

## Smoketest before scaling

After implementation, on the next available H100 worker:
1. Re-run the sciq B=4 cell smoketest. Compare wall-time vs. B=4 with CPU ICR (if we have a baseline).
2. Diff `icr_scores.npy` against a B=1 numpy-only baseline run on the same data — must match within `1e-4`.
3. If both pass, GPU ICR is safe for Phase 1.

## Expected outcome

End-to-end batched capture goes from ~1.3 sec/sample (CPU ICR) to ~0.85 sec/sample (GPU ICR) at B=4 on H100. Full grid totals:

| | Per model | Both models | 3 parallel workers wall |
|---|---|---|---|
| B=4, CPU ICR (today after sonnet F merge) | ~135 h | ~270 h | ~90 h ≈ 3.7 d |
| **B=4 + GPU ICR (this spec)** | **~88 h** | **~176 h** | **~59 h ≈ 2.5 d** |

~1.2-day wall-time savings on the full grid for ~half a sonnet-day of engineering.
