# FP8 Activation Cache — Risk Analysis

**Status:** analysis only, no implementation.
**Goal:** halve in-memory cache footprint by loading fp16 activations as fp8 at training time, so the SearchQA-scale (~200k sample) cache fits in 2 TB RAM alongside additional non-hallucinated examples.

## Scope of the change

- Disk format stays **fp16** in Zarr. Nothing about generation, logging, or the canonical record changes.
- The classifier-side cache (the in-memory tensor used by `activation_research/training.py` / `trainer.py`) becomes **fp8** — either by mmap-ing the fp16 file and downcasting once on load, or by maintaining a separate fp8 mirror.
- Training math itself stays at fp32/fp16 inside the classifier; we upcast on the way into the model.
- This is **post-hoc quantization for memory only**, not fp8 training. It does not speed up training (as the user noted — classifier weights are fp32).

This framing matters because most published fp8 work is about *training in fp8 on H100/B200* (Transformer Engine, MS-AMP, FP8-LM). Their problems (gradient underflow, optimizer state precision, master-weight drift) **do not apply here**. Our risks are narrower: representation noise on already-computed activations and how that noise propagates through a contrastive loss.

## TL;DR of the risks

1. **Outliers will clip or quantize to zero.** Transformer hidden states have well-known outlier features (10–100× the typical magnitude). E4M3 saturates at 448 and E5M2 at 57344; per-tensor naive casting will either clip the outliers (E4M3) or wipe the small components (E5M2). Per-layer (and ideally per-channel) scales are basically required.
2. **Mantissa precision drops from 10 bits → 3 bits (E4M3) or 2 bits (E5M2).** Relative quantization error goes from ~10⁻³ to ~6–12 %. For contrastive losses that rely on small cosine-similarity gaps between hard positives/negatives, this is the main thing to validate empirically.
3. **No native CPU fp8 ops.** PyTorch has `torch.float8_e4m3fn` / `torch.float8_e5m2` as storage dtypes, but compute kernels are essentially GPU-only (and even there are limited). Numpy has no fp8 — you need `ml_dtypes` (Google) or hand-rolled uint8 packing. So the cache lives as fp8 *bytes*, and DataLoader workers (or a GPU-side cast) upcast to fp16/bf16 just before the forward pass.
4. **Memmap with fp8 needs a uint8-backed file plus a manual scale sidecar**, since numpy memmap can't speak `float8_e4m3fn` without `ml_dtypes`. This is solvable but is real plumbing — and it means the fp8 cache is not just `arr.astype(...)` over the existing zarr store.
5. **`float8_e4m3fn` has no Inf and no NaN** (the "fn" = "finite"). Sentinel-based code that checks `isnan` on activations will silently miss them. Worth auditing the loader path for NaN handling before flipping over.
6. **Determinism & reproducibility shift.** Numbers in the paper run at fp16 cache will not bit-match the fp8 cache run. This is fine but should be flagged in any results table that mixes the two.

## fp8 format choice

PyTorch / hardware exposes two formats:

| Format | Exponent / Mantissa | Max | Min normal | Subnormal min | Notes |
|--------|---------------------|------|-----------|----------------|-------|
| `e4m3fn` (E4M3) | 4 / 3 | 448 | 2⁻⁶ ≈ 0.0156 | 2⁻⁹ ≈ 0.00195 | No Inf/NaN; used for forward activations in fp8 training |
| `e5m2`   (E5M2) | 5 / 2 | 57344 | 2⁻¹⁴ | 2⁻¹⁶ | Has Inf/NaN; used for gradients |

For our use case (forward activations after layer norm + residual, post-hoc cast):

- **E4M3 is the right default.** Better mantissa precision (3 bits vs 2) for the bulk of the distribution, which is what the contrastive head looks at. Range of 448 is enough if we apply a per-layer scale so the typical max sits around 100–200.
- **E5M2 is only better if outliers are extreme and we refuse to scale per-layer.** Possible fallback for layers with very heavy tails, but I would not start here.

## What "fp16 → fp8" actually does to our activations

Reference points (no measurement done yet — these are theory):

- **Mantissa noise.** With 3 mantissa bits + implicit leading 1, every interval `[2^k, 2^{k+1})` is split into 8 levels. Quantization error is up to ~6.25 % relative, often closer to 3 %. After layer norm, residual-stream values are typically 𝒪(1)–𝒪(10), so absolute error per element is ~0.03–0.6.
- **Outlier features.** In Llama-3.1 / Qwen3-class models there are a handful of feature dimensions with magnitudes 10–100× the rest, especially at deeper layers. With a single per-tensor scale chosen to fit those outliers into 448, the small features (which are most of them, and most of what the contrastive head learns from) end up with very few mantissa bits of useful signal. With a scale chosen to preserve the bulk, the outliers clip to 448. Either choice is bad on its own.
- **Mitigation: per-channel or per-token scales.** This is what SmoothQuant / LLM.int8() / FP8 inference recipes do. Storing one fp16/bf16 scale per (layer, hidden-dim channel) is cheap (e.g. `16 layers × 4096 dims × 2 B = 128 KB per dataset`) and recovers most of the quality.
- **Cosine similarity drift.** Contrastive learning cares about angles, not magnitudes. fp8 cast preserves direction reasonably well in the bulk, but small angular gaps between hard positives can be eaten by mantissa noise. The number to watch is the *separation* of the positive and negative similarity distributions, not raw activation MSE.

**Concrete validation experiment we should run before committing to fp8:**
1. Pick one already-trained activation cache (e.g. `popqa_qwen3` test split — small).
2. Compute fp8-cast tensors with: (a) per-tensor scale, (b) per-layer scale, (c) per-(layer, channel) scale.
3. Report (i) max abs error vs fp16, (ii) per-token cosine similarity vs fp16, (iii) classifier eval AUROC delta when scoring with each cast variant using a classifier trained on fp16. If (c) costs <1 AUROC point we ship it; if even (c) costs >2 points the gain isn't worth the noise.

## Storage & I/O plumbing

**Cache build path (one-time per dataset, on the CPU node):**

```
zarr fp16 → memmap fp16 → compute per-channel amax → per-layer scales → cast → fp8 bytes (uint8)
                                                                        → save scales sidecar (.npz / .json)
```

The fp8 bytes can sit in a numpy memmap of dtype `uint8` of identical shape, plus a small scales file. At training time the DataLoader does `bytes → torch.float8_e4m3fn (view) → bf16/fp16 (cast) → multiply by scale → batch`.

Practical notes:

- **`ml_dtypes`** (`pip install ml_dtypes`) gives numpy `float8_e4m3fn` and `float8_e5m2` dtypes; that's the cleanest way to avoid hand-rolling uint8 codecs. PyTorch can `from_numpy` an `ml_dtypes.float8_e4m3fn` array on recent versions, but verify on this env's torch first.
- **Conversion CPU cost.** A 1 TB fp16 → 500 GB fp8 cast at maybe 2–4 GB/s on a CPU node is 5–10 minutes per dataset. Negligible vs generation time, but it does mean fp8 builds are not free — we want to cache them on disk too if we're going to load the same cache repeatedly.
- **DataLoader upcast cost.** Casting fp8 → bf16 on CPU is bandwidth-bound and roughly free vs the network/disk bandwidth we're already spending. Doing it on GPU after a pinned-memory copy is even cheaper.
- **Memory math (sanity check).** 1 TB fp16 cache → 500 GB fp8 cache. SearchQA at ~2× current largest dataset (~200k samples) → ~1 TB fp8, fits comfortably with ~1 TB headroom for non-hallucinated additions and other datasets. This *is* the win.

## Risks I'd watch for

| Risk | Likelihood | Severity | Notes |
|------|-----------|----------|-------|
| Outlier clipping degrades AUROC by >1 pt | Medium | High if it happens | Per-channel scale almost certainly fixes it; validate empirically |
| Hard-negative cosine gap collapses under quantization | Medium | High | Test on a real classifier eval, not just MSE |
| `float8_e4m3fn` `isnan` semantics break some loader assumption | Low | Medium | Audit any `torch.isnan(activation)` / `torch.isfinite` calls |
| `ml_dtypes` interop with this torch build is flaky | Low–Medium | Low | Falls back to manual uint8 codec; maybe 1 day of plumbing |
| Reproducibility: fp16 vs fp8 numbers in the same paper table | Certain | Low | Just label clearly; rerun key baselines if needed |
| Scale sidecar drifts out of sync with fp8 file | Low | Medium | Embed scales inside the same file (e.g. zarr group) rather than a side .npz |
| Two layers with very different magnitude regimes share one scale | Low if per-layer | Medium | Per-layer is the minimum we should use; per-channel is safer |
| Subnormal underflow on small features | Medium | Low | E4M3 min normal ~0.0156 — features below that round to subnormal/zero, expected and usually fine |

## What this does *not* solve

- **Disk footprint.** Disk stays fp16. If disk pressure becomes the issue too, that's a separate decision (would need to commit fp8 as the on-disk format and accept the irreversibility). Recommend keeping disk fp16 for now — it's the source of truth and we can re-derive any in-memory format from it.
- **Training speed.** As the user already noted, classifier weights are fp32; fp8 cache only buys memory.
- **Whether SearchQA at 200k actually fits.** It looks like it does (≈1 TB at fp8, 2 TB RAM, leaving headroom for non-hallucinated examples), but worth confirming with an actual byte count from one shard before committing. If it's tight, sub-sampling SearchQA's train split is still a reasonable fallback.

## Recommendation

1. **Build the validation experiment first** (per-channel-scaled E4M3 vs fp16 baseline, on `popqa_qwen3`). One day of work; gates the whole effort.
2. If AUROC delta is < 1 point, build the fp8 cache as a *derived* artifact: keep fp16 zarr canonical, generate `activations.fp8.zarr` (or sidecar) with embedded per-channel scales, and load that in the trainer.
3. **Default to E4M3 with per-(layer, channel) scales.** Don't ship per-tensor scales — they will leave ~1 AUROC on the table for nothing.
4. **Don't go below fp8** (no int4 or int8 quantization for now). The contrastive setting is sensitive enough that fp8 is the aggressive end of what I'd try without a much larger ablation.
5. Keep an explicit `--cache-dtype {fp16,fp8_e4m3}` flag in the trainer so we can A/B at any time and so paper-table reruns are reproducible.
