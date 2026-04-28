# NQ Qwen3-8B Activation Generation — Handoff Notes

**Date:** 2026-04-28  
**Status:** Batched inference wired up and profiled; recommended batch_size=128; ready for full run

---

## What Was Done

Added Natural Questions (NQ) train/test split support and Qwen3-8B activation generation. Changes span three files:

| File | Change |
|------|--------|
| `tasks/llmsknow/natural_questions.py` | Added 80/20 train/test split (seed=42, same as PopQA); refactored `run_inference` to use `exp.run_exp`; added `run_inference_batched` (mirrors PopQA pattern) |
| `scripts/run_with_server.py` | NQ block now passes `split`, `logger_type`, `activations_path`, `resume`, `batch_size`; `get_task_name` returns `natural_questions_train` for train split; added `--batch-size` CLI arg |
| `activation_logging/server.py` | Fixed `enable_thinking` — moved to `_format_chat_prompt_for_model` via `apply_chat_template(..., enable_thinking=False)` for Qwen3 models |

New configs:
- `configs/datasets/nq_test_qwen3.json` → `output/natural_questions/Qwen3-8B/`
- `configs/datasets/nq_train_qwen3.json` → `output/natural_questions_train/Qwen3-8B/`

---

## Pain Points Encountered

1. **`enable_thinking` wrong API** — the plan called for injecting `enable_thinking=False` into `model.generate()` kwargs, but this parameter is only accepted by `tokenizer.apply_chat_template()`. Fixed in `_format_chat_prompt_for_model`.

2. **`--N` defaults to 1** — `run_with_server.py`'s argparse has `--N` with `default=1`, not `None`. Any dispatch script that omits `--N` silently processes exactly 1 sample. Always pass `--N 100000` (or larger than the dataset) for full runs.

3. **`data_dir=None` propagation** — `run_experiment()` explicitly sets `data_dir=None` in `task_kwargs`, so `kwargs.get("data_dir", "fallback")` always returns `None`. Fixed with `or` fallback: `kwargs.get("data_dir") or "external/LLMsKnow/data"`.

4. **Server path is ~45s/sample** — the default `exp.run_exp` path hits the HF activation server one request at a time. For 16k+ samples this is ~200 hours. Must use `--batch-size` to bypass the server and use `HFTransformersAdapter` directly.

5. **SSH dispatch timeout** — `gpu_dispatch.py` has a 60s SSH timeout for dispatching jobs. Model-loading startup (5 shards, ~4s) occasionally causes the nohup + echo PID to time out. Workaround: dispatch directly via `ssh alphagpu19 "nohup bash /tmp/script.sh > log 2>&1 & echo \$!"`.

6. **Stale zarr not cleaned** — deleting `generation.jsonl` is not enough; the zarr from previous failed runs persists. Always clean `shared/natural_questions_*_qwen3_8b/activations.zarr/` before a fresh run.

---

## Current Setup (Batched Mode)

```bash
# Train split
ssh alphagpu19 "nohup bash /tmp/nq_qwen3_train.sh > shared/logs/nq_train.log 2>&1 & echo \$!"

# /tmp/nq_qwen3_train.sh:
/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python scripts/run_with_server.py \
    --step all \
    --task naturalquestions \
    --model Qwen/Qwen3-8B \
    --split train \
    --N 100000 \
    --batch-size 128 \
    --activations-path shared/natural_questions_train_qwen3_8b/activations.zarr \
    2>&1 | tee /tmp/nq_qwen3_train.log

# Test split (run after train finishes)
/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python scripts/run_with_server.py \
    --step all \
    --task naturalquestions \
    --model Qwen/Qwen3-8B \
    --split test \
    --N 100000 \
    --batch-size 128 \
    --activations-path shared/natural_questions_qwen3_8b/activations.zarr \
    2>&1 | tee /tmp/nq_qwen3_test.log
```

Note: `--batch-size` bypasses the vLLM server entirely. No need to start or stop a server process.

---

## Profiling Results (2026-04-28, alphagpu19 H200)

**Run:** 20 train samples, batch_size=8, Qwen3-8B fp16

| Metric | Value |
|--------|-------|
| Throughput | **6.34s / batch of 8 = 0.79s/sample** |
| Samples/sec | **~1.26 samples/sec** |
| GPU utilization | **72%** during inference |
| VRAM used | **32 GB** (16 GB model + ~16 GB activation buffers) |
| Storage per sample | **~74 MB** (37 layers × 2 arrays × 1 MB chunks, uncompressed) |

**Estimated full-run time at batch_size=8:**

| Split | Samples | ETA |
|-------|---------|-----|
| Train | 16,617 | ~3.7 hours |
| Test | 4,155 | ~55 minutes |
| **Total** | 20,772 | **~4.6 hours** |

**Storage estimate (calibrated against existing zarrs):**

Existing Llama-3.1-8B-Instruct zarrs (33 layers) for reference:
- `shared/hotpotqa/activations.zarr`: 635 GB / 7,405 samples = **85.7 MB/sample**
- `shared/mmlu/activations.zarr`: 871 GB / 10,225 samples = **85.2 MB/sample**

Qwen3-8B has 37 layers (vs 33 for Llama), so expect ~(37/33) × 86 MB ≈ **96 MB/sample**.

| Split | Samples | Estimated size |
|-------|---------|----------------|
| Train | 16,617 | ~1.6 TB |
| Test  | 4,155  | ~400 GB |

Cluster has ~14 PB free — storage is not a concern.

### GPU Utilization Analysis

72% GPU utilization at batch_size=8. See extended profiling below for the full picture.

---

## Extended Profiling Results (2026-04-28)

Scripts: `scripts/profile_inference_throughput.py`, `scripts/profile_disk_throughput.py`  
Logs: `shared/logs/profile_inference.log`, `shared/logs/profile_disk.log`

### 1. GPU Inference Cap (pure inference, no disk, no activation logging)

50 batches per config, max_new_tokens=64, greedy, Qwen3-8B fp16 on H200.

| Batch size | Batch time (s) | Samples/sec | Output tokens/sec | VRAM |
|-----------|---------------|-------------|------------------|------|
| 8         | 1.649         | 4.85        | 310              | 16.4 GB |
| 16        | 1.655         | 9.67        | 619              | 16.4 GB |
| 32        | 1.662         | 19.25       | 1232             | 16.4 GB |
| 64        | 1.657         | 38.62       | 2472             | 16.4 GB |
| 128       | 1.795         | **71.30**   | **4563**         | 16.4 GB |

**Critical observation: batch time is nearly flat** across BS 8→128 (1.65s → 1.80s). The H200 is barely loaded — only 16.4 GB of 140 GB VRAM used, and throughput scales almost linearly with batch size. The per-batch overhead (tokenization, KV-cache init, autoregressive decode) dominates; the actual GPU compute is essentially free at this model size.

BS=64 is still perfectly flat (1.657s). BS=128 shows the first slight increase (+8.5%) but stays linear — no memory pressure at all. Could push to 256+ without issue.

At BS=128 with pure GPU: ~71 samp/s. For 16617 train samples: ~234s = **~4 minutes** (no activation logging).

### 2. Disk Throughput (simulated zarr writes, no GPU)

20 trials per config. Each sample: 37 layers × 64 seq × 4096 hidden × float16 × 2 arrays = **38.8 MB/sample**.

Three strategies:
- **zarr_per_sample** — current approach: write one sample at a time, chunk_shape=(1,1,64,4096)
- **zarr_batch_slice** — batch-level slice assignment into zarr, chunk_shape=(BS,1,64,4096)
- **npy_batch** — single `np.save()` per batch (theoretical ceiling, no zarr overhead)

| Strategy | BS | Batch time (s) | Samp/s | MB/s |
|----------|-----|---------------|--------|------|
| zarr_per_sample | 8 | 0.485 | 16.5 | 640 |
| zarr_batch_slice | 8 | 0.348 | 23.0 | 891 |
| npy_batch | 8 | 0.159 | 50.3 | 1952 |
| zarr_per_sample | 16 | 0.956 | 16.7 | 649 |
| zarr_batch_slice | 16 | 0.646 | 24.8 | 961 |
| npy_batch | 16 | 0.332 | 48.2 | 1868 |
| zarr_per_sample | 32 | 1.932 | 16.6 | 643 |
| zarr_batch_slice | 32 | 1.432 | 22.4 | 867 |
| npy_batch | 32 | 0.691 | 46.3 | 1796 |

Key observations:
- **zarr_per_sample throughput is batch-size-invariant** (~16-17 samp/s, ~640 MB/s) because it writes one chunk per sample regardless of batch size — no batching benefit.
- **zarr_batch_slice** is ~40% faster than per-sample writes (~23-25 samp/s, ~900 MB/s).
- **npy_batch is 3x faster than zarr** (~48 samp/s, ~1900 MB/s) but loses the structured zarr format.
- The disk ceiling (~1900 MB/s) appears to be the NFS/network filesystem limit, not local SSD.

### 2b. zarr_block chunk shape — tested and rejected

Tested chunk shape `(BS, N_LAYERS, SEQ, HIDDEN)` — collapsing all layers into one chunk to reduce writes from 74 to 2 per batch.

| BS | Strategy | Time(s) | Samp/s | MB/s | Chunk size | Writes/batch |
|----|----------|---------|--------|------|------------|--------------|
| 8 | zarr_block (BS,L,T,H) | 0.781 | **10.25** | 398 | 155 MB | 2 |
| 8 | zarr_batch_slice (BS,1,T,H) | 0.383 | **20.89** | 810 | 0.5 MB | 74 |

`zarr_block` is **2× slower** despite 37× fewer NFS operations. Root cause: zarr must allocate, encode, and serialize a 155 MB contiguous buffer per chunk before writing; at that size the codec pipeline stalls and memory allocation overhead dominates. Small 4 MB chunks in `zarr_batch_slice` pipeline through the encoder efficiently.

**Chunk shape should remain `(BS, 1, T, H)`.** Higher BS does not help here — the zarr bandwidth ceiling is ~900 MB/s (~24 samp/s at 38 MB/sample) regardless of chunk count once you're in the batch-slice regime.

Note: float16 activations are near-random bits — compression (zstd etc.) saves <5% and is not worth the CPU overhead.

### 3. Bottleneck Analysis

Combining GPU + disk, the expected throughput at BS=8:
- Pure GPU: 4.85 samp/s
- Disk (zarr_per_sample): 16.5 samp/s
- Expected combined (sequential): ~3.8 samp/s

**Measured actual (BS=8, with full activation logging): 1.26 samp/s** — ~3x slower than the theoretical ceiling.

The gap is **activation extraction overhead** in `HFTransformersAdapter`:
- With `output_hidden_states=True`, all 37 layer hidden states are accumulated during the 64-step autoregressive loop
- At BS=8: 37 layers × 64 tokens × 8 batch × 4096 × float16 = ~310 MB must be copied GPU→CPU per batch
- This copy + numpy reshape + per-layer slicing accounts for most of the ~5s overhead

### 4. Recommendations

| | Current | Recommended |
|--|---------|-------------|
| Batch size | 8 | **32** |
| Zarr write strategy | per-sample | **batch_slice** `(BS,1,T,H)` chunks (needs `ZarrActivationsLogger` change) |
| Activation extraction | per-layer blocking `.cpu()` | **pinned buffer + `non_blocking=True`** (stack_single is slower — see profiling) |
| Expected samp/s | 1.26 | **~18–20** |

**Target: ~20 samp/s at BS=32.** At BS=32 with wins 1+2 + async overlap:
- GPU ceiling: 19.3 samp/s
- Disk ceiling (zarr_batch_slice): 22.4 samp/s
- Pipeline is GPU-bound — disk is slightly faster, so async overlap (next GPU batch starts while disk write runs) drives sustained throughput to ~18–20 samp/s.

**Do not use BS=128 for activation logging.** At BS=128 the disk ceiling is still ~24 samp/s (zarr NFS bandwidth limit) while GPU does 71 samp/s — GPU idles 3× longer than it computes. BS=32 is the sweet spot where GPU and disk are matched. Higher BS only helps for pure inference benchmarks.

**Longer-term win 1 — Zarr batch writes:** switch `ZarrActivationsLogger` to batch-level writes (chunk the first axis by batch_size). Brings disk throughput from 16.5 → 22 samp/s.

**Longer-term win 2 — Pinned memory + `non_blocking=True` (3.5–10× activation extraction speedup):**

Profiled on alphagpu19 H200 (`scripts/profile_gpu_transfer.py`, 50 trials, Qwen3-8B config: 37 layers × 64 seq × 4096 hidden × float16):

| Strategy | BS | Batch(s) | Samp/s | GB/s | vs loop |
|----------|-----|----------|--------|------|---------|
| per_layer_loop | 8 | 0.0107 | 746 | 14.5 | baseline |
| stack_single | 8 | 0.0348 | 230 | 4.5 | **3.2× slower** |
| pinned_nonblock | 8 | 0.0030 | 2640 | 51.2 | **3.5× faster** |
| per_layer_loop | 32 | 0.0513 | 624 | 12.1 | baseline |
| stack_single | 32 | 0.1399 | 229 | 4.4 | **2.7× slower** |
| pinned_nonblock | 32 | 0.0120 | 2665 | 51.7 | **5.3× faster** |
| per_layer_loop | 64 | 0.2535 | 253 | 4.9 | baseline |
| stack_single | 64 | 0.2799 | 229 | 4.4 | **1.1× slower** |
| pinned_nonblock | 64 | 0.0238 | 2694 | 52.3 | **10.7× faster** |

**Key findings:**

- `stack_single` is **slower** than the per-layer loop at all batch sizes. `torch.stack` allocates a new `[37, BS, seq, hidden]` tensor on GPU and copies all 37 layers into it before the PCIe transfer — the GPU-side allocation overhead exceeds the savings from fewer transactions.
- `pinned_nonblock` achieves **51–52 GB/s** — near the H200 PCIe 5.0 ceiling (~64 GB/s). The per-layer loop only reaches 12–15 GB/s because pageable memory forces the driver to temporarily lock pages on every transfer.
- The speedup grows with batch size: 3.5× at BS=8, 10.7× at BS=64.

**Implementation** — pre-allocate a page-locked CPU buffer at `ZarrActivationsLogger` init, replace the per-layer loop with a single `copy_(..., non_blocking=True)`:

```python
# At ZarrActivationsLogger init — allocate once:
self._activation_buffer = torch.empty(
    (n_layers, batch_size, max_seq_len, hidden_size),
    dtype=torch.float16,
    pin_memory=True,          # page-locked: enables true async DMA
)

# During log_activations — single async transfer:
stacked = torch.stack(list(hidden_states), dim=0)   # [n_layers, BS, seq, hidden] on GPU
self._activation_buffer.copy_(stacked, non_blocking=True)
# GPU can start next batch here while PCIe transfer runs
torch.cuda.synchronize()                            # wait only when data is needed
arr = self._activation_buffer.numpy()
```

Move `synchronize()` into the `AsyncActivationWriter` thread so the PCIe transfer overlaps with GPU compute on the next batch.

**Updated bottleneck analysis with pinned_nonblock:**

With `pinned_nonblock`, activation extraction at BS=32 drops to 0.012s/batch — negligible vs GPU inference at 1.66s/batch. The bottleneck shifts:

| Stage | BS=32 batch time | Samp/s |
|-------|-----------------|--------|
| GPU inference (pure) | 1.662s | 19.3 |
| Activation extraction (pinned_nonblock) | 0.012s | 2665 — no longer a bottleneck |
| Disk zarr_per_sample | 1.932s | 16.5 |
| Disk zarr_batch_slice | 1.432s | 22.4 |

With wins 1+2 at BS=32, the pipeline is GPU (19 samp/s) + disk (22 samp/s) in sequence → ~10–12 samp/s. With async overlap (PCIe transfer and next GPU batch running concurrently) → up to ~18 samp/s.

**Revised ETA with wins 1+2 at BS=32:**

| Split | Samples | Optimistic (18 samp/s) | Conservative (10 samp/s) |
|-------|---------|------------------------|--------------------------|
| Train | 16,617 | ~15 min | ~28 min |
| Test  | 4,155  | ~4 min  | ~7 min  |

---

## What's Left for NQ

- [ ] Clean old zarr: `rm -rf shared/natural_questions_train_qwen3_8b/activations.zarr shared/natural_questions_qwen3_8b/activations.zarr`
- [ ] Run train split (~15–28 min with wins 1+2; ~70 min at BS=32 with current code)
- [ ] Run test split (~4–7 min with wins 1+2; ~17 min at BS=32 with current code)
- [ ] Create experiment config: `configs/experiments/baseline_comparison_nq_qwen3.json`
- [ ] Dispatch training runs via `gpu_dispatch.py`

## What's Left for Other Datasets (Qwen3)

The 6 remaining datasets (mmlu, popqa, sciq, searchqa, hotpotqa, movies) already have `run_step` + `batch_size` support. Before running them, implement the same speedups so they benefit too.

**Code changes (shared, implement once):**

- [ ] `ZarrActivationsLogger`: add pinned CPU buffer + `non_blocking=True` extraction (win 2 — 3.5–10× activation speedup)
- [ ] `ZarrActivationsLogger`: switch to batch-level zarr writes (win 1 — 40% disk speedup)
- [ ] `AsyncActivationWriter`: move `torch.cuda.synchronize()` into the writer thread to overlap PCIe transfer with GPU compute

**Per-dataset runs (after code changes):**

| Dataset | Est. samples (train+test) | Est. time @ 18 samp/s |
|---------|--------------------------|----------------------|
| mmlu | ~10,225 test + train TBD | ~10 min test |
| popqa | TBD | TBD |
| sciq | TBD | TBD |
| searchqa | TBD | TBD |
| hotpotqa | ~7,405 test + train TBD | ~7 min test |
| movies | TBD | TBD |

- [ ] Run each dataset at `--batch-size 32` with updated `ZarrActivationsLogger`
- [ ] Create experiment configs: `configs/experiments/baseline_comparison_{dataset}_qwen3.json` for each
