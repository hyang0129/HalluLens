# Batch Write Optimization Plan

**Target:** ~20 samp/s (up from 1.26 samp/s)  
**Batch size:** 32  
**Two changes only — no pinned buffer needed**

---

## Why 20 samp/s is achievable at BS=32

At BS=32 with batch writes:

| Stage | Time/batch | Samp/s |
|-------|-----------|--------|
| GPU inference | 1.662s | 19.3 |
| Activation extraction (per_layer_loop) | 0.051s | — |
| Disk (zarr_batch_slice) | 1.432s | 22.4 |

GPU (1.713s total) and disk (1.432s) are nearly matched. `AsyncActivationWriter` already runs zarr writes in a background thread, so disk for batch N overlaps with GPU for batch N+1. Sustained throughput → **~18–20 samp/s**, GPU-bound.

Pinned buffer (`non_blocking=True`) saves 0.039s/batch at BS=32 — ~2% gain, not worth the complexity.

---

## Change 1 — `ZarrActivationsLogger.log_batch()`

**File:** `activation_logging/zarr_activations_logger.py`

Currently `log_entry` is called once per sample inside the batch loop. Each call writes 74 individual zarr chunks (37 layers × 2 arrays), hitting the NFS IOPS ceiling at ~16 samp/s regardless of batch size.

Add a `log_batch(entries: list[tuple[str, dict]])` method that:
1. Stacks all BS samples' activations into arrays of shape `(BS, n_layers, T, H)`
2. Writes them as a single slice assignment: `self._prompt_activations[start:end] = batch_arr`

With chunk shape `(BS, 1, T, H)`, this maps the slice perfectly to chunk boundaries — 74 chunk writes total per batch instead of 74 × BS. Disk throughput shifts from IOPS-limited (16 samp/s) to bandwidth-limited (~22 samp/s).

The chunk shape is already configurable via `activation_chunk_shape`. Pass `(BS, 1, response_max_tokens, hidden_size)` at logger init. The `_resolve_activation_chunk_shape` method handles this; just ensure callers pass the right shape.

`log_entry` stays unchanged — still used by the vLLM server path (non-batched).

---

## Change 2 — `AsyncActivationWriter.enqueue_batch()`

**File:** `activation_logging/server.py`

Currently the batch loop calls `writer.enqueue(key, entry)` per sample ([hotpotqa.py:265](../../tasks/llmsknow/hotpotqa.py#L265), same pattern in all other tasks). Each enqueue becomes a separate `log_entry` call in the drain thread — per-sample writes, no batching benefit.

Add `enqueue_batch(entries: list[tuple[str, dict]])` that puts a single item on the queue representing the whole batch. The drain thread calls `log_batch` instead of `log_entry` when it dequeues a batch item.

```python
# queue item type: (key, entry, metadata_only) for single
#                  (entries_list, None, None)   for batch — drain detects by type
```

Or cleaner: use a dataclass/namedtuple to distinguish single vs batch queue items.

---

## Change 3 — Update call sites

**Files:** all `run_inference_batched` methods

Change from:
```python
for i, result in enumerate(results):
    writer.enqueue(key, build_entry(result))
```

To:
```python
batch_entries = [(key(result), build_entry(result)) for result in results]
writer.enqueue_batch(batch_entries)
```

Affected files:
- `tasks/llmsknow/hotpotqa.py` — `run_inference_batched`
- `tasks/llmsknow/mmlu.py` — `run_inference_batched`
- `tasks/llmsknow/natural_questions.py` — `run_inference_batched`
- Any other tasks with the same pattern

---

## What does NOT change

- `log_entry` — keep for the vLLM server path
- `AsyncActivationWriter.enqueue` — keep for single-sample paths
- Activation extraction in `HFTransformersAdapter` — per_layer_loop is fine at BS=32
- Chunk layer dimension — stays at 1; `(BS, N_LAYERS, T, H)` chunks are 2× slower (tested)

---

## Expected outcome

| Config | Samp/s |
|--------|--------|
| Current (BS=8, per-sample writes) | 1.26 |
| BS=32, per-sample writes (no change) | ~3–4 |
| BS=32, batch writes (this plan) | **~18–20** |

NQ train (16,617 samples): ~14 min  
NQ test (4,155 samples): ~3.5 min
