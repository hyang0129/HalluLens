# Inference bottleneck: 19s/sample on H100 due to sequential single-request pipeline

## Problem

Running Llama-3.1-8B-Instruct (fp16) on an H100, inference takes **~19 seconds per sample** with only **67% GPU utilization**. For 7,405 HotpotQA samples, that's ~39 hours wall time.

Expected throughput for an 8B model on H100 should be **1–3 seconds per sample**, roughly a 6–10x improvement.

## Root Cause Analysis

There are three bottlenecks stacking on top of each other:

### 1. Sequential single-request client (`utils/exp.py`)

`process_with_incremental_save()` sends **one request at a time** in a `for` loop, waiting for the full response before sending the next:

```python
# utils/exp.py:37-39
for idx, prompt in enumerate(tqdm(prompts, desc=desc)):
    generation = inference_fn(prompt)  # blocks until done
    f.write(json.dumps(record) + '\n')
```

The `max_workers` parameter is documented as `DEPRECATED` (line 86) — the system is intentionally single-threaded.

### 2. Global inference lock on the server (`activation_logging/server.py:557`)

Even if the client sent concurrent requests, a `threading.Lock()` serializes all inference on the server:

```python
# server.py:557
inference_lock = threading.Lock()

# server.py:1901
with inference_lock:
    response_text, model_outputs, input_length, trim_pos = run_inference(...)
```

This completely prevents any batching — the GPU processes one sequence at a time, which explains the 67% utilization (idle between requests).

### 3. Synchronous activation logging blocks the response (`server.py:1930-1959`)

After each inference, the server writes activations to Zarr **synchronously** before returning the HTTP response. For 32 layers × 4096 hidden dims, each Zarr `log_entry` call involves multiple array resizes and numpy conversions that block the next inference from starting.

### Why the server uses HF Transformers, not vLLM

This is a critical architectural constraint: the custom server uses `model.generate(output_hidden_states=True)` to capture per-layer hidden states for activation logging. The vLLM inference path explicitly **does not provide hidden states** (`server.py:254`):

> "vLLM inference path does NOT provide hidden states, so activation logging will be skipped"

So the inference lock isn't just being conservative — `model.generate()` is not thread-safe on a shared GPU model, and activation capture requires the HF transformers path.

## Proposed Solution

### Part 1: Server-side request batching (`BatchInferenceQueue`)

Instead of processing one prompt at a time, accumulate incoming requests and run a single **batched** `model.generate()` call:

- A `BatchInferenceQueue` collects incoming requests up to `max_batch_size` (default: 8) or a short timeout (e.g., 50ms)
- Prompts are tokenized together with left-padding into a single `(batch_size, max_seq_len)` tensor
- `model.generate()` runs once on the full batch — the GPU processes multiple sequences simultaneously
- Hidden states are split per-sequence on `dim=0` (the batch dimension) using each request's index

This is the highest-impact change: batching gives the GPU more work per forward pass, directly improving utilization.

#### Data integrity guarantees for batching

The critical invariant: **each activation log entry must be keyed to exactly the prompt+response that produced it.**

1. **Entry key binding**: Each request's `entry_key` is computed from its prompt hash *before* it enters the batch. The key travels with the request through the full pipeline: batching → inference → activation split → Zarr write.

2. **Hidden state demuxing**: HF `model.generate()` with `batch_size > 1` returns hidden states with shape `(batch_size, seq_len, hidden_dim)`. We split on `dim=0` using each request's index in the batch — no cross-contamination is possible since `dim=0` is the batch dimension. Each request's `input_length` is tracked individually for correct prompt/response splitting.

3. **Per-request output extraction**: Generated token sequences are split using each request's known prompt length (`input_ids.shape[1]` per sample after padding), ensuring the correct response text is paired with the correct activation entry.

### Part 2: Async activation I/O (`AsyncActivationWriter`)

Decouple activation logging from the inference→response path:

- After batch inference completes, each `(entry_key, entry_dict)` pair is pushed onto a write queue
- A dedicated background writer thread drains the queue into Zarr
- The HTTP response is returned to the client immediately, without waiting for Zarr writes
- Graceful shutdown drains the queue before server exit, ensuring no activations are lost

Each queued entry is self-contained (key + all data needed for `log_entry`), so write ordering doesn't affect correctness — Zarr entries are keyed, not ordered.

### Part 3: Concurrent client requests (`utils/exp.py`)

Update the client to send multiple requests concurrently:

- Replace the sequential `for` loop with `asyncio` + `openai.AsyncOpenAI`
- Use a semaphore to limit in-flight requests (e.g., 16) matching the server's batch capacity
- Results are still saved incrementally to JSONL as they complete (resume-safe)
- Backward compatible: `max_concurrent=1` preserves current sequential behavior

### Component summary

| Component | Location | Role |
|---|---|---|
| `BatchInferenceQueue` | `activation_logging/server.py` | Collects requests, runs batched `model.generate()`, returns per-request results |
| `AsyncActivationWriter` | `activation_logging/server.py` | Background thread draining a write queue into Zarr |
| `process_with_concurrent_requests` | `utils/exp.py` | Client sends N requests concurrently via asyncio |

## Expected Impact

| Metric | Current | After fix |
|---|---|---|
| Per-sample latency | ~19s | ~2–4s |
| GPU utilization | ~67% | ~90%+ |
| HotpotQA full run (7,405 samples) | ~39 hours | ~4–8 hours |
| Throughput (samples/sec) | ~0.05 | ~0.3–0.5 |

## Risks & Mitigations

- **Memory pressure from batching**: Larger batches use more GPU memory for KV cache + hidden states. Mitigate with configurable `max_batch_size` (start with 4–8 for 8B model on H100).
- **Padding overhead**: Prompts of varying length need padding. Left-padding with attention masks is standard for causal LMs and well-supported by HF.
- **Async write failures**: If the writer thread crashes, activations in the queue are lost. Mitigate by catching exceptions per-entry and logging failures without killing the thread.
- **Backward compatibility**: All new parameters have defaults matching current behavior (`max_batch_size=1`, `max_concurrent=1`), so existing scripts work unchanged.
