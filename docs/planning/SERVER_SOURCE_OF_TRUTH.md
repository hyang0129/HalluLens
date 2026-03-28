# Server as Single Source of Truth for Inference Records

**Issues:** [#6](https://github.com/hyang0129/HalluLens/issues/6) (single source of truth), [#7](https://github.com/hyang0129/HalluLens/issues/7) (inference bottleneck)
**Status:** Draft plan — pending review

---

## 1. Problem Statement

During `run_with_server.py` inference, both the client and server independently write records:

| Writer | File | Contents |
|--------|------|----------|
| **Client** (`utils/exp.py`) | `generation.jsonl` | All QA metadata + prompt + generation text |
| **Server** (`zarr_activations_logger.py`) | `index.jsonl` + Zarr arrays | prompt + response + activations + logprobs + logging config |

A crash between the server write and the client write leaves the two stores inconsistent. Resume requires cross-checking both. The upcoming selfcheck step (multiple samples per prompt) multiplies this problem.

Additionally, this dual-write architecture is a direct contributor to the inference bottleneck ([#7](https://github.com/hyang0129/HalluLens/issues/7)): the client blocks on each response to write JSONL before sending the next request, and the server blocks on synchronous Zarr writes before returning the response. Solving the source-of-truth problem unlocks the batching and concurrency improvements needed to fix throughput.

### What's already in Zarr index metadata

The server already stores prompt and response text in `index.jsonl` — the `log_entry` metadata filter (line 697-709) only strips tensor fields (`model_outputs`, `all_layers_activations`, logprob tensors). Everything else from the entry dict is persisted, including:

- `prompt` (the full prompt text)
- `response` (the full response text)
- `prompt_hash`, `model`, `input_length`, `trim_position`
- `multi_sample`, `sample_group_id`, `sample_index`, `request_id`

### What's in `generation.jsonl` that the server doesn't have

The client writes `all_prompts.iloc[idx].to_dict()` plus `generation` — this includes QA dataset fields:

- `answer` (gold truth)
- `title`, `pageid`, `revid`, `description`, `reference`, `categories`, `h_score_cat`

These are **input dataset fields**, not inference outputs. They originate from the QA generation step and are already stored in the QA output file that the task loads before inference.

---

## 2. Design Principle

**The server is the single source of truth for inference outputs** (response text + activations). Dataset fields (gold answers, metadata) stay in the dataset layer — they don't need to be duplicated into inference records.

`generation.jsonl` is eliminated as a client-written file. Instead:

- **Resume** reads the server's Zarr index to determine which prompts are complete
- **Eval** joins the QA dataset with server records on `prompt_hash`
- **Training** (`ActivationParser`) joins similarly — it already loads both `inference_json` and Zarr

---

## 3. Current Data Flow (what changes)

### 3.1 Inference step today

```
QA file ─→ precise_wikiqa.py ─→ run_exp() ─→ call_vllm_api() ─→ server
                                     │                              │
                                     ▼                              ▼
                              generation.jsonl              Zarr arrays + index.jsonl
                              (client writes)               (server writes)
```

**Resume:** `run_exp()` loads `generation.jsonl`, extracts completed prompt set, filters remaining prompts (`exp.py:156-159`).

### 3.2 Inference step after

```
QA file ─→ precise_wikiqa.py ─→ run_exp() ─→ call_vllm_api() ─→ server
                                     │                              │
                                     │ (reads for resume)           ▼
                                     └──────────────────── Zarr arrays + index.jsonl
                                                           (server writes — single source)
```

**Resume:** `run_exp()` queries Zarr index for completed `prompt_hash` values, filters remaining prompts.

**Eval/Training:** join QA dataset + Zarr index on `prompt_hash` to reconstruct what `generation.jsonl` provided.

### 3.3 Combined architecture (single source + batching)

Solving #6 and #7 together eliminates two sequential bottlenecks from the same pipeline. The current chain:

```
request → server inference → server Zarr write (sync) → HTTP response → client JSONL write → next request
```

Becomes:

```
Client (async, N concurrent)           Server
────────────────────────               ──────
send request 1 ──────────────→  ┐
send request 2 ──────────────→  ├──→ BatchInferenceQueue
send request 3 ──────────────→  │      collects up to max_batch_size or timeout
  ...                           │
                                ├──→ model.generate(batch)
                                │      hidden_states shape: (batch, seq, hidden)
                                │
                                ├──→ per request in batch:
                                │      compute entry_key (prompt_hash)
                                │      enqueue (key, entry) → AsyncWriteQueue
                                │      return HTTP response immediately
                                │
←── response 1 (text + id) ────┘
←── response 2 (text + id) ────┘
←── response 3 (text + id) ────┘

                                     AsyncActivationWriter (background thread)
                                       drains queue → log_entry() or log_metadata()
                                       Zarr index.jsonl is the single resume record

Client: no writes — just sends requests
Resume: read Zarr index for completed prompt_hashes
```

**Why the two issues are synergistic:**

1. **Async activation writes become clean.** The `AsyncActivationWriter` pushes entries onto a background queue. With #6, the server owns all records — the HTTP response returns immediately, the client fires the next request, and the background writer handles persistence. No client-side write coordination needed.

2. **Concurrent client becomes trivial.** With concurrent in-flight requests, thread-safe JSONL writes with ordering guarantees would be needed. With #6, the client writes nothing — concurrency is fire-and-forget.

3. **Resume with concurrent requests just works.** A crash with N in-flight requests leaves ambiguous state if the client owns the JSONL. With #6, resume reads the Zarr index — it sees exactly what the server successfully wrote, regardless of client state at crash time.

4. **Batched inference + `log_metadata` for selfcheck.** The batch can mix activation-logged and text-only requests in one `model.generate()` call. The async writer dispatches `log_entry()` or `log_metadata()` per request based on its flags.

---

## 4. Required Changes

### 4.1 `ZarrActivationsLogger` — metadata-only entries

**File:** `activation_logging/zarr_activations_logger.py`

Add `log_metadata(key, metadata)` for entries that have no activation arrays (text-only selfcheck samples, or any future use case where we want the record without the heavy arrays).

```python
def log_metadata(self, key: str, metadata: Dict[str, Any]):
    """Write an index entry with no activation arrays."""
    if self.read_only:
        raise ValueError("Cannot log entries in read-only mode")
    self._ensure_dirs()

    metadata_entry = {
        k: v for k, v in metadata.items()
        if k not in {"model_outputs", "all_layers_activations",
                     "response_token_ids", "response_token_logprobs",
                     "response_topk_token_ids", "response_topk_logprobs"}
    }
    metadata_entry["key"] = key
    metadata_entry["sample_index"] = None   # no Zarr array row
    metadata_entry["has_activations"] = False

    with open(self._index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata_entry) + "\n")
    self._index[key] = metadata_entry
```

**Why:** This decouples "record exists" from "activations exist." The selfcheck step needs metadata-only entries for text-only samples. Even for the current inference step, having this capability means the server can record a response even if activation extraction fails.

### 4.2 `ZarrActivationsLogger` — key-exists overwrite

**File:** `activation_logging/zarr_activations_logger.py`

Add a key-existence check to `log_entry`. If the key already exists and has a Zarr row (`sample_index is not None`), overwrite that row instead of appending a new one.

```python
# At the top of log_entry(), after activation extraction:
if key in self._index:
    existing = self._index[key]
    existing_idx = existing.get("sample_index")
    if existing_idx is not None:
        idx = existing_idx  # reuse existing Zarr row
    else:
        # Upgrading a metadata-only entry to a full entry
        idx = self._prompt_activations.shape[0]
        # resize arrays...
else:
    idx = self._prompt_activations.shape[0]
    # resize arrays...
```

**Why:** Prevents orphaned Zarr rows on retry/resume. If a crash occurs after the server writes but before the client acknowledges, the retry writes to the same row.

### 4.3 `ZarrActivationsLogger.get_entry` — handle metadata-only entries

**File:** `activation_logging/zarr_activations_logger.py`, lines 740-746

Current code raises `KeyError` when `sample_index is None` (line 742-743) **before** checking `metadata_only` (line 745-746). Reorder:

```python
def get_entry(self, key: str, metadata_only: bool = False) -> Dict[str, Any]:
    if key not in self._index:
        raise KeyError(f"Key {key} not found in Zarr index")

    meta = dict(self._index[key])
    idx = meta.get("sample_index")

    if metadata_only or idx is None:
        return meta  # metadata-only entries or explicit metadata request

    # ... existing activation retrieval code ...
```

**Why:** Metadata-only entries (from `log_metadata`) have `sample_index=None`. The current code crashes on them even when the caller only wants metadata.

### 4.4 `utils/exp.py` — resume from Zarr index

**File:** `utils/exp.py`, `run_exp()` function, lines 140-184

Replace the `generation.jsonl`-based resume with Zarr index-based resume:

```python
# Current (lines 145-159):
if resume and Path(generations_file_path).exists():
    existing_generations = pd.read_json(generations_file_path, lines=True)
    existing_prompts = set(existing_generations['prompt'].tolist())
    mask = ~all_prompts['prompt'].isin(existing_prompts)

# New:
if resume and activations_path:
    from activation_logging.zarr_activations_logger import ZarrActivationsLogger
    logger = ZarrActivationsLogger(activations_path, read_only=True)
    completed_hashes = set()
    for key, meta in logger._index.items():
        ph = meta.get("prompt_hash")
        if ph:
            completed_hashes.add(ph)
    logger.close()

    all_prompts['_prompt_hash'] = all_prompts['prompt'].apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    mask = ~all_prompts['_prompt_hash'].isin(completed_hashes)
    all_prompts = all_prompts[mask].drop(columns=['_prompt_hash'])
```

**Note:** `run_exp` needs the `activations_path` parameter added to its signature. When `activations_path` is not provided (non-activation runs), fall back to `generation.jsonl` resume for backward compatibility.

**`generation.jsonl` is no longer written** by the client during inference. The `process_with_incremental_save` call is removed from the inference path. If a `generation.jsonl` file is needed for eval or other downstream use, it can be **exported** from the Zarr index after inference completes (see §4.6).

### 4.5 `activation_logging/activation_parser.py` — load from Zarr index

**File:** `activation_logging/activation_parser.py`, `_load_metadata()`, lines 753-793

Currently loads `inference_json` (generation.jsonl) and merges with `eval_json`. Change to load inference data from the Zarr index and join with the QA dataset for gold answers:

```python
def _load_metadata(self) -> pd.DataFrame:
    # Load inference records from Zarr index (server source of truth)
    index_records = []
    for key, meta in self.activation_logger._index.items():
        if meta.get("sample_index") is not None:  # has activations
            index_records.append({
                "prompt_hash": meta.get("prompt_hash", key.split("_")[0]),
                "prompt": meta.get("prompt"),
                "generation": meta.get("response"),
            })
    gendf = pd.DataFrame(index_records)

    # Load eval labels
    with open(self.eval_json, 'r') as f:
        data = json.loads(f.read())

    # Join with QA dataset for gold answers if inference_json provided
    if self.inference_json:
        qa_df = pd.read_json(self.inference_json, lines=True)
        qa_df['prompt_hash'] = qa_df['prompt'].apply(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        )
        # Merge QA metadata (answer, title, etc.) onto Zarr-sourced records
        gendf = gendf.merge(
            qa_df[['prompt_hash', 'answer', 'title', 'h_score_cat', 'pageid',
                   'revid', 'description', 'categories', 'reference']],
            on='prompt_hash', how='left'
        )

    # ... rest unchanged (add abstain/halu, deduplicate, split) ...
```

**Backward compatibility:** If `inference_json` is provided (existing workflows), it still works — the QA metadata is joined in. The `generation` field now comes from Zarr rather than from the JSONL, but contains the same text.

### 4.6 Export utility — reconstruct `generation.jsonl` from Zarr

For any downstream script that expects `generation.jsonl`, provide an export function:

```python
def export_generation_jsonl(activations_path: str, qa_file: str, output_path: str):
    """Reconstruct generation.jsonl from Zarr index + QA dataset."""
    logger = ZarrActivationsLogger(activations_path, read_only=True)
    qa_df = pd.read_json(qa_file, lines=True)
    # ... join and write ...
```

This is a convenience for backward compatibility, not a required part of the pipeline.

### 4.7 `tasks/shortform/precise_wikiqa.py` — eval reads from Zarr or export

**File:** `tasks/shortform/precise_wikiqa.py`, `PreciseWikiQAEval.__init__`, line 124

The eval step reads `generation.jsonl` via `pd.read_json(self.generations_file_path, lines=True)`. Two options:

**Option A (minimal change):** After inference completes, export `generation.jsonl` from Zarr index before eval starts. Eval code unchanged.

**Option B (clean):** Eval loads from Zarr index + QA dataset directly, same join pattern as §4.5.

**Recommended: Option A** for the initial implementation — it's less invasive and keeps eval decoupled from the Zarr layer. Option B can follow later.

### 4.8 `AsyncActivationWriter` — decouple Zarr writes from inference

**File:** `activation_logging/server.py` (new class)

Currently, `log_entry()` runs synchronously inside the request handler — the HTTP response waits for Zarr array resizes and numpy conversions. Decouple this:

```python
class AsyncActivationWriter:
    """Background thread that drains a write queue into Zarr."""

    def __init__(self, logger: ZarrActivationsLogger, max_queue_size: int = 256):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._logger = logger
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._shutdown = threading.Event()
        self._thread.start()

    def enqueue(self, key: str, entry: dict, metadata_only: bool = False):
        """Non-blocking enqueue. Called from request handler after inference."""
        self._queue.put((key, entry, metadata_only))

    def _drain(self):
        """Writer thread — processes entries until shutdown."""
        while not self._shutdown.is_set() or not self._queue.empty():
            try:
                key, entry, metadata_only = self._queue.get(timeout=0.1)
                if metadata_only:
                    self._logger.log_metadata(key, entry)
                else:
                    self._logger.log_entry(key, entry)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"AsyncWriter failed for key {key}: {e}")
                # Log error but don't kill the thread — other entries can still succeed

    def shutdown(self, timeout: float = 30.0):
        """Drain remaining entries and stop."""
        self._shutdown.set()
        self._thread.join(timeout=timeout)
        remaining = self._queue.qsize()
        if remaining > 0:
            logger.warning(f"AsyncWriter shutdown with {remaining} entries still in queue")
```

The request handler becomes:

```python
# Before (synchronous — blocks response):
if not model_name.endswith('.gguf') and model_outputs is not None:
    logger_to_use.log_entry(entry_key, {...})

# After (async — response returns immediately):
if not model_name.endswith('.gguf') and model_outputs is not None:
    async_writer.enqueue(entry_key, {...})
```

**Resume safety:** The Zarr index is the resume record. If the server crashes with entries in the queue, those prompts will be re-sent on resume (they won't appear in the index). The key-exists overwrite (§4.2) handles the case where the entry was partially written.

### 4.9 `BatchInferenceQueue` — server-side request batching

**File:** `activation_logging/server.py` (new class)

Replace the global `inference_lock` with a batching queue that accumulates requests and runs a single `model.generate()` call on the batch:

```python
class BatchInferenceQueue:
    """Collects incoming requests, runs batched model.generate(), returns per-request results."""

    def __init__(self, model, tokenizer, max_batch_size: int = 8, max_wait_ms: float = 50):
        self._model = model
        self._tokenizer = tokenizer
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._pending = []           # list of (prompt, params, future)
        self._lock = threading.Lock()
        self._batch_ready = threading.Event()

    def submit(self, prompt: str, params: dict) -> concurrent.futures.Future:
        """Submit a single request. Returns a Future with (response_text, model_outputs, input_length, trim_pos)."""
        future = concurrent.futures.Future()
        with self._lock:
            self._pending.append((prompt, params, future))
            if len(self._pending) >= self._max_batch_size:
                self._batch_ready.set()
        return future

    def _collect_batch(self) -> list:
        """Wait for batch to fill or timeout, then return collected requests."""
        self._batch_ready.wait(timeout=self._max_wait)
        with self._lock:
            batch = self._pending[:self._max_batch_size]
            self._pending = self._pending[self._max_batch_size:]
            self._batch_ready.clear()
            return batch

    def _run_batch(self, batch: list):
        """Tokenize, run model.generate() on batch, split results per request."""
        prompts = [b[0] for b in batch]
        params_list = [b[1] for b in batch]
        futures = [b[2] for b in batch]

        # Left-pad tokenization
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self._model.device)

        # Track per-request input lengths (before padding)
        input_lengths = [len(self._tokenizer.encode(p)) for p in prompts]

        # Batched generation
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max(p.get("max_tokens", 512) for p in params_list),
                temperature=params_list[0].get("temperature", 0.0) or 1.0,
                do_sample=any(p.get("temperature", 0.0) > 0 for p in params_list),
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Split results per request
        for i, future in enumerate(futures):
            try:
                # Extract per-request generated tokens and hidden states
                gen_ids = outputs.sequences[i]
                response_text = self._tokenizer.decode(
                    gen_ids[inputs.input_ids.shape[1]:], skip_special_tokens=True
                )
                # Extract per-request hidden states (dim=0 split)
                per_request_outputs = _extract_per_request_outputs(outputs, i)
                future.set_result((response_text, per_request_outputs, input_lengths[i], None))
            except Exception as e:
                future.set_exception(e)
```

The request handler submits to the queue and waits on the future:

```python
# Before:
with inference_lock:
    response_text, model_outputs, input_length, trim_pos = run_inference(prompt, ...)

# After:
future = batch_queue.submit(prompt, params)
response_text, model_outputs, input_length, trim_pos = future.result()
```

**Data integrity:** Each request's `entry_key` is computed from its `prompt_hash` before entering the batch. Hidden states are split on `dim=0` (the batch dimension) — no cross-contamination. Each request's `input_length` is tracked individually for correct prompt/response splitting.

**Mixed temperature batches:** If a batch contains both greedy (t=0) and stochastic (t>0) requests, the simplest approach is to separate them into two sub-batches by temperature. A more advanced approach uses per-token temperature masking but adds complexity.

### 4.10 Concurrent client requests

**File:** `utils/exp.py` (new function)

Replace `process_with_incremental_save` with an async variant for the inference path:

```python
async def process_concurrent(all_prompts, model, activations_path,
                             max_concurrent=16, temperature=0.0, **kwargs):
    """Send requests concurrently. Server owns all record-keeping."""
    client = openai.AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def send_one(prompt):
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return response.choices[0].message.content

    tasks = [send_one(row.prompt) for _, row in all_prompts.iterrows()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Errors logged, not raised — server has the record for successful ones
    return results
```

**No client writes.** The server records every completed inference in the Zarr index. The client's only job is to send requests and observe completion. On resume, the client reads the Zarr index for completed `prompt_hash` values and only sends remaining prompts.

**Backward compatible:** `max_concurrent=1` with `asyncio.run()` is functionally equivalent to the current sequential loop.

---

## 5. What Stays the Same

- Server API contract (same endpoints, same request/response models)
- Zarr array format and layout
- `generate` step (QA generation — no activations, no server involvement)
- `eval` step (reads `generation.jsonl` — provided by export in Option A)
- HF Transformers inference path with `output_hidden_states=True` (required for activation capture)
- All new parameters default to current behavior (`max_batch_size=1`, `max_concurrent=1`)

---

## 6. Migration / Backward Compatibility

- Existing `generation.jsonl` files continue to work — `ActivationParser` falls back to them when provided
- `run_exp` supports both resume paths: Zarr index (when `activations_path` given) and JSONL (when not)
- The export utility (§4.6) can regenerate `generation.jsonl` from any existing Zarr store

---

## 7. Implementation Order

### Phase 1: Zarr logger foundation (§4.1–4.3) — enables both #6 and #7

| Step | Change | Files | Risk |
|------|--------|-------|------|
| 1a | `log_metadata()` method | `zarr_activations_logger.py` | Low — additive |
| 1b | Key-exists overwrite in `log_entry()` | `zarr_activations_logger.py` | Medium — changes write behavior |
| 1c | Fix `get_entry()` for metadata-only entries | `zarr_activations_logger.py` | Low — reorder two lines |

### Phase 2: Async activation writer (§4.8) — immediate latency improvement

| Step | Change | Files | Risk |
|------|--------|-------|------|
| 2 | `AsyncActivationWriter` + wire into request handlers | `server.py` | Medium — moves writes off request path |

This is a pure server change. Even with `batch_size=1` and a sequential client, it immediately reduces per-request latency by decoupling Zarr I/O from the HTTP response.

### Phase 3: Client becomes stateless (§4.4–4.7) — can parallelize with Phase 2

| Step | Change | Files | Risk |
|------|--------|-------|------|
| 3a | Resume from Zarr index in `run_exp()` | `utils/exp.py` | Medium — changes resume path |
| 3b | Stop writing `generation.jsonl` during inference | `utils/exp.py` | Medium — removes client writes |
| 3c | Export utility + wire into `--step all` | `utils/exp.py`, `run_with_server.py` | Low — new code |
| 3d | `ActivationParser` reads from Zarr index | `activation_parser.py` | Low — additive |

### Phase 4: Server-side batching (§4.9) — biggest throughput impact

| Step | Change | Files | Risk |
|------|--------|-------|------|
| 4 | `BatchInferenceQueue` replacing `inference_lock` | `server.py` | High — changes core inference path |

Requires Phase 2 (async writes) to be effective. Requires careful testing of hidden state demuxing and left-padding correctness.

### Phase 5: Concurrent client (§4.10) — feeds the batch queue

| Step | Change | Files | Risk |
|------|--------|-------|------|
| 5 | `process_concurrent` with `AsyncOpenAI` | `utils/exp.py` | Medium — new async code |

Trivial once the client is stateless (Phase 3). The semaphore (`max_concurrent`) should be tuned to match `max_batch_size` on the server.

### Dependency graph

```
Phase 1 (Zarr logger)
   ├──→ Phase 2 (async writer)  ──→ Phase 4 (batching) ──→ Phase 5 (concurrent client)
   └──→ Phase 3 (stateless client) ──────────────────────↗
```

Phases 2 and 3 can be developed in parallel. Phase 4 depends on Phase 2. Phase 5 depends on Phases 3 and 4.

---

## 8. Open Questions

1. **Should the server store the `answer` (gold truth)?** Currently no — it's a dataset field, not an inference output. The client could pass it as extra metadata in the request, but this mixes concerns. Recommendation: keep it in the dataset layer and join at read time.

2. **Should `call_vllm_api` return structured data?** It currently returns only the text string. Returning a dict/dataclass with `{text, id, prompt_hash}` would make the entry_key accessible to the client without recomputing. This is useful for selfcheck but not strictly required for this issue.

3. **LMDB logger parity.** The LMDB logger (`activations_logger.py`) has a similar metadata flow. Should it get the same `log_metadata` / key-exists changes? Recommendation: defer — LMDB is the older format and Zarr is preferred for new experiments.

4. **Mixed temperature batches.** The `BatchInferenceQueue` may receive both greedy (t=0) and stochastic (t>0) requests simultaneously (e.g., during selfcheck). Options: (a) separate into sub-batches by temperature, (b) per-token temperature masking, (c) require clients to coordinate. Recommendation: (a) — simple, correct, minor efficiency cost.

5. **Async writer queue backpressure.** If inference is faster than Zarr I/O (likely with batching), the write queue grows unboundedly. The `maxsize` parameter on `Queue` provides backpressure — when full, `enqueue()` blocks, which slows the request handler. Acceptable tradeoff: Zarr I/O is the actual bottleneck and should match inference rate for steady state.

6. **`max_batch_size` tuning.** Depends on model size, sequence length, and GPU memory. For 8B model on H100 (80GB), batch sizes of 4–8 should be safe with 512 max_tokens. Need empirical testing with activation capture enabled (hidden states for all layers multiply memory usage).

---

## 9. Expected Impact (combined #6 + #7)

| Metric | Current | Phase 2 only | Phase 2+3 | All phases |
|--------|---------|--------------|-----------|------------|
| Per-sample latency | ~19s | ~15s | ~15s | ~2–4s |
| GPU utilization | ~67% | ~70% | ~70% | ~90%+ |
| HotpotQA (7,405 samples) | ~39h | ~31h | ~31h | ~4–8h |
| Resume correctness | Fragile (dual store) | Fragile | Robust (single source) | Robust |
| Concurrent requests | 1 | 1 | 1 | 8–16 |

Phase 2 alone gives a modest improvement (Zarr I/O off the critical path). The big win comes from Phase 4 (batching) + Phase 5 (concurrent client) together — the GPU finally stays busy.
