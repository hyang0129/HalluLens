# Implementation Plan: Server Source of Truth + Inference Batching

**Issues:** [#6](https://github.com/hyang0129/HalluLens/issues/6), [#7](https://github.com/hyang0129/HalluLens/issues/7)
**Design doc:** [SERVER_SOURCE_OF_TRUTH.md](SERVER_SOURCE_OF_TRUTH.md)
**Bottleneck analysis:** [INFERENCEISSUE.MD](INFERENCEISSUE.MD)

---

## Phase 1: Zarr Logger Foundation

**Goal:** Make `ZarrActivationsLogger` capable of metadata-only entries and safe overwrites.
**File:** `activation_logging/zarr_activations_logger.py`
**Risk:** Low — additive changes, no existing behavior modified.

### 1a. Add `log_metadata()` method

**After** line 727 (end of `log_entry`), add:

```python
def log_metadata(self, key: str, metadata: Dict[str, Any]):
    """Write an index-only entry with no activation arrays.

    Use for text-only selfcheck samples or any record where activations
    are not available/needed. The entry is visible to list_entries(),
    get_entry(metadata_only=True), and _build_group_index().
    """
    if self.read_only:
        raise ValueError("Cannot log entries in read-only mode")
    self._ensure_dirs()

    metadata_entry = {
        k: v for k, v in metadata.items()
        if k not in {
            "model_outputs", "all_layers_activations",
            "response_token_ids", "response_token_logprobs",
            "response_topk_token_ids", "response_topk_logprobs",
        }
    }
    metadata_entry["key"] = key
    metadata_entry["sample_index"] = None
    metadata_entry["has_activations"] = False

    with open(self._index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata_entry, ensure_ascii=False) + "\n")
    self._index[key] = metadata_entry
```

Also add the method to the `ActivationsLogger` wrapper class in `activations_logger.py` so it delegates to the backend:

```python
def log_metadata(self, key: str, metadata: Dict[str, Any]):
    if self._backend is not None:
        return self._backend.log_metadata(key, metadata)
    raise NotImplementedError("log_metadata only supported for Zarr backend")
```

**Test:** Write a metadata-only entry, verify it appears in `list_entries()`, verify `get_entry(key, metadata_only=True)` returns it, verify `get_entry(key, metadata_only=False)` returns it without crashing (see 1c).

### 1b. Add key-exists overwrite to `log_entry()`

**At** line 622 (currently `idx = self._prompt_activations.shape[0]`), replace with:

```python
# Check if key already exists — overwrite existing row to prevent orphans
if key in self._index:
    existing_meta = self._index[key]
    existing_idx = existing_meta.get("sample_index")
    if existing_idx is not None:
        # Reuse existing Zarr array row
        idx = existing_idx
        logger.info(f"Overwriting existing entry {key} at Zarr row {idx}")
    else:
        # Upgrading metadata-only entry to full entry — append new row
        idx = self._prompt_activations.shape[0]
        self._prompt_activations.resize((idx + 1, num_layers, self.prompt_max_tokens, hidden_size))
        self._response_activations.resize((idx + 1, num_layers, self.response_max_tokens, hidden_size))
        self._prompt_len.resize((idx + 1,))
        self._response_len.resize((idx + 1,))
        self._sample_key.resize((idx + 1,))
        # ... resize logprob arrays if present ...
else:
    idx = self._prompt_activations.shape[0]
    self._prompt_activations.resize((idx + 1, num_layers, self.prompt_max_tokens, hidden_size))
    # ... existing resize block (lines 625-637) ...
```

The existing resize block at lines 625-637 moves into the `else` branch. The overwrite branch skips resizing — the row already exists at the right shape.

**Note:** `_ensure_arrays` (line 622) must be called before the key check since we need `num_layers` and `hidden_size` from activation extraction (lines 613-620). The key check goes between `_ensure_arrays` and the current resize block.

**Test:** Log an entry with key "foo", log again with key "foo" and different response text. Verify only one Zarr row exists, verify the index points to the updated metadata, verify no orphaned rows.

### 1c. Fix `get_entry()` for metadata-only entries

**At** lines 740-746, change:

```python
# Current (lines 740-746):
meta = dict(self._index[key])
idx = meta.get("sample_index")
if idx is None:
    raise KeyError(f"Missing sample_index for key {key}")

if metadata_only:
    return meta
```

To:

```python
meta = dict(self._index[key])
idx = meta.get("sample_index")

# Return early for metadata-only requests or entries without activations
if metadata_only or idx is None:
    return meta
```

**Test:** Create a metadata-only entry via `log_metadata()`. Call `get_entry(key)` (no flags) — should return metadata dict without crashing. Call `get_entry(key, metadata_only=True)` — same. Call `get_entry(key, metadata_only=False)` — same (idx is None, returns metadata).

---

## Phase 2: Async Activation Writer

**Goal:** Decouple Zarr I/O from the HTTP response path. Immediate latency improvement.
**File:** `activation_logging/server.py`
**Risk:** Medium — moves writes off the request thread.

### 2a. Replace per-request logger with persistent logger

**Problem:** `get_logger_for_request()` (line 1384) creates a new `ActivationsLogger` per request. The docstring says "to avoid LMDB assertion errors" — this is a legacy LMDB concern. With Zarr-only (enforced at line 1399), a single persistent logger is safe and required for the async writer.

**At** server startup (`startup_event`, line 1992), create a persistent logger:

```python
# New globals
_persistent_logger: Optional[ActivationsLogger] = None
_async_writer: Optional[AsyncActivationWriter] = None

@app.on_event("startup")
async def startup_event():
    global _persistent_logger, _async_writer
    # ... existing startup code ...

    # Create persistent Zarr logger
    activations_path = DEFAULT_ACTIVATIONS_PATH
    if activations_path and activations_path.strip().endswith(".zarr"):
        target_layers = os.environ.get("ACTIVATION_TARGET_LAYERS", "all")
        sequence_mode = os.environ.get("ACTIVATION_SEQUENCE_MODE", "all")
        _persistent_logger = ActivationsLogger(
            lmdb_path=activations_path,
            map_size=DEFAULT_MAP_SIZE,
            target_layers=target_layers,
            sequence_mode=sequence_mode,
        )
        _async_writer = AsyncActivationWriter(_persistent_logger)
        logger.info(f"Persistent Zarr logger created at {activations_path}")
```

**At** shutdown (`shutdown_event`, line 2097):

```python
@app.on_event("shutdown")
async def shutdown_event():
    global _async_writer
    # ... existing shutdown code ...

    if _async_writer:
        _async_writer.shutdown(timeout=30.0)
        logger.info("AsyncActivationWriter drained and stopped")
```

**Modify** `get_logger_for_request()` to return the persistent logger when the path matches, falling back to per-request creation for non-default paths:

```python
def get_logger_for_request(request_params):
    activations_path = request_params.get('activations_path', request_params.get('lmdb_path'))
    # ... validation ...

    # Use persistent logger if path matches
    if _persistent_logger and activations_path == DEFAULT_ACTIVATIONS_PATH:
        return _persistent_logger, None, False  # None = no custom logger to close

    # Otherwise create per-request (for custom paths)
    # ... existing creation code ...
```

Update callers to skip `logger_to_use.close()` when the custom logger is `None`:

```python
logger_to_use, custom_logger, _ = get_logger_for_request(params)
try:
    # ... log_entry ...
finally:
    if custom_logger:
        custom_logger.close()
```

### 2b. Add `AsyncActivationWriter` class

**Add** to `activation_logging/server.py` (after imports, before endpoint definitions):

```python
import queue
import threading

class AsyncActivationWriter:
    """Background thread that drains activation write requests into Zarr.

    Entries are self-contained (key + all data for log_entry/log_metadata),
    so write ordering doesn't affect correctness.
    """

    def __init__(self, logger_instance, max_queue_size: int = 256):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._logger = logger_instance
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="async-activation-writer", daemon=True)
        self._errors: int = 0
        self._written: int = 0
        self._thread.start()

    def enqueue(self, key: str, entry: dict, metadata_only: bool = False):
        """Non-blocking enqueue. Called from request handler after inference completes."""
        self._queue.put((key, entry, metadata_only))

    def _drain(self):
        while not self._shutdown.is_set() or not self._queue.empty():
            try:
                key, entry, metadata_only = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if metadata_only:
                    self._logger.log_metadata(key, entry)
                else:
                    self._logger.log_entry(key, entry)
                self._written += 1
            except Exception as e:
                self._errors += 1
                logger.error(f"AsyncWriter failed for key {key}: {e}", exc_info=True)

    def shutdown(self, timeout: float = 30.0):
        """Signal shutdown and wait for queue to drain."""
        self._shutdown.set()
        self._thread.join(timeout=timeout)
        remaining = self._queue.qsize()
        if remaining > 0:
            logger.warning(f"AsyncWriter shutdown with {remaining} entries still in queue")
        logger.info(f"AsyncWriter stats: {self._written} written, {self._errors} errors")

    @property
    def pending(self) -> int:
        return self._queue.qsize()
```

### 2c. Wire async writer into request handlers

**In** the chat completions endpoint (lines 1930-1959), replace:

```python
# Before (synchronous):
if not model_name.endswith('.gguf') and model_outputs is not None:
    logger_to_use, _, _ = get_logger_for_request(params)
    try:
        logger_to_use.log_entry(entry_key, {
            "prompt": prompt,
            "response": response_text,
            "model_outputs": model_outputs,
            ...
        })
    finally:
        logger_to_use.close()
```

With:

```python
# After (async — response returns immediately):
if not model_name.endswith('.gguf') and model_outputs is not None:
    if _async_writer:
        _async_writer.enqueue(entry_key, {
            "prompt": prompt,
            "response": response_text,
            "model_outputs": model_outputs,
            "input_length": input_length,
            "model": model_name,
            "messages": [msg.model_dump() for msg in request.messages],
            "trim_position": trim_pos,
            "prompt_hash": prompt_key,
            "multi_sample": multi_sample,
            "sample_group_id": sample_group_id,
            "sample_index": request.sample_index,
            "request_id": resolved_request_id,
        })
    else:
        # Fallback: synchronous write (no persistent logger configured)
        logger_to_use, custom_logger, _ = get_logger_for_request(params)
        try:
            logger_to_use.log_entry(entry_key, { ... })
        finally:
            if custom_logger:
                custom_logger.close()
```

**Same change** in the completions endpoint (lines 1785-1808).

**Test:** Run inference with async writer enabled. Verify all entries appear in Zarr index after server shutdown. Kill the server mid-run, restart, verify resume picks up from correct state.

---

## Phase 3: Stateless Client

**Goal:** Client stops writing `generation.jsonl` during inference. Resume reads Zarr index.
**Files:** `utils/exp.py`, `scripts/run_with_server.py`, `tasks/shortform/precise_wikiqa.py`
**Risk:** Medium — changes resume behavior.

### 3a. Resume from Zarr index in `run_exp()`

**In** `utils/exp.py`, `run_exp()` function, **replace** the resume block (lines 140-184):

```python
# Current resume (lines 145-159):
if resume and Path(generations_file_path).exists():
    existing_generations = pd.read_json(generations_file_path, lines=True)
    existing_prompts = set(existing_generations['prompt'].tolist())
    mask = ~all_prompts['prompt'].isin(existing_prompts)
    remaining_prompts = all_prompts[mask].copy()
    ...
```

With:

```python
if resume:
    already_completed_count = 0
    remaining_prompts = all_prompts

    # Primary resume path: read Zarr index (server source of truth)
    if activations_path and Path(activations_path).exists():
        from activation_logging.zarr_activations_logger import ZarrActivationsLogger
        import hashlib

        resume_logger = ZarrActivationsLogger(activations_path, read_only=True, verbose=False)
        completed_hashes = set()
        for key, meta in resume_logger._index.items():
            ph = meta.get("prompt_hash")
            if ph:
                completed_hashes.add(ph)
        resume_logger.close()

        if completed_hashes:
            all_prompts_with_hash = all_prompts.copy()
            all_prompts_with_hash['_ph'] = all_prompts_with_hash['prompt'].apply(
                lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
            )
            mask = ~all_prompts_with_hash['_ph'].isin(completed_hashes)
            remaining_prompts = all_prompts[mask].copy()
            already_completed_count = len(all_prompts) - len(remaining_prompts)

            print(f"  Resume from Zarr index: {already_completed_count}/{len(all_prompts)} complete")

    # Fallback resume path: read generation.jsonl (backward compat / non-activation runs)
    elif generations_file_path and Path(generations_file_path).exists():
        existing_generations = pd.read_json(generations_file_path, lines=True)
        existing_prompts = set(existing_generations['prompt'].tolist())
        mask = ~all_prompts['prompt'].isin(existing_prompts)
        remaining_prompts = all_prompts[mask].copy()
        already_completed_count = len(all_prompts) - len(remaining_prompts)

        print(f"  Resume from generation.jsonl: {already_completed_count}/{len(all_prompts)} complete")

    if len(remaining_prompts) == 0:
        print(f"  All {len(all_prompts)} prompts already processed!")
        if return_gen:
            # Reconstruct from Zarr index for return value
            return _load_generations_from_zarr(activations_path, all_prompts)
        return None

    all_prompts = remaining_prompts
```

### 3b. Stop writing `generation.jsonl` during inference

**In** `utils/exp.py`, the `process_with_incremental_save` call (lines 211-217) is replaced with a simpler loop that just sends requests — no file writes:

```python
# Current (lines 211-217):
all_prompts = process_with_incremental_save(
    all_prompts=all_prompts,
    inference_fn=inference_fn,
    generations_file_path=generations_file_path,
    desc=f"{inference_method.upper()} inference",
    resume_mode=(existing_generations is not None)
)

# New:
if activations_path:
    # Server-as-source-of-truth path: client just sends requests
    prompts = all_prompts.prompt.to_list()
    for idx, prompt in enumerate(tqdm(prompts, desc=f"{inference_method.upper()} inference")):
        inference_fn(prompt)  # Server logs the result
else:
    # Legacy path: client writes generation.jsonl (non-activation runs)
    all_prompts = process_with_incremental_save(
        all_prompts=all_prompts,
        inference_fn=inference_fn,
        generations_file_path=generations_file_path,
        desc=f"{inference_method.upper()} inference",
        resume_mode=(existing_generations is not None)
    )
```

### 3c. Export `generation.jsonl` from Zarr index

**Add** to `utils/exp.py`:

```python
def export_generation_jsonl(activations_path: str, all_prompts: pd.DataFrame,
                            output_path: str):
    """Reconstruct generation.jsonl from Zarr index + QA dataset.

    Joins Zarr index metadata (prompt_hash → response) with the original
    QA dataset to produce a file identical to what the old client wrote.
    """
    import hashlib
    from activation_logging.zarr_activations_logger import ZarrActivationsLogger

    zlogger = ZarrActivationsLogger(activations_path, read_only=True, verbose=False)

    # Build prompt_hash → response mapping from Zarr
    zarr_records = {}
    for key, meta in zlogger._index.items():
        ph = meta.get("prompt_hash")
        if ph and meta.get("response"):
            zarr_records[ph] = meta["response"]
    zlogger.close()

    # Hash the QA prompts and join
    qa = all_prompts.copy()
    qa['_ph'] = qa['prompt'].apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    qa = qa[qa['_ph'].isin(zarr_records)]
    qa['generation'] = qa['_ph'].map(zarr_records)
    qa = qa.drop(columns=['_ph'])

    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in qa.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')

    print(f"  Exported {len(qa)} records to {output_path}")
```

### 3d. Wire export into `--step all` between inference and eval

**In** `scripts/run_with_server.py`, the step loop (lines 753-764):

```python
# Current:
if step == "all":
    steps = ["inference", "eval"] if task in no_generate_tasks else ["generate", "inference", "eval"]
    for s in steps:
        result = run_task_step(s, task, model, **task_kwargs)

# New:
if step == "all":
    steps = ["inference", "eval"] if task in no_generate_tasks else ["generate", "inference", "eval"]
    for s in steps:
        # Export generation.jsonl from Zarr after inference, before eval
        if s == "eval" and activations_path and generations_file_path:
            from utils.exp import export_generation_jsonl
            qa_prompts = _load_qa_prompts(task, **task_kwargs)  # reload QA dataset
            export_generation_jsonl(activations_path, qa_prompts, generations_file_path)

        result = run_task_step(s, task, model, **task_kwargs)
```

The `_load_qa_prompts` helper reuses the task's prompt-loading logic (already exists in each task's `run_step`). This may require a small refactor to expose the QA DataFrame loading as a standalone function in each task module.

**Test:** Run `--step all` end-to-end. Verify `generation.jsonl` is exported between inference and eval. Verify eval results match a baseline run with the old client-write path.

### 3e. `ActivationParser` reads from Zarr index

**In** `activation_logging/activation_parser.py`, `_load_metadata()` (lines 753-793):

```python
def _load_metadata(self) -> pd.DataFrame:
    # Load inference records from Zarr index (server source of truth)
    index_records = []
    for key, meta in self.activation_logger._index.items():
        ph = meta.get("prompt_hash", key.split("_")[0])
        index_records.append({
            "prompt_hash": ph,
            "prompt": meta.get("prompt"),
            "generation": meta.get("response"),
            "has_activations": meta.get("sample_index") is not None,
        })
    zarr_df = pd.DataFrame(index_records)

    # Join with QA dataset for gold answers and dataset metadata
    if self.inference_json and Path(self.inference_json).exists():
        qa_df = pd.read_json(self.inference_json, lines=True)
        qa_df['prompt_hash'] = qa_df['prompt'].apply(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        )
        # Keep only QA metadata columns (not prompt/generation — those come from Zarr)
        qa_cols = ['prompt_hash', 'answer', 'title', 'h_score_cat', 'pageid',
                   'revid', 'description', 'categories', 'reference']
        qa_cols = [c for c in qa_cols if c in qa_df.columns]
        zarr_df = zarr_df.merge(qa_df[qa_cols], on='prompt_hash', how='left')

    gendf = zarr_df[zarr_df['has_activations']].drop(columns=['has_activations'])

    # Load eval labels
    with open(self.eval_json, 'r') as f:
        data = json.loads(f.read())
    gendf['abstain'] = data['abstantion'][:len(gendf)]
    gendf['halu'] = data['halu_test_res'][:len(gendf)]

    # ... rest unchanged: deduplicate, train/test split ...
```

**Backward compatibility:** When `inference_json` points to an existing `generation.jsonl`, the QA metadata is joined in. When it's absent, the method still works — it just won't have gold answers (acceptable for inference-only workflows).

---

## Phase 4: Server-Side Batching

**Goal:** Replace the global `inference_lock` with a batch queue. Biggest throughput impact.
**File:** `activation_logging/server.py`
**Risk:** High — changes core inference path. Requires Phase 2.

### 4a. Add `BatchInferenceQueue` class

```python
import concurrent.futures

class BatchInferenceQueue:
    """Accumulates incoming requests into batches for model.generate()."""

    def __init__(self, max_batch_size: int = 8, max_wait_ms: float = 50.0):
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._pending: List[Tuple[str, dict, concurrent.futures.Future]] = []
        self._lock = threading.Lock()
        self._batch_event = threading.Event()
        self._shutdown = False
        self._worker = threading.Thread(target=self._batch_loop, name="batch-inference", daemon=True)
        self._worker.start()

    def submit(self, prompt: str, params: dict) -> concurrent.futures.Future:
        """Submit a single inference request. Returns Future resolving to
        (response_text, model_outputs, input_length, trim_pos)."""
        future = concurrent.futures.Future()
        with self._lock:
            self._pending.append((prompt, params, future))
            if len(self._pending) >= self._max_batch_size:
                self._batch_event.set()
        return future

    def _batch_loop(self):
        while not self._shutdown:
            # Wait for batch to fill or timeout
            self._batch_event.wait(timeout=self._max_wait)
            with self._lock:
                if not self._pending:
                    self._batch_event.clear()
                    continue
                batch = self._pending[:self._max_batch_size]
                self._pending = self._pending[self._max_batch_size:]
                self._batch_event.clear()

            self._run_batch(batch)

    def _run_batch(self, batch):
        """Run batched model.generate() and dispatch results to futures."""
        prompts = [b[0] for b in batch]
        params_list = [b[1] for b in batch]
        futures = [b[2] for b in batch]

        try:
            model_name = params_list[0].get("model_name", DEFAULT_MODEL)
            model, tokenizer = get_model_and_tokenizer(model_name)

            # Left-pad tokenization for causal LM batching
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True).to(model.device)
            input_lengths = [
                len(tokenizer.encode(p, add_special_tokens=True))
                for p in prompts
            ]

            max_new_tokens = max(p.get("max_tokens", 512) for p in params_list)
            temperature = params_list[0].get("temperature", 0.0)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Split results per request
            for i, future in enumerate(futures):
                try:
                    pad_len = inputs.input_ids.shape[1] - input_lengths[i]
                    gen_start = inputs.input_ids.shape[1]  # after all padding + prompt
                    gen_ids = outputs.sequences[i, gen_start:]
                    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                    # Extract per-request model_outputs for activation logging
                    per_request_outputs = _split_hidden_states(outputs, i, input_lengths[i], pad_len)

                    future.set_result((response_text, per_request_outputs, input_lengths[i], None))
                except Exception as e:
                    future.set_exception(e)

        except Exception as e:
            # If the entire batch fails, fail all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
```

### 4b. Add `_split_hidden_states` helper

```python
def _split_hidden_states(outputs, batch_idx: int, input_length: int, pad_length: int):
    """Extract per-request hidden states from batched model output.

    HF model.generate with output_hidden_states=True returns hidden states
    per generation step. We need to reconstruct per-layer activations for
    the full sequence (prompt + response) for a single batch item.

    Returns a model_outputs-like object compatible with ZarrActivationsLogger.extract_activations().
    """
    # Implementation depends on HF generate output format.
    # outputs.hidden_states is a tuple of (num_gen_steps,) where each step
    # has (num_layers,) tensors of shape (batch, seq, hidden).
    #
    # This is the most complex part of batching and requires careful testing
    # with the specific model architecture.
    #
    # TODO: Implement and test with Llama-3.1-8B-Instruct
    raise NotImplementedError("Hidden state splitting requires model-specific implementation")
```

**This is the hardest part.** HF `model.generate()` with `output_hidden_states=True` returns hidden states per generation step, not as a single tensor. The exact format depends on whether `use_cache=True` (default) and the model architecture. This needs a dedicated implementation + test pass with the target model.

### 4c. Wire batch queue into request handlers

**Replace** the `inference_lock` block in chat completions (lines 1900-1916):

```python
# Before:
with inference_lock:
    response_text, model_outputs, input_length, trim_pos = run_inference(
        prompt=prompt, max_tokens=effective_max_tokens,
        temperature=params['temperature'], top_p=params['top_p'],
        model_name=model_name, auth_token=request.auth_token
    )

# After:
if _batch_queue and not model_name.endswith('.gguf'):
    future = _batch_queue.submit(prompt, {
        "max_tokens": effective_max_tokens,
        "temperature": params['temperature'],
        "top_p": params['top_p'],
        "model_name": model_name,
        "auth_token": request.auth_token,
    })
    response_text, model_outputs, input_length, trim_pos = future.result()
else:
    # Fallback: sequential inference (GGUF models, or batching disabled)
    with inference_lock:
        response_text, model_outputs, input_length, trim_pos = run_inference(...)
```

**Initialize** at startup:

```python
_batch_queue: Optional[BatchInferenceQueue] = None

@app.on_event("startup")
async def startup_event():
    global _batch_queue
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "1"))  # default: no batching
    if max_batch_size > 1:
        _batch_queue = BatchInferenceQueue(max_batch_size=max_batch_size)
        logger.info(f"Batch inference enabled: max_batch_size={max_batch_size}")
```

**Test:** Run with `MAX_BATCH_SIZE=1` — should behave identically to current. Run with `MAX_BATCH_SIZE=4` — verify throughput improvement and activation correctness.

---

## Phase 5: Concurrent Client

**Goal:** Client sends multiple requests concurrently to feed the batch queue.
**File:** `utils/exp.py`
**Risk:** Medium — requires Phases 3 and 4.

### 5a. Add `process_concurrent()` function

```python
import asyncio
import openai

async def process_concurrent(prompts: List[str], model: str, port: int = 8000,
                             max_concurrent: int = 16, temperature: float = 0.0,
                             max_tokens: int = 512, max_retries: int = 3):
    """Send inference requests concurrently. Server owns all record-keeping.

    Args:
        prompts: List of prompt strings to process
        model: Model name for the API request
        port: Server port
        max_concurrent: Maximum in-flight requests (should match server batch capacity)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Retry count for failed requests
    """
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0

    async def send_one(prompt: str, idx: int):
        nonlocal completed, failed
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    completed += 1
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Request {idx} failed after {max_retries} attempts: {e}")
                        failed += 1
                    else:
                        await asyncio.sleep(2 ** attempt)

    tasks = [send_one(p, i) for i, p in enumerate(prompts)]

    # Progress reporting
    total = len(tasks)
    gather_task = asyncio.gather(*tasks)
    await gather_task

    print(f"  Concurrent inference: {completed}/{total} succeeded, {failed} failed")
    await client.close()
```

### 5b. Wire into `run_exp()`

```python
# In run_exp(), replace the inference loop (Phase 3b) with:
if activations_path:
    prompts = all_prompts.prompt.to_list()
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "1"))

    if max_concurrent > 1:
        asyncio.run(process_concurrent(
            prompts=prompts, model=model_path, port=port or 8000,
            max_concurrent=max_concurrent, temperature=0.0,
            max_tokens=max_tokens, max_retries=max_retries,
        ))
    else:
        # Sequential fallback
        for prompt in tqdm(prompts, desc=f"{inference_method.upper()} inference"):
            inference_fn(prompt)
```

**Tuning:** `MAX_CONCURRENT_REQUESTS` should be ~2x `MAX_BATCH_SIZE` to keep the batch queue fed. For `MAX_BATCH_SIZE=8`, use `MAX_CONCURRENT_REQUESTS=16`.

**Test:** Run with `MAX_CONCURRENT_REQUESTS=1` — identical to Phase 3 behavior. Run with `MAX_CONCURRENT_REQUESTS=16` + `MAX_BATCH_SIZE=8` — verify throughput improvement, verify all entries appear in Zarr index, verify resume works after interruption.

---

## Dependency Graph

```
Phase 1 (Zarr logger: log_metadata, overwrite, get_entry)
   │
   ├──→ Phase 2 (async writer)
   │       │
   │       └──→ Phase 4 (batch inference queue)
   │               │
   │               └──→ Phase 5 (concurrent client)
   │
   └──→ Phase 3 (stateless client, Zarr resume, export)
               │
               └──→ Phase 5 (concurrent client)
```

Phases 2 and 3 can be developed in parallel on separate branches.

---

## Test Strategy

| Phase | Test | What to verify |
|-------|------|----------------|
| 1 | Unit tests on `ZarrActivationsLogger` | `log_metadata`, key overwrite, `get_entry` with `None` idx |
| 2 | Integration: run inference with async writer | All entries in Zarr after shutdown; crash recovery |
| 3 | End-to-end: `--step all` with Zarr resume | Exported `generation.jsonl` matches old client-write baseline |
| 4 | Batch correctness: compare batch=1 vs batch=4 | Per-sample activations and response text must be identical |
| 5 | Throughput: benchmark batch=8, concurrent=16 | Measure samples/sec on H100, compare to baseline ~0.05 s/s |
