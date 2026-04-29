# Adding New Models to the Inference Pipeline

This document covers what was changed to support Qwen3-8B and what any agent (or developer)
needs to know when adding another model (Gemma, Mistral, etc.).

---

## What we changed to support Qwen3

### 1. Model registration (`CLAUDE.md`)

Added Qwen3-8B to the supported models table:

```
| Qwen/Qwen3-8B | Thinking mode disabled automatically |
```

No code change was needed for model loading — `get_model_and_tokenizer` in
`activation_logging/server.py` uses `AutoModelForCausalLM.from_pretrained` generically
and loads in `float16` on GPU. Any HuggingFace causal LM works out of the box.

### 2. Thinking mode (`activation_logging/model_adapter.py`)

Qwen3 supports a "thinking" mode where the model reasons step-by-step inside `<think>` tags.
We want it disabled for factual QA benchmarks.

**What does NOT work:** passing `enable_thinking=False` as a `generate()` kwarg — older
versions of transformers reject unknown kwargs with a `ValueError`.

**What works:** Qwen3 only activates thinking mode when prompted via the chat template with
the appropriate system message. Since all LLMsKnow tasks use **raw (non-chat-template)
prompts** like `"Answer the question concisely.\n\nQ: {q}\nA:"`, thinking mode is never
triggered. No kwarg needed.

The original code that added `generate_kwargs["enable_thinking"] = False` was removed.
If you ever use chat-template prompts with Qwen3 and need to suppress thinking, use a
system message: `{"role": "system", "content": "/no_think"}` in the messages list, or
set `thinking_budget=0` if your transformers version supports it.

### 3. GPU memory / activation queuing (`activation_logging/model_adapter.py`)

`model.generate(output_hidden_states=True)` returns all hidden states as GPU tensors.
These tensors were being held on GPU until the async zarr writer consumed them from the
queue. With a queue depth of 256 batches, GPU memory filled up and caused OOM on long runs.

**Fix:** Call `.cpu()` on each activation tensor immediately after extraction, before
returning from `_extract_activations`. Also `del outputs` explicitly after the per-sample
extraction loop to free the large generate output before activations are queued.

```python
# _extract_activations — end of loop
activations.append(act.cpu())   # move off GPU immediately

# infer_batch — after extraction loop
del outputs                     # free generate output before queuing
return results
```

This is essential for any model. Without it, long runs (>10k samples) will OOM.

### 4. Zarr token dimension (`activation_logging/zarr_activations_logger.py` callers)

**Root cause:** `--max_tokens` CLI default in `run_with_server.py` was `1024`, which flowed
into `activation_chunk_shape=(BS, 1, 1024, H)` for every task. Each batch then wrote
`32 × 37 × 1024 × 4096 × 2 bytes = 9.8 GB` of activations — 16× too large, making
I/O the bottleneck.

**Fix:** `--max_tokens` default changed to `None`. `--max_inference_tokens` default changed
from `256` to `64`. Per-task fallbacks (`kwargs.get("max_tokens", 64)`) now fire correctly
because `None` values are stripped from `task_kwargs` before dispatch.

All LLMsKnow task defaults were also updated from `max_tokens=128` to `max_tokens=64` for
consistency with NQ.

**Result:** Correct chunk shape `(32, 1, 64, 4096)` — each batch writes ~610 MB, keeping
the async zarr writer from falling behind GPU inference.

### 5. `--N` default (`run_with_server.py`)

`--N` CLI default was `1`, causing all LLMsKnow tasks to process only 1 sample. Changed
to `None` (all samples). PreciseWikiQA is unaffected — its handler has its own fallback
`kwargs.get("N", 1)`.

---

## What an agent needs to add a new model (e.g. Gemma-3)

### Step 1: Verify HuggingFace loading works

`get_model_and_tokenizer` in `activation_logging/server.py` loads any `AutoModelForCausalLM`.
No changes needed unless the model has special loading requirements (e.g. requires a custom
`trust_remote_code`, specific dtype, or multi-GPU `device_map`).

Check:
- Does the model load cleanly with `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")`?
- Does `AutoTokenizer.from_pretrained(model_id)` work?
- Does `tokenizer.pad_token_id` exist? If not, set it to `tokenizer.eos_token_id`.

### Step 2: Check for generate() incompatibilities

Some models add non-standard kwargs to their generate() method, or reject standard ones.
The current generate kwargs are:

```python
generate_kwargs = dict(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=...,
    do_sample=...,
    output_hidden_states=True,
    output_scores=True,          # for logprobs
    return_dict_in_generate=True,
    pad_token_id=tokenizer.pad_token_id,
)
```

Run a quick smoke test — if `model.generate(**generate_kwargs)` raises `ValueError` about
unknown kwargs, remove the offending one or wrap it conditionally on model name.

Common issues:
- `output_scores` not supported by some models → set `enable_logprobs=False`
- Left-padding required for batch inference — verify tokenizer uses left-padding
  (the adapter sets `tokenizer.padding_side = "left"` in `server.py`)

### Step 3: Check the hidden states structure

The adapter assumes HF's standard output format:

```
outputs.hidden_states[0]   → prompt hidden states: tuple of L tensors (B, prompt_len, H)
outputs.hidden_states[1:]  → per-generation-step: list of tuples of L tensors (B, 1, H)
```

Most decoder-only models (Llama, Mistral, Qwen, Gemma) follow this. Verify with:

```python
outputs = model.generate(..., output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=2)
print(len(outputs.hidden_states))           # should be 3 (prompt + 2 steps)
print(outputs.hidden_states[0][0].shape)    # (batch, prompt_len, hidden)
print(outputs.hidden_states[1][0].shape)    # (batch, 1, hidden)
```

If the structure differs (e.g. encoder-decoder models), `_extract_activations` in
`model_adapter.py` will need updating.

### Step 4: Check hidden size and layer count

The zarr store is sized by `(n_samples, n_layers, max_tokens, hidden_size)`. The adapter
infers `n_layers` and `hidden_size` from the first batch's output. No manual config needed.

For reference:
- Llama 3.1 8B: 33 layers (32 transformer + 1 embedding), hidden=4096
- Qwen3-8B: 37 layers, hidden=4096
- Gemma-3 9B: verify with `len(outputs.hidden_states[0])`

### Step 5: Update `CLAUDE.md`

Add the model to the supported models table with its HuggingFace ID and any notes (e.g.
"thinking mode", "requires trust_remote_code").

### Step 6: Test with a small run

```bash
python scripts/run_with_server.py \
    --step inference \
    --task naturalquestions \
    --model <new-model-id> \
    --activations-path /tmp/test_zarr \
    --split test \
    --N 64          # small sample to verify end-to-end
```

Check:
- generation.jsonl has 64 lines with non-empty `generation` fields
- zarr store shape is `(64, <n_layers>, 64, <hidden_size>)`
- No OOM, no tensor errors

---

## Key invariants to preserve when adding a model

| Property | Value | Why |
|----------|-------|-----|
| `max_tokens` (response) | 64 | zarr chunk alignment; changing breaks existing stores |
| `activation_chunk_shape` | `(32, 1, 64, H)` | batch-aligned chunks = single zarr write per batch |
| Activations moved to CPU | immediately after extraction | prevents GPU OOM on long runs |
| `output_hidden_states=True` | always | required for activation capture |
| Left-padding | always for batched inference | right-padding corrupts prompt activation indexing |
