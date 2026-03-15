# SelfCheckGPT Implementation Investigation

**Paper:** [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs (Manakul et al., 2023)](https://arxiv.org/abs/2303.08896)
**Repo:** https://github.com/potsawee/selfcheckgpt
**Bucket:** Black-box / sampling-based self-consistency (Section D in SOTA_TRACKER)

---

## 1. How SelfCheckGPT Works

The core idea: a hallucinated fact tends to be inconsistent across multiple stochastic samples, while a true fact tends to appear consistently.

**Algorithm:**
1. Generate `k` stochastic samples for each prompt (temperature > 0, no top-1 greedy).
2. For each sentence in the main (greedy) response, compute an inconsistency score against the `k` samples.
3. The response-level hallucination score = aggregate (e.g. mean) of sentence-level scores.

**Scoring variants (cheapest → best):**
| Variant | Method | Extra model? |
|---|---|---|
| n-gram (BM25) | Sentence appears verbatim/near-verbatim in samples | None |
| BERTScore | Semantic similarity between sentence and samples | HF encoder (e.g. `roberta-large`) |
| NLI | Entailment score: samples entail the sentence | DeBERTa MNLI (`potsawee/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) |
| LLM-prompting | Ask GPT-style model whether sample supports sentence | External LLM API |

---

## 2. Storage Design: Appending to Existing Zarr Stores

The critical requirement is that SelfCheck sampling can run **after** a primary inference run has already been logged to Zarr — a separate, append-only step that doesn't touch the greedy-pass data.

### 2.1 Zarr key naming convention

The existing codebase already has multi-sample grouping infrastructure:

- `ZarrActivationsLogger` stores entries by arbitrary string key.
- `ActivationParser._build_group_index()` groups entries by `sample_group_id` metadata field, falling back to `key.split("_")[0]`.
- `ActivationParser.select_primary_key()` picks the entry with `sample_index == 0` as the greedy answer.
- `ActivationParser.get_group_keys(prompt_hash)` returns **all** keys for a group.

**We can use this directly.** SelfCheck samples are stored in the same Zarr with **content-addressed keys** — a double hash of prompt + response:

```
{prompt_hash}                     ← existing greedy entry (sample_index=0)
{prompt_hash}_sc_{response_hash}  ← stochastic sample  (sample_index=1..k)
```

Where `response_hash = SHA256(response_text)[:16]` (first 16 hex chars — collision-safe within a prompt group, keeps keys readable).

**Why double hash instead of sequential indices (`_sc_0`, `_sc_1`, ...)?**
Content-addressed keys cryptographically bind each Zarr row's activations to the exact response text that produced them. With index-based keys, a crash-and-resume could silently pair activations from one generation with text from a different generation at the same index. Double hashing makes this impossible — the key itself proves the activation/text correspondence.

Each `_sc_` entry's Zarr metadata (written to `meta/index.jsonl`) includes:
```json
{
  "key": "{prompt_hash}_sc_{response_hash}",
  "sample_group_id": "{prompt_hash}",
  "sample_index": 1,
  "is_selfcheck_sample": true,
  "temperature": 0.7,
  "generation": "stochastic answer text",
  "prompt": "..."
}
```

`_build_group_index()` groups by `sample_group_id`, so `get_group_keys(prompt_hash)` returns all of them. `select_primary_key()` returns the `sample_index == 0` greedy entry, leaving existing code paths unchanged. SHA-256 hex digests contain no underscores, so `key.split("_")[0]` in the fallback path still correctly recovers the prompt hash as the group ID.

### 2.2 Selective activation logging: `selfcheck_k` vs `activation_log_count`

Generating 20 stochastic samples at inference time but only storing activations for 5 of them requires decoupling the **text sampling count** from the **activation logging count**.

**Parameters:**
- `--selfcheck-k 20` — number of **extra** stochastic samples to generate beyond the greedy pass (21 total: 1 greedy + 20 stochastic). All 20 texts stored in `selfcheck_samples.jsonl`.
- `--selfcheck-log-activations 0` (default) — **minimum** number of stochastic samples that should have activations logged to Zarr. If the current count is below this minimum, new samples are generated with activation logging until the minimum is met. These new samples are appended (increasing effective `k`) rather than retroactively adding activations to existing text-only samples (see §2.6.4 activation integrity rule). The greedy Zarr row already exists and is not counted here.

**Key challenge — hash is prompt-level, not answer-level:**
The existing Zarr is keyed by `prompt_hash` (SHA-256 of the prompt text). This works because each prompt had exactly one answer. For multiple answers to the same prompt, the Zarr key **must include the response** to be unique. The content-addressed `{prompt_hash}_sc_{response_hash}` scheme solves this — the key is unique per distinct response, and `_build_group_index()` groups by the prefix, so the greedy entry at `{prompt_hash}` and stochastic entries at `{prompt_hash}_sc_{response_hash}` are all grouped together correctly.

**Selfcheck inference loop (per prompt):**

```python
import hashlib

def response_hash(text: str) -> str:
    """Short content hash for keying selfcheck samples."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# 1. Greedy pass — already logged by the inference step (or a prior run), skip
greedy_key = prompt_hash  # existing Zarr row

# 2. Stochastic passes — k and log_activations are minimums
# If we already have enough, this loop is a no-op
existing_count = len(existing_samples)
existing_act_count = sum(1 for s in existing_samples if s.get("has_zarr_activations"))
need_more_text = max(0, selfcheck_k - existing_count)
need_more_acts = max(0, selfcheck_log_activations - existing_act_count)
# New samples to generate: enough to satisfy both minimums
new_sample_count = max(need_more_text, need_more_acts)

for j in range(new_sample_count):
    log_activations = (j < need_more_acts)  # first need_more_acts new samples get Zarr rows

    if log_activations:
        # Call activation-logging server (FastAPI + vLLM hooks)
        # Server computes sc_key = f"{prompt_hash}_sc_{response_hash(text)}" AFTER generation
        response = call_logging_server(prompt, temperature=temp)
        sc_key = f"{prompt_hash}_sc_{response_hash(response.text)}"
    else:
        # Call vLLM directly (no hooks, no Zarr write) — text only
        response = call_vllm_direct(prompt, temperature=temp)
        sc_key = f"{prompt_hash}_sc_{response_hash(response.text)}"

    # Text always goes to selfcheck_samples.jsonl regardless
    append_selfcheck_text(prompt_hash, sc_key, response.text, logged=log_activations)
```

**What gets logged per sample type** (example: first run k=20 la=0, then re-run k=20 la=2):

| Sample | Key | Zarr row? | Text? | When generated |
|---|---|---|---|---|
| Greedy | `{hash}` | Yes (existing) | generation.jsonl | inference step |
| SC 0..19 | `{hash}_sc_{rhash_i}` | No | selfcheck_samples.jsonl | first run (text-only) |
| SC 20..21 | `{hash}_sc_{rhash_j}` | Yes | selfcheck_samples.jsonl | second run (with activations) |

Total after second run: effective k=22 (20 text-only + 2 with Zarr rows). The 2 new samples have activations that exactly match their text — guaranteed by the content-addressed key.

**Logprobs for text-only samples:** vLLM can return token logprobs, but this requires explicitly passing `logprobs=True` (and optionally `top_logprobs=N`) in the API request — it is not returned by default. The text-only call path must include these parameters. We store logprobs inline in `selfcheck_samples.jsonl` rather than Zarr:

```json
{
  "prompt_hash": "abc123...",
  "selfcheck_samples": [
    {
      "zarr_key": "abc123..._sc_0",
      "generation": "sample text",
      "has_zarr_activations": true,
      "token_logprobs": null
    },
    {
      "zarr_key": "abc123..._sc_5",
      "generation": "sample text",
      "has_zarr_activations": false,
      "token_logprobs": [-0.12, -1.4, -0.03, ...],
      "topk_logprobs": [[(-0.12, 1234), (-0.9, 5678), ...], ...]
    }
  ]
}
```

For samples with Zarr rows, `token_logprobs` is `null` in the JSONL (read from Zarr instead to avoid duplication). This keeps `selfcheck_samples.jsonl` as the single lookup point regardless of whether a sample has activations.

**Storage estimate** (example: 1000 prompts, 32 layers, hidden=4096, R_max=64):
- Per-row Zarr cost ≈ 32 × 64 × 4096 × 2 bytes (float16) ≈ 16 MB
- Default (`selfcheck_log_activations=0`): zero extra Zarr storage — text + logprobs only in JSONL.
- With 5 activation rows: 1000 prompts × 5 rows ≈ 80 GB — significant, opt-in only.

**No changes needed to `ZarrActivationsLogger`** — it already supports arbitrary string keys and append-only writes.

### 2.3 Companion text file: `selfcheck_samples.jsonl`

Alongside the Zarr, write a plain JSONL file (one line per prompt) for human-readable inspection and to support text-only scorers without touching Zarr:

```json
{
  "prompt_hash": "abc123...",
  "prompt": "Answer in one sentence. Q:...\n A:",
  "generation": "primary greedy answer",
  "selfcheck_samples": [
    {"zarr_key": "abc123..._sc_a1b2c3d4e5f67890", "generation": "sample 1", "temperature": 0.7},
    {"zarr_key": "abc123..._sc_f0e1d2c3b4a59876", "generation": "sample 2", "temperature": 0.7}
  ]
}
```

This is written incrementally (one line at a time) with resume support keyed on `prompt_hash`.

### 2.4 Client-side request construction

The client (`utils/lm.py`) calls the server via the OpenAI Python SDK:

```python
# current call_vllm_api — no custom fields, temperature fixed at 0.0
chat_completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=max_tokens,
    temperature=temperature,   # always 0.0 in run_exp's inference_fn lambda
    top_p=top_p
)
```

Three issues for selfcheck:

**Issue 1 — Custom server fields require `extra_body`.**
The server's Pydantic model accepts `multi_sample`, `request_id`, `sample_group_id`, `sample_index` — but these are not standard OpenAI fields. The OpenAI Python SDK passes them via `extra_body`:

```python
chat_completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,                    # must be > 0 for stochastic samples
    extra_body={
        "multi_sample": True,
        "sample_group_id": prompt_hash, # groups with the existing greedy row
        "sample_index": i + 1,          # 0 = greedy (already logged)
        # request_id is NOT set here — the server computes entry_key
        # from prompt_hash + response_hash AFTER generation (see §2.5)
    }
)
```

Note: `call_vllm_api` currently has no `extra_body` parameter. Adding `extra_body: Optional[dict] = None` is backward-compatible (all existing callers use keyword args), but the internal `client.chat.completions.create()` call must be updated to pass `**(extra_body or {})` as well.

**Issue 2 — Temperature is hardcoded to 0.0 in `run_exp`.**
`run_exp` builds `inference_fn = lambda p: call_vllm_api(p, ..., temperature=0.0, ...)`. Selfcheck needs a separate callable with `temperature=selfcheck_temperature`. This is a minor change — the selfcheck step has its own loop and does not go through `run_exp`.

**Issue 3 — No per-request opt-out of activation logging on the server.**
The server always logs activations for non-GGUF models:
```python
if not model_name.endswith('.gguf') and model_outputs is not None:
    logger_to_use.log_entry(entry_key, {...})
```
There is no `skip_activation_logging` field. For the text-only samples (indices `selfcheck_log_activations..selfcheck_k-1`), the two options are:

| Option | Pros | Cons |
|---|---|---|
| **A. Add `skip_activation_logging` field to server** | Clean separation, no second server | Requires server change; server still runs the model (activations are computed, just not persisted) |
| **B. Call a second raw vLLM port** (if `vllm_serve.py` exposes one) | No server code change; truly skips logging | Requires a second server process or a separate port on the same vLLM instance |

**Recommended: Option A** — add `skip_activation_logging: bool = False` to the request model and wrap the `log_entry` call in `if not request.skip_activation_logging`. This is a one-line server change and keeps a single server endpoint for all calls.

For text-only selfcheck samples, the client then sends:
```python
extra_body={
    "multi_sample": True,
    "sample_group_id": prompt_hash,
    "sample_index": i + 1,
    "skip_activation_logging": True,   # ← new field
}
```

No Zarr row is written. The content-addressed key (prompt+response hash) is still computed server-side and returned in the response `id` field for the client to record in JSONL.

### 2.5 Server-side key assignment — content-addressed keys

The server (`activation_logging/server.py`) builds Zarr keys via `build_entry_key()`:

```python
def build_entry_key(prompt, request_id, multi_sample):
    prompt_key = prompt_hash(prompt)          # SHA256 of prompt text
    if not multi_sample:
        return prompt_key, prompt_key, None   # key = prompt hash — OVERWRITES on repeat calls

    resolved_request_id = request_id or str(uuid.uuid4())[:8]   # random if not supplied
    entry_key = f"{prompt_key}_{resolved_request_id}"
    return entry_key, prompt_key, resolved_request_id
```

**Normal single-inference path (existing pipeline):** `multi_sample=False` → `build_entry_key` returns `prompt_key` directly. Unaffected.

**Selfcheck path — content-addressed key from prompt + response:**

Instead of the client passing a deterministic `request_id`, the server computes the entry key **after generation** using a hash of the response text. This binds the Zarr key to the exact text that produced the activations, making activation/text mismatches structurally impossible.

**Required server change:** Move `build_entry_key()` call to **after** inference in the chat completions endpoint (currently at line 1890, before inference at line 1907). In the completions endpoint it's already after inference. Updated `build_entry_key`:

```python
def build_entry_key(prompt: str, request_id: Optional[str], multi_sample: bool,
                    response_text: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    prompt_key = prompt_hash(prompt)
    if not multi_sample:
        return prompt_key, prompt_key, None

    if request_id:
        # Explicit request_id — use as-is (backward compatible)
        resolved_request_id = request_id
    elif response_text is not None:
        # Content-addressed: key includes response hash
        resolved_request_id = f"sc_{hashlib.sha256(response_text.encode()).hexdigest()[:16]}"
    else:
        # Fallback (should not happen for selfcheck path)
        resolved_request_id = str(uuid.uuid4())[:8]

    entry_key = f"{prompt_key}_{resolved_request_id}"
    return entry_key, prompt_key, resolved_request_id
```

The client sets `multi_sample=True` and `sample_group_id=prompt_hash` but does **not** set `request_id`. The server computes the key after generation. The response `id` field returns the computed `entry_key`, which the client records in `selfcheck_samples.jsonl`.

SHA-256 hex digests contain no underscores, so `key.split("_")[0]` in `_build_group_index()` correctly recovers the prompt hash as the group ID. The `_sc_` infix is a fixed marker separating prompt hash from response hash.

### 2.6 Declarative single-command design

Instead of separate `--step` invocations, `run_with_server.py` becomes declarative: you specify **what outputs you want** and the script figures out what's already done and fills in the rest. The same command is both the initial run and the resume command.

#### 2.6.1 Usage

```bash
# Full run: greedy inference + selfcheck + eval — single command
python scripts/run_with_server.py \
    --step all \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 1000 \
    --selfcheck-k 20 \
    --selfcheck-temperature 0.7 \
    --logger-type zarr \
    --activations-path shared/goodwiki/activations.zarr

# Resume after crash — exact same command, picks up where it left off
# (same command as above)

# Add activation-logged samples to a completed text-only run (k=20, la=0):
# k minimum already met, but la deficit = 2. Generates 2 NEW samples
# (indices 20–21) with activations. Effective k becomes 22.
python scripts/run_with_server.py \
    --step all ... --selfcheck-k 20 --selfcheck-log-activations 2

# Existing behavior preserved: no selfcheck flags = greedy-only (current default)
python scripts/run_with_server.py --step all --task precisewikiqa ...
```

When `--selfcheck-k` is present, `--step all` expands to: `generate → inference → selfcheck → eval` (two-pass — all greedy inference completes before selfcheck begins). When absent, behavior is unchanged: `generate → inference → eval`.

`--step selfcheck` also works standalone for post-hoc use against an existing run.

#### 2.6.2 Two-pass execution model

The implementation uses two sequential passes, preserving the existing inference step untouched:

**Pass 1 — Greedy inference** (existing `inference` step, unchanged):
Iterates all prompts, generates greedy responses, logs activations to Zarr, writes `generation.jsonl`. Resumes via existing `generation.jsonl` prompt_hash check.

**Pass 2 — Selfcheck sampling** (new `selfcheck` step):
Reads completed `generation.jsonl`, iterates all prompts, generates `selfcheck_k` stochastic samples per prompt, writes `selfcheck_samples.jsonl`. Resumes via per-prompt completion check.

Both passes run within a single `--step all` invocation. The server stays up across both.

#### 2.6.3 Per-prompt completion model (selfcheck pass)

`selfcheck_k` and `selfcheck_log_activations` are **minimums**. A prompt is complete when both are satisfied. If more samples are needed (for text or activations), new samples are appended — existing samples are never re-run or modified.

```
for prompt_hash in generation.jsonl:

    existing = load_selfcheck_entry(prompt_hash)  # from selfcheck_samples.jsonl
    existing_count = len(existing)
    existing_act_count = sum(1 for s in existing if s.get("has_zarr_activations"))

    # Corruption check: verify activation samples have matching Zarr rows
    for sample in existing:
        if sample.get("has_zarr_activations"):
            zarr_entry = activation_logger.get_entry_metadata(sample["zarr_key"])
            if zarr_entry is None or zarr_entry.get("response") != sample["generation"]:
                # Zarr/JSONL mismatch — discard this prompt and redo entirely
                log.warning(f"Corrupt entry for {prompt_hash}: Zarr/JSONL mismatch, resetting")
                delete_zarr_sc_keys(prompt_hash)  # remove all _sc_ keys for this prompt
                existing = []
                existing_count = 0
                existing_act_count = 0
                break

    need_more_text = max(0, selfcheck_k - existing_count)
    need_more_acts = max(0, selfcheck_log_activations - existing_act_count)
    new_sample_count = max(need_more_text, need_more_acts)

    if new_sample_count == 0:
        continue  # both minimums satisfied

    for j in range(new_sample_count):
        log_activations = (j < need_more_acts)  # activation samples generated first

        if log_activations:
            # Server computes content-addressed key after generation
            response = call_logging_server(prompt, temperature=sc_temp)
            sc_key = response.id  # server returns {prompt_hash}_sc_{response_hash}
        else:
            response = call_vllm_direct(prompt, temperature=sc_temp)
            sc_key = f"{prompt_hash}_sc_{response_hash(response.text)}"

        append_sample(existing, sc_key, response, has_activations=log_activations)

    write_selfcheck_jsonl(prompt_hash, existing)  # per-prompt batch write
```

#### 2.6.4 Activation integrity rule

**Activations are bound to their generation at creation time and are never retroactively added.**

Stochastic sampling produces different text each time, so re-running inference for a sample index would produce activations for a *different* answer than the text stored in `selfcheck_samples.jsonl`. To prevent this silent mismatch:

- Existing samples are never re-run or modified. New samples are always appended.
- When `selfcheck_log_activations` exceeds the current activation count, new samples are generated *with* activation logging to make up the deficit. These new samples increase effective `k`.
- Every Zarr row's activations correspond exactly to the text stored alongside it — enforced structurally by the content-addressed key (`prompt_hash + response_hash`).
- On resume, a corruption check verifies that each activation-flagged JSONL entry has a matching Zarr row with identical response text. If any mismatch is found, the entire prompt's selfcheck entries are discarded and regenerated.

**Example — adding activations after a text-only run:**

| Run | `selfcheck_k` | `selfcheck_log_activations` | Result |
|---|---|---|---|
| First | 20 | 0 | 20 text-only samples (indices 0–19) |
| Second | 20 | 2 | `k` minimum already met (20 ≥ 20), but `la` deficit = 2. Generates 2 new samples (indices 20–21) with activations. Effective k = 22. |
| Third | 20 | 2 | Both minimums met (22 ≥ 20, 2 ≥ 2). No-op. |
| Fourth | 25 | 2 | `k` deficit = 3, `la` already met. Generates 3 new text-only samples (indices 22–24). Effective k = 25. |

#### 2.6.5 Resume edge cases

**Crash mid-prompt:** JSONL is written per-prompt after all samples for that prompt complete. A crash mid-prompt loses at most `k` samples for one prompt — acceptable given `k` is typically 20 and regeneration is cheap. On resume, the incomplete prompt has no JSONL entry and is simply retried from scratch.

**Corruption detection:** On resume, every activation-flagged JSONL entry is checked against Zarr (see corruption check in §2.6.3). If the JSONL says `has_zarr_activations=True` but the Zarr row is missing or has different response text, the entire prompt's selfcheck data is discarded and regenerated. This handles:
- Crash after Zarr write but before JSONL write (orphan Zarr rows cleaned up)
- Crash after partial JSONL write (truncated JSON line → parse error → discard)
- Any future data corruption

**Greedy is never re-generated:** The selfcheck pass never touches the greedy Zarr row at `prompt_hash`. It only writes `{prompt_hash}_sc_{response_hash}` keys. `generation.jsonl` is read-only input to the selfcheck phase.

### 2.7 `ActivationParser` additions

Add one method to expose selfcheck samples to downstream consumers. **Note:** `get_group_keys()` only returns samples with Zarr rows. For a complete view (including text-only samples), this method also reads `selfcheck_samples.jsonl`:

```python
def get_selfcheck_entries(
    self,
    prompt_hash: str,
    selfcheck_jsonl_path: Optional[str] = None,
    include_activations: bool = False,
    include_logprobs: bool = True,
) -> List[Dict[str, Any]]:
    """Return metadata (+ optional activations/logprobs) for all selfcheck
    samples of a given prompt. Returns empty list if none logged.

    Merges two sources:
    - selfcheck_samples.jsonl (all samples, text + logprobs)
    - Zarr group index (only samples with activations)
    """
    results = []

    # Primary source: JSONL has all samples (text-only and activation-logged)
    if selfcheck_jsonl_path:
        jsonl_entry = load_selfcheck_entry(prompt_hash, selfcheck_jsonl_path)
        for sample in jsonl_entry:
            entry = dict(sample)
            if include_activations and sample.get("has_zarr_activations"):
                zarr_data = self.activation_logger.get_entry(sample["zarr_key"])
                if zarr_data:
                    entry.update(zarr_data)
            results.append(entry)
        return results

    # Fallback: Zarr-only (returns only activation-logged samples)
    all_keys = self.get_group_keys(prompt_hash)
    sc_keys = [k for k in all_keys if "_sc_" in k]
    for key in sorted(sc_keys):
        if include_activations:
            entry = self.activation_logger.get_entry(key)
        else:
            entry = self.activation_logger.get_entry_by_key(key, metadata_only=True) or {}
        if include_logprobs and not include_activations:
            entry["logprobs"] = self.get_response_logprobs(key)
        results.append(entry)
    return results
```

`ActivationDataset.__getitem__` optionally attaches selfcheck entries when `include_selfcheck=True`.

### 2.8 SelfCheck scorer (offline, text-only)

A new evaluator class that reads `selfcheck_samples.jsonl` (no Zarr needed for n-gram/BERTScore):

```python
for row in selfcheck_samples_jsonl:
    sentences = sentence_tokenize(row["generation"])
    samples = [s["generation"] for s in row["selfcheck_samples"]]
    scores = [selfcheck_ngram_score(sentence, samples) for sentence in sentences]
    row["selfcheck_score"] = mean(scores)   # higher = more hallucinated
```

Output: `selfcheck_results.json` parallel to `eval_results.json`.

---

## 3. Required Code Changes (Scoped)

| File | Change | Priority |
|---|---|---|
| `activation_logging/server.py` | (1) Add `skip_activation_logging: bool = False` to `CompletionRequest` and `ChatCompletionRequest`; wrap `log_entry` call in `if not request.skip_activation_logging`. (2) Add `response_text` param to `build_entry_key()`; when `multi_sample=True` and no `request_id`, compute content-addressed key as `{prompt_hash}_sc_{SHA256(response)[:16]}`. (3) Move `build_entry_key()` call to after inference in chat completions endpoint (already post-inference in completions endpoint). | High |
| `scripts/run_with_server.py` | Add `--selfcheck-k`, `--selfcheck-temperature`, `--selfcheck-log-activations` flags; when present, insert `selfcheck` into `--step all` sequence (`generate → inference → selfcheck → eval`); add `selfcheck` as a valid `--step` choice for standalone use | High |
| `tasks/shortform/precise_wikiqa.py` | `run_step_selfcheck()`: iterate prompts from `generation.jsonl`, check per-prompt completion state (§2.6.3) including corruption check, call server with `extra_body` including `skip_activation_logging=True` for text-only samples; server returns content-addressed key in response `id` | High |
| `utils/lm.py` | Add `extra_body: Optional[dict] = None` param to `call_vllm_api`; pass `**(extra_body or {})` to `client.chat.completions.create()`. Add `logprobs=True` support for selfcheck text-only calls. | High |
| `activation_logging/activation_parser.py` | Add `get_selfcheck_entries()` method that merges JSONL (all samples) with Zarr (activation-logged only); add `include_selfcheck` param to `ActivationDataset` | Medium |
| `tasks/shortform/precise_wikiqa.py` | New `SelfCheckEval` class (n-gram scorer first, BERTScore/NLI later) | Medium |
| `requirements.txt` | `selfcheckgpt` package (from pypi); `bert_score` (optional) | Low |

---

## 4. Open Questions

1. **Temperature for samples:** 0.7 is the paper default; expose as `--selfcheck-temperature` flag.
2. **Sentence tokenizer:** paper uses spacy (`en_core_web_sm`); `nltk.sent_tokenize` may suffice and avoids adding the spacy dependency — check if nltk is already in the env.
3. **Greedy vs sampled:** Keep `temperature=0` greedy as the primary (already logged); run k stochastic samples separately. Paper evaluates greedy answer against stochastic samples.
4. **k budget:** Paper uses `k=20`; at what `k` does AUROC plateau? Plan a sweep over k={5, 10, 20}.
5. **Activation logging for samples:** Controlled by `--selfcheck-log-activations N` — N extra Zarr rows beyond the existing greedy row. Text-only samples skip the activation-logging server and call vLLM directly. Storage cost for N=5, k=20, 1000 prompts ≈ 80 GB of extra Zarr data — expose this clearly in docs.
6. **Comparable eval labels:** SelfCheckGPT produces a continuous score. Report AUROC against our binary `halu_test_res` labels from `eval_results.json`.
7. **Resume logic:** `selfcheck_samples.jsonl` is the resume record. If a line exists for a `prompt_hash` and has `len(selfcheck_samples) >= k`, skip it. On resume, activation-flagged entries are verified against Zarr — if either side is corrupt or mismatched, the entire prompt is discarded and regenerated (see §2.6.5).

---

## 5. Recommended Implementation Order

1. **Storage schema** — agree on the `generation.jsonl` format (see §2.1 above).
2. **Inference extension** — Option A: `k` sequential extra calls with temperature.
3. **n-gram scorer** — easiest SelfCheck variant, no extra models.
4. **BERTScore scorer** — stronger signal, needs `bert_score` dep.
5. **NLI scorer** — strongest black-box variant, needs DeBERTa model.
6. **Sweep experiment** — vary `k` and temperature, report AUROC on PreciseWikiQA.

---

## 6. Comparability Notes (for SOTA_TRACKER)

- SelfCheckGPT is **bucket D** (multi-sample + optional verifier model).
- Must report `k` (number of samples), temperature, and which variant (n-gram/BERTScore/NLI/LLM).
- **Not directly comparable** to single-pass activation-probe results without noting the extra inference cost.
- Paper results use GPT-3.5/4 (LLM-prompting variant) — our n-gram/NLI results will be lower-bound.
