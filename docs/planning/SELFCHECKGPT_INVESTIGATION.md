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
| NLI | Entailment score: samples entail the sentence | DeBERTa MNLI (`potsawee/longformer-large-4096-answering-race` or similar) |
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

**We can use this directly.** SelfCheck samples are stored in the same Zarr with keys:

```
{prompt_hash}           ← existing greedy entry (sample_index=0)
{prompt_hash}_sc_0      ← stochastic sample 0  (sample_index=1)
{prompt_hash}_sc_1      ← stochastic sample 1  (sample_index=2)
...
{prompt_hash}_sc_{k-1}  ← stochastic sample k-1
```

Each `_sc_` entry's Zarr metadata (written to `meta/index.jsonl`) includes:
```json
{
  "key": "{prompt_hash}_sc_0",
  "sample_group_id": "{prompt_hash}",
  "sample_index": 1,
  "is_selfcheck_sample": true,
  "selfcheck_sample_index": 0,
  "temperature": 0.7,
  "generation": "stochastic answer text",
  "prompt": "..."
}
```

`_build_group_index()` groups by `sample_group_id`, so `get_group_keys(prompt_hash)` returns all of them. `select_primary_key()` returns the `sample_index == 0` greedy entry, leaving existing code paths unchanged.

### 2.2 Selective activation logging: `selfcheck_k` vs `activation_log_count`

Generating 20 stochastic samples at inference time but only storing activations for 5 of them requires decoupling the **text sampling count** from the **activation logging count**.

**Parameters:**
- `--selfcheck-k 20` — number of **extra** stochastic samples to generate beyond the greedy pass (21 total: 1 greedy + 20 stochastic). All 20 texts stored in `selfcheck_samples.jsonl`.
- `--selfcheck-log-activations 5` — how many of the 20 stochastic samples to also log to Zarr (default: 0). The greedy Zarr row already exists and is not counted here.

**Key challenge — hash is prompt-level, not answer-level:**
The existing Zarr is keyed by `prompt_hash` (SHA-256 of the prompt text). This works because each prompt had exactly one answer. For multiple answers to the same prompt, the Zarr key **must include the sample index** to be unique. The proposed `{prompt_hash}_sc_{i}` scheme solves this — the existing `_build_group_index()` groups by the prefix (splitting on `_`), so the greedy entry at `{prompt_hash}` and stochastic entries at `{prompt_hash}_sc_0`, `{prompt_hash}_sc_1` etc. are all grouped together correctly.

**Inference loop (per prompt):**

```python
# 1. Greedy pass — already logged in the prior inference run, skip
greedy_key = prompt_hash  # existing Zarr row

# 2. Stochastic passes (selfcheck_k=20 extra samples)
for i in range(selfcheck_k):
    sc_key = f"{prompt_hash}_sc_{i}"
    log_activations = (i < selfcheck_log_activations)  # first N get full Zarr row

    if log_activations:
        # Call activation-logging server (FastAPI + vLLM hooks)
        # Zarr row written with key sc_key
        response = call_logging_server(prompt, temperature=temp, zarr_key=sc_key)
    else:
        # Call vLLM directly (no hooks, no Zarr write) — text only
        response = call_vllm_direct(prompt, temperature=temp)

    # Text always goes to selfcheck_samples.jsonl regardless
    append_selfcheck_text(prompt_hash, sc_key, response.text, logged=log_activations)
```

**What gets logged per sample type** (example: selfcheck_k=20, selfcheck_log_activations=5):

| Sample | Key | Zarr row? | Activations? | Logprobs in Zarr? | Logprobs in JSONL? | Text? |
|---|---|---|---|---|---|---|
| Greedy (existing) | `{hash}` | Yes (existing) | Yes | Yes | — | generation.jsonl |
| SC sample 0..4 | `{hash}_sc_0` .. `{hash}_sc_4` | Yes (new) | Yes | Yes | Yes | selfcheck_samples.jsonl |
| SC sample 5..19 | `{hash}_sc_5` .. `{hash}_sc_19` | No | No | No | **Yes** | selfcheck_samples.jsonl |

Total: 1 existing greedy row + 5 new stochastic Zarr rows + 15 text+logprob-only.

**Logprobs for text-only samples:** vLLM returns token logprobs as part of every response payload — no activation hooks required. The direct vLLM call already gets them. We store them inline in `selfcheck_samples.jsonl` rather than Zarr:

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
- 1000 prompts × 5 extra activation rows ≈ 80 GB — significant, expose `--selfcheck-log-activations` clearly.

**No changes needed to `ZarrActivationsLogger`** — it already supports arbitrary string keys and append-only writes.

### 2.3 Companion text file: `selfcheck_samples.jsonl`

Alongside the Zarr, write a plain JSONL file (one line per prompt) for human-readable inspection and to support text-only scorers without touching Zarr:

```json
{
  "prompt_hash": "abc123...",
  "prompt": "Answer in one sentence. Q:...\n A:",
  "generation": "primary greedy answer",
  "selfcheck_samples": [
    {"zarr_key": "abc123..._sc_0", "generation": "sample 1", "temperature": 0.7},
    {"zarr_key": "abc123..._sc_1", "generation": "sample 2", "temperature": 0.7}
  ]
}
```

This is written incrementally (one line at a time) with resume support keyed on `prompt_hash`.

### 2.4 Server-side key assignment — how the server handles repeated calls for the same prompt

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

`multi_sample` mode is triggered when any of `multi_sample=True`, `sample_index`, `sample_group_id`, or `request_id` is set on the request.

**Problem:** if the client calls the server k times for the same prompt without passing a deterministic `request_id`, each call gets a random UUID suffix → keys like `{prompt_hash}_a3f2b1c0`, `{prompt_hash}_99de12ab` — unpredictable, unscannable, not resumable.

**Solution: the client must pass a deterministic `request_id` per sample index.**

For selfcheck sample `i`, the client sets `request_id = f"sc_{i}"`:

```python
# results in entry_key = f"{prompt_hash}_sc_{i}"
response = call_logging_server(
    prompt,
    multi_sample=True,
    request_id=f"sc_{i}",
    sample_group_id=prompt_hash,
    sample_index=i + 1,   # 0 = greedy (already logged), 1..k = stochastic
    temperature=temp,
)
```

This gives exactly the `{prompt_hash}_sc_{i}` key scheme the resume logic depends on. SHA-256 hashes are pure hex (no underscores), so `key.split("_")[0]` in `_build_group_index()` correctly recovers the prompt hash as the group ID.

**No server changes needed** — `request_id` is already an accepted field on both `/v1/completions` and `/v1/chat/completions`. The client just needs to set it explicitly rather than letting the server generate a random one.

### 2.5 New inference step: `--step selfcheck`

A standalone step that can be appended to any existing run:

```bash
python scripts/run_with_server.py \
    --step selfcheck \
    --inference-json shared/hotpotqa/generation.jsonl \
    --activations-path shared/hotpotqa/activations.zarr \
    --selfcheck-k 20 \
    --selfcheck-log-activations 5 \
    --selfcheck-temperature 0.7
```

#### Resume / reuse logic

The scenario: HotpotQA was already fully run (greedy inference + activation logging). Now we add selfcheck. The step must reuse all existing saved data and be safely restartable.

**Sources of truth (checked in order):**

1. `selfcheck_samples.jsonl` — one line per prompt hash. Tracks text + logprobs for each sample, plus `has_zarr_activations` per entry. The line also records the parameters it was generated with: `selfcheck_k` and `selfcheck_log_activations`.

2. Zarr index (`meta/index.jsonl`) — tracks which `_sc_` keys have been written. Used as a fallback if `selfcheck_samples.jsonl` is missing or partially written (e.g., crash mid-write).

**A prompt is fully done only when BOTH conditions hold:**
- `len(jsonl_entry["selfcheck_samples"]) == selfcheck_k` — all text samples present
- `sum(1 for s in jsonl_entry["selfcheck_samples"] if s["has_zarr_activations"]) == selfcheck_log_activations` — correct number of activation-logged rows

This matters for the case where a prior run used `selfcheck_log_activations=3` and the current run specifies `selfcheck_log_activations=5` — the text samples are complete but the activation count is insufficient, so the missing activation rows must be generated.

**Resume decision tree per prompt:**

```
for prompt_hash in generation.jsonl:

    # Greedy — always reuse, never re-run
    assert prompt_hash in zarr_index   # written by original inference run

    # Load existing JSONL entry (may be None, partial, or complete)
    jsonl_entry = selfcheck_samples_jsonl.get(prompt_hash)
    existing_samples = jsonl_entry["selfcheck_samples"] if jsonl_entry else []

    # Count what we already have
    existing_text_indices = {i for i, s in enumerate(existing_samples)}
    existing_activation_indices = {
        i for i, s in enumerate(existing_samples)
        if s.get("has_zarr_activations")
        or f"{prompt_hash}_sc_{i}" in zarr_index  # Zarr fallback for crash recovery
    }

    # Both conditions must be satisfied to skip
    text_complete = (len(existing_text_indices) == selfcheck_k)
    activations_complete = (len(existing_activation_indices) == selfcheck_log_activations)
    if text_complete and activations_complete:
        continue  # fully done

    # Generate missing samples
    for i in range(selfcheck_k):
        needs_text = i not in existing_text_indices
        needs_activations = (i < selfcheck_log_activations) and (i not in existing_activation_indices)

        if not needs_text and not needs_activations:
            continue  # already have everything for this index

        sc_key = f"{prompt_hash}_sc_{i}"
        if needs_activations:
            # Must (re-)call activation-logging server to get Zarr row
            response = call_logging_server(prompt, key=sc_key)
        elif needs_text:
            # Text-only: call vLLM directly
            response = call_vllm_direct(prompt)

        # Update in-memory sample list, merge with existing where present
        upsert_sample(existing_samples, i, sc_key, response)

    # Rewrite JSONL line atomically once all k samples are complete
    write_selfcheck_jsonl(prompt_hash, existing_samples)
```

**Key edge case — upgrading activation count:** if a sample index `i < selfcheck_log_activations` already has a text entry in JSONL (`needs_text=False`) but no Zarr row (`needs_activations=True`), the logging server is called and only the activation/logprob data is written to Zarr. The text in the JSONL entry is unchanged. This lets you upgrade `selfcheck_log_activations` on a prior run without re-generating any text.

**Greedy is never re-generated:** `--step selfcheck` never touches the greedy Zarr row at `prompt_hash`. It only writes `{prompt_hash}_sc_{i}` keys and `selfcheck_samples.jsonl`. The original `generation.jsonl` is read-only input.

### 2.6 `ActivationParser` additions

Add one method to expose selfcheck samples to downstream consumers:

```python
def get_selfcheck_entries(
    self,
    prompt_hash: str,
    include_activations: bool = False,
    include_logprobs: bool = True,
) -> List[Dict[str, Any]]:
    """Return metadata (+ optional activations/logprobs) for all selfcheck
    samples of a given prompt. Returns empty list if none logged."""
    all_keys = self.get_group_keys(prompt_hash)
    sc_keys = [k for k in all_keys if "_sc_" in k]
    results = []
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

### 2.7 SelfCheck scorer (offline, text-only)

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
| `scripts/run_with_server.py` | Add `selfcheck` step; wire `--selfcheck-k`, `--selfcheck-temperature`, `--selfcheck-log-activations` flags | High |
| `tasks/shortform/precise_wikiqa.py` | `run_step_selfcheck()`: iterate prompts, route first N samples through activation-logging server, remaining through vLLM directly, write `selfcheck_samples.jsonl` | High |
| `activation_logging/activation_parser.py` | Add `get_selfcheck_entries()` method; add `include_selfcheck` param to `ActivationDataset` | Medium |
| `tasks/shortform/precise_wikiqa.py` | New `SelfCheckEval` class (n-gram scorer first, BERTScore/NLI later) | Medium |
| `requirements.txt` | `selfcheckgpt` package (from pypi); `bert_score` (optional) | Low |

---

## 4. Open Questions

1. **Temperature for samples:** 0.7 is the paper default; expose as `--selfcheck-temperature` flag.
2. **Sentence tokenizer:** paper uses spacy (`en_core_web_sm`); need to check if it's in the env or add it.
3. **Greedy vs sampled:** Keep `temperature=0` greedy as the primary (already logged); run k stochastic samples separately. Paper evaluates greedy answer against stochastic samples.
4. **k budget:** Paper uses `k=20`; at what `k` does AUROC plateau? Plan a sweep over k={5, 10, 20}.
5. **Activation logging for samples:** Controlled by `--selfcheck-log-activations N` — N extra Zarr rows beyond the existing greedy row. Text-only samples skip the activation-logging server and call vLLM directly. Storage cost for N=5, k=20, 1000 prompts ≈ 80 GB of extra Zarr data — expose this clearly in docs.
6. **Comparable eval labels:** SelfCheckGPT produces a continuous score. Report AUROC against our binary `halu_test_res` labels from `eval_results.json`.
7. **Resume logic:** `selfcheck_samples.jsonl` is the resume record. If a line exists for a `prompt_hash` and has `len(selfcheck_samples) == k`, skip it. The Zarr keys also act as a fallback resume check via `get_group_keys`.

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
