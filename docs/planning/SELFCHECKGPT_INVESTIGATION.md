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
- `--selfcheck-k 20` — number of stochastic text responses to generate (all stored in `selfcheck_samples.jsonl`)
- `--selfcheck-log-activations 5` — how many of those to also log to Zarr (default: 1, i.e., greedy only; set to N to get greedy + N-1 stochastic)

**Key challenge — hash is prompt-level, not answer-level:**
The existing Zarr is keyed by `prompt_hash` (SHA-256 of the prompt text). This works because each prompt had exactly one answer. For multiple answers to the same prompt, the Zarr key **must include the sample index** to be unique. The proposed `{prompt_hash}_sc_{i}` scheme solves this — the existing `_build_group_index()` groups by the prefix (splitting on `_`), so the greedy entry at `{prompt_hash}` and stochastic entries at `{prompt_hash}_sc_0`, `{prompt_hash}_sc_1` etc. are all grouped together correctly.

**Inference loop (per prompt):**

```python
# 1. Greedy pass — already logged, skip if exists in Zarr
greedy_key = prompt_hash
# already logged in prior inference run

# 2. Stochastic passes
for i in range(selfcheck_k):
    sc_key = f"{prompt_hash}_sc_{i}"
    log_activations = (i < selfcheck_log_activations - 1)  # -1 because greedy already counts

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

**What gets logged per sample type:**

| Sample | Key | Zarr row? | Activations? | Logprobs? | Text? |
|---|---|---|---|---|---|
| Greedy (existing) | `{hash}` | Yes (existing) | Yes | Yes | generation.jsonl |
| SC sample 0..N-2 | `{hash}_sc_0` .. `{hash}_sc_{N-2}` | Yes (new) | Yes | Yes | selfcheck_samples.jsonl |
| SC sample N-1..k-1 | `{hash}_sc_{N-1}` .. `{hash}_sc_{k-1}` | No | No | No | selfcheck_samples.jsonl |

For N=5, k=20: 1 greedy + 4 stochastic with activations + 16 text-only.

**Storage estimate** (example: 1000 prompts, 32 layers, hidden=4096, R_max=64):
- Per-row Zarr cost ≈ 32 × 64 × 4096 × 2 bytes (float16) ≈ 16 MB
- 1000 prompts × 4 extra activation rows ≈ 64 GB — significant, expose `--selfcheck-log-activations` clearly.

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

### 2.4 New inference step: `--step selfcheck`

A standalone step that can be appended to any existing run:

```bash
python scripts/run_with_server.py \
    --step selfcheck \
    --inference-json shared/goodwiki/generation.jsonl \
    --activations-path shared/goodwiki/activations.zarr \
    --selfcheck-k 10 \
    --selfcheck-temperature 0.7
```

Internal flow:
1. Read existing `generation.jsonl` to get prompts + prompt hashes.
2. Read `selfcheck_samples.jsonl` (if exists) to skip already-sampled prompts (resume).
3. For each unsampled prompt, make `k` API calls at `temperature > 0` to the running vLLM server.
4. Log each sample to the existing Zarr with key `{prompt_hash}_sc_{i}`.
5. Append a line to `selfcheck_samples.jsonl`.

### 2.5 `ActivationParser` additions

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

### 2.6 SelfCheck scorer (offline, text-only)

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
5. **Activation logging for samples:** Controlled by `--selfcheck-log-activations N` (greedy already counts as 1). Text-only samples skip the activation-logging server and call vLLM directly. Storage cost for N=5, k=20, 1000 prompts ≈ 64 GB of extra Zarr data — expose this clearly in docs.
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
