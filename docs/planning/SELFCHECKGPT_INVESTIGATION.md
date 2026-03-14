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

## 2. What Needs to Change in HalluLens

### 2.1 Inference step — multi-sampling

Currently: **one generation per prompt** in `generation.jsonl`.

We need: **k stochastic samples per prompt**, stored alongside the greedy answer.

**Proposed schema change to `generation.jsonl`:**

```json
{
  "index": 42,
  "prompt": "Answer in one sentence. Q:...\n A:",
  "answer": "gold answer",
  "generation": "primary greedy answer",
  "selfcheck_samples": [
    "sampled answer 1 (temp=0.7)",
    "sampled answer 2 (temp=0.7)",
    "sampled answer 3 (temp=0.7)"
  ]
}
```

- `generation` stays as the greedy/primary response (used for all other evaluators).
- `selfcheck_samples` is a new array of k stochastic samples at temperature > 0.

This is backward compatible: existing loaders ignore unknown fields.

### 2.2 How to generate the samples

**Option A — Extra API calls during inference (simplest):**
- After each greedy generation, make `k` additional calls with `temperature=0.7` (or similar).
- Already supported by the OpenAI-protocol server in `activation_logging/server.py`.
- Activated with a flag like `--selfcheck-samples k`.
- Adds `k × inference_time` cost per prompt.

**Option B — Batch multi-sampling via vLLM `n` parameter:**
- vLLM supports `n` completions per request. Pass `n=k+1` with temperature > 0.
- First completion = samples; take greedy separately (or use `best_of`).
- Much faster (single forward pass for batch), but requires changes to how we call the server.
- More invasive server changes.

**Recommendation for investigation phase: Option A** — minimal code change, lets us validate the approach before optimizing.

### 2.3 Evaluation step — SelfCheck scorer

A new offline scorer that reads the multi-sample `generation.jsonl` and produces a SelfCheck score per prompt. This fits naturally as a new step or a new evaluator class alongside `PreciseQAEval`.

**Pseudocode:**
```python
for row in generation_jsonl:
    sentences = sentence_tokenize(row["generation"])
    samples = row["selfcheck_samples"]
    scores = [selfcheck_score(sentence, samples) for sentence in sentences]
    row["selfcheck_score"] = mean(scores)   # higher = more hallucinated
```

Output: `selfcheck_results.json` parallel to `eval_results.json`.

### 2.4 Activation logging — no change needed (initially)

SelfCheckGPT is black-box: it only needs the text of samples. No activation changes required.
Later, if we want to study *why* SelfCheck works, we can log activations for the samples too.

---

## 3. Required Code Changes (Scoped)

| File | Change | Priority |
|---|---|---|
| `tasks/shortform/precise_wikiqa.py` | Add `--selfcheck-samples k` flag; make `k` extra stochastic calls per prompt; write `selfcheck_samples` array to `generation.jsonl` | High |
| `utils/exp.py` | Thread-safe / incremental save of multi-sample arrays | High |
| `tasks/shortform/precise_wikiqa.py` | New `SelfCheckEval` class (n-gram scorer first, BERTScore/NLI later) | High |
| `scripts/run_with_server.py` | Wire `--selfcheck-samples` into the inference and eval steps | Medium |
| `requirements.txt` | `selfcheckgpt` package (from pypi), `bert_score` (optional, for BERTScore variant) | Low |

---

## 4. Open Questions

1. **Temperature for samples:** 0.7 is the paper default; should we expose this as a flag?
2. **Sentence tokenizer:** paper uses spacy (`en_core_web_sm`); is that already in our env?
3. **Greedy vs sampled:** Do we want a true greedy pass (temperature=0) + k sampled, or k+1 sampled? The paper uses the greedy answer as "the response to evaluate."
4. **k budget:** Paper uses `k=20` samples; at what `k` does performance plateau? Should we run a sweep?
5. **Activation logging for samples:** Should activations be logged for the stochastic samples too (for future SE / SEP comparisons)? Would significantly increase storage.
6. **Comparable eval labels:** SelfCheckGPT produces a continuous score; our `eval_results.json` labels are binary (CORRECT/INCORRECT). We need to threshold or report AUROC directly.
7. **Resume logic:** The current resume key is `prompt`. With multi-samples, resume should check both `prompt` and whether `selfcheck_samples` has length `k`.

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
