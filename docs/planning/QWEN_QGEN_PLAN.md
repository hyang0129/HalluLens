# Plan: Qwen2.5 Question Generation Support

## Context

LLaMA 70B has been unreliable as the question generator for PreciseWikiQA/LongWiki. Qwen2.5-72B-Instruct-GPTQ-Int8 is the target replacement. It's already mentioned in the `run_with_server.py` docstring as "supported" but was missing from `model_map` in `utils/lm.py`, causing a `KeyError` when `call_vllm_api` tries to look up the server URL.

**Most infrastructure already works:**
- `activation_logging/server.py` already routes GPTQ models to vLLM automatically via `_should_use_vllm_backend()` which checks `if "gptq" in model_l`. No changes needed to `server.py` or `vllm_serve.py`.
- `call_vllm_api` already uses the **chat completions** endpoint (`client.chat.completions.create`), sending `messages=[{"role": "user", "content": prompt}]`. vLLM applies the loaded model's native chat template server-side, so Qwen2.5 will automatically get its `<|im_start|>` template — no prompt formatting changes needed.
- The question generation prompts (`PRECISE_Q_GENERATION_PROMPT`, `ANSWERABILITY_PROMPT`) are clean plain-text instructions with no LLaMA-specific tokens baked in, so they are model-agnostic.

The fix is entirely in `utils/lm.py`.

---

## Changes Made

### `utils/lm.py` — added to `model_map`

```python
# Qwen2.5 models (q_generator replacements for LLaMA 70B)
'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8': {'name': 'qwen2.5_72B_gptq_int8', ...},
'Qwen/Qwen2.5-14B-Instruct':            {'name': 'qwen2.5_14B', ...},
'Qwen/Qwen2.5-7B-Instruct':             {'name': 'qwen2.5_7B', ...},
```

All three use `http://{CUSTOM_SERVER}:8000/v1` (same pattern as all other models).

GPTQ routing: `server.py` auto-detects `"gptq"` in the model name and loads via vLLM backend. No explicit `--quantization gptq` flag required from the caller.

---

## Chat Template / Prompt Format Notes

`call_vllm_api` sends prompts via the **chat completions** API (not raw text completions). vLLM applies the model's native chat template server-side:

| Model | Template applied |
|-------|-----------------|
| LLaMA 3.x | `<\|start_header_id\|>user<\|end_header_id\|>\n...<\|eot_id\|>...` |
| Qwen2.5 | `<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n` |

Since `PRECISE_Q_GENERATION_PROMPT` and `ANSWERABILITY_PROMPT` are clean plain-text instructions (no embedded model-specific tokens), the template difference is invisible to output quality — both models receive the same semantic instruction in their native format.

---

## Known Minor Debt (not blocking)

**Hardcoded LLaMA tokenizer in `WikiQA.__init__`** (`utils/generate_question.py` line 179):
```python
self.encoding = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B-Instruct', ...)
```
Used for document length gating/chunking. Qwen2.5 has a larger vocabulary (~150k vs ~128k tokens) so tokenizes more efficiently — chunk boundaries will be very slightly off when using Qwen as qgen. Effect on quality is negligible since this is length estimation, not exact counting.

**Follow-up**: pass `q_generator` model ID into `WikiQA.__init__` and load the actual model's tokenizer dynamically.

---

## Smoke Test

See `notebooks/qwen_qgen_smoketest.ipynb`. Uses the imported `run_experiment` function:

```python
from scripts.run_with_server import run_experiment

run_experiment(
    step="generate",
    task="precisewikiqa",
    model="meta-llama/Llama-3.1-8B-Instruct",
    q_generator="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
    N=5,
    quick_debug_mode=True,
)
```

**What to check:**
1. Server starts without error (GPTQ model detected, vLLM backend loads)
2. No `KeyError` on model lookup
3. 5 QA pairs generated — verify question quality looks reasonable (not garbled or repetitive)
4. Check the QA output file for well-formed questions ending in `?`
