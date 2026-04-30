# SmolLM3-3B Integration Plan

**Model:** `HuggingFaceTB/SmolLM3-3B`  
**Reference:** `docs/ADDING_NEW_MODELS.md` (general model integration guide)

---

## Architecture overview

| Property | Value | Impact |
|----------|-------|--------|
| Parameters | 3B | ~6 GB in float16 — well within H100/H200 VRAM |
| Architecture | Decoder-only, GQA, NoPE | Standard hidden_states format expected |
| Hidden size | ~2048 (TBC) | Zarr stores will be 2x smaller than Llama/Qwen3 (4096) |
| Layers | ~28 (TBC) | Slightly fewer chunks per batch write |
| Context | 128k (YARN extrapolation) | No issue at max_tokens=64 |
| Thinking mode | Yes (dual-mode) | Must disable — same class of problem as Qwen3 |
| transformers req | ≥ 4.53.0 | **Must verify on target node before running** |

---

## Risks and unknowns

### 1. Thinking mode (HIGH — must verify)
SmolLM3 has reasoning/thinking mode like Qwen3. The Qwen3 fix (use raw non-chat-template
prompts) likely works here too, but needs confirmation. If SmolLM3 activates thinking even
on raw prompts, the generated text will contain `<think>...</think>` blocks that corrupt
eval string matching.

**Verification:** Run 5 samples through `run_with_server.py --N 5` and inspect the
`generation` field in generation.jsonl. If it contains `<think>`, the model is thinking
on raw prompts and we need a suppression strategy (system message or `thinking_budget=0`).

### 2. NoPE architecture (LOW — verify hidden states format)
SmolLM3 uses NoPE (No Positional Encoding) instead of RoPE. This only affects attention
internals; the hidden states returned by `output_hidden_states=True` should follow the
standard format. Verify with the smoke test below.

### 3. transformers version (MEDIUM — check before running)
SmolLM3 requires transformers ≥ 4.53.0. The GPU nodes may have an older version.

**Check:** `ssh alphagpu12 "/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python -c 'import transformers; print(transformers.__version__)'"` 

If < 4.53.0: `pip install --upgrade transformers` in the p311 env on the target node.

### 4. Hidden size (LOW — verify before creating zarr stores)
SmolLM2-1.7B has hidden_size=2048. SmolLM3-3B is likely larger. The zarr logger infers
this automatically from the first batch — no manual config needed. Just confirm the actual
value from the smoke test so zarr store sizes can be estimated.

---

## Implementation steps

### Step 1: Environment check
```bash
ssh alphagpu12 "/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python -c \
  'import transformers; print(transformers.__version__)'"
```
If < 4.53.0, upgrade transformers. No other dependency changes expected.

### Step 2: Add to CLAUDE.md
Add to the supported models table:
```
| SmolLM3 3B | HuggingFaceTB/SmolLM3-3B | Thinking mode disabled via raw prompts (verify) |
```

### Step 3: Smoke test (64 samples, no activations)
```bash
python scripts/run_with_server.py \
    --step inference \
    --task naturalquestions \
    --model HuggingFaceTB/SmolLM3-3B \
    --split test \
    --N 64
```
Check:
- `output/natural_questions/SmolLM3-3B/generation.jsonl` has 64 lines
- `generation` fields are clean factual answers (no `<think>` tags)
- No `ValueError` from generate() about unknown kwargs

### Step 4: Smoke test with activations
```bash
python scripts/run_with_server.py \
    --step inference \
    --task naturalquestions \
    --model HuggingFaceTB/SmolLM3-3B \
    --activations-path /tmp/smollm3_smoke/activations.zarr \
    --split test \
    --N 64
```
Check zarr store:
```python
import zarr
z = zarr.open('/tmp/smollm3_smoke/activations.zarr', 'r')
arr = z['arrays']['prompt_activations']
print(arr.shape)    # (64, <n_layers>, 64, <hidden_size>)
print(arr.chunks)   # (32, 1, 64, <hidden_size>)
```
Confirm `hidden_size` and `n_layers` match expected values.

### Step 5: Create dataset configs
Create `configs/datasets/nq_test_smollm3.json` and `configs/datasets/nq_train_smollm3.json`
mirroring the Qwen3 versions (`nq_test_qwen3.json`) but with:
- `model_name`: `SmolLM3-3B`
- `input_dim`: actual hidden_size from Step 4
- `activations_path`: `shared/natural_questions_smollm3_3b/activations.zarr`

### Step 6: Full generation run
Add SmolLM3-3B to `scripts/generate_all_qwen3.sh` or create a parallel
`scripts/generate_all_smollm3.sh` with the same structure, substituting the model name
and zarr paths.

---

## Thinking mode suppression (if needed)

If Step 3 reveals `<think>` tags in raw-prompt outputs, suppress thinking via a system
message in the task's `format_prompt` function:

```python
# Option A: chat-template with /no_think system message
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": f"Answer concisely.\n\nQ: {question}\nA:"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This would require a new per-model prompt path in each task's `format_prompt`. The cleanest
approach is a model-name check in a shared utility, similar to how Qwen3's thinking was
handled.

---

## Expected zarr store sizes (estimate)

Assuming hidden_size=2048, ~28 layers, 64 tokens, float16:

| Dataset | Samples | Size |
|---------|---------|------|
| NQ test | 4,155 | ~4 GB |
| NQ train | 16,617 | ~16 GB |
| hotpotqa val | 7,405 | ~7 GB |
| hotpotqa train | 90,447 | ~87 GB |
| Full stack | ~400k | ~380 GB |

Approximately 2x smaller than Qwen3/Llama stores (which use hidden_size=4096).
