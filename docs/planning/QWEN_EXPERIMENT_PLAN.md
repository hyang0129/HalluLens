# Plan: Replicate Llama3 Experiments for Qwen3-8B

## Model Selection

**Chosen: `Qwen/Qwen3-8B`** (released mid-2025, successor to Qwen2.5-7B-Instruct)

`Qwen/Qwen3.5-9B` was also evaluated but rejected: its hybrid attention architecture (8 standard + 24 Gated DeltaNet layers) produces non-uniform activation shapes across layers, which breaks our classifier stack that assumes uniform hidden dim `D` at every layer.

Key specs vs. Llama 3.1-8B-Instruct:

| Property | Llama 3.1-8B | Qwen3-8B |
|----------|-------------|----------|
| Hidden dim | 4096 | 4096 |
| Num layers | 32 | 36 |
| Context | 128K | 32K native / 131K w/ YaRN |
| HF model ID | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen3-8B` |

Hidden dim matches (4096), so `input_dim` in dataset configs stays the same — no model architecture changes needed.

Quantized options if VRAM is tight: `Qwen/Qwen3-8B-AWQ` or `Qwen/Qwen3-8B-FP8` (no GPTQ-Int8 available, unlike the 72B).

Fallback if Qwen3-8B causes problems: `Qwen/Qwen2.5-7B-Instruct` (already in `model_map`, hidden_dim=3584 → would require `input_dim=3584` in all dataset configs).

---

## Thinking Mode Decision: OFF

**Decision: run with `enable_thinking=False` for all primary experiments.**

### Why thinking OFF

Qwen3's thinking tokens are plain autoregressive output — the same weights and layers process them as the answer. The `</think>` token (ID 151668) is just a delimiter; there is no architectural separation between the two phases.

The research evidence strongly favors disabling thinking for activation-based hallucination detection:

- A 2025 EMNLP paper ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.67/)) found CoT prompting *degrades* activation-based hallucination detection: it shifts internal states and washes out the factual-recall signal classifiers are trained on.
- Thinking tokens dominate sequence length. If logged as part of the full sequence (our current setup), they dilute the answer-phase activations that carry the hallucination signal.
- Thinking OFF is a clean apples-to-apples comparison with Llama3, which has no thinking.

### Sampling parameters

Our Llama3 experiments use greedy decoding (`temperature=0.0`). Qwen3 with thinking OFF performs well with greedy. Keep `temperature=0.0` for consistency.

If thinking is ever enabled in a future ablation, Qwen3 docs recommend `temp=0.6, top_p=0.95, top_k=20` — greedy with thinking ON degrades output quality.

### Future ablation (not in scope now)

A possible follow-up: enable thinking with `thinking_token_budget=1024` (vLLM `SamplingParams`) and log activations from **answer-phase tokens only** (after `</think>`). The final hidden state of `</think>` may encode an uncertainty summary worth probing. This requires segmenting the activation sequence at token ID 151668 — a non-trivial change to the logger — so it is deferred.

---

## Datasets in Scope

All 7 datasets, each with train + test splits:

| Dataset | Train split config | Test split config |
|---------|-------------------|------------------|
| mmlu | mmlu_train | mmlu_test |
| popqa | popqa_train | popqa_test |
| sciq | sciq_train | sciq_test |
| searchqa | searchqa_train | searchqa_test |
| hotpotqa | hotpotqa_train | hotpotqa |
| movies | — (test only) | movies |
| natural_questions | nq_train_qwen3 (new) | nq_test_qwen3 (new) |

**NQ note:** The existing Llama NQ data landed in `shared/natural_questions_logprob/` for historical reasons. For Qwen, use the standard convention: `output/natural_questions[_train]/Qwen3-8B/` and `shared/natural_questions[_train]_qwen3_8b/activations.zarr`. NQ also currently lacks a train-split config — create both train and test configs fresh.

---

## What Needs to Change

### 1. Add Qwen3-8B to `utils/lm.py` model_map

`Qwen/Qwen3-8B` is not currently in `model_map`. Add it alongside the existing Qwen2.5 entries (~line 755):

```python
'Qwen/Qwen3-8B': {'name': 'qwen3_8B',
                   'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
'Qwen/Qwen3-8B-AWQ': {'name': 'qwen3_8B_awq',
                       'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
```

### 2. Disable thinking mode in `activation_logging/server.py`

Qwen3 thinking mode must be explicitly disabled. The HF transformers path (used for non-quantized models, which is what we want for activation logging) calls `model.generate()`. The fix: detect `"qwen3"` in the model name and inject `enable_thinking=False` into the generation kwargs.

Concretely, find where `model.generate()` is called in the HF inference path and add:

```python
if "qwen3" in model_name.lower():
    generate_kwargs["enable_thinking"] = False
```

This is a **new requirement** that didn't exist for Llama3. Without it, responses will contain `<think>...</think>` preambles that corrupt eval scoring and contaminate activations.

### 3. Run inference + activation logging for all datasets

For each dataset (train and test), run `scripts/run_with_server.py` with `--model Qwen/Qwen3-8B`. Output paths follow the existing convention automatically:

- Generations: `output/{dataset}/Qwen3-8B/generation.jsonl`
- Eval results: `output/{dataset}/Qwen3-8B/eval_results.json`
- Activations: `shared/{dataset}_qwen3_8b/activations.zarr`

Node to use: **alphagpu19** (H200, ~116 GB free VRAM — Qwen3-8B at fp16 is ~16 GB, AWQ is ~5 GB).

### 4. Create new dataset configs

Create `configs/datasets/{name}_train_qwen3.json` and `configs/datasets/{name}_test_qwen3.json` for all 7 datasets (~13 files total — movies and NQ train are new). Each mirrors the Llama version with these fields changed:

```json
{
  "name": "{dataset}_train_qwen3",
  "inference_json": "output/{dataset}_train/Qwen3-8B/generation.jsonl",
  "activations_path": "shared/{dataset}_train_qwen3_8b/activations.zarr",
  "eval_json": "output/{dataset}_train/Qwen3-8B/eval_results_for_training.json",
  "raw_eval_jsonl": "output/{dataset}_train/Qwen3-8B/raw_eval_res.jsonl",
  "model_name": "Qwen3-8B",
  "input_dim": 4096
}
```

### 5. Create new experiment configs

Create `configs/experiments/baseline_comparison_{dataset}_qwen3.json` for all 7 datasets. Copy the existing Llama version and update `"dataset"` to point to the new Qwen3 dataset config name. Methods and seeds stay identical:

```json
{
  "experiment_name": "baseline_comparison_{dataset}_qwen3",
  "dataset": "{dataset}_qwen3",
  "methods": ["contrastive_logprob_recon", "linear_probe", "multi_layer_linear_probe", "token_entropy", "logprob_baseline"],
  "split_seed": 42,
  "training_seeds": [0, 5, 26, 42, 63],
  "device": "auto",
  "num_workers": 4,
  "persistent_workers": true,
  "output_dir": "runs"
}
```

### 6. Run training experiments

After inference + activations are complete, run `scripts/experiment_status.py` to verify data is ready, then dispatch training jobs via `gpu_dispatch.py` for each experiment config — same as the Llama3 workflow.

---

## Layer Range Notes

Qwen3-8B has **36 layers** vs. Llama 3.1-8B's **32 layers**. The analogous layer range for training would be approximately layers 17–35 (instead of 14–29). For eval-layer spot checks, use layers 26 and 32.

These are passed as CLI flags to `train_activation_model.py` (`--train-layers`, `--eval-layers`), not baked into configs, so no config changes needed — just update dispatch scripts.

---

## Known Pre-existing Debt (not blocking)

- `utils/generate_question.py:179` — hardcoded LLaMA tokenizer used for document chunking in question generation. Negligible effect on output quality; tracked in `QWEN_QGEN_PLAN.md`.

---

## Execution Order

1. **Code** — Add Qwen3-8B to `utils/lm.py` model_map + disable thinking mode in `server.py`
2. **Inference** — Run all 13 dataset splits on alphagpu19 (train first, test second; NQ train is new)
3. **Configs** — Create ~13 dataset configs + 7 experiment configs
4. **Training** — Dispatch training runs via `gpu_dispatch.py`
5. **Evaluation** — Compare results against Llama3 baseline using existing result-reporting scripts
