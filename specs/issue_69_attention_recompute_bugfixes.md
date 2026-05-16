# Attention Recompute — GPU Smoketest Bugfix Pass (2026-05-15)

First end-to-end run of `scripts/smoketest_attention_recompute.sh` on a
Jupyter GPU node (alphagpu03-8889, H100 80GB) caught two latent bugs in
`activation_logging/attention_recompute.py` that only surface when the
recompute path is exercised against a real HF model in fp16 on the installed
transformers version (>= ~4.50). Both were fixed in commit
`8e99376` on branch `feat/issue-69-icr-probe-attn-infra`.

## Bug 1 — dtype mismatch in `recompute_block_attention`

### Symptom

Phase 1 (`--validate-first`) crashed immediately on the first block of the
first sample:

```
RuntimeError: expected mat1 and mat2 to have the same dtype,
              but got: float != c10::Half
```

Traceback bottomed out in `transformers/models/llama/modeling_llama.py:236`,
the `q_proj(hidden_states)` linear projection.

### Root cause

`recompute_block_attention` was casting `h_prev` to a hardcoded `torch.float32`
before feeding it through the block:

```python
h_prev = h_prev.to(device=device, dtype=torch.float32)
```

The HF model in this codebase loads with `torch_dtype=torch.float16` (see
`recompute_attention.py:_load_model`), so the block's `q_proj.weight` is fp16
and the fp32 hidden state could not flow through it.

### Fix

Infer the dtype from the block parameters and cast both `h_prev` and the
additive causal mask to match:

```python
block = block.to(device).eval()
model_dtype = next(block.parameters()).dtype
h_prev = h_prev.to(device=device, dtype=model_dtype)
causal_mask = _build_causal_mask(T, device).to(dtype=model_dtype)
```

The cast happens after `block.to(device)` so we read the dtype of the
device-placed weights, not the (possibly different) on-CPU staging copy.
`_head_average_resp_to_resp` already up-casts the attention tensor with
`.float()` before returning, so the public output contract
(`(R, R) float32`) is unchanged.

## Bug 2 — `rotary_emb` moved to model level in newer transformers

### Symptom

After Bug 1 was fixed, the next run failed slightly later:

```
File ".../modeling_llama.py", line 240, in forward
    cos, sin = position_embeddings
TypeError: cannot unpack non-iterable NoneType object
```

So `block.self_attn(...)` was now receiving `position_embeddings=None`, which
the newer LlamaAttention.forward requires to be a `(cos, sin)` tuple.

### Root cause

`_call_self_attn` tried two paths:

1. **Old API path** — call `self_attn` without `position_embeddings` and let
   it derive RoPE from `position_ids` internally. This raises `TypeError` on
   newer transformers (the kwarg is required), which we caught.
2. **New API path** — look up `rotary_emb` on the block or on `self_attn`,
   call it to get `(cos, sin)`, and pass `position_embeddings=(cos, sin)`.

An interactive introspection on the GPU node confirmed that in the installed
transformers version:

```
block type: LlamaDecoderLayer
block has rotary_emb: False
self_attn type: LlamaAttention
self_attn has rotary_emb: False
block rotary attrs: []
self_attn rotary attrs: []
model.model attrs with rotary: ['rotary_emb']
```

`LlamaRotaryEmbedding` was promoted to the `LlamaModel` level. Individual
decoder blocks no longer own one, so both `hasattr` checks failed and the
fallback passed `position_embeddings=None` straight into `self_attn`.

### Fix

Thread the model-level rotary embedding into the recompute path:

* `recompute_block_attention(... rotary_emb: Optional[nn.Module] = None)` —
  new optional parameter. When provided, the function pre-computes
  `position_embeddings = rotary_emb(normed, position_ids)` and forwards it
  to `_call_self_attn`.
* `_call_self_attn(... position_embeddings=None)` — new optional parameter
  short-circuits the legacy block-level / `self_attn`-level lookup when the
  caller already has `(cos, sin)`.
* All three call sites in `scripts/recompute_attention.py` (validate path,
  fused recompute loop, and self-check loop) now pass
  `rotary_emb=getattr(model.model, "rotary_emb", None)`.

The `getattr(..., None)` keeps the legacy fallback alive: on older
transformers releases where `rotary_emb` lives on the block instead of the
model, `model.model` won't have the attribute, `None` is forwarded, and
`_call_self_attn` walks the original lookup chain.

## Merge conflict during pull

The local fixes were stashed and the upstream Wave 4b rewrite of
`recompute_attention.py` was pulled in. The stash pop produced a conflict
where the stashed hunk landed in the middle of `_load_key_list` (a function
that doesn't call `recompute_block_attention` at all). The resolution was to
discard the misplaced stash fragment and re-apply the `rotary_emb=...` kwarg
to the three call sites in the new file. No semantic content was lost.

## Smoketest outcome — all 4 phases green

Job ID `e884eda1c12b` on alphagpu03-8889 (H100 80GB), default args
(`MODEL=meta-llama/Llama-3.1-8B-Instruct DATASET=hotpotqa SAMPLES=20
R_MAX=64`):

| Phase | What it checks | Result |
|------:|----------------|--------|
| 1 | `--validate-first` — 4 samples × 32 blocks, max\|A_recomp − A_full\| < 1e-3 | **PASS** (all blocks under tolerance) |
| 2 | `--max-samples 20` — fused recompute writes `attention/` store + `icr_scores.npy` | **PASS** (exit 0) |
| 3 | Attention store readback — shapes / dtype / first-row non-zero | **PASS** |
| 4 | `icr_scores.npy` readback — shape `(20, 32)`, 0/20 all-zero rows | **PASS** |

Final script status: `phase_fail=0`.

### Timing (wall clock)

* Phase 1: 43.1 s (4 samples + full forward × per-block recompute)
* Phase 2: 45.8 s (20 samples through the fused recompute + ICR loop)

GPU utilization was sparse (mostly 0% in the 5-second sampler, occasional
bursts to 2% with ~17 GB resident) — expected for the recompute path, which
is dominated by attention kernels on short response sequences, not by long
contiguous compute.

### Artifacts

* Log: `reports/smoketest_attention_recompute/smoketest_llama_3_1_8b_instruct_20260515_231027.log`
* Phase 1 / Phase 2 `/usr/bin/time -v` reports:
  `reports/smoketest_attention_recompute/time_llama_3_1_8b_instruct_20260515_231027.txt.phase{1,2}`
* GPU sampler: `reports/smoketest_attention_recompute/gpu_llama_3_1_8b_instruct_20260515_231027.log`
* Smoke output store + scores: `reports/smoketest_attention_recompute/hotpotqa_llama_3_1_8b_instruct/`

## Follow-ups

* The same smoketest should be repeated with `MODEL="Qwen/Qwen3-8B"` once
  Qwen activations are available, to confirm the `rotary_emb` lookup also
  works on `Qwen3Model`. The `getattr(model.model, "rotary_emb", None)`
  pattern is generic, but the underlying `LlamaRotaryEmbedding` vs.
  `Qwen3RotaryEmbedding` call signatures should be re-checked end-to-end.
* The Phase 1 numerical tolerance is `1e-3` per block — tight enough to
  catch the kind of subtle layout / RoPE mismatches Bug 2 produced
  (where it would have passed shape checks but produced garbage).
