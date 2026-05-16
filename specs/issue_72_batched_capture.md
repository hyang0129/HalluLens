# Batched capture (B>1) for #72

**Status:** proposal
**Branch target:** `feat/issue-72-inference-capture-rewrite` (or a child branch)
**Motivation:** B=1 with eager attention + per-layer attention dumping is GPU-bound — H100 sits at ~10-30% utilization at B=1. Batching to B=4 yields ~3× wall-time reduction (not 4× because variable-length decoding wastes GPU on early-EOS samples in the batch). For sciq smoketest scale this isn't worth it; for full Phase 1 of #70 (HotpotQA train 80k × 2 models) it cuts ~3 days off the grid.

This spec is *separate from* the current capture path which lands in PR #73. Implement on top of #73 after the merge.

## Goal

End-to-end capture path that runs `model.generate(input_ids=[B, max_prompt_len], ...)` with `B > 1` and produces byte-identical memmap output to the B=1 path, modulo numerical fp16 drift from padding-token attention bleeding into adjacent samples (which we mask before stitching anyway).

## Contracts that DO NOT change

| | |
|---|---|
| `InferenceCaptureWriter` API | Per-sample. Orchestrator calls `writer.append(...)` B times after each batched forward. **Zero changes.** |
| Cell JSON / manifest format | Adds an optional `batch_size` field with default 1 — see §4. |
| `ICRDataset(mode="memmap")` | Memmap layout is unchanged; only orchestrator runs batched. |
| `activation_research/icr_score.py::compute_icr_score` | Per-sample per-layer scalar. Orchestrator loops B × L calls. **Zero changes.** |
| Apples-to-apples comparison with upstream | The equivalence gate runs each batch sample as B=1 against upstream's `_pre_process_attn` and asserts byte-identical output. |
| Capture file format | Memmap layout in `specs/issue_72_inference_capture_rewrite.md` "Data layout". Unchanged. |
| Resume semantics | Prompt-hash keying in `meta.jsonl`. A batch may include some already-written samples; orchestrator filters those out before calling generate. |

## Contracts that DO change

### 1. `activation_logging/generate_capture.py` stitching primitives

Each function gains a batched form. Current signatures squeeze the B=1 dim; new signatures keep it and handle per-sample `response_len`.

#### `stitch_response_to_response`

**Today:**
```python
def stitch_response_to_response(
    attentions,           # tuple of length 1 + response_len
    prompt_len: int,
    r_max: int,
    response_len: int | None = None,
) -> np.ndarray:         # (num_layers, r_max, r_max) fp16
```

**Batched:**
```python
def stitch_response_to_response_batched(
    attentions,           # tuple of length 1 + max_response_len
    prompt_lens: np.ndarray,    # (B,) int — different prompts may have different padded lengths in attention_mask
    response_lens: np.ndarray,  # (B,) int — different samples EOS at different steps
    r_max: int,
) -> np.ndarray:         # (B, num_layers, r_max, r_max) fp16
```

For each `t >= 1` in `attentions`:
- Layer-attention shape is `(B, num_heads, 1, prompt_pad_len + t)` (HF pads attention to the longest sequence in the batch up to step t)
- For each sample `b`:
  - Skip if `t > response_lens[b]` (this sample already EOS'd; padding step)
  - Slice key dimension at `[prompt_lens[b] : prompt_lens[b] + r_max]`
  - Head-average → place into `out[b, layer, t-1, :]`

**Subtle:** when sample `b` has EOS'd early at step `t_b`, HF still emits attention rows for it at steps `t > t_b`, but they're for `pad_token_id`. We MUST skip these — otherwise we contaminate the response-to-response sub-block with pad-token attention. The skip is `t > response_lens[b]` check.

#### `stitch_response_hidden_states`

Analogous. Skip pad steps per sample. Returns `(B, num_layers + 1, max_response_len, hidden_dim)` fp16.

#### `stitch_prompt_hidden_states`

Easier — only reads `hidden_states[0]` (prefill). Per-sample slice is `[:, :prompt_lens[b]]`. Returns `(B, num_layers + 1, max_prompt_len, hidden_dim)` fp16.

#### `extract_logprobs`

Same skip-after-EOS logic. Returns three `(B, max_response_len, ...)` arrays.

### 2. `scripts/capture_inference.py` orchestrator

#### `_run_capture` outer loop

Today: iterates `for i, sample in enumerate(dataset)`, builds one prompt, runs generate, writes one row.

Batched:
```python
batch_size = args.batch_size  # default 1
pending_samples = []  # collected, then flushed when len == batch_size

for i, sample in enumerate(dataset):
    prompt = build_prompt(sample, task_module)
    prompt_hash = sha256(prompt)
    if writer.is_written(prompt_hash):
        continue
    pending_samples.append((i, sample, prompt, prompt_hash))
    if len(pending_samples) == batch_size:
        _run_batch(pending_samples, ...)
        pending_samples = []

if pending_samples:  # tail batch < B
    _run_batch(pending_samples, ...)
```

#### `_run_batch` — the new function

1. **Tokenize with padding (left side, for causal LMs):**
   ```python
   tokenizer.padding_side = "left"
   batch = tokenizer([s.prompt for s in pending_samples],
                     padding=True, truncation=True,
                     max_length=args.max_prompt_len,
                     return_tensors="pt").to(model.device)
   ```
   `batch.input_ids: (B, padded_prompt_len)`, `batch.attention_mask: (B, padded_prompt_len)`.

2. **Per-sample real prompt length:**
   ```python
   prompt_lens = batch.attention_mask.sum(dim=1).cpu().numpy()  # (B,)
   ```

3. **`model.generate`:**
   ```python
   out = model.generate(
       batch.input_ids,
       attention_mask=batch.attention_mask,
       max_new_tokens=args.max_response_len,
       do_sample=False,
       output_attentions=True,
       output_hidden_states=True,
       output_scores=True,
       return_dict_in_generate=True,
       pad_token_id=tokenizer.eos_token_id,
   )
   ```
   `out.sequences: (B, padded_prompt_len + actual_response_len)` where `actual_response_len <= max_response_len` and may be shorter if all samples EOS'd.

4. **Per-sample real response length:**
   ```python
   response_lens = np.empty(B, dtype=np.int32)
   for b in range(B):
       resp_b = out.sequences[b, padded_prompt_len:]
       # Find first EOS after the prompt; if no EOS, response_len == max_new_tokens
       eos_positions = (resp_b == tokenizer.eos_token_id).nonzero()
       if len(eos_positions) > 0:
           response_lens[b] = eos_positions[0].item() + 1  # include the EOS token
       else:
           response_lens[b] = resp_b.shape[0]
   ```
   (Or use HF's `out.sequences[b].argmax(== pad_token)` if pad_token != eos_token; with `pad_token_id=eos_token_id` we have to track first-EOS-after-prompt.)

5. **Stitch:**
   ```python
   resp_attn = stitch_response_to_response_batched(out.attentions, prompt_lens, response_lens, args.r_max)
   resp_hs   = stitch_response_hidden_states_batched(out.hidden_states, prompt_lens, response_lens, args.max_response_len)
   prompt_hs = stitch_prompt_hidden_states_batched(out.hidden_states, prompt_lens, args.max_prompt_len)
   token_lp, topk_ids, topk_lp = extract_logprobs_batched(out.scores, out.sequences, prompt_lens, response_lens, args.top_k)
   ```

6. **Per-sample loop for ICR + writer.append:**
   ```python
   for b in range(B):
       resp_ids = out.sequences[b, padded_prompt_len : padded_prompt_len + response_lens[b]]
       decoded = tokenizer.decode(resp_ids, skip_special_tokens=True)
       hallucinated = not task_adapter.is_correct(decoded, pending_samples[b].sample)
       icr = compute_icr_per_layer(resp_attn[b], resp_hs[b], response_lens[b])
       writer.append(sample_index=..., prompt_hash=..., ..., icr_score_per_layer=icr, ...)
   ```

#### Padding-side gotcha

Left-padding is mandatory for decoder-only causal LMs with `model.generate()`. If you use right-padding, the model generates from the pad tokens and produces garbage. Verify `tokenizer.padding_side == "left"` and `tokenizer.pad_token` is set (often `pad_token = eos_token` for Llama; check Qwen3 specifically since it has a separate pad).

### 3. Cell JSON adds `batch_size`

```json
{
  ...
  "batch_size": 4
}
```

Default in `generate_manifest.py`: `batch_size=1` (no change to existing manifests). Phase 1 manifests pass `--batch-size 4`.

## Equivalence gate at B>1

Critical: byte-identical to B=1 within fp16 tolerance.

`tests/test_capture_equivalence_batched.py`:
1. Build a tiny-gpt2 model fixture.
2. Pick 4 short prompts of varying length.
3. Run capture path with B=4. Save outputs.
4. Run capture path with B=1 four times, one per prompt. Save outputs.
5. Assert: `max|batched_resp_attn[b] - unbatched_resp_attn[b]| < 1e-3` for each `b`.
6. Same for hidden states and logprobs.

This is a stronger gate than the existing `test_capture_equivalence.py` (which checks against upstream's `_pre_process_attn`). It catches batching-specific bugs (padding, per-sample slicing, EOS handling).

The existing `test_capture_equivalence.py` continues to run unchanged at B=1.

## Edge cases

| Case | Handling |
|---|---|
| Tail batch (< B samples) | Flush at end of loop. Generate handles short batch transparently. |
| All B samples already in `meta.jsonl` | Skip batch entirely; never call generate. |
| Some samples in batch already written, others not | Filter to only-new before tokenization. Effective batch size temporarily drops. |
| Sample with `response_len == 0` (immediate EOS) | response_lens[b] = 0; stitch returns zero rows; writer writes zero rows; meta.jsonl `hallucinated = not is_correct("", answer)` (usually True). Same as B=1. |
| OOM at B=4 | Document recommended B per (model, max_response_len, max_prompt_len). Llama-3.1-8B at max_prompt=512, max_response=64: B=4 fits in 80GB H100. Qwen3-8B at same: B=4 likely fits, B=8 borderline. |
| Padding token == EOS token (Llama default) | First-EOS-after-prompt detection accounts for this. Be careful with `attention_mask.sum()` since `(pad_token_id == eos_token_id)` doesn't propagate naturally. |

## Throughput expectations (Llama-3.1-8B on H100)

| B | max_response_len | Sec/sample | Wall for sciq_train (12k) |
|---|---|---|---|
| 1 | 256 | 6.7 | 22h |
| 1 | 64 | 1.7 | 5.5h |
| 4 | 64 | 0.6 | 2h |
| 8 | 64 | 0.4 (OOM risk) | 1.3h |

Phase 1 HotpotQA train (~80k samples) at B=4, max_response_len=64: **~13h per model**, parallelizable to ~4h with 3 workers.

## Files to create / modify

| File | Action | LOC |
|---|---|---|
| `activation_logging/generate_capture.py` | Add `*_batched` variants alongside existing | +200 |
| `scripts/capture_inference.py` | Add `_run_batch`; route via `args.batch_size` | +80 |
| `scripts/dispatch/generate_manifest.py` | Add `--batch-size` CLI flag; threadable through cell JSON | +5 |
| `scripts/dispatch/worker.sh` | Forward `batch_size` from cell JSON to `--batch-size` arg | +3 |
| `tests/test_generate_capture_batched.py` | Stitch primitives at B>1 against B=1 reference | +200 |
| `tests/test_capture_equivalence_batched.py` | End-to-end batched vs. unbatched | +100 |

Total ~600 LOC; about 1.5 days for a sonnet (or a focused day for opus).

## Risks

| Risk | Mitigation |
|---|---|
| Pad-token attention bleeds into stitched response sub-block, contaminating ICR scores | Mandatory `response_lens[b]` check in every stitching loop; gate test asserts numerical equivalence with B=1 path |
| Left-padding bug (model treats pad as part of context) | HF's `model.generate` with `attention_mask` handles this; verify with a test that left-padded and right-padded same prompt yield the same decoded response |
| Variable-length wasted GPU | Accepted; we get ~3× not 4× speedup |
| OOM at larger B with longer max_response_len | Per-model B recommendation table in CLI help; OOM is fail-fast (no silent corruption) |
| Qwen3 tokenizer has different pad token semantics | Verify in a unit test; document Qwen3-specific config in capture_inference.py |
| The numerical-equivalence gate at B>1 doesn't hold within fp16 tolerance | Diagnose: is it the prompt-padding side, the EOS detection, the per-sample stitching? Should be a 1-day debug, not a redesign |

## Out of scope

- Dynamic batching (filling the next batch with samples claimed mid-flight). Static batches are simpler and ~95% of the win.
- Different B per cell. One B per worker (CLI arg).
- Mixing batch-and-non-batch in the same worker. Worker picks B from cell JSON or CLI arg, sticks to it for the cell.
- vLLM, flash attention, SDPA — these all break the equivalence gate.

## Smoketest before scaling

Before launching Phase 1 with B=4:
1. Run `tests/test_capture_equivalence_batched.py` — must pass.
2. Run one HotpotQA validation cell (7400 samples) at B=4 on a single worker. Expected wall: ~70 min. Compare ICR score distribution mean/std to the B=1 SciQ smoketest baseline: should be similar (within ~10%).
3. If both pass, B=4 is safe for the full grid.

If either fails: drop back to B=1 for Phase 1. The capture path is the bottleneck for Phase 1 anyway; B=1 + 3 parallel workers ships in ~36h instead of ~13h, still tractable for an EMNLP submission timeline.
