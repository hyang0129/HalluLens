# Inference Capture Rewrite — single-process, memmap-native attention + activation logging

**Status:** draft / proposal
**Supersedes (partially):** Wave 4 of #69 (recompute-from-cached-states approach)
**Owner:** unassigned
**Branch:** to be created

## Motivation

Issue #69's recompute approach (cache hidden states during inference, recompute attention from them later) is conceptually clever but has been fragile in practice. Each of the following surfaced as a separate debugging session:

1. fp16 dtype mismatch in `recompute_block_attention` — fixed (commit `8e99376`).
2. `LlamaRotaryEmbedding` moved from block to model level across transformers versions — fixed (commit `8e99376`).
3. `--validate-first` returned vacuous PASS when every candidate sample had `response_len < 1` — fixed (commit `8d8a52d`).
4. `--validate-first` self-check fallback prints a numerical-equivalence PASS message even though no full-forward cross-check ran (sibling of #3, still open).
5. `shared/hotpotqa_qwen3_8b/activations.zarr` has 1472 blank leading rows — data anomaly that blocked the Qwen3 smoketest.
6. Many existing zarrs are missing `prompt_token_ids` / `response_token_ids` / `input_ids`, so the numerical-equivalence check can't run on them at all (confirmed today: `mmlu_qwen3_8b` is missing token IDs; Phase 1 of the Qwen3 smoketest could only run shape self-checks).
7. `gpu_dispatch.py run` joins argv into a remote `bash -c`, so local-shell-side `MODEL=...` env vars are silently dropped — documented (commit `50fec70`).

Each individual fix is small. The meta-problem is that recompute is fragile **by construction**: we are reconstructing attention from a snapshot taken by a different code path, under a different (and drifting) set of HF model internals, against zarr stores that were not contracted to support this use case.

A simpler architecture: capture attention as a byproduct of the same forward pass that produces hidden states, using the same `model.generate(output_attentions=True, return_dict_in_generate=True)` call the upstream ICR Probe reference uses. No version skew, no missing token-IDs, no recompute primitive to keep in sync with HF internals, no post-hoc validation step needed because the attention *is* the model's attention. And — critically for apples-to-apples comparison — the attention numbers come from the same code path the paper's authors used.

**Implication for the open `_run_validate` self-check bug (#4 above):** don't fix it. The recompute path goes away in this rewrite; spending time on the message is rearranging deck chairs. Leave the spec bug-postmortem accurate as documentation; delete the code when PR #71 is cut down.

## Scope

**In scope**
- Single-process inference script for the datasets/models the ICR probe needs.
- Captures, per sample, into `np.memmap`-backed files:
  - response activations (per layer, per response token).
  - response-to-response attention slice (`r_max × r_max`, per layer).
  - response + prompt token IDs.
  - prompt length, response length.
  - ICR scores (computed inline as sidecar).
- `meta.jsonl` as authoritative valid-rows list, written append-only after each sample's bytes are flushed.
- Resume semantics: on restart, read last `meta.jsonl` line, continue from `sample_index + 1`.

**Out of scope**
- OpenAI-compatible server / vLLM / FastAPI. Generation runs in-process with HF Transformers in `eager` attention mode.
- Re-capturing existing zarrs that aren't needed for the ICR paper experiments.
- Replacing `activation_logging/zarr_activations_logger.py` for other (non-ICR) consumers — they keep working with their existing stores.
- Logprob capture for SE / P(true) baselines (separate concern; can be added later if needed).

**Open**
- Should we also capture **prompt** activations? Existing zarrs do; for ICR we strictly only need response activations. Tradeoff: ~5× storage saved if we drop prompt activations, but loses backward compatibility for any baseline that reads them. Default proposal: yes, mirror existing zarr contract for prompt activations to avoid forcing other code to fork. Revisit if storage explodes.

## Non-goals

- **Not** a wholesale replacement of the existing inference + activation logging path. Existing benchmarks (PreciseWikiQA, LLMsKnow) that use the OpenAI-compatible server continue to work. The new path is a purpose-built capture pipeline for ICR-probe data, runnable independently.
- **Not** a guarantee that the new memmap contract matches the old zarr contract. The new layout is designed for memmap-native reads in `ICRDataset` and downstream probes; converters are out of scope.

## Architecture

Single Python process, single GPU per run. Generation and attention capture happen in one `model.generate()` call — matching the upstream ICR Probe reference implementation rather than introducing a separate teacher-forced pass.

```
load_model_eager(model_name)                # HF, attn_implementation='eager'
load_evaluator(task)                         # per-dataset is_correct; substring/regex for LLMsKnow
prepare_dataset(task)                        # existing task module's iterator
for sample in samples:                       # B=1 to start; see Risks #1 for batching
    inputs = tokenize(sample.prompt)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            ...
        )
    # out.attentions is a heterogeneous tuple — see "Upstream stitching contract".
    resp_attn      = stitch_response_to_response(out.attentions, prompt_len, r_max)
    resp_hs        = stitch_response_hidden_states(out.hidden_states, prompt_len)
    icr            = compute_icr_score(resp_attn, resp_hs, ...)
    generated_text = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
    hallucinated   = not is_correct(generated_text, sample.reference_answer)  # inline, per-dataset
    writer.append(sample, resp_hs, resp_attn, icr,
                  token_ids=out.sequences[0], hallucinated=hallucinated)
writer.finalize()   # synthesizes eval_results.json from meta.jsonl for compat
```

The key shifts from the current architecture:
- No subprocess server, no port binding, no async hooks.
- One `generate()` call yields tokens, hidden states, *and* attention in a single forward+decode pass — matches the paper's reference implementation byte-for-byte.
- Token IDs come from `out.sequences` and are written alongside everything else; no risk of going missing.
- Per-sample memmap appends are paired with `meta.jsonl` append; either both succeed or the sample doesn't count.

## Label contract — what counts as "hallucinated"

Apples-to-apples with the upstream paper applies to **feature computation** (ICR score formula, attention stitching, JSD normalization — all resolved in [`notes/icr_probe_paper_notes.md`](../notes/icr_probe_paper_notes.md)). It does **not** apply to label production:

- Upstream `src/icr_probe.py` ships only the trainer; it takes pre-built `(X, y)` data loaders. There is no upstream labeling code to match.
- The paper text (arXiv:2507.16488) references training data as `(q, a, y)` with `y ∈ {0, 1}` and treats labels as given — no criterion specified for HaluEval-QA / SQuAD / HotpotQA / TriviaQA.

The contract we must enforce is internal: **labels must be the same across every baseline on the same `(dataset, model, sample)`**. Otherwise the comparison between ICR Probe and SE / P(true) / SelfCheckGPT / SEP / the contrastive baseline isn't measuring relative method quality — it's also measuring label drift.

Concretely:

1. **Generation step** of the new capture path produces `generation.jsonl` (text + metadata + label), `response_activations.npy`, `response_attention.npy`, `response_token_ids.npy`, `icr_scores.npy`, and `meta.jsonl`. Each `meta.jsonl` line includes a `hallucinated` field (bool).
2. **Labels are computed inline** — after `model.generate()` returns for each sample, call the task module's `is_correct(generated_text, reference_answer)` and write `hallucinated = not is_correct(...)` into that sample's `meta.jsonl` line and into `generation.jsonl`. For LLMsKnow tasks this is a substring/regex check in `tasks/llmsknow/<dataset>.py`, pure CPU, negligible latency relative to generation. At the end of the run, `writer.finalize()` synthesizes `eval_results.json` from `meta.jsonl` for backward compatibility with other baseline consumers (SE, P(true), SelfCheckGPT, SEP, contrastive) that read that file today.
3. **No separate `--step eval` call** is needed for label production. The capture script's default mode produces the label as part of the generation loop. A `--step eval-only` flag is kept for re-labeling an existing `generation.jsonl` without re-running inference (in case the evaluator logic changes), but it is a convenience, not the primary path.
4. **ICR Probe training** loads labels from `meta.jsonl` (via the `hallucinated` field) or from the synthesized `eval_results.json` — same data either way. No new evaluator, no new threshold.

**What this rules out:**
- Adding a "did the answer hallucinate?" check inside the new capture loop that uses any criterion *other than* the task module's evaluator. If a task's evaluator is wrong or noisy, fix it in the task module — don't shadow it.
- Re-evaluating with a different judge model when the existing one stays put. If we change evaluators we re-evaluate every baseline, not just ICR.
- Using sample-level fields the paper might mention (e.g. some HaluEval rows have a built-in `hallucination` column) instead of our task-module's label. Datasets that ship pre-labels are fine to use, but only if the same column is used for *every* baseline on that dataset.

**What this allows:**
- A `--step eval-only` mode that re-runs the per-dataset evaluator over an existing `generation.jsonl` without re-capturing (useful if the evaluator changes and we need to re-label without re-running GPU inference). This is a convenience path.
- The evaluator for LLMsKnow tasks is cheap enough to run inline with no batching needed. If PreciseWikiQA (LLM judge) is ever added to the ICR capture scope, batch the judge calls at end-of-run rather than per-sample inline.

## Upstream stitching contract (load-bearing)

This is the contract our capture path must match for apples-to-apples comparison with the published ICR Probe baseline.

When `generate(..., output_attentions=True, return_dict_in_generate=True)` runs, `out.attentions` is a tuple of length `1 + response_len`:

| Index | Source | Shape (per layer) |
|---|---|---|
| `0` | Prefill forward pass on the prompt | `(batch, num_heads, prompt_len, prompt_len)` |
| `t ≥ 1` | Decode step that emits response token `t-1` | `(batch, num_heads, 1, prompt_len + t)` |

The prefill chunk is square (`prompt_len × prompt_len`). Each decode-step chunk is a single attention row from the newly-emitted query token to all previously-seen positions. Upstream's `_pre_process_attn` ([upstream `src/icr_score.py:53-102`](../notes/icr_probe_paper_notes.md#9-cross-region-attention-masking--only-response-to-response-attention-contributes)) zero-pads each piece to a uniform `(token_num, token_num)` shape and concatenates along the query dimension to form the unified causal matrix.

For our capture, we don't need to materialize the full `(T, T)` matrix. The ICR Probe only uses response-to-response attention (notes §9 — cross-region positions get zeroed before top-p anyway). We stream-slice as follows:

- **Skip `out.attentions[0]` entirely.** The prefill chunk is prompt-to-prompt; ICR doesn't consume it.
- For each `out.attentions[t]` (t ≥ 1), the query row is for response position `t-1`. Slice the key dimension at `[prompt_len : prompt_len + r_max]` to keep only response-to-response keys. The resulting `(num_heads, 1, r_max)` slice is the t-th row of the per-layer `(r_max, r_max)` attention block we persist.
- Stack response rows over `t = 1 .. min(response_len, r_max)`, head-average per layer (notes §4: simple mean over all heads), write to memmap.

This bounds peak GPU memory at `O(L × H × r_max)` per decode step, not `O(L × H × T²)`.

**Numerical-equivalence assertion (not vacuous, unlike `_run_validate`):** for any sample with `response_len > 0`, our stream-stitched `response_attention[s, l]` must equal upstream's `output_attentions[l, input_lens:input_lens+r, input_lens:input_lens+r].mean(head)` to within fp16 tolerance (`< 1e-3`). The integration test runs both paths on the same prompt and asserts this. If it fails, the stitching contract is broken and the numbers won't be comparable to the paper.

## Data layout

Per dataset × model:

```
shared/icr_capture/{dataset}_{model_slug}/
  config.json                  # model_name, num_layers, hidden_dim, r_max, dtype, ...
  meta.jsonl                   # one line per fully-written sample (authoritative)
  response_activations.npy     # memmap (N, num_layers+1, max_response_len, hidden_dim) fp16
  response_attention.npy       # memmap (N, num_layers, r_max, r_max) fp16
  prompt_activations.npy       # memmap (N, num_layers+1, max_prompt_len, hidden_dim) fp16   [optional]
  prompt_token_ids.npy         # memmap (N, max_prompt_len) int32  (-1 padded)
  response_token_ids.npy       # memmap (N, max_response_len) int32  (-1 padded)
  prompt_len.npy               # memmap (N,) int32
  response_len.npy             # memmap (N,) int32
  icr_scores.npy               # (N, num_layers) fp32  — full numpy save, no memmap needed (~1.4 MB / 10k samples)
  generation.jsonl             # (carried over) prompt + generated text + sampling metadata
```

`meta.jsonl` line format:
```json
{"sample_index": 42, "key": "mmlu_42", "prompt_len": 137, "response_len": 28, "hallucinated": true, "wrote_at": "2026-05-16T..."}
```

Resume on restart: open `meta.jsonl`, take `max(sample_index)`, restart from `+1`. If a memmap row was partially written before crash, it gets overwritten with no harm — we never trust a row without a meta line.

## Storage budget (rough)

Per sample, Qwen3-8B (36 layers, hidden_dim=4096, max_response_len=256, r_max=64, fp16):
- response_activations: 37 × 256 × 4096 × 2 ≈ 76 MB
- response_attention:   36 × 64 × 64 × 2  ≈ 0.3 MB
- prompt_activations:   37 × max_prompt_len × 4096 × 2 ≈ similar order to response_activations
- token IDs, lengths, ICR scores: < 1 KB

→ ~150 MB/sample dominated by activations. 10k-sample dataset ≈ 1.5 TB.

This is the same order of magnitude as existing zarr stores. **The memmap rewrite is not a storage win** — it's a code-quality and correctness win. If we want to shrink storage, the right knobs are:
- Truncate `max_response_len` (need to audit how long responses actually are; 256 is generous).
- Save activations at a subset of layers (e.g. the layers probes actually train on, per Wave 4's `--train-layers 14-29`).
- Drop `prompt_activations` if no consumer needs them.

These are storage-side decisions, separable from the memmap rewrite. Default v1: keep behavior parity with existing zarr (all layers, full response window up to `max_response_len`).

## Datasets + models to re-capture (v1)

Subset needed for the EMNLP probe-paper experiments. Likely:

| Dataset    | Llama-3.1-8B | Qwen3-8B |
|------------|:------------:|:--------:|
| MMLU       | re-capture   | re-capture |
| HotpotQA   | re-capture   | re-capture (existing zarr has 1472 blank leading rows) |
| PopQA      | re-capture   | re-capture |
| Natural Questions | re-capture | re-capture |
| SciQ       | re-capture   | re-capture |
| SearchQA   | re-capture   | re-capture |

Plus train splits for the datasets the probe is trained on (per `configs/experiments/baseline_comparison_*.json`).

At ~16 samp/s steady-state, MMLU test (10k) ≈ 10 min/model; MMLU train (100k) ≈ 100 min/model. Aggregate: order of one to two days of H100 wall time across the layout. Bounded but not free.

**Open question:** do we need to re-capture *test* splits, or only the splits that the recompute-based zarrs currently lack token IDs / have anomalies on? Worth running `audit_zarr_prefix.py` extended to also report token-ID-array presence before committing to the full re-capture.

## Risks

1. **Peak GPU memory during `generate()`.** With `output_attentions=True`, each decode step emits one attention row per layer of shape `(num_heads, 1, prompt_len + t)`. Aggregate intermediate state across L layers, H heads, and response_len decode steps is bounded — `O(L × H × T)` per step, not `O(L × H × T²)` — but HF still holds the prefill chunk `(L, H, P, P)` for the duration of the call. For Qwen3-8B with `prompt_len=400`: 36 × 32 × 400² × 4 bytes ≈ 740 MB just for the prefill attention. Manageable at B=1, tight at B=4+. **Mitigation:** start at B=1 and tune up only if a node has headroom. Stream-slicing (see Upstream stitching contract) discards each decode step's attention as soon as we extract the response-to-response slice, so it doesn't accumulate.

2. **Stitching contract divergence from upstream.** The bytes we persist as `response_attention[s, l]` must match upstream's processed attention to fp16 tolerance — otherwise ICR scores aren't apples-to-apples. The contract has at least three places to get wrong: (a) the index offset between `out.attentions[t]` and response token `t-1`, (b) whether we slice key positions at `[prompt_len : prompt_len + r_max]` or `[input_lens : input_lens + r_max]` (these are the same iff our tokenizer matches upstream's BOS/EOS handling), (c) head aggregation order (mean of per-head slices vs. slice of head-mean — equivalent mathematically but only if dtype handling is identical). **Mitigation:** the numerical-equivalence assertion in the Upstream stitching contract section above is the canary; it must run as part of CI / smoketest before any re-capture run starts.

3. **Memmap corruption blast radius.** Per Q4 above: one corrupt memmap = lose that dataset run. Mitigation is `meta.jsonl` + resume; we accept the risk in v1 and add sharding later if needed.

4. **Storage we'd commit to.** Re-capturing the full layout at present granularity is ~10-15 TB. Worth confirming we have the disk for it before kicking off, and that we're OK with the existing zarrs becoming dead weight (or scheduled for deletion).

5. **Throwing away Waves 1–4 of #69.** The recompute primitive, the AttentionMemmapWriter, the fused recompute loop, the smoketest harness — Wave 4 in particular is fresh work. Some pieces port forward: `compute_icr_score()`, `ICRDataset`, the memmap reader, the smoketest's Phase 3/4 readback structure. The recompute primitive itself (`recompute_block_attention`, `_call_self_attn`, the rotary_emb threading) goes away. PR #71's scope shrinks dramatically; #69 may be closed in favor of this issue.

## Concrete plan (sketch)

1. **Audit token-ID presence** across existing zarrs (extend `scripts/audit_zarr_prefix.py` to also list which arrays exist per store). Identifies which datasets currently have token IDs (useful as a numerical-equivalence reference against the new path) and which would have needed re-capture anyway.
2. **Port the upstream stitching primitive.** Implement `stitch_response_to_response(out.attentions, prompt_len, r_max)` in a new module `activation_logging/generate_capture.py`. The implementation mirrors upstream `_pre_process_attn` lines 53–102 but stream-slices to response-to-response keys instead of materializing the full `(T, T)` matrix. Unit-test it on a tiny model (e.g. `sshleifer/tiny-gpt2`) against a hand-stitched reference.
3. **Prototype the capture script** on one (dataset × model) pair — `sciq_qwen3_8b` (1000 samples, smallest, fast iteration). End-to-end: load model, run generation with `output_attentions=True`, stream-stitch, write memmap + meta, compute ICR inline, read back. No probe training yet.
4. **Build the `MemmapInferenceCapture` writer** as the abstraction (mirrors `AttentionMemmapWriter` from Wave 4 but with two memmap streams + token IDs + ICR sidecar).
5. **Numerical-equivalence gate.** Run the new capture path on one sample, then run upstream's reference `_pre_process_attn` (downloaded ad-hoc from the upstream repo, or via a vendored copy) on the same `model.generate()` output, and assert `max|ours - upstream| < 1e-3` on the response-to-response sub-block. This is the canary that protects the apples-to-apples comparison. **Must pass before any full re-capture run starts.**
6. **Port `compute_icr_score` and `ICRDataset`** from Wave 4 to read the new layout. Both should be small diffs — the readers don't care that the writer used a hook vs. a stitch, only that the bytes are in the right shape.
7. **Re-capture the layout** dataset by dataset (parallelizable across nodes via `gpu_dispatch.py`). Start with test splits; train splits in parallel as nodes free up.
8. **Cut PR #71 down** to whatever still applies, mark the rest superseded.

## Decision needed before starting

- Disk budget — do we have ~15 TB to commit to the re-capture, or do we need to truncate first?
- Lifetime of existing zarrs — schedule them for deletion after the new capture lands, or keep them around as historical?
- Whether to vendor the upstream `_pre_process_attn` (so the numerical-equivalence gate has an in-tree reference) or fetch it ad-hoc each time the gate runs.
