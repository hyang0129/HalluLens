# Implementation Spec: ICR Probe — Attention-Map Generation & Storage Infrastructure (Issue #69)

**Goal:** Produce the data substrate needed to evaluate ICR Probe (Zhang et al., ACL 2025; [arXiv:2507.16488](https://arxiv.org/abs/2507.16488)) on our existing inference runs, **without re-running full inference**. Specifically: for every sample already stored in `shared/*_qwen3_8b/activations.zarr` and `shared/*_llama-3.1-8b-instruct/activations.zarr` (or however they are slugged on disk), generate, store, and expose for loading the per-layer attention information required by the ICR Score, by recomputing attention sublayers from the cached hidden states that are already on disk.

This issue covers **data infrastructure only**. The probe model, training loop, and evaluation live in Issue #70 and depend on this.

The ICR Probe baseline was added to the must-add list of [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §11.3 on 2026-05-15. Reviewers at EMNLP are likely to ask why this baseline was omitted; this issue is the prerequisite for being able to answer "we ran it and here are the numbers."

---

## 1. Scope

### In scope
- A standalone CPU/GPU script that, for each entry in an existing activations zarr store, loads its cached per-layer hidden states and recomputes that sample's per-layer attention probabilities for a defined set of query positions, then writes them to a new zarr layout.
- A numerical-equivalence validation step that proves the recomputed attention matches a full-model forward pass to defined tolerance, on a held-out batch, **before** running at scale.
- A reader API analogous to `ActivationParser` that exposes attention probs alongside hidden states for the same sample.
- Resume semantics consistent with existing zarr loggers.

### Out of scope (deferred to Issue #70)
- The ICR Score computation itself (JSD between projection and attention distributions).
- The probe MLP and its training/eval.

### Phase plan
- **Phase 1 (this issue, mandatory):** HotpotQA only, both models (Llama-3.1-8B-Instruct, Qwen3-8B), test + train splits.
- **Phase 2 (this issue, optional, only after Phase 1 numbers from Issue #70 reproduce the ICR Probe paper's trend):** the remaining 5 datasets (NQ, MMLU, PopQA, SciQ, SearchQA), both models, both splits.

Per `PAPER_ROADMAP.md` §11.3.4: never cut Phase 1.

---

## 2. Paper / code reading — RESOLVED (2026-05-15)

Resolved by direct reading of the released code at [github.com/XavierZhang2002/ICR_Probe](https://github.com/XavierZhang2002/ICR_Probe). Full answers with file:line citations live in [`notes/icr_probe_paper_notes.md`](../notes/icr_probe_paper_notes.md). Headline answers and the changes they force on this spec:

| Question | Resolved answer | Spec section affected |
|---|---|---|
| 1. Query positions | All response tokens; per-token ICR then averaged over tokens to a single L-vector probe input | §4.1, §5.2, §6.1 |
| 2. Module granularity | One ICR per (token, layer); not separated by attn / MLP | §5.2 (drop `mlp_updates`) |
| 3. Top-k semantics | Top-k of head-averaged attention at the query token; **`top_p=0.1` overrides `top_k=20` in the released config**, so effective k = `floor(0.1 × len(attn_row))` and varies with sequence length | §5.2 (don't lock k at storage time), §10 (#70) |
| 4. Head aggregation | Arithmetic mean over heads. The `use_induction_head=True` default is parametrically a no-op given the README's loose `skew_threshold=0, entropy_threshold=1e5`. README explicitly: *"not used in the final version"* | §5.2 |
| 5. Projection direction | **Residual-stream**, not vocab unembedding. Dot product of `Δh_ℓ` against `h^{ℓ-1}` at top-k context positions, normalized by `‖h_j^{ℓ-1}‖`. **No `W_U` needed.** Projection target is the **previous** layer's hidden state, not the post-block one | §4.2, §4.3, §5.2 |
| 6. Layer indexing | Code's `layer ℓ ∈ [0, L)` maps to HF transformer block `ℓ`. Block 0's score uses HF `hidden_states[0]` (embedding output). If our cached zarr lacks the embedding layer, either re-derive it (cheap) or drop block 0 from the probe | §3 item 3, §4.3 |
| 7. Probe input | L-dim vector per sample; MLP `(L, 128, 64, 32, 1)` <16K params (Issue #70 scope) | §11 (out of scope) |
| 8. Sequence handling | Code is agnostic to greedy vs. sampled generation; our greedy zarrs are fine | — |

Two findings were not on the original §2 list but materially change storage:

| Finding | Source | Spec section affected |
|---|---|---|
| Cross-region attention is **zeroed before top-k** — only response-to-response and prompt-to-prompt attention is kept; we only need response-to-response for the probe path | `icr_score.py:104-127` | §5.1 (storage shape collapses from `(L, R, T)` to `(L, R, R)`) |
| `js_divergence` standardizes inputs (z-score → softmax → JSD) on the k-dim top-k subset; not the paper's clean Eq. 7 formulation | `icr_score.py:258-267` | §10 (Issue #70 must replicate exactly) |

---

## 3. Preconditions — verified against on-disk zarrs on 2026-05-15

Verified by `tasks/llmsknow/*.py` code inspection AND a live scan of `/mnt/home/hyang1/LLM_research/shared/hotpotqa_*` zarrs over the Empire AI SSH config in `.ssh/config` (host `empire-ai`):

1. **`sequence_mode="all"` was used to produce all existing zarr stores.** Verified via `tasks/llmsknow/{hotpotqa,natural_questions,mmlu,popqa,sciq,searchqa,movies}.py` (all pass `sequence_mode="all"` explicitly to `HFTransformersAdapter`), `scripts/natural_questions.py:397`, and `utils/lm.py:192` (server path hardcodes `ACTIVATION_SEQUENCE_MODE=all`).
2. **Actual cached shapes (HotpotQA test split):**
   - Llama-3.1-8B-Instruct: `prompt_activations.shape = (7405, 33, 64, 4096)`, `response_activations.shape = (7405, 33, 64, 4096)`. **L = 33, P_max = 64, R_max = 64.** Llama-3.1-8B has 32 transformer blocks → `L = 33 = num_hidden_layers + 1`, so **HF embedding output is cached at index 0**.
   - Qwen3-8B: `(8877, 37, 64, 4096)` for both prompt and response. Qwen3-8B has 36 transformer blocks → `L = 37 = num_hidden_layers + 1`, **embedding cached at index 0**.
3. **Layer-index alignment is settled (per item 2):** `arrays/*_activations[s, ℓ, :, :]` for `ℓ = 0` is the embedding output, and `ℓ ∈ [1, L)` are transformer-block outputs. To score block `b ∈ [0, num_blocks)` per the ICR code, we use `arrays/*_activations[s, b, :, :]` as `h^{ℓ-1}` (block input) and `arrays/*_activations[s, b+1, :, :]` as `h^ℓ` (block output). **Block 0 is scoreable directly from cache; no embedding rederivation needed.**
4. **Per-sample lengths (HotpotQA test, 7405 written rows each):**
   - **Llama:** `prompt_len mean=32.4, max=64, 12/7405 (0.16%) at P_max`. `response_len uniformly 63` for every sample (max generation budget hit).
   - **Qwen3:** `prompt_len mean=31.9, max=81, 11/7405 at P_max=64` (a few prompts truncated). `response_len mean=134, max=255, median=63` — **half of Qwen3 samples generate beyond the cached R_max=64**. Only the first 64 response tokens have cached hidden states / are recomputable.
5. **Variable-length samples are zero-padded to `P_max=64` / `R_max=64`.** Implementer must mask out padding using `prompt_len` and `response_len` when feeding hidden states into the attention recomputation.

**Implication of finding 4 (Qwen3 response truncation):** the ICR Probe re-implementation on Qwen3 evaluates the score on the **first 64 response tokens only** for any sample whose true response is longer. This is a real bias in the reproduction. Mitigations (decision deferred to PR description):
- (a) Accept and document — token-wise mean of ICR scores over the truncated prefix is still a valid feature; the probe sees a slightly noisier signal for long-response samples. Median Qwen3 response_len is 63, so the median sample is unaffected.
- (b) Re-run Qwen3 HotpotQA inference at `response_max_tokens=256` to capture full responses. ~24 GPU-hours per `PAPER_ROADMAP.md` §11.3.3 — costly but addresses the bias cleanly.

The PR must report the bias and Issue #70's evaluation must include a sensitivity check (e.g. ICR Probe AUROC restricted to samples where `response_len ≤ 64`).

---

## 4. Generation: recomputing attention from cached hidden states

### 4.1 What we need to compute, per sample

Per the resolved answers in §2 (cross-region masking + all-response-token query positions):

For each sample `s`, each transformer block `ℓ ∈ [0, L)`, and each response query position `q ∈ [0, response_len_s)`:
- A head-averaged attention vector `A_{s,ℓ,q} ∈ ℝ^{R_max}` over **response key positions only** (key positions `[response_start, response_start + R_max)`, equivalently the last `R_max` columns of the model's per-block attention matrix). Cross-attention to prompt positions is unused (zeroed by `set_other_attn_scores_to_zero` in `icr_score.py:104-127` before top-k).

Storage shape per sample: `(L, R_max, R_max)`. Mask by `response_len_s` on read.

> Note on the `prompt_to_prompt` region: ICR Probe's released code masks-keep prompt-to-prompt attention as well (for an empirical-study branch that scores prompt tokens). We do not need this for the probe path; only response-to-response is required.

### 4.2 The recomputation, mathematically

For a standard Llama / Qwen decoder block, the attention sublayer at layer ℓ takes input `h_{ℓ-1}` (the residual stream entering this block) and produces attention probabilities as:

```
x        = RMSNorm_ℓ_attn(h_{ℓ-1})                     # pre-attention norm
Q        = (x · W_Q_ℓ).reshape(..., n_heads, head_dim)
K        = (x · W_K_ℓ).reshape(..., n_heads, head_dim)
Q, K     = apply_RoPE(Q, K, positions)
A_raw    = Q · K^T / sqrt(head_dim)                    # (n_heads, T, T)
A_raw    = A_raw + causal_mask(T)
A        = softmax(A_raw, dim=-1)                      # (n_heads, T, T)
A_avg    = A.mean(dim=0)                               # (T, T)  head-averaged
```

The implementer must produce `A` (or `A_avg`, depending on §2 answer 4) and slice the rows at `q ∈ Q_s`.

### 4.3 Sourcing `h_{ℓ-1}` from cached zarr

For each sample, the cached `h_{ℓ-1}` over the full sequence is reconstructed as:

```
prompt_h    = arrays/prompt_activations  [s, ℓ-1, :prompt_len_s, :]      # (P_s, H)
response_h  = arrays/response_activations[s, ℓ-1, :response_len_s, :]    # (R_s, H)
h           = concat([prompt_h, response_h], dim=0)                       # (T_s, H)  T_s = P_s + R_s
```

**Layer-index alignment** (per §3 item 3, cross-referenced with `notes/icr_probe_paper_notes.md` §6): the ICR Score for transformer block `ℓ ∈ [0, L)` consumes the *block-input* hidden state at HF index `ℓ` (which equals the embedding output for `ℓ=0`). Two possibilities for our cache:

- **`prompt_activations.shape[1] == num_hidden_layers + 1`** (HF `hidden_states[0..L]` cached): block-`ℓ` input is `arrays/*_activations[s, ℓ, :, :]` directly. Best case, no extra work for block 0.
- **`prompt_activations.shape[1] == num_hidden_layers`** (HF `hidden_states[1..L]` cached, no embedding): the embedding output for block 0 is **missing**. Required mitigation: re-derive from `arrays/response_token_ids` + retokenized prompt, run only the model's `embed_tokens` layer (one matmul per sample, trivial). Alternative: drop block 0 from the ICR Probe input entirely (smaller `L`, small documented accuracy hit).

Empirical check pending — see §10 step 3.

**Projection target — non-obvious**: per `icr_score.py:233`, the *projection target* `current_layer_all_hs = origin_hidden_states[layer]` is the **previous** layer's hidden state (HF `ℓ`), not the post-block one. Issue #70's score-compute step pairs `Δh_ℓ = h^ℓ - h^{ℓ-1}` (block `ℓ`'s residual update) with projection onto `h^{ℓ-1}` at top-k positions. This is captured in `AttentionParser.get_paired` so the alignment lives in one place (§6.1).

### 4.4 Sequence-length sufficiency check

`prompt_max_tokens=512`, `response_max_tokens=64`. Run a full scan of `arrays/prompt_len` and `arrays/response_len` for both Llama and Qwen3 HotpotQA stores. Report:
- Histogram of `prompt_len` (bins around 512) and `response_len` (bins 0–64).
- Truncation rate at `prompt_len == 512` and `response_len == 64`.
- Mean `response_len` (for storage compression sizing).

If >2% of samples are truncated at `R_max=64` for any (model, dataset), flag in the PR and decide whether ICR Probe is being evaluated on a biased subsample (truncated responses lose late-token attention rows). Prompt-side truncation is less concerning since cross-region attention is unused (§4.1), but should still be reported.

### 4.5 Numerical-equivalence validation (mandatory gate before running at scale)

Before running over any full zarr store, the implementer must demonstrate that the recomputed attention matches a fresh full-model forward pass to a defined tolerance. This is a 1-day task that protects every downstream number.

Procedure:
1. Pick 4 samples (2 hallucinated + 2 non-hallucinated) from `shared/hotpotqa_llama_3_1_8b_instruct/activations.zarr`.
2. For each sample, re-tokenize the original prompt + response (recover the exact token sequence from stored token IDs in `arrays/response_token_ids` plus a separately retokenized prompt).
3. Run the full model with `output_attentions=True, output_hidden_states=True` on this sequence. Capture per-block attention `A^full_ℓ` (shape `(n_heads, T, T)`), then **slice and head-average to get the response-to-response sub-block** `A^full_resp_ℓ ∈ ℝ^{R × R}`.
4. Independently, load `h^{ℓ-1}` from the zarr store, run only the attention sublayer of block ℓ (with the same model weights loaded), and capture `A^recomp_ℓ`. Slice and head-average to the same response-to-response sub-block `A^recomp_resp_ℓ ∈ ℝ^{R × R}`.
5. For each block ℓ: assert `max |A^full_resp_ℓ - A^recomp_resp_ℓ| < 1e-3` (fp16) or `< 1e-5` (fp32 computation).
6. Additionally assert `argmax_k A^full_resp_ℓ[q, k]` matches `argmax_k A^recomp_resp_ℓ[q, k]` for every response query position `q`.

If the tolerance is exceeded, diagnose before proceeding. Common failure modes to watch for:
- RoPE position offset (off-by-one from prompt left-padding handling in `_extract_activations`).
- RMSNorm vs LayerNorm confusion across model families (Llama-3.1 uses RMSNorm; verify Qwen3).
- Pre- vs post-norm interpretation of cached `h`.
- fp16 vs bf16 numerical precision in stored vs recomputed.

Record validation results in `notes/icr_probe_validation.md` with concrete numbers.

### 4.6 Path A fallback (re-run full inference with attention logged)

Only if §4.5 fails irrecoverably: re-run inference on HotpotQA test + train, both models, with `output_attentions=True` and direct attention logging. This is ~24 GPU-hours total per `PAPER_ROADMAP.md` §11.3.3. The implementer must NOT pre-emptively choose Path A; it is reserved for the case where Path B numerical equivalence cannot be achieved.

---

## 5. Storage

### 5.1 Design

Attention data lives in a **new** zarr store next to the existing activations store, not inside it. Rationale: the existing store has its own pre-allocation schema, and writing into it requires its mode='a' write semantics, which we want to avoid touching to keep the existing artifacts read-only-safe.

Path convention:
```
shared/<dataset>_<model_slug>/
    activations.zarr/         (existing — DO NOT modify)
    attention.zarr/           (new, this issue)
```

### 5.2 Zarr layout for `attention.zarr`

```
attention.zarr/
  arrays/
    response_attn     shape (N, L, R_max, R_max)  dtype float16
                      # Per-sample per-block head-averaged response-to-response
                      # attention sub-block. Mask by response_len on read.
                      # Cross-region attention is unused (zeroed in the score
                      # path per icr_score.py:104-127).
    sample_key        shape (N,)                  dtype |S64
    response_len      shape (N,)                  dtype int32  (mirror of activations.zarr)
    prompt_len        shape (N,)                  dtype int32  (mirror; needed for projection-target indexing)
  meta/
    index.jsonl                                   # one JSON object per sample
                                                  # {"key", "sample_index", ...}
    config.json                                   # see below
```

`mlp_updates` from the prior draft is **dropped** — ICR Score uses the full residual `Δh_ℓ = h_ℓ − h_{ℓ-1}`, which is derivable from cached layer-output hidden states alone (§2 answer 2).

`config.json` fields (mandatory, fail-fast on mismatch at read time):
```json
{
  "source_activations_zarr": "../activations.zarr",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "num_layers": 32,
  "num_heads": 32,
  "head_dim": 128,
  "attention_region": "response_to_response",
  "query_position_rule": "all_response_tokens",
  "head_aggregation": "mean",
  "use_induction_head": false,
  "projection_kind": "residual_stream",
  "projection_target_layer": "previous",
  "projection_normalization": "l2_on_target",
  "score_top_k": null,
  "score_top_p": 0.1,
  "jsd_input_normalization": "zscore_then_softmax",
  "include_block_0": "tbd_after_layer_index_check",
  "dtype": "float16",
  "r_max": 64,
  "recomputed_from_cached_states": true,
  "validation_max_abs_diff": 4.2e-4,
  "icr_probe_repo_commit": "github.com/XavierZhang2002/ICR_Probe@<sha>"
}
```

The `score_*`, `projection_*`, `jsd_input_normalization`, and `use_induction_head` fields are advisory at this stage (Issue #69 stores attention only); they document the contract Issue #70 must satisfy and let #70 fail fast on mismatch.

### 5.3 Chunking

`response_attn` chunks: `(1, 1, R_max, R_max)`. Per-sample, per-layer chunks match the natural access pattern (the probe reads all layers for one sample at a time). This is similar to `activations.zarr`'s `(1, 1, prompt_chunk_tokens, H)` chunking pattern in [`zarr_activations_logger.py:498`](../activation_logging/zarr_activations_logger.py). Use zstd compression — padding past `response_len` is zeros and compresses well.

### 5.4 Storage size estimate

Worst-case allocated, fp16, response-to-response sub-block per sample at `R_max=64`:

| Dataset | Split | N | L | Size |
|---|---|---|---|---|
| HotpotQA | test  | 7400  | 28 | ~1.7 GB (Qwen3) |
| HotpotQA | test  | 7400  | 32 | ~1.9 GB (Llama-3.1) |
| HotpotQA | train | ~88K  | 28 | ~21 GB (Qwen3) |
| HotpotQA | train | ~88K  | 32 | ~24 GB (Llama-3.1) |

**Phase 1 total (both models, test + train): ~49 GB allocated.** Real on-disk after zstd compression of `response_len << R_max` padding zeros should be well under half of that. Phase 2 extrapolates to ~100–150 GB across the full 6-dataset grid — comfortable on NFS.

---

## 6. Reading API

### 6.1 New file: `activation_logging/attention_parser.py`

```python
class AttentionParser:
    """Reads recomputed attention probs from attention.zarr, paired with
    the activations zarr that they were derived from."""

    def __init__(
        self,
        attention_zarr_path: str,
        activations_parser: Optional[ActivationParser] = None,
    ):
        """If activations_parser is None, construct one from the
        source_activations_zarr field in config.json."""

    def get_attention(self, key: str) -> dict:
        """Returns:
            {
                "response_attn": Tensor (L, R_max, R_max) float32,
                "response_len": int,
                "prompt_len": int,
            }
        Caller must mask attention positions >= response_len."""

    def get_paired(self, key: str, relevant_layers: list[int]) -> dict:
        """Returns hidden states and attention for one sample, with explicit
        pairing of (h^{ℓ-1} as projection target, attn^ℓ as the attention
        distribution being matched against). Per icr_score.py:231-234:
            - block ℓ's score uses h^ℓ (post-block) and h^{ℓ-1} (block input)
            - projection target = h^{ℓ-1} (the "previous" layer)
            - block 0's input = HF embedding output (see §4.3)

        Output:
            {
                "h_block_input":  dict[layer_idx → Tensor(R, H)],   # h^{ℓ-1} at response positions only
                "delta_h":        dict[layer_idx → Tensor(R, H)],   # h^ℓ − h^{ℓ-1} at response positions
                "response_attn":  dict[layer_idx → Tensor(R, R)],   # head-averaged response-to-response
                "response_len": int,
                "prompt_len": int,
            }"""

    def list_keys(self) -> list[str]: ...
    def __len__(self) -> int: ...
```

`get_paired` is the primary API for Issue #70. It enforces the layer-index alignment in one place so the probe code does not have to think about it.

### 6.2 Dataset wrapper

A torch `Dataset` subclass that mirrors `ActivationsDataset` and yields:
```python
{
    "hashkey": str,
    "halu": int,
    "response_attn":  Tensor (L, R_max, R_max),  # response-to-response, masked by response_len
    "h_block_input":  Tensor (L, R_max, H),       # h^{ℓ-1} at response positions
    "delta_h":        Tensor (L, R_max, H),       # h^ℓ − h^{ℓ-1} at response positions
    "response_len": int,
}
```

`h_block_input` and `delta_h` are computed on-the-fly from cached layer-output hidden states at load time (cheap — one subtraction per layer). Not stored. For `ℓ=0`, `h_block_input` is the embedding output (cached if available, else re-derived per §4.3).

Place this class in `activation_research/icr_dataset.py` (new file).

---

## 7. Files to create / modify

| File | Action | Purpose |
|---|---|---|
| `notes/icr_probe_paper_notes.md` | Create | Answers to §2 open questions, with paper quotes |
| `notes/icr_probe_validation.md` | Create | §4.5 numerical-equivalence run results |
| `activation_logging/attention_recompute.py` | Create | Forward pass through one attention block from cached `h` |
| `activation_logging/attention_zarr_logger.py` | Create | Writer for `attention.zarr` |
| `scripts/recompute_attention.py` | Create | CLI driver: input `<activations_zarr>`, output `<attention_zarr>` |
| `activation_logging/attention_parser.py` | Create | Reader API |
| `activation_research/icr_dataset.py` | Create | Torch Dataset for paired (Δh, attention) loading |
| `tests/test_attention_recompute.py` | Create | §4.5 validation packaged as a test, plus unit tests for layer alignment |
| `tests/test_attention_parser.py` | Create | Round-trip write/read, mask correctness |

No edits to existing zarr loggers or `ActivationParser` are required. The new infra reads from the existing store and writes alongside it.

---

## 8. CLI for `scripts/recompute_attention.py`

```
--activations-zarr   str, required    Path to existing activations.zarr
--attention-zarr     str, required    Path to write new attention.zarr
--model              str, required    HF model ID; must match the inference model
--query-position-rule str, default    Resolved from §2 answer 1, e.g. "last_response_token"
--batch-size         int, default 8   Samples per recomputation batch
                                       (one sample = one full sequence through one attn block)
--device             str, default "cuda"
--dtype              str, default "float16"
--validate-first     flag             Run §4.5 validation on 4 samples and exit
--resume             flag             Skip samples already present in attention.zarr
--max-samples        int, optional    For smoke testing
--num-workers        int, default 2
```

`--validate-first` is the gating run. The implementer must run it for both models on HotpotQA and paste output into the PR description before launching the full job.

---

## 9. Compute budget

Per-sample cost (Path B): one forward pass through one attention sublayer per layer = `L × (one attention block forward)`. For Llama-3.1-8B that's `32 × ~1ms = ~32ms` on H200, plus model load (~30s, amortized).

HotpotQA Qwen3-8B test (7.4K samples × ~32ms ≈ 4 min) + Llama (similar) → well under 1 GPU-hour per (model, dataset) for test splits, ~12 GPU-hours total if Phase 2 also runs.

This is consistent with the `PAPER_ROADMAP.md` §11.3.3 estimate (~half a day per (model, dataset), conservative).

---

## 10. Implementation order

1. ~~Read paper + code~~ — **DONE 2026-05-15**, `notes/icr_probe_paper_notes.md` committed.
2. ~~Sequence-length check~~ — **DONE 2026-05-15**, results in §3 item 4.
3. ~~Layer-index check~~ — **DONE 2026-05-15**, results in §3 items 2–3 (cached `L = num_hidden_layers + 1`; embedding cached at index 0; block 0 scoreable directly).
4. **`attention_recompute.py`** — single-sample, single-block attention recomputation given `h^{ℓ-1}` and a loaded HF model. Output: head-averaged response-to-response attention sub-block. Verify on one sample by eye against `output_attentions=True`.
5. **Validation (§4.5)** — 4-sample numerical equivalence test, both models, restricted to the response-to-response sub-block. Write `notes/icr_probe_validation.md`. **Gate: do not proceed past this point if tolerance fails.**
6. **`attention_zarr_logger.py`** — writer with chunking from §5.3 and config.json from §5.2.
7. **`scripts/recompute_attention.py`** — CLI. Run with `--validate-first` first, then `--max-samples 32` smoke, then full HotpotQA test (both models), then HotpotQA train.
8. **`attention_parser.py` + `icr_dataset.py`** — reader API + dataset. Tests in `tests/`. Encode the layer-index alignment from §6 of the notes (block ℓ scoring uses HF index ℓ as projection target) in one place.
9. **PR**, with validation numbers, response-length histogram, layer-index check output, and a one-sample `get_paired()` printout pasted into the description.

---

## 11. Out-of-scope explicitly

- ICR Score computation, probe training, probe evaluation — Issue #70.
- Other datasets beyond HotpotQA — Phase 2, gated on Issue #70 results reproducing the paper's trend.
- Cross-model recomputation (e.g. running Llama attention from Qwen-cached hidden states) — nonsensical, do not attempt.
- Modifying existing `activations.zarr` stores.
- Re-running inference unless §4.5 explicitly fails.

---

## 12. Risk register (issue-local)

| Risk | Likelihood | Mitigation |
|---|---|---|
| §4.5 numerical equivalence fails due to RoPE / padding handling | medium | Diagnose against single-sample HF `output_attentions=True` ground truth (sliced to response sub-block); fix offset; rerun |
| Qwen3 response truncation (50%+ of samples have true `response_len > 64`) biases the ICR signal | medium | Documented in §3 item 4 + PR description. Issue #70 evaluation must include a sensitivity check restricted to `response_len ≤ 64`. Re-running Qwen3 inference at `R_max=256` is the clean fix (~24 GPU-hr) but deferred to a decision call |
| ~~§3 item 3 layer indexing~~ | resolved | Cached `L = num_hidden_layers + 1`; embedding output at index 0; same convention for both Llama and Qwen3 |
| ~~Storage layout becomes obsolete~~ | resolved | §5.2 layout aligned to code's response-to-response attention + L+1 hidden-state convention before any writer code is written |
| Implementer skips §4.5 validation under time pressure | medium | This issue's PR template requires pasting validation numbers; reviewer enforces |
| Issue #70 implements `js_divergence` per paper Eq. 7 instead of code's z-score-then-softmax variant | medium | `attention.zarr/meta/config.json` records `jsd_input_normalization: "zscore_then_softmax"` as the contract; #70's score-compute test must verify against the released code on at least one cached example |
