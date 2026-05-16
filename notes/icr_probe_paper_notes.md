# ICR Probe — Paper & Code Notes

**Paper:** Zhang et al., ACL 2025. *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs.*
[arXiv:2507.16488](https://arxiv.org/abs/2507.16488) · [code: XavierZhang2002/ICR_Probe](https://github.com/XavierZhang2002/ICR_Probe) · [ACL anthology](https://aclanthology.org/2025.acl-long.880/)

**Purpose of this file.** Resolve the 8 open questions in [`specs/issue_69_icr_probe_attention_infra.md`](../specs/issue_69_icr_probe_attention_infra.md) §2 *before* writing recompute / storage code. Each answer cites the released source code at the file:line level. The released code is the authoritative reproducibility artifact and is the source of truth where it differs from the paper formulation (it does, in a few places).

This is the gating deliverable for issue #69. Storage layout in the spec depends on these answers and must be revised based on the conclusions in §10 below.

**Files read.** `src/icr_score.py` (268 lines), `src/utils.py` (57 lines), `src/icr_probe.py` (166 lines), `README.md`. Downloaded from raw.githubusercontent.com/XavierZhang2002/ICR_Probe/main on 2026-05-15.

---

## 1. Query position(s) at which the ICR Score is evaluated

**Answer: every response token; per-token ICR scores then averaged over tokens before being fed to the probe.**

Source: `icr_score.py:217-220` (`compute_icr`):

```python
for layer in range(len(self.pooling_attentions)):
    icr_scores_layer = []
    for token in range(len(self.pooling_attentions[layer])):
        current_token_attn = self.pooling_attentions[layer][token]
        ...
```

`pooling_attentions[layer]` ranges over **output (response) tokens only** — set up in `_pre_process_attn` (lines 53–102) by extracting attention rows for positions in `[input_lens:, :]`. Then `_pooling_attn` pools across heads per layer.

The token-wise average is performed downstream (in the probe input pipeline, not in `compute_icr` itself — `compute_icr` returns `icr_scores_item` of shape `[L][response_len]`). The probe consumes a fixed-length `L`-vector, so the averaging step is implicit but consistent with the paper.

**Implication for storage:** the spec's `Q_max = 1` is wrong. The intermediate attention quantity needed is per-response-token, per-layer: shape `(L, R, ?)` per sample.

---

## 1.5 Where the attention comes from — `generate()` returns

**Answer: attention is captured during `model.generate(..., output_attentions=True, return_dict_in_generate=True)`. `out.attentions` is a heterogeneous tuple of length `1 + response_len` that `_pre_process_attn` stitches into one unified `(L, H, T, T)` causal matrix.**

Source: `icr_score.py:62-86` (`_pre_process_attn`). The per-piece shapes:

| `out.attentions[k]` | Source | Shape per layer (after squeezing batch) |
|---|---|---|
| `k = 0` | Prefill forward pass on the prompt | `(num_heads, prompt_len, prompt_len)` |
| `k ≥ 1` | Decode step that emitted response token `k − 1` | `(num_heads, 1, prompt_len + k)` |

The prefill chunk is square — full prompt-to-prompt attention from the initial forward. Each decode-step chunk is a *single attention row* (one query position emitting one token), with key length growing by 1 each step as the autoregressive context grows.

Upstream's stitching:
1. Right-pad each prefill row to `token_num = prompt_len + response_len` with zeros (`F.pad(..., (0, padding_size))` at `icr_score.py:69-72`).
2. Right-pad each decode-step row to `token_num` similarly (`icr_score.py:84-87`).
3. Concatenate prefill rows + decode-step rows along the query dimension to get `(L, H, T, T)`.
4. Zero out cross-region positions via `set_other_attn_scores_to_zero` (§9 below).

**Implication for our re-capture path (issue: inference rewrite).** We never need to materialize the full `(T, T)` matrix. For the ICR Probe, only response-to-response is consumed (§9). We can stream-slice as decode steps emit:

- Skip `out.attentions[0]` entirely — prompt-to-prompt is masked out anyway.
- For each `out.attentions[t]` with `t ≥ 1`, take the key slice `[prompt_len : prompt_len + r_max]`. The remaining `(num_heads, 1, r_max)` row is the t-th query position's response-to-response attention.
- Stack `t = 1 .. min(response_len, r_max)` rows → `(num_heads, r_max, r_max)` per layer. Mean over heads (§4) → `(r_max, r_max)`. This is what gets persisted.

Peak memory becomes `O(L × H × r_max)` per decode step instead of `O(L × H × T²)`. The prefill chunk that HF holds internally is still `(L, H, P, P)` — that's the bottleneck during the prefill step, not the per-token loop.

**Caveat for the apples-to-apples gate.** Our stream-slice must produce the same bytes as upstream's "stitch first, then mask" sequence. They are mathematically equivalent — the masked positions in upstream's full matrix are exactly the positions we never materialize — but the contract requires a numerical-equivalence assertion on a real prompt before we trust the rewrite.

---

## 2. Module granularity (one score per layer vs. attn + MLP separately)

**Answer: one ICR score per (response-token, layer). Not separated by attention / MLP sublayer.**

Source: `icr_score.py:231-235`:

```python
current_token_hs = self.output_hidden_states[layer + 1][token]
previous_token_hs = self.output_hidden_states[layer][token]
...
hs_diff = (current_token_hs - previous_token_hs)
```

`hs_diff` is the **full residual update across one transformer block** (attention + MLP combined). The code does not decompose it.

**Implication:** spec's optional `mlp_updates` array is not needed. The `Δh_ℓ = h_ℓ − h_{ℓ-1}` quantity is derivable from cached layer-output hidden states alone.

---

## 3. Meaning of "top-k attention-weighted tokens" and the **k vs. top_p** subtlety

**Answer: top-k key positions of the head-averaged attention vector at the query token. But `top_p` overrides `top_k` when given.**

Source: `icr_score.py:225-229`:

```python
top_k = min(top_k, len(current_token_attn)) if (top_k is not None) else len(current_token_attn)
top_k = top_k if top_p is None else int(top_p * len(current_token_attn))
top_p_token = top_k/max(len(current_token_attn),1e-6)
top_p_layer.append(top_p_token)
current_token_attn_topk, current_token_attn_topk_idx = torch.topk(current_token_attn, k=top_k)
```

README quickstart (`README.md:51-58`): `top_k=20, top_p=0.1`. Because `top_p=0.1` is non-None, line 226 makes `top_k = int(0.1 * len(current_token_attn))`. **The `top_k=20` argument is silently overridden.**

For a `current_token_attn` of length `T` (full sequence, padded with zeros for prompt-side positions — see §6 below), the effective k is `floor(0.1 * T)`. For our HotpotQA with `P_max + R_max = 576`, effective k ≈ 57 per token.

**Implication:** the paper's "k=20" headline is misleading. Effective k scales with sequence length. If we re-implement with `top_p=0.1`, we don't need to commit to a fixed k at storage time — we store the full attention row and apply `top_p` at score-compute time.

---

## 4. Head aggregation

**Answer: mean across heads.**

Source: `icr_score.py:194-201`:

```python
if pooling == 'mean':
    pooled_layer = torch.mean(stacked_heads, dim=0)
elif pooling == 'max':
    pooled_layer = torch.max(stacked_heads, dim=0)[0]
elif pooling == 'min':
    pooled_layer = torch.min(stacked_heads, dim=0)[0]
```

README default: `pooling='mean'`.

**Caveat — `use_induction_head` filter:** when `use_induction_head=True` (README default), only heads classified as "induction heads" by skewness + entropy thresholds are included in the mean. However, the README also sets `skew_threshold=0, entropy_threshold=1e5`, which are loose enough that **every head passes the filter** (skewness ≥ 0 is almost always true, entropy ≤ 1e5 is always true). The fallback at lines 170–173 (force-pick top `num_heads // 8` by skewness if too few heads pass) also doesn't trigger when all heads pass.

Net effect: in the released-paper configuration, `use_induction_head` is effectively a no-op and the result is **simple arithmetic mean over all heads**.

The README confirms: *"Parameters for Induction Head, but not used in the final version"* (`README.md:39`).

**Implication for our reimplementation:** ignore the induction-head logic; aggregate by mean over all heads.

---

## 5. Projection direction (vocab unembedding vs. residual-stream) AND the projection target's layer index

**Answer: residual-stream projection — NO vocab unembedding. AND: projection target is the *previous* layer's hidden state h^{ℓ-1}, not the post-block one.**

Source: `icr_score.py:231-238`:

```python
current_token_hs = self.output_hidden_states[layer + 1][token]
previous_token_hs = self.output_hidden_states[layer][token]
current_layer_all_hs = self.origin_hidden_states[layer]            # ← projection target
current_token_hs_topk = current_layer_all_hs[current_token_attn_topk_idx]
hs_diff = (current_token_hs - previous_token_hs)
w_i = torch.sum(hs_diff * current_token_hs_topk, dim=1) / (
        torch.norm(current_token_hs_topk, dim=1) + 1e-8)
```

Two non-obvious details:

1. **`current_layer_all_hs = origin_hidden_states[layer]`** — the projection target is the **input-side** hidden state of the block being scored, i.e. `h^{ℓ-1}` in HF numbering. Not the post-block hidden state.

2. **Length normalization is by `||x_j^ℓ||`**, the L2 norm of the *projection target* (top-k context positions' hidden state), not by `||Δh_i||` or by `||x_j^ℓ|| · ||Δh_i||`.

So the per-context-position score is `(Δh_i^ℓ · h_j^{ℓ-1}) / ||h_j^{ℓ-1}||`. No softmax at this stage — softmax happens later inside `js_divergence` (see §11 below).

---

## 6. Layer indexing convention — and why the embedding hidden state matters

**Answer: code's layer index `ℓ` (0-indexed, 0..L-1) corresponds to HF transformer block `ℓ`. Scoring block 0 requires the EMBEDDING OUTPUT (`HF hidden_states[0]`).**

Source: `icr_score.py:42-51` (`_pre_process_hs`):

```python
hidden_states_input = torch.stack(hidden_states[0], dim=0)  # [layer, batch, input_size, hidden_size]
hs_input = hidden_states_input[:, 0, :]                     # [layer, input_size, hidden_size]
hidden_states_output = torch.stack([torch.stack(layer) for layer in hidden_states[1:]], dim=0)
hs_output = torch.cat([hidden_states_output[i, :, 0] for i in range(len(hidden_states_output))], dim=1)
hs_all = torch.cat([hs_input, hs_output], dim=1)           # [layer, total_seq_len, hidden_size]
```

HF's `hidden_states[0]` from the prompt forward pass is a tuple of length `L+1` (embedding output at index 0, plus one tensor per transformer block at indices 1..L). The code preserves this `L+1` first dimension into `origin_hidden_states`.

Then `compute_icr` iterates `layer in range(len(self.pooling_attentions))` where `pooling_attentions` has `L` entries (HF attention output has one entry per transformer block, no embedding). For `layer = 0`:
- `current_token_hs = output_hidden_states[1][token]` = block-0's output hidden state
- `previous_token_hs = output_hidden_states[0][token]` = **embedding output** for this token
- `current_layer_all_hs = origin_hidden_states[0]` = **embedding output** for the full sequence

**Critical implication.** To score the first transformer block, the code consumes the embedding output. If our cached zarr stores only `L = num_hidden_layers` tensors (block outputs, no embedding), **block 0 cannot be scored from cache alone**. Mitigations, in preference order:

1. **Re-derive the embedding output** from cached `response_token_ids` (and re-tokenized prompts) + the loaded model's embedding layer. One matrix multiply per sample, trivial cost. Preserves block-0 in the probe.
2. **Drop block 0** from the L-dim probe input (the probe would be `(L−1, 128, 64, 32, 1)`). Small accuracy hit, documented as a deviation.

Decision must be made after confirming `prompt_activations.shape[1]` on NFS (an `L` vs `L+1` empirical check).

---

## 7. Probe input shape

**Answer: `L`-dim per-sample vector (one score per layer, after token-wise mean). Probe is `(L, 128, 64, 32, 1)` MLP, BCE loss, Adam + ReduceLROnPlateau.**

Source: `utils.py:5-26`:

```python
class ICRProbe(nn.Module):
    def __init__(self, input_dim=32):
        super(ICRProbe, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        ...
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
```

Source: `icr_probe.py:38-46`:

```python
input_dim = next(iter(self.train_loader))[0].shape[1]
self.model = ICRProbe(input_dim=input_dim).to(self.device)
self.criterion = nn.BCELoss()
self.optimizer = torch.optim.Adam(
    self.model.parameters(),
    lr=self.config.learning_rate,
    weight_decay=self.config.weight_decay
)
```

Confirms: probe consumes a single L-dim feature vector per sample. Hidden layers 128 → 64 → 32 with BatchNorm + LeakyReLU(0.01) + Dropout(0.3); sigmoid output. Issue #70 scope.

Reported metrics (`icr_probe.py:138-145`): F1, ROC-AUC, PCC, accuracy, precision, recall. Apples-to-apples comparable with our existing pipeline's `predictions.csv → eval_metrics`.

---

## 8. Sequence handling (greedy vs. sampled generation)

**Answer: not constrained by the code. The score is computed on a single generated sequence's hidden states; how that sequence was produced is irrelevant to the score formula.**

Our existing zarrs are from greedy generation. No divergence concern.

---

## 9. Cross-region attention masking — only response-to-response attention contributes

**Critical finding not anticipated by the spec.** The code zeros out cross-region attention before top-k.

Source: `icr_score.py:104-127` (`set_other_attn_scores_to_zero`), called from `_pre_process_attn:97`:

```python
a, b, c = self.core_positions['user_prompt_start'], self.core_positions['user_prompt_end'], self.core_positions['response_start']
mask = torch.zeros((layer_num, head_num, token_num, token_num), dtype=torch.bool)
mask[:, :, a:b, a:b] = True       # user-prompt-to-user-prompt
mask[:, :, c:, c:] = True         # response-to-response
attn_all[~mask] = 0
```

Note that the response-to-prompt region (`mask[:, :, c:, a:b]`) is **NOT** kept. For a response query token, attention to all prompt positions is zeroed out before top-k.

**Implications:**

- Top-k indices for response-token queries are **always within the response region** (prompt-side scores are zero, so torch.topk skips them in any non-pathological case).
- The projection target (`current_layer_all_hs[current_token_attn_topk_idx]`) ends up indexing **response-side hidden states only**.
- We only need to store the response-to-response attention sub-block per layer, not the full causal attention matrix.
- **Storage requirement collapses** from `(N, L, R_max, T_max)` to `(N, L, R_max, R_max)` — a ~9× reduction at our defaults (`R_max=64, T_max=576`).

The `core_positions['user_prompt_start':'user_prompt_end']` masking exists for an ablation that scores prompt tokens, which we don't reproduce; we only need the response side.

---

## 10. JSD on standardized softmax of top-k subsets — NOT the paper's clean formulation

**Critical finding.** The `js_divergence` function does not implement the paper's Eq. 7 verbatim. It standardizes the inputs first.

Source: `icr_score.py:258-267`:

```python
def js_divergence(p, q):
    # standardize: p, q -> N(0, 1)
    p = (p - p.mean()) / max(p.std(), 1e-8)
    q = (q - q.mean()) / max(q.std(), 1e-8)
    # softmax: p, q -> [0, 1]
    p = F.softmax(p, dim=0)
    q = F.softmax(q, dim=0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

So the actual computation, given top-k raw attention values `a_topk` and raw projection lengths `w_topk` of dimension `k`:

```
a_norm  = softmax(zscore(a_topk))            # k-dim distribution
w_norm  = softmax(zscore(w_topk))            # k-dim distribution
m       = 0.5 * (a_norm + w_norm)
ICR     = 0.5 * KL(a_norm || m) + 0.5 * KL(w_norm || m)
```

This is **not** the paper's `Proj_i^ℓ = softmax([p_{i,j}^ℓ]_{j=1}^N)` over the full sequence. The released code computes JSD on a `k`-dim subset (top-k attention positions), with z-score temperature normalization, then softmax. This matters because:

- The standardization changes the "temperature" of the distribution per-sample.
- The softmax inside JSD means the raw attention values (already a probability distribution from softmax in the model's attention) get re-normalized.

**Implication:** for our reimplementation to match the published baseline (apples-to-apples), we must replicate `js_divergence` exactly — not rewrite to the paper's notational form. This belongs in Issue #70 (score computation), not #69 (storage). But it informs what we store: we need raw attention values (not pre-softmax logits), since the JSD function takes raw inputs and processes them internally.

We also need raw `w_i` (the projection length vector before any softmax). That's computed inline at score-compute time from cached hidden states, so nothing to store.

---

## 11. Resulting changes to the spec storage layout (`specs/issue_69_icr_probe_attention_infra.md` §5)

The answers above require revising the spec's storage layout. Summary of deltas:

### 11.1 `attn_probs` shape

**Spec:** `(N, L, Q_max=1, T_max=576)` — single attention row per sample.
**Correct:** `(N, L, R_max=64, R_max=64)` — response-to-response attention sub-block per sample, masked by `response_len`.

The `R_max × R_max` shape is the consequence of §9 (cross-region attention masking) — we do not need the prompt-side key positions at all.

### 11.2 Storage size estimate (§5.4)

**Spec estimate:** ~240–273 MB per (model, dataset, test split) at `Q_max=1`.
**Revised** (per-sample response-to-response attention):

| Dataset | Split | N | L | R_max | R_max | dtype | Size |
|---|---|---|---|---|---|---|---|
| HotpotQA | test  | 7400  | 28 | 64 | 64 | fp16 | ~1.7 GB |
| HotpotQA | test  | 7400  | 32 | 64 | 64 | fp16 | ~1.9 GB |
| HotpotQA | train | ~88K  | 28 | 64 | 64 | fp16 | ~21 GB |
| HotpotQA | train | ~88K  | 32 | 64 | 64 | fp16 | ~24 GB |

**HotpotQA Phase 1 total (both models, test + train):** ~49 GB worst-case allocated.

zstd compression on padded zeros past `response_len_s` should bring real on-disk size down to ~10–20 GB total. Phase 2 (5 more datasets) would bring it to ~100–150 GB across the full grid — comfortable on NFS.

This is now **smaller** than the spec's worst-case estimate of 270 MB × 6 datasets in a different sense — the spec was right that worst-case is manageable, just not on the same axis.

### 11.3 §5.2 `attention.zarr` config.json fields

```diff
  "source_activations_zarr": "../activations.zarr",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "num_layers": 32,
  "num_heads": 32,
  "head_dim": 128,
- "query_position_rule": "last_response_token",
+ "query_position_rule": "all_response_tokens",
+ "attention_region": "response_to_response",     // §9: cross-region zeroed before top-k
+ "score_top_p": 0.1,                             // effective k = floor(top_p * effective_seq_len)
+ "projection_kind": "residual_stream",           // §5: NOT vocab unembedding
+ "projection_target_layer": "previous",          // §5: h^{ℓ-1} as projection target
+ "projection_normalization": "l2_on_target",     // §5: ||h_j^{ℓ-1}||
+ "jsd_input_normalization": "zscore_then_softmax", // §10: matches js_divergence exactly
+ "include_block_0": "tbd_after_layer_index_check", // §6
  "head_aggregation": "mean",
  "use_induction_head": false,                    // §4: README sets thresholds to no-op
  "dtype": "float16",
  "recomputed_from_cached_states": true,
  "validation_max_abs_diff": "tbd",
  "icr_probe_paper_commit": "github.com/XavierZhang2002/ICR_Probe@<sha>"
```

Drop the optional `mlp_updates` field — never needed (see §2 above).

### 11.4 §6.1 `AttentionParser.get_paired` signature

Updated `attn_probs` shape: `(L, R_max, R_max)` not `(L, 1, T_max)`. Caller masks by `response_len`. Storage and reader are aligned in dimensions.

### 11.5 §9 compute budget

**Unchanged.** One attention-sublayer forward pass per (sample, layer); the response-to-response sub-block is sliced from the output.

### 11.6 §4.5 numerical-equivalence test scope

The test gate must compare the recomputed attention to a full-model `output_attentions=True` forward pass **restricted to the response-to-response sub-block**, not the full causal attention matrix. The full matrix's prompt-side values are irrelevant for ICR Probe (they get zeroed in §9).

---

## 12. Remaining open questions for sign-off

Not resolved by paper/code reading — need empirical check on NFS or a decision call:

1. **Cached layer-0 convention.** Is `arrays/prompt_activations.shape[1] == num_hidden_layers` (block outputs only) or `num_hidden_layers + 1` (with embedding output at index 0)? Need NFS scan against one HotpotQA zarr for each of Llama-3.1-8B and Qwen3-8B. Decides whether block-0 scoring requires an embedding-layer rederivation (cheap, recommended) or block-0 is dropped from the probe input.

2. **HotpotQA response-length distribution.** Mean `response_len` directly affects on-disk zstd compression ratio. Need NFS scan of `arrays/response_len`.

3. **`use_induction_head` reproduction toggle.** Set to `False` in our reimplementation by default (per §4 above), but worth confirming with paper authors if their reported headline numbers actually disabled the path or used the thresholds in the released README. (Low priority — the released config is the artifact.)

4. **k vs. top_p commitment.** Plan: store full `(R_max, R_max)` attention and apply `top_p=0.1` at score-compute time (Issue #70). This preserves k-ablation reproducibility without locking storage to a specific k.

5. **Storage option choice (was a decision in v1 of these notes).** Resolved: the response-to-response storage layout in §11.1 above is small enough that the v1's Option A / B / C tradeoff is moot. Full attention sub-block at fp16 fits comfortably; recommend storing it.

---

## 13. Citations index

| Source | Used for |
|---|---|
| `icr_score.py:42-51` (`_pre_process_hs`) | §2, §5, §6 |
| `icr_score.py:53-102` (`_pre_process_attn`) | §1, §6 |
| `icr_score.py:104-127` (`set_other_attn_scores_to_zero`) | §9 |
| `icr_score.py:150-179` (`_is_induction_head`) | §4 |
| `icr_score.py:181-210` (`_pooling_attn`) | §4 |
| `icr_score.py:212-251` (`compute_icr`) | §1, §2, §3, §5, §6 |
| `icr_score.py:258-267` (`js_divergence`) | §10 |
| `utils.py:5-26` (`ICRProbe`) | §7 |
| `icr_probe.py:38-46, 138-145` (trainer) | §7 |
| `README.md:51-58` (quickstart) | §3, §4 |
| `README.md:39` ("not used in final version") | §4 |
