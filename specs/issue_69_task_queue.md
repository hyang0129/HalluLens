# Task Queue: Issue #69 — ICR Probe Attention Infrastructure

**Tier:** 2 (multi-area, clear requirements, no architecture ambiguity)  
**ADR:** N/A — all design decisions settled in `specs/issue_69_icr_probe_attention_infra.md`  
**Branch:** `feat/issue-69-icr-probe-attn-infra`  
**One PR:** yes

---

## File Ownership Table

| File | Owner | Wave |
|---|---|---|
| `activation_logging/attention_recompute.py` | Coder A | 1 |
| `activation_logging/attention_zarr_logger.py` | Coder B | 1 |
| `tests/test_attention_recompute.py` | Tester | 1 |
| `scripts/recompute_attention.py` | Coder C | 2 |
| `activation_logging/attention_parser.py` | Coder D | 2 |
| `tests/test_attention_parser.py` | Tester 2 | 2 |
| `activation_research/icr_dataset.py` | Coder E | 3 |

No file appears in more than one coder's scope. Wave 1 tasks are independent and can run in parallel. Wave 2 depends on Wave 1 completing. Wave 3 depends on Wave 2 completing.

---

## Wave 1 — Task A: `activation_logging/attention_recompute.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Implement a function that, given the cached hidden state h^{ℓ-1}
for a full sequence (prompt concat response) and a loaded HF transformer block,
returns the head-averaged response-to-response attention sub-block for that block.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§4.2, §4.3, §4.5)
      notes/icr_probe_paper_notes.md (§9 — cross-region masking)
      activation_logging/zarr_activations_logger.py (understand stored shape)
      activation_logging/activation_parser.py (understand key/index conventions)
  - Prior artifacts: none (Wave 1, no upstream code dependencies)

Output:
  - Deliverable: activation_logging/attention_recompute.py
  - Must include:
      recompute_block_attention(h_prev, block, prompt_len, response_len,
                                  position_ids=None, device="cpu") -> Tensor
        - h_prev:        Tensor (T, H) float32 — the full seq hidden state entering block
        - block:         a single HF transformer block (e.g. model.model.layers[ℓ])
        - prompt_len:    int — number of prompt tokens
        - response_len:  int — number of response tokens (T = prompt_len + response_len)
        - position_ids:  Tensor (T,) or None — if None, infer as arange(T)
        - Returns:       Tensor (response_len, response_len) float32
                         head-averaged response-to-response attention probabilities
                         (rows = query response tokens, cols = key response tokens)
      Helper: _build_causal_mask(T, device) -> Tensor (1, 1, T, T) bool or float
      Helper: _head_average_resp_to_resp(attn, prompt_len, response_len) -> Tensor
        - attn: (n_heads, T, T) full attention after softmax
        - Returns: (response_len, response_len)
      Module-level docstring explaining: cross-region attention is intentionally
        discarded (per icr_score.py:104-127; ICR Score only uses response-to-response).

  Notes on implementation:
    - Load the block in eval() mode. Call block.self_attn directly to get attention weights.
    - Some HF attention impls only return attn_weights when output_attentions=True is passed.
      For Llama: LlamaSdpaAttention does NOT return weights; use LlamaAttention (eager) by
      patching or by calling with attn_implementation="eager" at model load time.
    - The function should NOT load the model itself — the caller passes the block.
    - h_prev is float32 at runtime (cast if needed); stored zarr tensors are float16.
    - For RoPE: construct position_ids = arange(T) unless caller overrides. Llama and Qwen3
      both use RoPE; position_ids must match the original inference positions.
      Prompt positions: 0 .. prompt_len-1. Response positions: prompt_len .. T-1.
    - Do not apply RMSNorm inside this function — h_prev is already the block input
      (pre-norm), so the block's internal forward handles normalization.
    - Causal mask: standard lower-triangular (True = attend, False = mask out), matching
      HF's convention for the model family.

Scope (files you may edit):
  - activation_logging/attention_recompute.py  (create new)

Out of scope (do not touch):
  - Any existing file in activation_logging/
  - Any file outside activation_logging/
  - Model loading — caller is responsible for loading the model and passing the block

Acceptance criteria:
  - [ ] Function signature exactly matches the spec above
  - [ ] Returns Tensor of shape (response_len, response_len), dtype float32
  - [ ] Returns correct shape for edge cases: response_len=1, response_len=64, prompt_len=0
  - [ ] _head_average_resp_to_resp slices the response-to-response quadrant only
        (rows prompt_len:, cols prompt_len:) before averaging heads
  - [ ] Module-level docstring cites icr_score.py:104-127 for cross-region zeroing
  - [ ] No model loading code inside this file
  - [ ] All functions have type annotations
  - [ ] Python 3.12 compatible

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not load models or tokenizers in this file
  - Do not import from scripts/ or tasks/
  - Do not add logging infrastructure — raise ValueError for bad inputs, no loguru
```

---

## Wave 1 — Task B: `activation_logging/attention_zarr_logger.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Implement a writer class that creates and incrementally fills an
attention.zarr store (layout per spec §5.2) with recomputed attention data,
with resume semantics and a config.json metadata file.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§5.1–§5.4)
      activation_logging/zarr_activations_logger.py (reference for zarr writer patterns,
        pre-allocation, chunking, resume logic, and compressor usage)
      activation_logging/compression.py (import ZstdCompression from here)

Output:
  - Deliverable: activation_logging/attention_zarr_logger.py
  - Must include:
      AttentionZarrLogger(zarr_path, mode, num_layers, r_max, config_dict,
                           expected_samples=None, dtype="float16")
        - zarr_path: str — path to attention.zarr directory to create/open
        - mode: "w" (overwrite) or "a" (append/resume)
        - num_layers: int — number of transformer blocks (NOT L+1; excludes embedding)
        - r_max: int — max response length (default 64)
        - config_dict: dict — written verbatim as meta/config.json
        - expected_samples: int or None — for pre-allocation
        - dtype: "float16" (stored dtype)

      write(sample_key, response_attn, response_len, prompt_len)
        - sample_key: str
        - response_attn: Tensor or ndarray (num_layers, r_max, r_max) float16
        - response_len: int
        - prompt_len: int
        - Appends one row; writes to arrays/response_attn, arrays/sample_key,
          arrays/response_len, arrays/prompt_len, meta/index.jsonl

      is_written(sample_key) -> bool
        - Returns True if this key is already in the store (for resume semantics)

      finalize()
        - Flush and close the store; write final meta/config.json if not yet written

      Zarr layout (exact array names and shapes):
        arrays/response_attn  shape (N, num_layers, r_max, r_max)  dtype float16
                               chunks (1, 1, r_max, r_max)  compressor=zstd
        arrays/sample_key     shape (N,)                    dtype |S64
        arrays/response_len   shape (N,)                    dtype int32
        arrays/prompt_len     shape (N,)                    dtype int32
        meta/index.jsonl      one JSON object per line: {"key": ..., "sample_index": ...}
        meta/config.json      verbatim copy of config_dict at init

  Notes on implementation:
    - Use ZstdCompression from activation_logging/compression.py for response_attn array.
    - Chunks (1, 1, r_max, r_max): per-sample per-layer — matches natural probe access.
    - Pre-allocate arrays to expected_samples if provided (avoids O(n) resizes).
    - Resume: on mode="a", load existing sample_key array and build a set for
      is_written() lookups. Never write a key twice.
    - config.json: write at __init__ time if creating ("w"), or verify it matches
      at __init__ time if resuming ("a"). Raise ValueError on mismatch.
    - meta/index.jsonl: open in append mode so resume writes continue from the end.
    - All path handling via pathlib.Path.
    - Do not depend on activation_logging/zarr_activations_logger.py at runtime
      (copy the pattern, don't import it).

Scope (files you may edit):
  - activation_logging/attention_zarr_logger.py  (create new)

Out of scope (do not touch):
  - activation_logging/zarr_activations_logger.py  (read-only reference)
  - Any existing file

Acceptance criteria:
  - [ ] Zarr layout matches spec §5.2 exactly (array names, shapes, dtypes, chunks)
  - [ ] config.json is written on "w" init, verified on "a" init
  - [ ] is_written() returns correct results for existing and missing keys
  - [ ] write() pads or truncates response_attn to (num_layers, r_max, r_max) shape
  - [ ] finalize() can be called multiple times without error
  - [ ] No dependency on zarr_activations_logger at runtime
  - [ ] Python 3.12 compatible, type-annotated

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not import ZarrActivationsLogger
  - Do not store mlp_updates (spec §5.2: dropped)
  - Do not create arrays for prompt activations or delta_h — those come from activations.zarr
```

---

## Wave 1 — Task C: `tests/test_attention_recompute.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Write spec-first unit tests for attention_recompute.py that cover
shape correctness, response-to-response slicing, head averaging, and edge cases.
Tests must be runnable without GPU and without loading real model weights.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§4.2, §4.5)
      notes/icr_probe_paper_notes.md (§9)
      (Do NOT read attention_recompute.py — write against the spec, not the implementation)

Output:
  - Deliverable: tests/test_attention_recompute.py
  - Must include:
      test_output_shape_standard: prompt_len=32, response_len=10, n_heads=4, head_dim=16
        → shape (10, 10)
      test_output_shape_single_response_token: response_len=1 → shape (1, 1)
      test_output_shape_r_max: response_len=64 → shape (64, 64)
      test_head_average_resp_to_resp: given a known (n_heads, T, T) attention tensor,
        verify the slice and average is numerically correct
      test_cross_region_discarded: the returned tensor contains only response-to-response
        rows and cols — verify indirectly by checking that prompt-row data is not present
      test_rows_sum_to_one: each row of the returned attention (after masking by response_len)
        sums to 1.0 ± 1e-5 (attention rows are probability distributions)
      test_causal_mask_lower_triangular: rows i < j should be zero in the returned attention
        (causal: a token cannot attend to future tokens)

  Mocking strategy:
    - Create a minimal FakeBlock that wraps a tiny nn.MultiheadAttention (or raw einsum)
      and returns known attention weights when called.
    - Alternatively, mock the block's self_attn to return a fixed pattern.
    - Do NOT use pytest.mock to bypass the actual softmax — test the real math.

Scope (files you may edit):
  - tests/test_attention_recompute.py  (create new)

Out of scope (do not touch):
  - Any file outside tests/

Acceptance criteria:
  - [ ] All 7 named tests present
  - [ ] Tests pass without GPU (cpu-only)
  - [ ] Tests pass without real HF model weights (use fake/toy weights)
  - [ ] No test imports zarr, model_adapter, or HuggingFace transformers (keep deps minimal)
  - [ ] pytest-compatible: python -m pytest tests/test_attention_recompute.py

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not read attention_recompute.py (spec-first means writing against the interface, not the impl)
  - Do not load any real model checkpoints
  - Do not skip the causal mask test
```

---

## Wave 2 — Task D: `scripts/recompute_attention.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Implement the CLI driver that loads an existing activations.zarr,
iterates over all samples, calls recompute_block_attention() for each block,
and writes results to attention.zarr via AttentionZarrLogger.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§8 for CLI spec, §4.3 for h^{ℓ-1} sourcing)
      activation_logging/attention_recompute.py  (MUST exist — Wave 2 depends on Wave 1)
      activation_logging/attention_zarr_logger.py  (MUST exist — Wave 2 depends on Wave 1)
      activation_logging/zarr_activations_logger.py  (read existing stores)
      activation_logging/activation_parser.py  (understand key/index conventions)
  - Prior artifacts:
      activation_logging/attention_recompute.py (Wave 1 Task A output)
      activation_logging/attention_zarr_logger.py (Wave 1 Task B output)

Output:
  - Deliverable: scripts/recompute_attention.py
  - CLI spec (exact argument names):
      --activations-zarr   str, required  Path to existing activations.zarr
      --attention-zarr     str, required  Path to write new attention.zarr
      --model              str, required  HF model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)
      --batch-size         int, default 8
      --device             str, default "cuda"
      --dtype              str, default "float16"
      --validate-first     flag  Run 4-sample validation and exit (see below)
      --resume             flag  Skip samples already in attention.zarr (uses is_written())
      --max-samples        int, optional  Smoke test limit
      --num-workers        int, default 2  DataLoader workers for reading activations

  Processing loop (per sample s):
    1. Load prompt_activations[s] shape (L, P_max, H) and response_activations[s] (L, R_max, H)
    2. Load prompt_len[s] and response_len[s]
    3. For each block b in 0..num_blocks-1:
         h_prev = concat(prompt_activations[s, b, :prompt_len, :],
                          response_activations[s, b, :response_len, :])  shape (T, H)
         attn_resp = recompute_block_attention(h_prev, model.model.layers[b],
                                                prompt_len, response_len, device=device)
         store attn_resp into response_attn[s, b, :response_len, :response_len]
    4. Pad remaining response_attn[s, b, response_len:, :] = 0.0
    5. Write to AttentionZarrLogger

  Layer-index note: activations.zarr stores L+1 layers (index 0 = embedding output).
    Block b's input h^{b-1} is at activations[s, b, :, :] (NOT b-1).
    So loop is: for b in range(num_blocks): h_prev from activations[s, b, ...].

  --validate-first behaviour:
    - Pick 4 samples (indices 0, 1, 2, 3)
    - For each sample: run recompute_block_attention() for ALL blocks
    - Also run full model forward with output_attentions=True on the same token sequence
      (re-tokenize from stored token IDs + stored prompt text, or from stored token arrays)
    - For each block: compute max |A_recomp[resp:, resp:] - A_full[resp:, resp:]| (head-averaged)
    - Print a table: sample × block → max_abs_diff
    - Assert max diff < 1e-3 for fp16 stored inputs; exit 0 if pass, exit 1 if fail
    - Print argmax alignment: for response query token 0, does argmax match?

  config.json to write into attention.zarr/meta/:
    {
      "source_activations_zarr": "<absolute path>",
      "model_name": "<--model arg>",
      "num_layers": <num_blocks>,
      "num_heads": <from model config>,
      "head_dim": <from model config>,
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
      "dtype": "float16",
      "r_max": 64,
      "recomputed_from_cached_states": true
    }

  Model loading:
    - Load with attn_implementation="eager" to ensure attention weights are returned.
    - Load in the requested dtype (float16 or bfloat16).
    - Use device_map="auto" if device="cuda".
    - Qwen3: set model.config.thinking_mode = False if the attribute exists.

  Progress: use tqdm for the outer sample loop.
  Logging: loguru, one INFO line per 100 samples.
  Resume: if --resume, call logger.is_written(key) before processing.

Scope (files you may edit):
  - scripts/recompute_attention.py  (create new)

Out of scope (do not touch):
  - activation_logging/attention_recompute.py  (use as-is)
  - activation_logging/attention_zarr_logger.py  (use as-is)
  - Any existing file

Acceptance criteria:
  - [ ] All CLI arguments match the spec above (exact names)
  - [ ] --validate-first exits 0 on pass, 1 on fail, with per-block diff table printed
  - [ ] --resume skips already-written samples (no re-computation)
  - [ ] --max-samples N stops after N samples
  - [ ] config.json is written with all required fields before the first sample
  - [ ] Layer-index alignment: activations[s, b] used as h^{b-1} for block b
  - [ ] Loads model with attn_implementation="eager"
  - [ ] Python 3.12 compatible; argparse-based CLI

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not re-implement zarr reading — read from the existing store directly via zarr.open()
  - Do not commit model weights or large data files
  - Do not use subprocess to call other scripts
```

---

## Wave 2 — Task E: `activation_logging/attention_parser.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Implement the AttentionParser reader API (spec §6.1) that reads from
attention.zarr and pairs attention data with hidden states from the source
activations.zarr for use by Issue #70's probe code.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§6.1 for API spec)
      notes/icr_probe_paper_notes.md (§5, §6 for layer-index alignment)
      activation_logging/attention_zarr_logger.py  (understand the layout it wrote)
      activation_logging/activation_parser.py  (mirror its patterns; import ActivationParser
        or its zarr-reading primitives for the activations side)
  - Prior artifacts:
      activation_logging/attention_zarr_logger.py (Wave 1 Task B output)

Output:
  - Deliverable: activation_logging/attention_parser.py
  - Must include:
      class AttentionParser:
        __init__(attention_zarr_path, activations_parser=None)
          - If activations_parser is None, read meta/config.json["source_activations_zarr"]
            and construct one.
          - Validate config.json["model_name"] matches activations metadata if available.

        get_attention(key: str) -> dict
          Returns:
            {
              "response_attn": Tensor (L, R_max, R_max) float32,  # upcast from float16
              "response_len": int,
              "prompt_len": int,
            }
          Caller is responsible for masking out positions >= response_len.
          L here = num_layers (blocks only, NOT L+1; embedding layer excluded).

        get_paired(key: str, relevant_layers: list[int]) -> dict
          Returns:
            {
              "h_block_input":  dict[int → Tensor(response_len, H)],  # h^{ℓ-1} at resp positions
              "delta_h":        dict[int → Tensor(response_len, H)],  # h^ℓ − h^{ℓ-1} at resp positions
              "response_attn":  dict[int → Tensor(response_len, response_len)],
              "response_len": int,
              "prompt_len": int,
            }
          Layer-index alignment (from notes §6, icr_score.py:42-51):
            For block b ∈ relevant_layers:
              h_block_input[b] = response_activations[key, b, :response_len, :]   # HF index b = block input
              h_block_output[b] = response_activations[key, b+1, :response_len, :]  # HF index b+1 = block output
              delta_h[b] = h_block_output[b] − h_block_input[b]
              response_attn[b] = attention_zarr[key, b, :response_len, :response_len]
          All tensors returned as float32 (upcast if needed).

        list_keys() -> list[str]
        __len__() -> int

  Notes:
    - get_paired() must read BOTH attention.zarr and activations.zarr for the same key.
    - The activations.zarr layer axis has L+1 entries (embedding at 0, block outputs at 1..L).
      Block b's input h^{b-1} is at activations[key, b, ...].
      Block b's output h^b is at activations[key, b+1, ...].
    - Fail with a clear KeyError if key is not found in either store.
    - Cache the key→sample_index mapping for O(1) lookup.
    - All returned tensors should be CPU float32 regardless of stored dtype.

Scope (files you may edit):
  - activation_logging/attention_parser.py  (create new)

Out of scope (do not touch):
  - activation_logging/activation_parser.py  (import from, do not edit)
  - activation_logging/attention_zarr_logger.py  (import from, do not edit)
  - Any file outside activation_logging/

Acceptance criteria:
  - [ ] get_attention() returns dict with exactly the three keys specified
  - [ ] get_paired() returns dict with all five keys; layer index alignment matches spec
  - [ ] delta_h[b] = response_activations[b+1] − response_activations[b] (not b+1 − b-1)
  - [ ] All returned tensors are float32 on CPU
  - [ ] list_keys() and __len__() are consistent with the store
  - [ ] AttentionParser(path) works without explicitly passing activations_parser
  - [ ] config.json mismatch raises ValueError at __init__ time
  - [ ] Python 3.12 compatible; type-annotated

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not re-implement zarr writing — read-only
  - Do not perform any attention recomputation in this file
  - Do not store tensors as float16 in the returned dicts
```

---

## Wave 2 — Task F: `tests/test_attention_parser.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Write integration-style tests for AttentionParser that create a synthetic
attention.zarr + activations.zarr, write known data, and verify get_attention()
and get_paired() return correct values.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§5.2, §6.1)
      activation_logging/attention_zarr_logger.py  (MUST exist — Wave 1 Task B)
      activation_logging/attention_parser.py  (MUST exist — Wave 2 Task E)
  - Prior artifacts:
      activation_logging/attention_zarr_logger.py (Wave 1 output)
      activation_logging/attention_parser.py (Wave 2 Task E output)

Output:
  - Deliverable: tests/test_attention_parser.py
  - Must include:
      fixture: tmp_stores(tmp_path) — creates a synthetic attention.zarr and a
        synthetic activations.zarr with known random data, 3 samples, 2 layers
        (num_layers=2 for speed), r_max=8, H=16

      test_get_attention_shape: AttentionParser.get_attention returns correct shapes
      test_get_attention_values: known written values round-trip correctly (within fp16 precision)
      test_get_attention_response_len: response_len in returned dict matches written value
      test_get_paired_shapes: h_block_input, delta_h, response_attn all correct shapes
      test_get_paired_delta_h: delta_h[b] == activations[b+1] - activations[b] numerically
      test_get_paired_layer_alignment: h_block_input[0] == activations[:, 0, :response_len, :]
        (embedding output = block 0 input)
      test_list_keys: list_keys() returns all 3 written keys
      test_missing_key_raises: get_attention("bad_key") raises KeyError
      test_config_mismatch_raises: AttentionParser with a store whose config.json has
        wrong model_name raises ValueError at __init__ time

Scope (files you may edit):
  - tests/test_attention_parser.py  (create new)

Out of scope (do not touch):
  - Any file outside tests/

Acceptance criteria:
  - [ ] All 9 named tests present
  - [ ] Tests run without GPU
  - [ ] Tests run without a real HF model (use synthetic attention.zarr only)
  - [ ] pytest-compatible: python -m pytest tests/test_attention_parser.py
  - [ ] test_get_paired_delta_h passes to within 1e-4 tolerance (fp16 round-trip)

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not load any real model checkpoints
  - Do not test GPU-dependent code paths (no CUDA assertions)
```

---

## Wave 3 — Task G: `activation_research/icr_dataset.py`

```
Issue: #69 — ICR Probe Attention-Map Generation & Storage Infrastructure
ADR: N/A

Objective: Implement a PyTorch Dataset that yields (response_attn, h_block_input,
delta_h, halu) tuples from a paired attention.zarr + activations.zarr + eval_results.json,
for use by Issue #70's probe training loop.

Input:
  - Files to read:
      specs/issue_69_icr_probe_attention_infra.md (§6.2 for Dataset spec)
      activation_logging/attention_parser.py  (MUST exist — Wave 2 Task E)
      activation_logging/activation_parser.py  (mirror the metadata/label loading pattern)
      activation_research/model.py  (understand ActivationsDataset conventions, relevant_layers)
  - Prior artifacts:
      activation_logging/attention_parser.py (Wave 2 Task E output)

Output:
  - Deliverable: activation_research/icr_dataset.py
  - Must include:
      class ICRDataset(torch.utils.data.Dataset):
        __init__(attention_zarr_path, eval_results_path, relevant_layers,
                   split="train", random_seed=42, val_fraction=0.15,
                   activations_parser=None)
          - eval_results_path: path to eval_results.json ({"key": ..., "label": 0|1} entries)
          - relevant_layers: list[int] — which block indices to include (e.g. [4..31])
          - split: "train" | "val" | "test"
          - Stratified split by label (consistent with ActivationDataset's pattern)

        __len__() -> int

        __getitem__(idx) -> dict:
          {
            "hashkey":       str,
            "halu":          int (0 or 1),
            "response_attn": Tensor (len(relevant_layers), R_max, R_max) float32,
            "h_block_input": Tensor (len(relevant_layers), R_max, H) float32,
            "delta_h":       Tensor (len(relevant_layers), R_max, H) float32,
            "response_len":  int,
          }
          - response_attn and delta_h are computed via AttentionParser.get_paired()
          - Stack only the relevant_layers entries into the output tensors
          - Positions past response_len are zero-padded (attention.zarr already stores zeros there)

  Notes:
    - Lazy loading: read from zarr per __getitem__, do not preload everything.
    - h_block_input and delta_h come from get_paired()["h_block_input"] and ["delta_h"].
    - relevant_layers must be a subset of the layers stored in attention.zarr.
    - Label loading: read eval_results.json as {key: {"halu": 0|1}} or similar;
      match the exact schema used by existing eval_results.json files in output/.
      Read one existing eval_results.json to confirm the schema before hardcoding it.

Scope (files you may edit):
  - activation_research/icr_dataset.py  (create new)

Out of scope (do not touch):
  - activation_logging/attention_parser.py  (import from, do not edit)
  - activation_research/model.py  (read-only reference)
  - Any existing file

Acceptance criteria:
  - [ ] __getitem__ returns dict with all 6 keys
  - [ ] response_attn shape is (len(relevant_layers), R_max, R_max)
  - [ ] h_block_input shape is (len(relevant_layers), R_max, H)
  - [ ] delta_h[i] == h_block_output - h_block_input for each layer (verified by inspection)
  - [ ] split="train"/"val"/"test" yields non-overlapping key sets
  - [ ] Stratified split: class balance roughly preserved across splits (within 5%)
  - [ ] Python 3.12 compatible; type-annotated

Tools allowed: Read, Edit, Write, Grep, Glob

Do not:
  - Do not copy-paste ActivationDataset — import its patterns by reference
  - Do not preload all zarr data into memory at __init__ time
  - Do not include any training logic in this file (Issue #70 scope)
```

---

## Post-Wave 3 — Validation Gate (GPU, user-run)

This step is NOT a coding task. After all 7 coding tasks are complete, the following must be run on a GPU node before the PR is opened:

```bash
python scripts/recompute_attention.py \
    --activations-zarr shared/hotpotqa_llama_3_1_8b_instruct/activations.zarr \
    --attention-zarr shared/hotpotqa_llama_3_1_8b_instruct/attention.zarr \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --validate-first

python scripts/recompute_attention.py \
    --activations-zarr shared/hotpotqa_qwen3_8b/activations.zarr \
    --attention-zarr shared/hotpotqa_qwen3_8b/attention.zarr \
    --model Qwen/Qwen3-8B \
    --validate-first
```

The PR description must include:
1. `--validate-first` output table (sample × block → max_abs_diff) for both models
2. Statement that all diffs are < 1e-3 (or a diagnosis if any fail)
3. Qwen3 response truncation acknowledgment (§3 item 4): 50%+ of Qwen3 samples have true response_len > 64; accepted per user decision 2026-05-15
4. `python -m pytest tests/test_attention_recompute.py tests/test_attention_parser.py` output

---

## Integration — Binary Check Order

Run in this order after each wave:

```bash
# After Wave 1
python -m pytest tests/test_attention_recompute.py -v

# After Wave 2
python -m pytest tests/test_attention_parser.py -v

# After Wave 3
python -m pytest tests/ -v --ignore=tests/test_attention_recompute.py -k "attention"

# Full sweep before PR
python -m py_compile activation_logging/attention_recompute.py
python -m py_compile activation_logging/attention_zarr_logger.py
python -m py_compile activation_logging/attention_parser.py
python -m py_compile activation_research/icr_dataset.py
python -m py_compile scripts/recompute_attention.py
```

---

## PR Checklist

- [ ] All 7 coding tasks delivered (check files exist and are non-empty)
- [ ] `python -m pytest tests/test_attention_recompute.py` passes
- [ ] `python -m pytest tests/test_attention_parser.py` passes
- [ ] `--validate-first` output pasted for both models (Llama + Qwen3)
- [ ] All max_abs_diff values < 1e-3 (fp16)
- [ ] No modifications to any existing file (check `git diff --name-only`)
- [ ] Qwen3 truncation documented in PR description
- [ ] PR description references Issue #69 and links `specs/issue_69_icr_probe_attention_infra.md`
