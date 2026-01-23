# Zarr Activation Logger Spec (No LMDB)

Date: 2026-01-22

## Goals / Non-Goals

### Goals
- **Fast random read** of one sample’s **one layer** activations across **all tokens** (i.e., return a `(seq_len, hidden)` tensor) without loading other layers/samples.
- **Thread/process safe reads** compatible with multiple PyTorch `DataLoader` workers.
- **No compression** (activations are high entropy).
- **Training batch access pattern**:
  - Randomly sample `b` samples.
  - For each sampled sample, pick **2 random layers**.
  - Read activations for those (sample, layer) pairs.
- Store activations as **FP16** (`np.float16` / `torch.float16`) to minimize IO and maximize throughput.
- **Metadata** support:
  - Global metadata (model, dataset version, schema, etc.).
  - Per-sample metadata (prompt, response, eval labels, lengths, ids, etc.).
- **Parallelizable writes** (multi-process writers). Sharded outputs are allowed, with an optional merge/consolidation step.

### Non-Goals
- Not trying to support arbitrary ragged tensor layouts without padding.
- Not trying to support concurrent *writes* from multiple processes into the same Zarr store (we avoid that on purpose).

## Data Model

The activation extractor currently produces (per request / per sample):
- `num_layers = L`
- `hidden_size = H` (fixed per model)
- prompt token length `prompt_len_i` (varies)
- response token length `response_len_i` (varies)
- per-layer activation tensor typically shaped like `(1, prompt_len_i + response_len_i, H)` for each layer when logging the full sequence.

This spec **forces fixed token lengths** at storage time and **separates prompt vs response activations** into distinct arrays:

- Prompt activations: `prompt_acts[i, l, t, h]` with fixed `t ∈ [0, P_max)`
- Response activations: `response_acts[i, l, t, h]` with fixed `t ∈ [0, R_max)`

Where:
- `P_max` is a chosen maximum prompt token length (pad/truncate)
- `R_max` is a chosen maximum response token length (pad/truncate), e.g. **64**

We still store `prompt_len[i]` and `response_len[i]` so downstream can mask padding.

## Storage Layout (Recommended)

### Zarr store type
- Use **Zarr v2** directory store for maximum ecosystem compatibility.
  - Store path: `…/activations.zarr/`
  - Use `zarr.open_group(path, mode='r'|'w'|'a')`

If you later need to reduce file count at very large scale, consider Zarr v3 + sharding. This spec focuses on a robust v2 design that meets the read pattern.

### Top-level group structure

```
activations.zarr/
  .zgroup
  .zattrs
  meta/
    config.json   (optional; or stored in attrs)
    sample_index.jsonl  (optional)
  arrays/
    prompt_activations/  (Zarr array)
    response_activations/(Zarr array)
    prompt_len/          (Zarr array, optional)
    response_len/        (Zarr array, optional)
    sample_key/          (Zarr array of fixed-length bytes, optional)
    hallu_label/         (Zarr array, optional)
    ... more numeric per-sample fields
  text/
    prompts.jsonl        (optional)
    responses.jsonl      (optional)
    extra_metadata.jsonl (optional)
```

We intentionally separate:
- **Large numeric tensors** → Zarr arrays (fast, chunked, mmap-like access).
- **Large variable-length strings** (prompt/response) → JSONL files (simple and robust), unless you confirm you want string arrays in Zarr.

## Dataset-Driven Parameter Selection

The exact choices for `P_max`, `R_max`, and chunking should be derived from the dataset’s length distribution and the downstream training access pattern.

### Definitions
- `prompt_len_i`: number of prompt tokens for sample `i`
- `response_len_i`: number of response tokens for sample `i`
- `P_max`: stored prompt token length (pad/truncate)
- `R_max`: stored response token length (pad/truncate)
- `H`: hidden size

### Choosing `R_max` (response)
- If your training always consumes the first K response tokens (or you only need short responses), pick a **constant** `R_max = K`.
- If responses vary and you want to retain most without truncation:
  - Pick `R_max = q_p(response_len)` where `q_p` is a high quantile (e.g., p=0.95 or 0.99).
- Operational default for many QA settings: **`R_max = 64`**, because it gives fixed-shape reads and typically covers short factual answers.

### Choosing `P_max` (prompt)
- If your prompt format is fixed-ish (templated tasks), `P_max` can be set to a safe constant.
- Otherwise choose `P_max = q_p(prompt_len)` where `p` is a high quantile (e.g., 0.95 or 0.99).

### Chunking guidance (token dimension)
Chunk size is primarily about minimizing IO per random query while keeping per-chunk file reads efficient.

- Response array:
  - Recommended: `T_chunk_resp = R_max` so each `(sample, layer)` response slice reads **exactly one chunk**.
- Prompt array:
  - Choose `T_chunk_prompt` so that chunk bytes are ~0.5–2MB:
    - bytes = `T_chunk_prompt * H * 2`
    - Example with `H=4096`: `T_chunk_prompt ∈ {64, 128, 256}`.

### Record truncation rate
For transparency and later analysis, record:
- `prompt_truncated_count`, `response_truncated_count`
- `prompt_truncated_fraction`, `response_truncated_fraction`

Store these in global metadata (`root.attrs` or `meta/config.json`).

## Activation Array Specification

### Datasets

#### `arrays/prompt_activations`
- Shape: `(N, L, P_max, H)`
- Dtype: `float16`
- Order: C-order
- Fill value: `0` (pad)
- Compression: **disabled** (`compressor=None`, `filters=None`)

#### `arrays/response_activations`
- Shape: `(N, L, R_max, H)`
- Dtype: `float16`
- Order: C-order
- Fill value: `0` (pad)
- Compression: **disabled** (`compressor=None`, `filters=None`)

Recommended default: `R_max = 64` to guarantee constant-shape response reads and stable batch assembly.

### Chunking (critical)
To support the exact read pattern (one sample, one layer, all tokens) for either prompt or response tensors, we want reads to touch as few chunks as possible while keeping each chunk a reasonable size.

Recommended chunk shape:
- `(1, 1, T_chunk, H)`

Where:
- `T_chunk` should be tuned so that each chunk is ~0.5–2MB.
- Chunk size in bytes = `1 * 1 * T_chunk * H * 2`.

Examples (assuming `H=4096`):
- `T_chunk=64`  → `64*4096*2 ≈ 512KB`
- `T_chunk=128` → `128*4096*2 ≈ 1MB`
- `T_chunk=256` → `256*4096*2 ≈ 2MB`

Read cost for one (sample, layer):
- Number of chunks ≈ `ceil(seq_len / T_chunk)`.

For response reads with fixed `R_max`, a good choice is often `T_chunk = R_max` (e.g., 64), which makes each (sample, layer) response slice read **exactly one chunk**.

This chunking also ensures that reading **two random layers** for the same sample does not force reading other layers.

### Reading API contract
Given sample index `i` and layer index `l`:
1. Prompt read:
  - Read `p = prompt_len[i]`
  - Load `P = prompt_activations[i, l, 0:p, :]` (or `0:P_max` if you want constant-shaped tensors)
2. Response read:
  - Read `r = response_len[i]`
  - Load `R = response_activations[i, l, 0:r, :]` (or `0:R_max` for constant-shaped tensors)
3. Return `P` and/or `R` as `np.ndarray` or `torch.Tensor`

This must not read `prompt_activations[i, :, :, :]` or `response_activations[i, :, :, :]` (no whole-layer-stack reads).

## Metadata Specification

### Required per-sample numeric metadata (Zarr arrays)
Store in `arrays/`:
- `prompt_len`: `(N,)` int32 (required for masking/prompt slicing)
- `response_len`: `(N,)` int32 (required; must be `<= R_max`)
- `hallu_label`: `(N,)` int8/bool (optional; 1=hallucinated, 0=not)
- `split`: `(N,)` int8 (optional; 0=train,1=val,2=test)

### Sample identity
One of:
- `sample_key`: `(N,)` fixed-length bytes (e.g., `|S64` SHA256 hex) in Zarr.
- Or store `sample_key` in `meta/sample_index.jsonl` and rely on row order.

### Variable-length text fields (recommended JSONL)
- `text/prompts.jsonl`: one JSON object per line, including at least `{ "i": <int>, "sample_key": "…", "prompt": "…" }`
- `text/responses.jsonl`: same layout.

Rationale: Zarr string arrays are possible but can introduce overhead and metadata contention; JSONL keeps the Zarr store optimized for tensor IO.

### Global metadata
Stored in group attrs (`root.attrs`) or `meta/config.json`:
- `schema_version`
- `model_id`
- `num_layers=L`
- `hidden_size=H`
- `T_max`
- `dtype=float16`
- chunk parameters
- creation time, git commit, dataset name, etc.

## Thread/Process Safety Requirements (Reads)

### Requirement
Must be safe for multiple PyTorch `DataLoader` workers (multiprocess) and potentially multi-threaded training.

### Rules
- **Read-only stores**: open Zarr with `mode='r'` in each worker process.
- Each worker should create its **own** Zarr Group/Array handles during `Dataset.__init__` or `worker_init_fn`.
- Avoid mutating `attrs` at runtime.
- Use consolidated metadata when possible:
  - After writing: run `zarr.consolidate_metadata(store)`.
  - At read time: `zarr.open_consolidated(store, mode='r')`.

This prevents repeated reads of many small `.zarray/.zattrs` JSON files.

## Parallel Write Strategy

### Core principle
**Do not** have multiple processes append into the same Zarr array. Zarr v2 directory stores don’t provide a clean, cross-platform, multi-process safe append story for high-throughput logging.

### Sharded write layout
Each writer produces a self-contained shard:

```
run_YYYYmmdd_HHMMSS/
  shards/
    shard-0000.zarr/
    shard-0001.zarr/
    ...
  manifest.json
  merged/   (optional final merged store)
```

Each `shard-xxxx.zarr` follows the same schema but with its own local `N_shard` samples:
- `activations`: `(N_shard, L, T_max, H)`
- `seq_len`: `(N_shard,)`
- etc.

### Efficiency notes
- Writers should pre-allocate arrays for their shard size when possible.
- Writes should be chunk-aligned: write in blocks that match `(1,1,T_chunk,H)` chunking.
- Keep activation dtype as `float16` end-to-end to avoid cast overhead.

### Enforcing fixed token lengths
- **Prompt**: choose `P_max` and store `min(prompt_len_i, P_max)` tokens; pad the rest with zeros.
- **Response**: choose `R_max` (recommended 64) and store `min(response_len_i, R_max)` tokens; pad the rest with zeros.
- Store `prompt_len[i]` and `response_len[i]` as the *true* lengths capped to `P_max/R_max` so downstream masking is consistent.

## Merge / Consolidation Options

You said merging is acceptable; you have two viable paths:

### Option A (Recommended): Multi-shard reader (no merge)
- Keep shards separate.
- Build a `Manifest` that maps global sample index → `(shard_path, local_index)`.
- Training dataset reads from the correct shard.

Pros:
- No expensive rewrite.
- Parallel writes scale linearly.
- Read performance remains good if shards are on fast storage.

Cons:
- Slightly more complex dataset code.

### Option B: Physical merge into one Zarr store
- Create final arrays sized `(N_total, L, T_max, H)`.
- Copy shard arrays into the final array along sample axis.
- Consolidate metadata.

Pros:
- Single store is simpler to deploy.

Cons:
- Requires a full pass rewrite (IO heavy).

### “Final output must meet read performance”
Both options meet the requirement as long as:
- Chunking is preserved.
- Consolidated metadata is used.
- Readers open stores in read-only mode.

## Performance Checklist / Guardrails

- **No compression / no filters** on activation arrays.
- Use `float16` activations.
- Use `zarr.consolidate_metadata` after writing shards (and again after merge).
- Keep chunk size ~0.5–2MB.
- Avoid storing per-sample metadata in `attrs` (attrs become a scalability bottleneck).

## Validation Criteria

### Correctness
- For random `(i,l)`, the returned tensor matches the originally logged activation slice.
- Returned token dimension length equals `seq_len[i]`.

### Read performance
- Random access benchmark:
  - For `k=10_000` random queries `(i, random layer)`, measure mean/median/p95 read time.
  - Run with multiple `DataLoader` workers (e.g., 0, 4, 8) to confirm no lock contention.

Suggested benchmark matrix (to pick final parameters):
- Try `R_max ∈ {32, 64, 128}` if your dataset supports it.
- Try `T_chunk_prompt ∈ {64, 128, 256}`.
- Measure:
  - mean/p95 latency for reading one `(sample, layer)` prompt slice
  - mean/p95 latency for reading one `(sample, layer)` response slice
  - end-to-end batch assembly time for the training pattern: `b` samples × 2 random layers.

### Write scalability
- With `W` writer processes, throughput scales approximately with `W` until IO saturation.

## Open Questions (to finalize exact parameters)

1. What are the dataset distributions of `prompt_len` and `response_len` (median/p95/p99)?
2. Choose a truncation policy: fixed constants vs quantile-based `P_max/R_max`.
3. Model hidden size `H` and number of layers `L` for your main runs?
4. Target storage backend: local NVMe, network FS, or object store?

These determine the best `T_chunk` choice and whether we should move to Zarr v3 sharding.
