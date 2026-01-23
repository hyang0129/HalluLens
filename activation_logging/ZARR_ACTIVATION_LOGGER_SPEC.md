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
- token sequence length `seq_len_i` (varies per sample)
- per-layer activation tensor typically shaped like `(1, seq_len_i, H)` for each layer.

We standardize stored activations per sample as:
- `acts[i, l, t, h]` where:
  - `i` is sample index
  - `l` is layer index
  - `t` is token index
  - `h` is hidden dimension

Because `seq_len_i` is variable, we store a fixed maximum token dimension `T_max` and keep `seq_len[i]` metadata.

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
    activations/         (Zarr array)
    seq_len/             (Zarr array)
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

## Activation Array Specification

### Dataset: `arrays/activations`
- Shape: `(N, L, T_max, H)`
- Dtype: `float16`
- Order: C-order
- Fill value: `0` (or `NaN` if you want explicit padding detection; `0` is faster)
- Compression: **disabled**
  - `compressor=None`
  - `filters=None` (no shuffle)

### Chunking (critical)
To support the exact read pattern (one sample, one layer, all tokens), we want reads to touch as few chunks as possible while keeping each chunk a reasonable size.

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

This chunking also ensures that reading **two random layers** for the same sample does not force reading other layers.

### Reading API contract
Given sample index `i` and layer index `l`:
1. Read `n = seq_len[i]`
2. Load `X = activations[i, l, 0:n, :]`
3. Return `X` as `np.ndarray` or `torch.Tensor`

This must not read `activations[i, :, :, :]` (no whole-layer-stack reads).

## Metadata Specification

### Required per-sample numeric metadata (Zarr arrays)
Store in `arrays/`:
- `seq_len`: `(N,)` int32
- `prompt_len`: `(N,)` int32 (optional but strongly recommended)
- `response_len`: `(N,)` int32 (optional)
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

### Write scalability
- With `W` writer processes, throughput scales approximately with `W` until IO saturation.

## Open Questions (to finalize exact parameters)

1. What is your expected `T_max` (e.g., 256, 512, 1024, 2048)?
2. Typical `seq_len` distribution (median/p95)?
3. Model hidden size `H` and number of layers `L` for your main runs?
4. Target storage backend: local NVMe, network FS, or object store?

These determine the best `T_chunk` choice and whether we should move to Zarr v3 sharding.
