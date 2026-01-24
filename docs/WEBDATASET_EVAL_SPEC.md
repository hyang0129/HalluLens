# WebDataset Evaluation Spec (Activation Logging)

Date: 2026-01-24
Owner: HalluLens Activation Logging

## Summary
Evaluate WebDataset (WDS) as a storage/read format for activation logging under network storage constraints. Provide a conversion path from existing Zarr stores to WDS that is compatible with the current training process. Assess the feasibility and tradeoffs of logging directly to WDS versus continuing with Zarr/JSON for resumable inference.

This spec focuses on **evaluation** and a **conversion-based integration**. It does not mandate replacing Zarr.

## Background
Current storage options (Zarr filestore + JSONL) perform similarly in practice. The working hypothesis is that **network file access (many small opens/seeks)** is the dominant bottleneck, not raw data throughput. WebDataset could reduce metadata and small-file overhead via large TAR shards, but may reduce random-access capabilities.

The training access pattern (per [docs/ZARR_ACTIVATION_LOGGER_SPEC.md](docs/ZARR_ACTIVATION_LOGGER_SPEC.md)) is:
- Randomly sample `b` samples.
- For each sample, pick 2 random layers.
- Read full token sequence for those layers.

## Goals
- Determine whether WDS yields **equal or better end-to-end training throughput** on network storage.
- Provide **Zarr → WDS conversion** compatible with current training datasets and sampling.
- Preserve per-sample metadata (prompt, response, eval labels, IDs, lengths).
- Maintain deterministic sample indexing for reproducibility.

## Non-Goals
- Replacing Zarr for all use cases.
- Implementing a new training data pipeline (beyond an adapter compatible with the current `Dataset` interface).
- Designing a new inference logging protocol.

## Constraints & Assumptions
- Storage backend is a **network filesystem** (potentially high latency for many small files).
- Training uses **random access** for samples and layers.
- Activations are **float16** and high entropy (avoid compression).
- Resumability for inference is important; incremental writes must be safe.

## WebDataset Design (Option A Focus)

### Option A: Sample-level records (primary target)
Each WDS sample stores **all layers** for that sample.
- Pros: Simple mapping; easy conversion from Zarr shard; minimal record count.
- Cons: Reading 2 layers requires reading the **entire sample tensor**. For large `L`, this is expensive and discards most read data.

**Decision rationale:** We accept the wasted reads because the main bottleneck appears to be **network file access (many opens/seeks)** rather than data throughput. Option A minimizes random lookups by pulling a single large record per sample, which should improve DataLoader throughput and GPU utilization.

### Non-targeted alternatives (de-prioritized)
- **Option B (layer-level)** is high-risk without pairing-aware random access and is not the focus for this evaluation.
- **Option C (layer-pack)** is a fallback if Option A does not improve DataLoader performance.

## Record Schema (Option A)
Each WDS record represents a full **sample** with **all layers**.

**Key**
- `__key__`: `<sample_key>` or `<sample_index>`

**Binary payloads**
- `prompt_acts.npy`: `float16` array `(L, P_max, H)`
- `response_acts.npy`: `float16` array `(L, R_max, H)`

**Metadata (JSON)**
- `meta.json` (per record; small):
  - `sample_index` (int)
  - `prompt_len` (int)
  - `response_len` (int)
  - `hallu_label` (0/1 or -1)
  - `split` (optional)
  - `sample_key` (optional)

**Optional (sample-level, separate sidecar JSONL)**
- `samples.jsonl` mapping `sample_index -> {prompt, response}`
  - Keep large text outside WDS tar. Additional metadata is **not required** for training.

## Sharding Strategy
- Shard size: **256–1024 MB** (tune based on storage and parallelism).
- Shard naming: `wds-00000.tar`, `wds-00001.tar`, ...
- For resuming inference: create **new shards** per run; do not append to existing tar files.

## Indexing for Random Access
Option A does **not** require random access to individual layers. We will **stream** from TAR shards with a **shuffle buffer** and do **not** require deterministic ordering.

## Conversion Pipeline (Zarr → WDS)
1. **Read Zarr** shard/store using consolidated metadata.
2. For each sample `i`:
   - Read `prompt_len[i]`, `response_len[i]`, labels.
  - Read **all layers** for the sample and write one WDS record with full `L`.
3. Emit `samples.jsonl` with text and extra metadata (if required).
4. Emit a **manifest** with shard paths and record counts.

**Expected outcome:** conversion is write-once, read-many. It is acceptable to run as an offline job.

## Direct Logging to WebDataset (Evaluation Only)
Direct WDS logging is feasible but **not ideal for resumable inference**:
- Tar files are not safely appendable under interruption.
- Resuming requires opening new shards and tracking progress externally.

If evaluated:
- Write a **shard-per-process** strategy with periodic shard rotation.
- Store a **checkpoint manifest** with last completed shard and record count.
- Prefer **conversion-based WDS** for training while keeping Zarr for inference logs.

## Compatibility With Current Training Process
To avoid changes to training code:
- Implement a `Dataset` adapter that mimics the current `__len__` and `__getitem__` behavior.
- For each index, return only the **training-relevant tensors and labels** (omit extra metadata).
- Use the manifest to locate the corresponding `(sample_id, layer_id)` records.

For Option A, use **streaming** reads with a **shuffle buffer**. Each iteration reads one record per sample from the stream and selects 2 random layers in-memory. This intentionally discards most layer data but minimizes file accesses and should improve DataLoader throughput and GPU utilization.

## Current Run Configuration
- Layers: `L = 32`
- Tokens: `R_max = 64` (response) and `P_max` as configured in Zarr
- Hidden size: `H = 4096`

## Evaluation Plan
### Metrics
- DataLoader throughput (samples/sec)
- GPU utilization during training step
- Read latency p50/p95 for `(sample, layer)` fetch
- End-to-end epoch time

### Benchmarks
- **Baseline**: Zarr+JSON on network drive
- **WDS streaming (Option A)**: sharded tars with a **shuffle buffer** (primary configuration)

### Decision Criteria
- If WDS Option A yields **≥ 10–15% throughput improvement** or **lower p95 latency** and increases GPU utilization, adopt for training data reads.
- If WDS performs similarly to Zarr, keep Zarr as canonical and WDS as optional export format.

## Risks
- Reading full-layer tensors may waste IO and memory if throughput becomes a bottleneck.
- Large WDS records could increase per-sample decode time; monitor CPU overhead.
- Data duplication if both Zarr and WDS are kept.

## Implementation Checklist
- [ ] Define WDS schema and shard layout (Option A).
- [ ] Implement Zarr → WDS converter (offline job).
- [ ] Implement WDS dataset adapter (sample-level reads, in-memory layer selection).
- [ ] Add evaluation script and metrics logging.
- [ ] Run benchmarks on network storage.

## Open Questions
1. What is the current `P_max` for prompts (to match Zarr)?
