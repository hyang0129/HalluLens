# Deterministic Sharding for Parallel Workers (No Coordinator)

## Context

The current inference pipeline processes a deterministically ordered task list sequentially in a single process. Each item produces a JSONL record (`generation.jsonl`) and optionally activations (Zarr/LMDB). The goal is to split this list across N workers on a shared filesystem — no master coordinator — and consolidate at the end to avoid write contention during inference.

---

## Why Generation Cannot Be Sharded

The `generate` step (PreciseWikiQA, LongWiki) is fundamentally incompatible with sharding:

1. **Single append-only output file** — all workers would write to `qa_<src>_<model>_dynamic.jsonl` concurrently, causing corruption or race conditions
2. **Resume logic reads the same file** — progress is tracked by counting lines in the output file; all shards would see the same count and duplicate work
3. **Bin-stratified random sampling** — generation uses `h_score_cat` bins with `random.shuffle`; no global state means shards can't coordinate their sampling to avoid overlap or gaps
4. **Non-idempotent filtering** — QA pairs are accepted/rejected via LLM evaluation; the number of accepted pairs per shard is unpredictable, making it impossible to assign a fixed quota per shard

**Resolution**: Generation runs once as a single-process pre-shard phase. Sharding applies only to `inference`. Tasks with pre-existing data (TriviaQA, NQ) skip directly to the shard phase.

---

## High-Level Design

### Core Principle: Pre-shard Generate → Shard Inference → Consolidate → Eval

```
Phase 0 — Generate (single process, runs once)
  generate step → data/precise_qa/save/qa_*.jsonl   (QA pairs, fixed list)

Phase 1 — Shard Inference (N workers, no coordinator)
  Read same QA file
    ├─ Worker 0: items [0::N] → shard_0_of_N/generation.jsonl + shard.done
    ├─ Worker 1: items [1::N] → shard_1_of_N/generation.jsonl + shard.done
    ├─ Worker 2: items [2::N] → shard_2_of_N/generation.jsonl + shard.done
    └─ Worker 3: items [3::N] → shard_3_of_N/generation.jsonl + shard.done

Phase 2 — Consolidate (single process, after all shard.done exist)
  consolidate_shards.py → merged/generation.jsonl (sorted by item_index)

Phase 3 — Eval (single process, on merged output)
  eval step → merged/eval_results.json
```

---

## Design Decisions

### 1. Shard Assignment (Static, Coordinator-Free)

- **Method**: round-robin interleaving — worker `i` of `N` processes items at indices `[i, i+N, i+2N, ...]`
- **Rationale**: Simpler than work-stealing. Spreads heterogeneous items evenly (avoids one shard getting all "hard" items if they cluster). Each worker fully determines its own slice from (`shard_index`, `num_shards`) alone.
- **Inputs**: `--shard-index i` and `--num-shards N` CLI args (fits existing arg pattern in `run_with_server.py`)
- **Stable index**: the original DataFrame row index (after deterministic load/sort) is the item's canonical identity. Preserved in output records as `item_index` field.

### 2. Per-Worker Output Isolation (No Write Contention)

Each worker writes exclusively to its own directory. No file is shared during inference.

```
output/{task}/{model}/
  shard_0_of_4/
    generation.jsonl          # only items belonging to this shard
    activations.zarr/         # only this shard's activations
    shard.done                # sentinel: written when worker finishes
  shard_1_of_4/
    ...
  merged/
    generation.jsonl          # final merged output
    eval_results.json
```

- **JSONL**: no contention — each worker appends only to its own file
- **Zarr**: no contention — each worker has its own store; Zarr does not support concurrent multi-writer access to the same array
- **Resume within a shard**: existing resume logic in `utils/exp.py` works as-is (filters by `prompt` string)

### 3. Completion Signaling (No Coordinator)

- Each worker writes a `shard.done` sentinel file upon successful completion of its step
- Consolidation script checks for all `N` sentinel files before merging; fails with a clear error if any are missing
- No polling, no locking — the sentinel is a plain file write (atomic on POSIX filesystems for small writes)

### 4. Consolidation Step (Single Process, End of Run)

`scripts/consolidate_shards.py`:

1. **Verify**: check all `shard_{i}_of_{N}/shard.done` files exist
2. **Merge JSONL**: load each shard's `generation.jsonl`, sort all records by `item_index` to restore original order, write merged `generation.jsonl`
3. **Zarr**: per-shard stores are left in place; downstream training code consumes them directly (merge utility can be added later if needed)

### 5. What Does Not Change

- `utils/exp.py` inference loop and resume logic — untouched
- Task modules (`run_step`) — untouched
- Eval code — runs on merged output, unchanged
- Zarr/LMDB logger internals — untouched
- Server lifecycle — each worker starts its own vLLM server on its own GPU

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `scripts/run_with_server.py` | Add `--shard-index` / `--num-shards` args; slice DataFrame before passing to task; write `shard.done` sentinel; route output to shard subdirectory |
| `utils/exp.py` | Add `item_index` field to each JSONL record during incremental save |
| `scripts/consolidate_shards.py` | New file: verify sentinels, merge JSONL sorted by `item_index` |
| Task modules (precise_wikiqa.py, triviaqa.py, natural_questions.py) | Pass `item_index` through DataFrame slice (likely no-op if slice is passed as-is) |

---

## Usage Example

```bash
# Phase 0: generate (only for tasks that need it — PreciseWikiQA, LongWiki)
python scripts/run_with_server.py --step generate --task precisewikiqa --N 400 ...
# → data/precise_qa/save/qa_goodwiki_..._dynamic.jsonl  (400 QA pairs, fixed)

# Phase 1: inference shards (run on separate machines/GPUs simultaneously)
python scripts/run_with_server.py --step inference --task precisewikiqa \
  --shard-index 0 --num-shards 4 \
  --generations-file-path output/.../shard_0_of_4/generation.jsonl \
  --activations-path output/.../shard_0_of_4/activations.zarr
# ... (workers 1, 2, 3 analogous with --shard-index 1/2/3)

# Inside each worker:
#   load full QA file (400 items)
#   assign item_index = 0..399
#   slice: items where item_index % 4 == shard_index  → 100 items
#   exp.run_exp() → writes shard_N/generation.jsonl (each record includes item_index)
#   write shard_N/shard.done

# Phase 2: consolidate (after all workers complete)
python scripts/consolidate_shards.py \
  --shard-dir output/{task}/{model}/ --num-shards 4
# → checks shard_*/shard.done, merges JSONL sorted by item_index → merged/generation.jsonl

# Phase 3: eval (unchanged)
python scripts/run_with_server.py --step eval --task precisewikiqa \
  --generations-file-path output/.../merged/generation.jsonl
```

---

## Confirmed Decisions

1. **Server**: Each worker runs its own vLLM server on its own GPU. Server lifecycle in `run_with_server.py` is unchanged.
2. **Eval**: Runs only once after consolidation on the merged `generation.jsonl`. No per-shard eval.
3. **Zarr**: Per-shard stores are kept as-is. No Zarr merge in consolidation.

---

## Verification

1. **Unit test**: run 2 workers on TriviaQA with N=10, shard-index=0/1, verify each shard has ~5 records with no overlap, consolidate and verify 10 records in correct order.
2. **Resume test**: interrupt a worker mid-run, restart with same args, verify no duplicate records in the shard file.
3. **Correctness**: compare metrics from single-process run vs. sharded+consolidated run on same N — results must be identical.
