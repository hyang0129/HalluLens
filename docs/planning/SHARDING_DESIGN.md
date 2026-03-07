# Deterministic Sharding for Parallel Workers (No Coordinator)

## Context

The current inference pipeline processes a deterministically ordered task list sequentially in a single process. Each item produces a JSONL record (`generation.jsonl`) and optionally activations (Zarr/LMDB). The goal is to split this list across N workers on a shared filesystem — no master coordinator — and consolidate at the end to avoid write contention during inference.

---

## Generation vs. Inference: Different Sharding Models

Generation and inference have fundamentally different constraints and require different parallelism strategies.

### Generation: Freeform Parallel with Filelock

Generation has no fixed input list — workers freely sample from the Wikipedia source corpus, call the LLM, and filter for answerability. There is no ordering requirement and no concept of "item i belongs to worker j". Key properties:

- **Duplicate source documents are acceptable**: the LLM uses `temperature=0.7` and randomly samples a section (`random.sample(sections, 1)`), so the same Wikipedia article will produce different questions across runs and workers.
- **No seeding**: `utils/generate_question.py` has no `random.seed()` calls, relying on Python's default OS-entropy initialization. This must be preserved — explicitly seeding workers would cause them to produce identical question sets. **Do not add seeding.**
- **Lock contention is negligible**: each item requires two sequential LLM API calls (generation + answerability check), taking seconds. File writes take milliseconds. The lock is held for a fraction of 1% of each item's wall time.
- **Overshoot is acceptable**: if multiple workers all read `count < N` simultaneously and each write one item, the final count may exceed N by at most `num_workers`. This is acceptable.

**Mechanism**: filelock-per-write on the shared output file. Each worker checks the line count before starting the next item; if `count >= N`, it stops. No atomic check+write required — the overshoot bound of `num_workers` items is fine.

```
Generation (freeform, any number of workers, shared file + filelock):
  Worker 0 ──┐                                   check count → write (filelock)
  Worker 1 ──┼──► qa_output.jsonl.lock            each worker stops when count ≥ N
  Worker 2 ──┘                               overshoot ≤ num_workers items
```

### Inference: Static Partition (No Duplicates)

The generated QA file is a fixed, deterministic list. Each item must be answered by exactly one worker. Static round-robin partition is required.

---

## High-Level Design

### Core Principle: Freeform Parallel Generate → Static-Partition Inference → Consolidate → Eval

```
Phase 0 — Generate (any number of workers, shared file + filelock)
  Worker 0, 1, 2, ... independently sample wiki docs and write accepted QA pairs
  Each worker stops when shared qa_output.jsonl line count ≥ N
  Final count = N ± num_workers (overshoot acceptable)
  → data/precise_qa/save/qa_*.jsonl

Phase 1 — Shard Inference (M workers, static partition, no coordinator)
  Read same QA file (fixed list)
    ├─ Worker 0: items [0::M] → shard_0_of_M/generation.jsonl + shard.done
    ├─ Worker 1: items [1::M] → shard_1_of_M/generation.jsonl + shard.done
    ├─ Worker 2: items [2::M] → shard_2_of_M/generation.jsonl + shard.done
    └─ Worker 3: items [3::M] → shard_3_of_M/generation.jsonl + shard.done

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
- **Inputs**: `--shards N` and `--shard-id i` CLI args (fits existing arg pattern in `run_with_server.py`)
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
| `utils/generate_question.py` | Add `filelock` around each accepted QA write in `per_bin_generation_batch`; check global count before each item to stop at N (overshoot ≤ num_workers) |
| `scripts/run_with_server.py` | Add `--shards` / `--shard-id` args (used by both steps); for inference: slice DataFrame and route output to shard subdirectory, write `shard.done` sentinel |
| `utils/exp.py` | Add `item_index` field to each JSONL record during incremental save |
| `scripts/consolidate_shards.py` | New file: verify sentinels, merge JSONL sorted by `item_index` |
| Task modules (precise_wikiqa.py, triviaqa.py, natural_questions.py) | Pass `item_index` through DataFrame slice (likely no-op if slice is passed as-is) |

---

## Usage Example

```bash
# Phase 0: generate (freeform parallel — run on as many machines as available simultaneously)
# Each worker independently samples wiki docs, stops when shared file reaches N
python scripts/run_with_server.py --step generate --task precisewikiqa --N 400 \
  --q-generator meta-llama/Llama-3.1-70B-Instruct ...
# → data/precise_qa/save/qa_goodwiki_..._dynamic.jsonl  (≥400 QA pairs, within 400+num_workers)
# Workers use filelock on qa_output.jsonl.lock; no seeding — OS entropy ensures diversity

# Phase 1: inference shards (run on separate machines/GPUs simultaneously)
python scripts/run_with_server.py --step inference --task precisewikiqa \
  --shards 4 --shard-id 0 \
  --generations-file-path output/.../shard_0_of_4/generation.jsonl \
  --activations-path output/.../shard_0_of_4/activations.zarr
# ... (workers 1, 2, 3 analogous with --shard-id 1/2/3)

# Inside each inference worker:
#   load full QA file (≥400 items)
#   assign item_index = 0..len-1
#   slice: items where item_index % shards == shard_id  → ~100 items
#   exp.run_exp() → writes shard_N/generation.jsonl (each record includes item_index)
#   write shard_N/shard.done

# Phase 2: consolidate (after all inference workers complete)
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
4. **Generation overshoot**: Final QA count may exceed N by at most `num_workers` items. Acceptable.
5. **No seeding in generation workers**: Python's `random` module must use OS-entropy initialization. Never add `random.seed()` to generation workers — doing so would cause workers to produce identical question sets when sampling the same documents.

---

## Verification

1. **Unit test**: run 2 workers on TriviaQA with N=10, shard-index=0/1, verify each shard has ~5 records with no overlap, consolidate and verify 10 records in correct order.
2. **Resume test**: interrupt a worker mid-run, restart with same args, verify no duplicate records in the shard file.
3. **Correctness**: compare metrics from single-process run vs. sharded+consolidated run on same N — results must be identical.

---

## Concrete Implementation: Generation Sharding

### CLI arguments

Two shared arguments used by both generation and inference steps:

```
--shards N     Total number of parallel workers. Required to enable sharding.
--shard-id i   This worker's index (0-based, 0 ≤ i < N). Must be specified when --shards is used.
```

Behaviour by step:
- **generate**: `shards` enables parallel mode (filelock + global count check). `shard-id` is required for consistency and used in logging; it does not partition generation work (workers sample freely).
- **inference**: `shard-id` determines which items this worker processes (`item_index % shards == shard_id`).

### Overview of changes

Three small, contained changes. Nothing else touches.

| File | Change |
|------|--------|
| `utils/generate_question.py` | Add `global_N` + `lock_path` params to `per_bin_generation_batch`; add `n_shards=None` to `precise_QA_generation_run_batch` and `longform_QA_generation_run_batch` |
| `tasks/shortform/precise_wikiqa.py` | Pass `n_shards=kwargs.get("shards")` into `precise_QA_generation_run_batch` |
| `scripts/run_with_server.py` | Add `--shards` / `--shard-id` args; validate `shard-id` is set when `shards` is set; pass `shards` and `shard_id` through kwargs |

`filelock` must be available (`pip install filelock`; add to `requirements.txt` if not already present).

---

### Change 1: `utils/generate_question.py`

#### `per_bin_generation_batch` — add global stop + filelock

Current signature (line 244):
```python
def per_bin_generation_batch(self, wiki_data, output_path, N, progress_bar=None):
```

New signature:
```python
def per_bin_generation_batch(self, wiki_data, output_path, N, progress_bar=None,
                              global_N=None, lock_path=None):
```

- `global_N`: when set, the real stop condition is `lines_in_file(output_path) >= global_N` rather than `len(local_QAs) >= N`
- `lock_path`: path for the `FileLock`; defaults to `output_path + ".lock"` when `global_N` is set

**Two targeted edits inside the method body:**

**(a) Chunk-loop guard** — replace the existing early-exit check at the top of the loop (line 257):

```python
# before (line 256-258):
for chunk_start in range(0, len(wiki_data), chunk_size):
    if len(QAs) >= N:
        break
```
```python
# after:
for chunk_start in range(0, len(wiki_data), chunk_size):
    if len(QAs) >= N:
        break
    if global_N is not None:
        try:
            with jsonlines.open(output_path) as _f:
                global_count = sum(1 for _ in _f)
        except Exception:
            global_count = 0
        if global_count >= global_N:
            break
```

This reads the file (no lock — a stale count is fine since overshoot of `≤ num_workers` is acceptable) before spending time on the next chunk.

**(b) Write section** — replace the bare write at lines 358-360:

```python
# before (lines 358-360):
with jsonlines.open(output_path, 'a') as writer:
    writer.write(valid_data[i])
```
```python
# after:
if global_N is not None:
    from filelock import FileLock
    _lp = lock_path or (output_path + ".lock")
    with FileLock(_lp):
        with jsonlines.open(output_path, 'a') as writer:
            writer.write(valid_data[i])
else:
    with jsonlines.open(output_path, 'a') as writer:
        writer.write(valid_data[i])
```

The filelock is held only for the duration of the append — milliseconds. It is not held during LLM calls.

#### `precise_QA_generation_run_batch` — add `n_shards` param

Current signature (line 370):
```python
def precise_QA_generation_run_batch(
        wiki_input_path, N=5000, q_generator=..., output_path="",
        from_scratch=False, max_workers=1, log_file=None):
```

New signature — add one parameter:
```python
def precise_QA_generation_run_batch(
        wiki_input_path, N=5000, q_generator=..., output_path="",
        from_scratch=False, max_workers=1, log_file=None, n_shards=None):
```

- `n_shards=None`: single-process mode, unchanged behaviour
- `n_shards=N` (N ≥ 2): parallel mode — passes `global_N=N` down to `per_bin_generation_batch`

In the bin loop body (line 421), change the `per_bin_generation_batch` call:

```python
# before (line 421):
bin_QAs = qa.per_bin_generation_batch(wiki_data, output_path, per_level_count, progress_bar=pbar)
```
```python
# after:
bin_QAs = qa.per_bin_generation_batch(
    wiki_data, output_path, per_level_count, progress_bar=pbar,
    global_N=(already_completed + effective_N) if n_shards is not None else None,
)
```

Note: `already_completed` and `effective_N` are already in scope at this point. `already_completed + effective_N` equals the original `N` argument, which is the absolute file target. Passing `effective_N` alone would be wrong in the resume case (the file already has `already_completed` lines, so `file_lines >= effective_N` would trigger immediately).

No other changes to `precise_QA_generation_run_batch`.

#### `longform_QA_generation_run_batch` — same pattern

Add `n_shards=None` to signature. Pass `global_N=N if n_shards is not None else None` to the `per_bin_generation_batch` call (line 474). Also remove the `assert len(bin_QAs) == per_level_count` (line 475) when `n_shards is not None` — early stops mean a bin may return fewer than `per_level_count` items.

---

### Change 2: `tasks/shortform/precise_wikiqa.py`

In the generate step call to `precise_QA_generation_run_batch` (around line 555), thread through the new param:

```python
precise_qa.precise_QA_generation_run_batch(
    wiki_input_path=...,
    N=remaining,
    output_path=QA_OUTPUT_PATH,
    from_scratch=False,
    max_workers=max_workers_qgen,
    n_shards=kwargs.get("shards"),   # ← add this line; None when not sharding
)
```

`longwiki_main.py` gets the analogous change for `longform_QA_generation_run_batch`.

---

### Change 3: `scripts/run_with_server.py`

Add two CLI args to the argparse block:

```python
parser.add_argument("--shards", type=int, default=None,
                    help="Total number of parallel workers.")
parser.add_argument("--shard-id", type=int, default=None,
                    help="This worker's shard index (0-based). Required when --shards is set.")
```

Add validation after parsing:

```python
if args.shards is not None and args.shard_id is None:
    parser.error("--shard-id is required when --shards is set")
if args.shard_id is not None and args.shards is None:
    parser.error("--shards is required when --shard-id is set")
```

Pass both through to task kwargs:

```python
kwargs["shards"] = args.shards      # None when not sharding
kwargs["shard_id"] = args.shard_id  # None when not sharding
```

---

### Worker invocation

All generation workers run the **identical command** except for `--shard-id`.

```bash
# Run on each machine simultaneously (shard-id differs per machine):
python scripts/run_with_server.py \
  --step generate \
  --task precisewikiqa \
  --N 400 \
  --shards 4 \
  --shard-id 0 \        # 1, 2, 3 on the other machines
  --generations-file-path shared/qa_output.jsonl \
  --log-file shared/server.log

# Workers stop autonomously when shared/qa_output.jsonl reaches 400 lines (±4).
# No consolidation step needed — the output is a single shared file.
```

The shared filesystem must support `FileLock` (requires `fcntl`-style locking — works on NFS with proper mount options, or any local/shared POSIX FS).

---

### Invariants preserved

- **No seeding**: no `random.seed()` is added anywhere. Each worker process gets independent OS-entropy random state.
- **Single-process mode unchanged**: when `--shards` is not set (`n_shards=None`), all code paths are identical to today. The `global_N` and `lock_path` parameters are never activated.
- **Resume**: `already_completed` is still computed from the file line count at startup — works correctly for parallel workers (each reads the same file, gets the same count, sets the same tqdm initial value; this is cosmetic and harmless).
- **Bin stratification**: retained as a soft guideline. In parallel mode, all workers iterate bins 8→9 in order. With enough workers and fast LLM calls, both bins will be sampled roughly equally before the global N is hit.
