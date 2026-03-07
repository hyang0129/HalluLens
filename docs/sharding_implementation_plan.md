# Sharding Implementation Plan

## Overview

Enable multiple nodes to run the same experiment config in parallel with no coordinator.
Each node is assigned `--shard-id` and `--total-shards`. All nodes independently sort
the input data the same way and interleave ownership by index.

**Core invariant:** given the same input file and `total_shards`, every node derives the
identical ordered list and owns a non-overlapping interleaved slice.

```
Sorted master list:  [item_0, item_1, item_2, item_3, item_4, item_5, ...]
Shard 0 owns:        [item_0,         item_2,         item_4,         ...]
Shard 1 owns:                [item_1,         item_3,         item_5, ...]
```

Resume is free: on restart a shard reads its own output, identifies completed items by
their stable key (`pageid` for generate, prompt identity for inference), and skips them.

---

## New Parameters

Added to `run_with_server.py` CLI and `run_experiment()` / `run_task_step()` signatures:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--shard-id` | `0` | This node's shard index |
| `--total-shards` | `1` | Total number of shards (1 = no sharding, identical to today) |
| `--seed` | `42` | Seed for within-article randomness (section selection) |

When `total_shards == 1`, all behaviour is identical to the current codebase.

---

## Master List Construction

### Generate step (wiki articles)

Current code: `sample(n=per_level_count, replace=True)` + non-deterministic shuffle.

New approach — direct interleave on sorted data:

```python
level_wiki = wiki_data_all[wiki_data_all['h_score_cat'] == bin]
level_wiki = level_wiki.sort_values('pageid').reset_index(drop=True)
shard_wiki  = level_wiki.iloc[shard_id::total_shards].reset_index(drop=True)
wiki_data   = shard_wiki.to_dict(orient='records')
```

Each shard owns a non-overlapping set of `pageid`s — ownership is implicit from position,
no tagging required.

**Resume:** read existing `qa.jsonl`, collect the set of `pageid`s already written,
skip any item in `wiki_data` whose `pageid` is already present.

```python
done_pageids = set()
if os.path.exists(output_path) and not from_scratch:
    with jsonlines.open(output_path) as reader:
        done_pageids = {rec['pageid'] for rec in reader}
wiki_data = [r for r in wiki_data if r['pageid'] not in done_pageids]
```

**Section selection** within an article uses a seeded RNG keyed to the article so it is
deterministic across re-runs:

```python
rng = random.Random(seed + record['pageid'])
section = rng.choice(sections)
```

**Target per shard:** each shard targets `ceil(per_level_count / total_shards)` accepted
QAs per bin. The last shard may produce slightly fewer if the bin does not divide evenly;
this is acceptable — total QAs across shards will be ≥ N.

### Inference step (QA rows)

```python
# Load full qa.jsonl — same on all nodes
QAs_df = pd.DataFrame([line for line in jsonlines.open(QA_OUTPUT_PATH)])
# ... existing quality filters ...
QAs_df = QAs_df.sort_values(['pageid', 'question']).reset_index(drop=True)

if total_shards > 1:
    QAs_df = QAs_df.iloc[shard_id::total_shards].reset_index(drop=True)
```

Resume is handled by the existing `exp.run_exp` logic which checks
`generations_file_path` for already-completed prompts.

---

## Output Strategy

### Generate step

All shards write to the **same shared `qa.jsonl`** protected by a `filelock`.
No merge step needed — the file is the final output once all shards finish.

Records are written as today (no new fields required). The `pageid` field already present
in each record is sufficient for resume.

```python
from filelock import FileLock

with FileLock(output_path + ".lock"):
    with jsonlines.open(output_path, 'a') as writer:
        writer.write(record)
```

### Inference step

Zarr cannot support concurrent multi-writer, so each shard writes to its own subdirectory:

```
output/run_name/
  shard_0/
    generation.jsonl
    activations.zarr
  shard_1/
    generation.jsonl
    activations.zarr
  ...
```

After all shards finish, `scripts/merge_shards.py` produces the unified output.

---

## File-by-file Changes

### `requirements.txt`

```
filelock>=3.12.0
```

---

### `scripts/run_with_server.py`

**`main()` argparse additions:**
```python
parser.add_argument('--shard-id',     type=int, default=0)
parser.add_argument('--total-shards', type=int, default=1)
parser.add_argument('--seed',         type=int, default=42)
```

**`run_experiment()` / `run_task_step()` signature additions:**
```python
def run_experiment(..., shard_id=0, total_shards=1, seed=42): ...
def run_task_step(..., shard_id=0, total_shards=1, seed=42): ...
```

`run_task_step` forwards `shard_id`, `total_shards`, `seed` as kwargs into the task
module's `run_step()`.

---

### `tasks/shortform/precise_wikiqa.py`

**`run_step()` signature addition:**
```python
def run_step(step, model, ..., shard_id=0, total_shards=1, seed=42):
```

**Generate step:** pass `shard_id`, `total_shards`, `seed` to
`precise_QA_generation_run_batch()`. `qa_output_path` is unchanged (shared file).

**Inference step:** apply interleave after existing filters, then derive shard-local
output paths:

```python
if total_shards > 1:
    QAs_df = QAs_df.sort_values(['pageid', 'question']).reset_index(drop=True)
    QAs_df = QAs_df.iloc[shard_id::total_shards].reset_index(drop=True)

    shard_tag = f"shard_{shard_id}"
    if generations_file_path:
        base, fname = os.path.split(generations_file_path)
        generations_file_path = os.path.join(base, shard_tag, fname)
    if activations_path:
        activations_path = os.path.join(activations_path, shard_tag)
```

**Eval step:** not sharded — runs once on merged data.

---

### `utils/generate_question.py`

**`precise_QA_generation_run_batch()` signature addition:**
```python
def precise_QA_generation_run_batch(
    wiki_input_path, N=5000, q_generator=..., output_path="",
    from_scratch=False, max_workers=1, log_file=None,
    shard_id=0, total_shards=1, seed=42,  # NEW
):
```

**Replace non-deterministic sampling with sorted interleave** (inside the per-bin loop):
```python
level_wiki = wiki_data_all[wiki_data_all['h_score_cat'] == bin]
level_wiki = level_wiki.sort_values('pageid').reset_index(drop=True)
shard_wiki  = level_wiki.iloc[shard_id::total_shards].reset_index(drop=True)
wiki_data   = shard_wiki.to_dict(orient='records')
```

**Replace count-based resume with pageid-based resume:**
```python
done_pageids = set()
if os.path.exists(output_path) and not from_scratch:
    with jsonlines.open(output_path) as reader:
        done_pageids = {rec['pageid'] for rec in reader}
wiki_data = [r for r in wiki_data if r['pageid'] not in done_pageids]
```

**Replace `random.sample` for section selection with seeded RNG:**
```python
rng = random.Random(seed + record['pageid'])
section = rng.choice(sections)
```

**Add filelock around the shared JSONL write** (replaces bare `jsonlines.open(..., 'a')`):
```python
from filelock import FileLock

with FileLock(output_path + ".lock"):
    with jsonlines.open(output_path, 'a') as writer:
        writer.write(record)
```

---

### `scripts/merge_shards.py` (new file)

Merges per-shard inference outputs into a single unified store.

```bash
python scripts/merge_shards.py \
    --output-dir output/run_name \
    --total-shards 4
```

**Steps:**

1. **Merge `generation.jsonl`**
   Concatenate `shard_*/generation.jsonl` line by line. Re-sort by `(pageid, question)`
   to restore the original QA ordering.

2. **Merge `activations.zarr`**
   For each shard store (read-only):
   - Concatenate `arrays/prompt_activations` and `arrays/response_activations` along axis 0
   - Concatenate all 1-D metadata arrays (`prompt_len`, `response_len`, `sample_key`, etc.)
   - Write into a new unified zarr store
   - Merge `meta/index.jsonl` files with remapped `sample_index` (0-based, contiguous)

3. **Sanity check**
   Assert sum of per-shard row counts equals merged row count.
   Assert no duplicate keys in merged `index.jsonl`.

---

## Usage Examples

### Single node (unchanged)
```bash
python scripts/run_with_server.py \
    --step all --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 1000
```

### Multi-node inference (4 shards)
```bash
# All nodes: identical config except --shard-id
python scripts/run_with_server.py \
    --step inference --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 1000 --seed 42 \
    --shard-id 0 --total-shards 4 \          # change per node
    --activations-path shared/run/activations.zarr \
    --generations-file-path shared/run/generation.jsonl
```

Output layout:
```
shared/run/shard_0/generation.jsonl
shared/run/shard_0/activations.zarr
shared/run/shard_1/generation.jsonl
shared/run/shard_1/activations.zarr
...
```

### After all shards complete
```bash
python scripts/merge_shards.py \
    --output-dir shared/run \
    --total-shards 4
```

---

## What Does Not Change

- `exp.run_exp` resume logic (operates per shard's local `generation.jsonl`)
- `ZarrActivationsLogger` (each shard has its own instance)
- Eval step (runs once on merged output)
- All other tasks (`triviaqa`, `naturalquestions`, `longwiki`) — shard params flow
  through `run_task_step` as kwargs and are ignored until those tasks are updated

---

## Open Questions

1. **Other tasks:** add shard support for `longwiki`, `triviaqa`, `naturalquestions` in
   a follow-up once the precisewikiqa pattern is validated.
