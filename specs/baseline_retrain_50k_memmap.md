# Implementation Spec: Apples-to-Apples 50k Re-Train on Memmap Captures

**Status:** draft / proposal
**Depends on:** #72 (merged 2026-05-16, capture writer landed), #75 (open, builds `MemmapContrastiveDataset`)
**Supersedes:** none (new issue)
**Owner:** unassigned
**Branch:** to be created

## 1. Goal

Migrate every trained probe to read training and test data from issue #72's memmap capture layout (`shared/icr_capture/{dataset}_{model_slug}/`) instead of the legacy zarrs, and re-run the per-(dataset × model) baseline comparison so that every trained method is fit on the **same training samples with the same hallucination labels** that the ICR probe is fit on. The HotpotQA and MMLU training cohorts end up capped at 50k samples (because that is what the memmap captures produce for those splits); the other four datasets retain their natural training sizes (all < 50k).

The re-trained cells are added to `results_table.json` **alongside** the existing zarr-trained cells. Both are reported. Old zarr cells become an internal training-corpus ablation view; the memmap-trained cells are the headline apples-to-apples comparison.

## 2. Scope

### 2.1 In scope (Phase 1 of this issue)

- New `MemmapActivationParser` wrapper in `activation_logging/memmap_activation_parser.py` exposing the `ActivationParser` surface that the per-method runners in `scripts/run_experiment.py` depend on (`get_dataset(split, **kwargs)`, `df`, `split_strategy`). Delegates to `MemmapContrastiveDataset` (from #75) and to the two new sibling modes below.
- New dataset modes covering the non-contrastive consumer shapes that #75 does not directly target:
  - `MemmapMultiLayerDataset` — deterministic ordered layers, returns `(num_layers, T, H)`. Mirrors `MultiLayerDeterministicDataset` at [activation_logging/activation_parser.py:435](activation_logging/activation_parser.py#L435).
  - `MemmapSingleLayerView` — fixed-layer `(1, T, H)`. Mirrors `SingleLayerView` at [activation_logging/activation_parser.py:380](activation_logging/activation_parser.py#L380).
  - These may be implemented as separate classes or folded into `MemmapContrastiveDataset` via a `mode="deterministic" / "single_layer"` parameter — whichever fits cleaner once #75's class is final. The spec is shape-driven, not file-layout-driven.
- Backend dispatch in `scripts/run_experiment.py` at the four `ActivationParser(...)` construction sites (lines 2166, 2213, 2249, 2262): when the dataset config has `"backend": "memmap"` and an `"icr_capture"` block, construct `MemmapActivationParser` instead. Same per-method runners; no per-runner code change.
- Phase 1 dataset configs (4 files): hotpotqa, natural_questions, each × Llama and Qwen3.
- Phase 1 experiment configs (4 files): same coverage.
- `scripts/results_table.py` extension to glob `runs/baseline_comparison_*_memmap/` and emit cells with a distinguishing key suffix.
- Unit tests for the wrapper and the new dataset modes.
- Phase 1 dispatch & verification plan.

### 2.2 In scope (Phase 2 of this issue, executed after Phase 1 lands)

- mmlu, popqa, sciq, searchqa memmap configs + re-training — as each dataset's icr_capture reaches a usable state.
- Move SEP's hotpotqa training to read from icr_capture (one dataset; the other datasets SEP runs against are already < 50k or skipped).

### 2.3 Out of scope

- New trained methods. Issue #75 owns the new contrastive+logprob+attn model.
- SmolLM3 — no icr_capture data exists for any dataset on SmolLM3. SmolLM3 retains its zarr-trained cells. Re-capture is a separate operational task.
- Sampling baselines (`token_entropy`, `logprob_baseline`, all `se_*`, all `selfcheck_*`, `p_true`) — they do not train on the data, so the cap does not change their scores.
- Re-capturing missing or partial icr_capture data (popqa stub, sciq partial, mmlu Qwen3 in progress) — this PR consumes whatever icr_capture contains; it does not produce captures.
- Deleting old zarrs — they stay for as long as their cells remain in `results_table.json`.

## 3. Dependencies

- **#72** (PR #73, merged 2026-05-16) — provides `InferenceCaptureWriter` and the memmap layout under `shared/icr_capture/`. ✓
- **#75** (open) — provides `MemmapContrastiveDataset`, the per-item dict contract, and the synthetic-capture helper used by tests. This PR composes #75's class and waits for it to merge before landing wiring code. Capped configs and the dispatch plan can be drafted in parallel.

The icr_capture entries `prompt_hash`/`sample_index`/`hallucinated` in `meta.jsonl` and the synthesized `eval_results.json` (array format) provide everything the existing probe trainers and evaluators consume after metadata synthesis (see §6.2 below).

## 4. Audit basis (re-verified on remote 2026-05-17)

Training- and test-sample counts per `(dataset, model)`. Numbers re-checked
on the remote via `ssh empire-ai "cd LLM_research/HalluLens; wc -l shared/icr_capture/*/meta.jsonl"`.

| Dataset (train) | Llama icr_capture | Qwen3 icr_capture | Test capture (both models) | Ready |
|---|---:|---:|---:|:---:|
| hotpotqa          | 49,996 (capped) | 49,996 (capped) |  7,405 | ✓ |
| natural_questions | 16,617 (full)   | 16,617 (full)   |  4,155 | ✓ |
| mmlu              | 49,666 (~cap)   | 37,062 (74% of cap, in progress) | 10,271 | partial |
| popqa             | 10,595 (full)   | 10,595 (full)   |  2,797 | ✓ |
| sciq              | 11,609 (full)   | 11,609 (full)   |  **1,000** (small — see §12.5) | ✓ |
| searchqa          | 49,959 (capped) | 49,959 (capped) | 21,609 | ✓ |

For reference, the old-zarr train-set sizes (used by the existing baselines)
were: hotpotqa 90k/137k (Llama/Qwen3), natural_questions 16,617 on both,
mmlu 94k/98k, popqa 11,189/10,595, sciq 11,455/11,609, searchqa 42,783/43,216.

**Single-phase rollout (revised from original two-phase plan).** 11 of 12
`(dataset, model)` cells are ready to sweep now. Only `mmlu × Qwen3` remains
in progress (~13k samples to go). The two-phase boundary in the original draft
was justified by partial popqa/sciq/searchqa captures that have since
completed; collapsing it removes artificial sequencing without changing any
code path. `mmlu × Qwen3` dispatches whenever its capture completes — same
config, same runner.

## 5. Methods to re-train

From `configs/experiments/baseline_comparison_*.json`:

| Method | Trained? | Re-run on memmap? | Dataset shape consumed |
|---|:---:|:---:|---|
| `contrastive`              | yes | yes | `(K, T, H)` views — `MemmapContrastiveDataset` |
| `contrastive_logprob_recon`| yes | yes | views + response logprob fields |
| `linear_probe`             | yes | yes | `(1, T, H)` single layer — `MemmapSingleLayerView` or `num_views=1, fixed_layer=...` |
| `saplma`                   | yes | yes | confirm in audit (§6.1) |
| `saplma_logprob_recon`     | yes | yes | confirm in audit (§6.1) |
| `llmsknow_probe`           | yes | yes | logprob-feature path; see §6.1 |
| `multi_layer_linear_probe` | yes | yes | deterministic `(num_layers, T, H)` — `MemmapMultiLayerDataset` |
| `sep`                      | yes (separate dispatch via `scripts/compute_sep.py`) | only where the 50k cap actually shrinks the training set — hotpotqa (both models) and mmlu (both); see §8.3 | natural_questions, popqa, sciq are under 50k already so SEP zarr scores remain valid; searchqa is at the cap (49,959) but the zarr was already at 43k — borderline, defer (§13.7) |
| `icr_probe`                | yes | **no rerun** (already trains on icr_capture) | — |
| `token_entropy`            | no  | n/a | — |
| `logprob_baseline`         | no  | n/a | — |

Phase 1 training runs: 7 trained methods × 2 datasets × 2 models × 5 seeds = **140 runs** + SEP hotpotqa runs.

## 6. New components

### 6.1 Pre-implementation audit — dataset-shape contract per probe

Before writing the wrapper, audit each per-method runner in `scripts/run_experiment.py` and record:
- Which `Dataset` subclass it ultimately receives from `ap.get_dataset(...)`.
- What keys it reads from each `__getitem__` dict.
- Whether it touches `ap.df` directly (e.g. for the OOD evaluator).

Output goes into a short table at the top of the PR. Drives whether each non-contrastive shape needs a dedicated `Memmap*` sibling class or can be served by `MemmapContrastiveDataset` parameters. This is a 30-minute exercise but it determines the wrapper's surface; doing it up front avoids carrying assumptions into code.

### 6.2 `MemmapActivationParser` — wrapper surface

File: `activation_logging/memmap_activation_parser.py` (new).

```python
class MemmapActivationParser:
    """Wrapper exposing the ActivationParser read surface over the icr_capture layout.

    Constructed from a single capture directory (one split per directory; train and test
    are separate captures, matching the icr_capture convention). Produces datasets via
    .get_dataset(split, ...) that emit the same per-item dict shape as zarr-backed
    datasets, so the existing per-method runners consume it without changes.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        random_seed: int = 42,
        val_fraction: float | None = None,
        split_strategy: Literal["none", "three_way"] = "three_way",
        verbose: bool = False,
    ):
        ...

    @property
    def df(self) -> pd.DataFrame:
        """Synthesizes a DataFrame from meta.jsonl, eagerly at construction.

        Required columns (verified by grep of every `activation_parser_df` and
        `ap.df` consumer in the repo — no other columns are read anywhere):

          - prompt_hash  (str)                  — sample identity; read by
                                                  evaluation.py:369,
                                                  webdataset_option_a.py:141
          - halu         (int/float)            — hallucination label. **Named `halu`,
                                                  NOT `hallucinated`**, to match
                                                  MultiMetricHallucinationEvaluator,
                                                  training.py:342, and the SingleLayer/
                                                  MultiLayer dataset constructors.
                                                  meta.jsonl stores it as `hallucinated`;
                                                  the wrapper renames on load.
          - split        ('train'|'val'|'test') — per-row assignment from the
                                                  stratified split. Required by
                                                  activation_parser.py:1447-1494,
                                                  build_memmap_cache.py:87-89,
                                                  webdataset_option_a.py:142,145.
                                                  Populated eagerly using the same
                                                  `_make_split_indices` call the dataset
                                                  classes use, with the same random_seed.
          - sample_index (int)                  — memmap row index from meta.jsonl
          - prompt_len   (int)                  — from prompt_len.npy
          - response_len (int)                  — from response_len.npy

        No lazy merge with generation.jsonl — nothing in the codebase reads
        generation-derived columns off `ap.df`.
        """

    @property
    def split_strategy(self) -> str: ...

    def get_dataset(
        self,
        split: Literal["train", "val", "test", "all"],
        *,
        # All kwargs accepted for API parity with ActivationParser.get_dataset.
        # Unknown / inapplicable kwargs are silently accepted to keep runners unchanged.
        relevant_layers: list[int] | None = None,
        num_views: int = 2,
        fixed_layer: int | None = None,
        pad_length: int | None = None,
        preload: bool = False,  # accepted; ignored — memmap is the model
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        deterministic: bool = False,
        check_ram: bool = False,
        **kwargs,
    ) -> torch.utils.data.Dataset:
        """Dispatch to MemmapContrastiveDataset / MemmapMultiLayerDataset /
        MemmapSingleLayerView based on the shape requested.
        """
```

Routing logic in `get_dataset()`:
- `deterministic=True` (any `num_views`) → `MemmapMultiLayerDataset`
- `num_views == 1` with `fixed_layer is not None` → `MemmapSingleLayerView` (or equivalent `MemmapContrastiveDataset` params)
- Otherwise → `MemmapContrastiveDataset`

All three accept `capture_dir`, `split`, `val_fraction`, `random_seed` with the same semantics as `ICRDataset` (which #75 reuses). Splits are stratified on `hallucinated`.

### 6.3 `MemmapMultiLayerDataset`

Mirrors `MultiLayerDeterministicDataset` at [activation_logging/activation_parser.py:435-472](activation_logging/activation_parser.py#L435-L472):

```python
def __getitem__(self, idx) -> dict:
    return {
        "hashkey":          self._prompt_hashes[idx],
        "halu":             torch.tensor(float(self._labels[idx]), dtype=torch.float32),
        "views_activations": self._read_response_activations(idx, self.layer_positions),  # (num_layers, T, H)
        "view_indices":     torch.tensor(self.layer_ids, dtype=torch.long),
    }
```

Reads `response_activations.npy` at the configured layer positions; pads/truncates to `pad_length` in the time dimension when set, or `max_response_len` from `config.json` otherwise.

### 6.4 `MemmapSingleLayerView`

Mirrors `SingleLayerDataset` at [activation_logging/activation_parser.py:397-432](activation_logging/activation_parser.py#L397-L432) (the spec previously referenced the wrong class name and line range):

```python
def __getitem__(self, idx) -> dict:
    return {
        "hashkey":          self._prompt_hashes[idx],
        "halu":             torch.tensor(float(self._labels[idx]), dtype=torch.float32),
        "views_activations": self._read_response_activations(idx, [self.layer_pos]),  # (1, T, H)
        "view_indices":     torch.tensor([self.layer_id], dtype=torch.long),
    }
```

### 6.5 Backend dispatch in `run_experiment.py`

Replace each of the four `ActivationParser(...)` constructions with a small helper:

```python
def _make_parser(dataset_cfg, *, activations_path, inference_json, eval_json,
                 random_seed, split_strategy, verbose):
    backend = dataset_cfg.get("backend", "zarr")
    if backend == "memmap":
        capture_dir = dataset_cfg["icr_capture"][...]  # train_dir or test_dir per call site
        return MemmapActivationParser(
            capture_dir=capture_dir,
            random_seed=random_seed,
            split_strategy=split_strategy,
            verbose=verbose,
        )
    return ActivationParser(
        inference_json=inference_json,
        eval_json=eval_json,
        activations_path=activations_path,
        logger_type=backend,
        random_seed=random_seed,
        split_strategy=split_strategy,
        verbose=verbose,
    )
```

Each of the four call sites becomes a single line invocation of this helper.

### 6.6 Wrapper invariants (cross-cutting requirements)

These invariants apply to `MemmapActivationParser` and the dataset classes it
produces. They were missed in the original draft and would cause silent data
loss or runtime errors if not respected.

#### 6.6.1 Train/test capture pairing → split mapping

The icr_capture layout stores train and test in **separate directories**, unlike
the legacy zarr where both live in one store. This breaks the assumption baked
into [`_make_split_indices`](activation_research/icr_dataset.py#L71) (used by
`MemmapContrastiveDataset` at [memmap_contrastive_dataset.py:309-312](activation_research/memmap_contrastive_dataset.py#L309-L312)),
which always allocates 80/10/20 → train/val/test.

| Call site | Construction | `get_dataset(split=...)` | Splitter behaviour |
|---|---|---|---|
| Test parser ([run_experiment.py:2166](scripts/run_experiment.py#L2166)) | `MemmapActivationParser(test_capture_dir, split_strategy="none")` | `get_dataset("test")` | All rows of the test capture; **no `_make_split_indices` call**. |
| Train parser ([run_experiment.py:2213, 2249](scripts/run_experiment.py#L2213)) | `MemmapActivationParser(train_capture_dir, split_strategy="three_way", random_seed=actual_split_seed)` | `get_dataset("train")` / `get_dataset("val")` | **90 / 10** stratified split on the train capture's `halu` labels; seed = `actual_split_seed`. |

**Do NOT delegate to the existing `_make_split_indices` 80/10/20 split unmodified
on train captures.** It would silently discard 20% of the data as a phantom
test set (e.g. 9,999 hotpotqa samples, 3,324 NQ samples per audit §4) because
that 20% is returned only when `split="test"` is requested — which never
happens on a train-capture parser (test rows live in the separate test
capture).

Preferred implementation: add a `splitter: Literal["90_10", "70_10_20"]`
parameter to `MemmapContrastiveDataset` (and the sibling deterministic /
single-layer modes), default `"90_10"`. The wrapper passes the value through
unchanged. Keeping the 70/10/20 mode available preserves the option to
re-create legacy-zarr-style splits for ablations.

#### 6.6.2 `--smoketest-memmap-cache` is disabled for backend=memmap

The smoketest path at [run_experiment.py:2101](scripts/run_experiment.py#L2101)
reads `ap.activation_logger._response_activations.shape[0]`, an attribute that
only exists on zarr-backed `ActivationParser` (see also
[activation_parser.py:1174,1214,1293,1326,1465](activation_logging/activation_parser.py#L1174)
and [build_memmap_cache.py:94,194](scripts/build_memmap_cache.py#L94)). The
memmap-cache machinery exists to materialise a memmap **from** a zarr; for
native icr_capture data the captures already are the memmap and there is
nothing to pre-cache.

The dispatch helper (§6.5) must skip the entire `--smoketest-memmap-cache`
code path when the resolved backend is `"memmap"`. Implementation: in
`run_experiment.py` wrap the existing smoketest branches with
`if dataset_cfg.get("backend") != "memmap":`, and log a one-line "memmap
backend — smoketest skipped" when the flag is set on a memmap config.

#### 6.6.3 Dataset `.df` attribute

[training.py:342](activation_research/training.py#L342) reads
`dataset.df["halu"]` (not `parser.df`). The current `MemmapContrastiveDataset`
does not expose a `df` attribute. Each of the three memmap dataset variants
(contrastive, multi-layer, single-layer) must expose a `.df` property
surfacing the same columns documented in §6.2, **already sliced to
`_split_indices`** (so `len(dataset.df) == len(dataset)`).

#### 6.6.4 `random_seed` plumbing — no production defaults

`MemmapContrastiveDataset(random_seed=42)` carries a default for tests, but
production wiring must always pass the per-fold `actual_split_seed`
explicitly. The wrapper's constructor should require `random_seed` as a
no-default keyword argument so that omitting it is a `TypeError` rather than
a silent fold collapse to seed 42 (which would make every "different seed"
run identical).

#### 6.6.5 `split_strategy` → split mapping in `get_dataset`

Concrete dispatch inside `MemmapActivationParser.get_dataset(split, ...)`:

```python
if self._split_strategy == "none":
    inner_split = "all"       # full capture, no carve
else:                          # "three_way"
    inner_split = split        # "train" | "val" | "test"
    # On a train capture there is no "test" partition by construction;
    # the runner should never request it. Raise if it does, to catch wiring
    # mistakes early.
    if inner_split == "test":
        raise ValueError(
            "MemmapActivationParser was constructed over a train capture "
            "(split_strategy='three_way'); request 'train' or 'val'. "
            "Test rows live in the separate test capture."
        )
```

## 7. Configuration changes

### 7.1 Dataset configs (11 new files; mmlu Qwen3 added later)

```
configs/datasets/
  hotpotqa_memmap.json                    (Llama)
  hotpotqa_qwen3_memmap.json              (Qwen3)
  natural_questions_memmap.json           (Llama)
  natural_questions_qwen3_memmap.json     (Qwen3)
  mmlu_memmap.json                        (Llama; mmlu_qwen3_memmap.json deferred until capture caps)
  popqa_memmap.json                       (Llama)
  popqa_qwen3_memmap.json                 (Qwen3)
  sciq_memmap.json                        (Llama)
  sciq_qwen3_memmap.json                  (Qwen3)
  searchqa_memmap.json                    (Llama)
  searchqa_qwen3_memmap.json              (Qwen3)
```

Schema (example, `hotpotqa_memmap.json`):

```json
{
  "name": "hotpotqa_memmap",
  "model_name": "Llama-3.1-8B-Instruct",
  "input_dim": 4096,
  "backend": "memmap",
  "outlier_class": 1,
  "icr_capture": {
    "train_dir": "shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000",
    "test_dir":  "shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct"
  }
}
```

The existing per-model configs (e.g. `hotpotqa_qwen3.json`) already carry an
`icr_capture` block from the issue #70 wiring; the new `*_memmap.json` configs
additionally set `"backend": "memmap"` to flip the runner. Sibling files (rather
than flipping `backend` on the existing config in place) are required so that
both zarr- and memmap-trained cells coexist in `results_table.json`,
distinguishable by dataset name — see §1, §9.

### 7.2 Experiment configs (11 new files; mmlu Qwen3 added later)

```
configs/experiments/
  baseline_comparison_hotpotqa_memmap.json
  baseline_comparison_hotpotqa_qwen3_memmap.json
  baseline_comparison_nq_memmap.json
  baseline_comparison_nq_qwen3_memmap.json
  baseline_comparison_mmlu_memmap.json                  (mmlu_qwen3 deferred)
  baseline_comparison_popqa_memmap.json
  baseline_comparison_popqa_qwen3_memmap.json
  baseline_comparison_sciq_memmap.json
  baseline_comparison_sciq_qwen3_memmap.json
  baseline_comparison_searchqa_memmap.json
  baseline_comparison_searchqa_qwen3_memmap.json
```

Each clones the corresponding non-memmap config and:
- Sets `"dataset": "<dataset>_memmap"`.
- Drops `token_entropy`, `logprob_baseline`, `icr_probe` from `methods` (sampling baselines unaffected; ICR already memmap-trained).
- Keeps the same 5 `training_seeds` and `split_seeds`.
- Output goes to `runs/baseline_comparison_<dataset>_memmap/`.

## 8. Dispatch plan

### 8.1 Smoketest (before any full sweep)

1 method × 1 seed × 1 dataset × 1 model:

```bash
python scripts/gpu_dispatch.py run --min-vram 30 -- \
  python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa_memmap.json \
    --methods linear_probe \
    --seeds 0
```

Acceptance:
- `eval_metrics.json["n_train"]` equals **90% of the icr_capture train size**
  per the §6.6.1 split rule (44,996 for hotpotqa Llama at 49,996 × 0.9;
  14,955 for natural_questions Llama at 16,617 × 0.9).
- `eval_metrics.json["n_test"]` equals the **full test capture size**
  (7,405 for hotpotqa, 4,155 for natural_questions).
- AUROC is within ±0.05 of the zarr-trained `linear_probe` seed 0 result for
  the same (dataset, model). Larger gaps are expected on hotpotqa
  (training set went from 90k→45k) and acceptable up to ±0.10.
- No errors opening any memmap file in `shared/icr_capture/...`.
- `run_manifest.json` is written and the git commit matches HEAD.

### 8.2 Full dispatch (single phase, 11 cells)

```bash
for cfg in hotpotqa hotpotqa_qwen3 nq nq_qwen3 mmlu \
           popqa popqa_qwen3 sciq sciq_qwen3 searchqa searchqa_qwen3; do
  python scripts/gpu_dispatch.py run --min-vram 30 -- \
    python scripts/run_experiment.py \
      --experiment configs/experiments/baseline_comparison_${cfg}_memmap.json
done
```

11 dispatches; each runs 7 trained methods × 5 seeds = 35 runs. Total:
**385 runs**. Dispatched in waves across available nodes — gpu_dispatch's
auto-selection handles node placement.

`mmlu_qwen3_memmap` dispatched as a 12th cell whenever the capture caps at
~50k (currently at 37k of 50k). No new code, just the missing config files.

### 8.3 SEP migration scope

SEP currently reads from the legacy zarr + `se_labels.json`. **It only needs
re-running when the 50k cap actually reduces the training set** — i.e. when
the icr_capture train size is materially smaller than the old zarr train
size. Affected (dataset × model) cells:

| Dataset | Llama: cap bites? | Qwen3: cap bites? |
|---|---|---|
| hotpotqa | ✓ (90k → 50k) | ✓ (137k → 50k) |
| mmlu | ✓ (94k → 50k) | ✓ (98k → 50k, deferred until cap) |
| searchqa | ✓ (43k → 50k? — under cap actually; **no rerun**) | ✓ (same; **no rerun**) |
| natural_questions, popqa, sciq | no — already under cap | no |

Net SEP re-runs: hotpotqa Llama, hotpotqa Qwen3, mmlu Llama, mmlu Qwen3
(deferred). searchqa is at the cap (49,959) but **was already at 42,783 in
zarr** — the cap doesn't reduce it materially; ICR-fold-consistency may
still warrant a rerun (see §13.7).

```bash
for model in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B; do
  for dataset in hotpotqa mmlu; do
    python scripts/gpu_dispatch.py run --min-vram 30 -- \
      python scripts/compute_sep.py --dataset $dataset --model $model
  done
done
```

Migration requires either:
- (a) Pointing the SEP data loader at the icr_capture directory + a re-derived `se_labels.json` over the 50k subset, or
- (b) Filtering the existing SEP training data to the prompt_hashes present in icr_capture's meta.jsonl.

Time-box: 1 day. If migration is non-trivial, hold SEP at zarr-trained and
mark the cells "zarr-trained; SEP-migration-pending" in `results_table.json`.

### 8.4 Cost estimate

From `results_table_remote.json` historical runs: per (method × seed) wall
time ranges from ~15 min (linear_probe) to ~2 h (contrastive variants).
Mean ≈ 45 min.

- Training (single-phase, 11 cells): 385 runs × 45 min ≈ **290 GPU-hours**.
- 11 dispatches across 4–6 nodes in waves: **~12–18 h wall time**.
- SEP: 4 runs × ~5 min ≈ negligible.

## 9. `results_table.py` changes

Extend the config glob in `scripts/results_table.py` to pick up `_memmap` variants and tag the cells:

```python
for cfg_path in sorted(glob("configs/experiments/baseline_comparison_*.json")):
    cfg = json.load(open(cfg_path))
    is_memmap_variant = cfg["dataset"].endswith("_memmap")
    # Key: keep the suffix in the dataset field so memmap-trained cells are
    # distinguishable from zarr-trained ones in the same table:
    #   {"dataset": "hotpotqa",        ...}  ← zarr-trained
    #   {"dataset": "hotpotqa_memmap", ...}  ← memmap-trained
```

Old cells stay in the table. The CSV gets an additional column-level distinction (or just rely on the dataset suffix).

## 10. Unit tests

### 10.1 `tests/test_memmap_activation_parser.py`

Synthesize a fake icr_capture directory (5 samples, num_layers=4, hidden_dim=8, max_response_len=6, max_prompt_len=8) — reuse #75's `_make_fake_capture()` helper if exported, otherwise duplicate.

Cases:
- `MemmapActivationParser(fake_dir, random_seed=42, split_strategy="three_way")` constructs without error.
- `.df` returns a DataFrame with `prompt_hash`, `hallucinated`, `sample_index`, `prompt_len`, `response_len` columns; row count matches meta.jsonl line count.
- `.get_dataset("train", num_views=2, relevant_layers=[1,2,3])` returns dataset whose `__getitem__(0)` dict has keys `{hashkey, halu, views_activations, view_indices}` and `views_activations.shape == (2, max_response_len, 8)`.
- `.get_dataset("train", num_views=1, fixed_layer=2)` returns dataset with `views_activations.shape == (1, max_response_len, 8)` and `view_indices == [2]`.
- `.get_dataset("train", deterministic=True, relevant_layers=[1,2,3])` returns `views_activations.shape == (3, max_response_len, 8)` and `view_indices == [1,2,3]`.
- `.get_dataset("train", include_response_logprobs=True, response_logprobs_top_k=4)` returns dict with `response_token_logprobs`, `response_topk_logprobs`, `response_topk_ids`, `response_token_mask`.
- Splits `train/val/test` partition the sample set with no overlap.

### 10.2 `tests/test_run_experiment_memmap_dispatch.py`

End-to-end dispatch test:
- Set up a fake icr_capture + fake dataset config + fake experiment config in a tmpdir.
- Run `run_experiment.py --experiment <fake_cfg> --methods linear_probe --seeds 0 --max-epochs 1`.
- Verify `runs/.../seed_0/eval_metrics.json` is written and `n_train` matches the fake capture's train split size.
- Verify `run_manifest.json` records the memmap backend in some discoverable way (or at least records the resolved dataset cfg).

### 10.3 `tests/test_memmap_evaluator_df.py`

Construct `MultiMetricHallucinationEvaluator(activation_parser_df=memmap_ap.df, ...)` against a fake parser; verify no `KeyError` and that the evaluator's `.compute()` runs end-to-end on the toy data.

## 11. Files to create / modify

| File | Action | Purpose |
|---|---|---|
| `activation_logging/memmap_activation_parser.py` | Create | `MemmapActivationParser` wrapper |
| `activation_research/memmap_multi_layer_dataset.py` | Create | `MemmapMultiLayerDataset` (deterministic ordered layers); may be folded into #75's module |
| `activation_research/memmap_single_layer_view.py` | Create | `MemmapSingleLayerView`; may be folded into #75's module |
| `activation_research/memmap_contrastive_dataset.py` | Extend (from #75) | Add `deterministic` and `single_layer` modes if the routing approach lives there |
| `scripts/run_experiment.py` | Modify | Backend dispatch helper + 4 call-site changes |
| `scripts/results_table.py` | Modify | Glob `_memmap` configs/runs, tag cells |
| `configs/datasets/{hotpotqa,hotpotqa_qwen3,natural_questions,natural_questions_qwen3,mmlu,popqa,popqa_qwen3,sciq,sciq_qwen3,searchqa,searchqa_qwen3}_memmap.json` | Create (11 files) | Dataset configs; `mmlu_qwen3_memmap.json` deferred until capture caps |
| `configs/experiments/baseline_comparison_{<same 11>}_memmap.json` | Create (11 files) | Experiment configs, one per dataset cell |
| `tests/test_memmap_activation_parser.py` | Create | Wrapper + dataset-shape tests |
| `tests/test_run_experiment_memmap_dispatch.py` | Create | End-to-end dispatch test |
| `tests/test_memmap_evaluator_df.py` | Create | Evaluator-from-`df` integration |
| `specs/baseline_retrain_50k_memmap.md` | Create (this file) | Implementation spec |

## 12. Verification & acceptance

The work is complete when:
1. All 11 `baseline_comparison_*_memmap.json` configs have produced full `runs/baseline_comparison_*_memmap/<dataset>/<method>/seed_{0..4}/eval_metrics.json` trees. `mmlu_qwen3_memmap` lands separately once its capture caps.
2. `python scripts/results_table.py` emits cells with both `dataset: "hotpotqa"` and `dataset: "hotpotqa_memmap"` (etc), both with `status: complete`, and the CSV view tabulates them side-by-side for the paper.
3. The smoketest acceptance bounds (§8.1) hold across all methods on every cell:
   - AUROC delta vs. zarr-trained counterpart is within ±0.10 per method. Anything beyond that is held and diagnosed before continuing.
4. SEP hotpotqa and mmlu runs are either complete on memmap (preferred) or explicitly marked as `zarr-trained; SEP-migration-pending` in `results_table.json`.
5. **sciq evaluation-size verification (one-off, do before publishing AUROC).** The sciq test capture is 1,000 samples — much smaller than its 11,609 train capture and smaller than every other test capture in the sweep. Confirm this is intentional (likely matches the dataset's canonical test split per `configs/datasets/sciq_test.json`) and not a partial/in-progress capture. If 1k is correct, document the asymmetry in the paper's table caption; if it's a stub, hold sciq cells until the test capture completes.

## 13. Decisions taken (2026-05-17, revised after remote audit)

1. **Use new icr_capture data completely; don't slice old zarrs.** The user-facing argument: the data is already written in memmap form, the dataset class (`MemmapContrastiveDataset`) is being built in #75, and slicing zarrs would create a single-use hack that doesn't compose with future captures. Cost is ~150 LOC of wrapper + wiring vs ~50 LOC for the zarr-slice path; the difference is negligible against the 50+ GPU-hours of re-training.
2. **Keep old zarr-trained cells in `results_table.json`** alongside the new memmap-trained ones. The old cells become an internal training-corpus ablation; the new cells are the paper's apples-to-apples view. No deletion at this stage.
3. ~~**Two-phase rollout.**~~ **Single-phase rollout.** *(revised 2026-05-17 after re-auditing the remote.)* Original draft staged hotpotqa + NQ first because popqa/sciq were stubs/partial. Re-verification on the remote shows 11/12 captures are now complete; only mmlu × Qwen3 remains in progress. Collapsed to one dispatch wave plus a deferred mmlu_qwen3 follow-up. See §4 audit table for current counts.
4. **Old-zarr labels for non-memmap baselines stay frozen.** When we report "memmap-trained vs zarr-trained" the labels differ between the two cohorts because the new inference run produced different generations. This is a feature, not a bug — the comparison is between training corpora as a whole, not feature-by-feature.
5. **90/10 train/val split on train-only captures, NOT the helper's 70/10/20.** *(new, after spotting the split-helper bug — see §6.6.1.)* The legacy zarr's 80/10/20 split assumes train+test are commingled; on separate train/test captures the 20% "test" carve would silently discard real training data. Decision: train captures get 90/10 train/val, test captures are used in full. Implementation: a `splitter` parameter on `MemmapContrastiveDataset` defaulting to `"90_10"`, the wrapper passes it through.
6. **No sibling-config collapse.** Considered flipping `"backend": "memmap"` on the existing dataset configs in place (avoids 11 new files). Rejected because (a) `results_table.py` keys cells by dataset name, so collapsing would mix zarr- and memmap-trained cells under one key and break the side-by-side comparison in §1; (b) the existing zarr-trained cells become hard to re-run for ablations. Cost is 11 small JSON files vs. weeks of paper-side disambiguation.
7. **searchqa SEP rerun deferred.** *(new.)* searchqa Llama is at the 50k cap (49,959) but the old zarr was at 42,783 — the cap doesn't materially shrink the training set, and SEP zarr scores should remain valid. If ICR-fold consistency on searchqa becomes a paper concern, fold this into a Phase 2 SEP migration along with mmlu Qwen3.

## 14. Out of scope follow-ups (separate issues / PRs)

1. **`mmlu_qwen3_memmap` dispatch** once its capture caps (currently 37k of 50k). Same configs/wiring, no new code.
2. **SmolLM3 re-capture + re-train** if SmolLM3 stays in the paper's primary table. Decision deferred; default is to leave SmolLM3 zarr-trained and footnote.
3. **Old-zarr deletion** once all paper cells are sourced from memmap-trained runs.
4. **`compute_sep.py` migration** to read from icr_capture as a first-class data source, retiring the legacy zarr+eval path.

## 15. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `MemmapContrastiveDataset` (from #75) doesn't generalize to the non-contrastive method shapes; more modes than expected | medium | The pre-implementation audit (§6.1) lists every per-method runner's dataset consumption; any gap turns into an explicit mode added in this PR. |
| `MultiMetricHallucinationEvaluator` consumes `df` columns the wrapper doesn't synthesize | medium | Grep the evaluator code, enumerate required columns, ensure `MemmapActivationParser.df` synthesizes all of them. Integration test in §10.3. |
| AUROC drift between zarr- and memmap-trained baselines exceeds plausible bound (>0.10 per method) on Phase 1 | low–medium | If observed, freeze and diagnose before continuing. Likely causes: feature-distribution shift from new inference run, label flips, padding-contract mismatch. |
| icr_capture per-sample ordering differs from the dataset iterator's order (so the new training data isn't the "first 50k" we expect for hotpotqa) | low | Verify on hotpotqa: read first 100 meta.jsonl entries, decode `prompt_hash` against the dataset iterator's first 100 prompts, confirm match. One-off check. |
| HotpotQA Qwen3 capture inherited the 1472-blank-leading-rows anomaly from #72's motivation | low | The new writer's resume semantics + meta.jsonl authoritativeness should prevent this; verify `meta.jsonl` minimum `sample_index == 0` and all rows are contiguous in the smoketest. |
| SEP migration takes longer than 1 day | medium | Time-box (§8.3). If non-trivial, hold SEP at zarr-trained for Phase 1, migrate in Phase 2 follow-up. |
| Old-zarr cells linger in `results_table.json` and confuse paper-side scripts that read it | low | Document the `dataset: foo` vs `dataset: foo_memmap` convention in §9 and in the paper's data-prep notes when written. |
