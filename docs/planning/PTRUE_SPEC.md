# P(true) Self-Evaluation Baseline — Implementation Spec

**Tracking issue:** [#50](https://github.com/hyang0129/HalluLens/issues/50)
**Paper:** Kadavath et al. 2022, "Language Models (Mostly) Know What They Know" ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221)), §3.2 "P(True)".
**Reference implementation:** [external/LLMsKnow/src/p_true_detection.py](../../external/LLMsKnow/src/p_true_detection.py)
**Roadmap context:** [PAPER_ROADMAP.md](../../PAPER_ROADMAP.md) — listed as a K=1-cluster baseline in the compute-matched comparison and as job 4 in the schedule.

---

## 1. Method

A prompt-based detector: ask the model whether its own greedy answer is correct, then read the binary `(A) True` vs. `(B) False` logits.

**Prompt template** (Kadavath, identical to LLMsKnow):

```text
Question: {question}
Possible answer: {answer}
Is the possible answer:
(A) True
(B) False
The possible answer is:
```

**Reversed variant** (sanity check for token-position bias):

```text
Question: {question}
Possible answer: {answer}
Is the possible answer:
(A) False
(B) True
The possible answer is:
```

**Score:**
1. Tokenize the prompt with the model's standard tokenizer (left-padded for batching).
2. Forward pass with `max_new_tokens=1`, capture `scores[0]` (logits for the first generated token).
3. `p_true = softmax(scores[0][[tok(" A"), tok(" B")]])[0]` for forward, `[1]` for reversed.
4. Higher `p_true` ⇒ model claims its answer is correct ⇒ should anti-correlate with `halu_label`.

**Prompt format decision:** raw string (no chat template). Matches Kadavath and LLMsKnow exactly, and the trailing `"The possible answer is:"` gives the model a leading-space context so `" A"` / `" B"` are single-token decodes on both Llama-3 and Qwen3 tokenizers. Acceptance test verifies this per model (§7).

---

## 2. Scope

**12 cells:** 6 datasets × 2 models, test split only. **All cells uncapped** — every dataset runs its full test partition.

| Dataset | Test rows | Notes |
|---|---|---|
| HotpotQA | 7,405 | Full test split |
| Natural Questions | 4,155 | Full test split |
| PopQA | 2,854 | Full test split |
| SciQ | 1,000 | Full test split |
| SearchQA | ~151,200 | Full pool — 151,140 (Llama) / 151,295 (Qwen3); see note below |
| MMLU | 10,225 | Full test split |
| **Per model** | **~176,839** | |
| **× 2 models** | **~353,678 forward passes** | |
| **+ reversed pass** | **~707,356 forward passes** | doubled for token-position-bias check |

> **No SearchQA cap.** Sampling baselines use [`output/searchqa_test_cap_<model>_seed42.json`](../../tasks/sampling_baselines/paths.py#L88-L89) (10K rows, [`SEARCHQA_TEST_CAP`](../../scripts/compute_subsets.py#L33)) because K=10 stochastic generations + semantic entropy on 151K is prohibitively expensive. P(true) is ~10× cheaper (one forward pass, `max_new_tokens=1`) and deterministic, so the cap brings no benefit here. Running uncapped tightens the SearchQA AUROC estimate by ~√15 ≈ 4× and removes a footnote from the main table. Sample-size differences between cells are handled correctly by the per-cell bootstrap CIs in §7.

> **SearchQA naming convention (historical inversion).** The "test" pool — `output/searchqa/{model}/generation.jsonl` — currently holds the HF **train** split (~151K rows, both Llama and Qwen3), and `output/searchqa_train/{model}/` holds the HF **test** split (~43K rows). This inversion is encoded as the canonical layout in [`scripts/audit_datasets.py:35-36`](../../scripts/audit_datasets.py#L35-L36) ("Qwen3 convention"). A partial fix in commit `f39d615` updated `generate_all_qwen3.sh` and added [`scripts/swap_searchqa_qwen3_splits.py`](../../scripts/swap_searchqa_qwen3_splits.py), but the swap was never run and the on-disk state remains inverted for both models. P(true) consumes this pool as-is. A real path/data rename is out of scope for this issue.

Models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`.

**Train split is out of scope.** P(true) is a detector (no training); only test-split AUROC is needed for the main table.

**No seed axis.** P(true) is deterministic given the (already-generated) greedy answer A: the readout prompt has no sampling, `max_new_tokens=1`, and reads logits directly. One run per (model, dataset) — no seed averaging needed.

---

## 3. GPU Time Estimate

Per example: one prefill of ~150–250 tokens (question + greedy answer + template), `max_new_tokens=1`. On H200 at `batch_size=64`, 8B-class models sustain ~50–100 ex/s for this shape.

| Phase | Wall (single H200) | Notes |
|---|---|---|
| Model load × 2 | 1–2 min | Load Llama, run 6 datasets, free, load Qwen3, run 6 datasets |
| Llama inference (forward + reversed, 6 datasets) | ~60–120 min | ~353,678 fwd passes / ~60 ex/s; SearchQA dominates (~302K of those) |
| Qwen3 inference (forward + reversed, 6 datasets) | ~60–120 min | Same shape |
| **Total end-to-end (one H200, serialized)** | **~2–4 hours** | |
| **3-node fanout** | **~50–90 min wall** | Round-robin 12 cells over 3 nodes; SearchQA cells dominate, so balance by putting SearchQA on its own node |

Comfortably under issue #50's stated ~6 GPU-hour budget. SearchQA accounts for ~85% of forward passes; the other 5 datasets together are rounding error.

---

## 4. File Layout

Mirror the `sampling_baselines` package layout for consistency with existing baselines.

```
tasks/p_true/
  __init__.py
  paths.py              # ptrue_scores_path(dataset, model_id, split) -> Path
  scorer.py             # PTrueScorer class — batched fwd pass, fwd + reversed

scripts/
  run_p_true_pass.py    # per-(dataset, split, model) driver, mirrors run_sampling_pass.py
  run_p_true_cell.sh    # bash wrapper, mirrors run_sampling_baselines_cell.sh
  run_p_true_shard.sh   # iterate cells in a shard file
  fanout_p_true.py      # multi-node fanout, mirrors fanout_sampling_baselines.py
  run_p_true_for_model.py  # OPTIONAL optimization: load model once, run all 6 ds
```

**Output paths** (one record per row, resumable):
```
output/p_true/{dataset_dir}/{model_name}/ptrue.jsonl
```
where `dataset_dir = "natural_questions"` for `nq`, `"searchqa"` for `searchqa`, etc., and `model_name = model_id.split("/")[-1]`. Train-split variant (not needed for the main table but pathing should support it) would write to `{dataset}_train/{model_name}/ptrue.jsonl`.

---

## 5. Module APIs

### `tasks/p_true/paths.py`

```python
from pathlib import Path
from tasks.sampling_baselines.paths import _dir_stem, model_name

def ptrue_output_dir(dataset: str, model_id: str, split: str = "test") -> Path:
    stem = _dir_stem(dataset)
    ds_dir = f"{stem}_train" if split == "train" else stem
    return Path("output") / "p_true" / ds_dir / model_name(model_id)

def ptrue_scores_path(dataset: str, model_id: str, split: str = "test") -> Path:
    return ptrue_output_dir(dataset, model_id, split) / "ptrue.jsonl"
```

### `tasks/p_true/scorer.py`

```python
class PTrueScorer:
    """Single-forward-pass P(true) detector.

    Loads model+tokenizer via activation_logging.server.get_model_and_tokenizer
    (same path as SamplingPass). Verifies " A" / " B" tokenize as single tokens
    on init and caches their IDs.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        ...

    def run(
        self,
        generation_jsonl: str,    # output/{ds}/{model}/generation.jsonl
        output_path: str,         # output/p_true/{ds}/{model}/ptrue.jsonl
        row_indices: list[int] | None = None,   # None = all rows (the default for every dataset)
        labels: list[int] | None = None,         # halu_test_res aligned to row_indices
    ) -> None:
        """For each row, write one JSON line:
            {"row_idx": int, "p_true": float, "p_true_reversed": float,
             "p_a": float, "p_b": float, "halu_label": 0|1}
        Resumable: skips rows whose row_idx is already in output_path.
        """
```

**Prompt construction:**
- `question = row["question"]` if column present else parse from `row["prompt"]`.
- `answer = row["generation"].strip()` — strip leading/trailing whitespace; do not truncate.
- For MMLU, where generations are letter answers ("A", "B", "C", "D"), the prompt still uses the literal generation as `{answer}` (matches Kadavath; the multi-choice nature of MMLU is orthogonal to P(true)).

**Batching:**
- Left-padded, `bs=64` default (parameterized via CLI).
- Forward and reversed prompts are run in separate batches (different prompts, different decoded position is the same — single-token decode).
- Both directions are written in a single output record so one resume marker covers both.

### `scripts/run_p_true_pass.py`

CLI surface mirrors `run_sampling_pass.py`:

```bash
python scripts/run_p_true_pass.py \
    --dataset hotpotqa --split test \
    --model meta-llama/Llama-3.1-8B-Instruct \
    [--batch-size 64] \
    [--smoketest]
```

Behaviour:
1. Resolve `generation_jsonl(dataset, model, split)` and `eval_results_json(dataset, model, split)` via `tasks.sampling_baselines.paths`.
2. Pass `row_indices=None` for every dataset (full test pool, no cap — including SearchQA).
3. Align `halu_label = eval["halu_test_res"][row_idx]` per row.
4. Call `PTrueScorer.run(...)`.
5. `--smoketest`: process first 50 rows only, then assert resumability (re-run is no-op).

### `scripts/run_p_true_for_model.py` (optimization)

Optional but recommended driver that loads the model once and iterates all 6 datasets:

```bash
python scripts/run_p_true_for_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --datasets hotpotqa,nq,popqa,sciq,searchqa,mmlu
```

Saves ~30 s × 5 of model-load overhead per model. Recommended over per-cell `.sh` if we're not parallelizing across nodes for the per-model batch.

---

## 6. Dispatch Plan

### Phase 0: smoke test (10 minutes)

```bash
# On one GPU node, verify tokenization + alignment on one cell, 50 rows.
python utils/jupyter_exec.py "
import subprocess
subprocess.check_call([
    '/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python',
    'scripts/run_p_true_pass.py',
    '--dataset', 'sciq', '--split', 'test',
    '--model', 'meta-llama/Llama-3.1-8B-Instruct',
    '--smoketest',
])
"
```

Acceptance: `output/p_true/sciq/Llama-3.1-8B-Instruct/ptrue.jsonl` has 50 rows, each with both `p_true` and `p_true_reversed` ∈ [0, 1]. Spot-check 5 rows.

### Phase 1: full grid (1 dispatch per node)

Two options, pick by node availability when the time comes:

**A. 3-node fanout** (parallel, ~20 min wall, total ~70 min GPU):

```bash
python scripts/fanout_p_true.py \
    --nodes alphagpu01-8888,alphagpu03-8887,alphagpu04-8884
```

Round-robin 12 cells across 3 nodes → 4 cells / node.

**B. 1-node serial** (~60 min wall, group-by-model optimization on):

```bash
python scripts/gpu_dispatch.py run --jupyter --node alphagpu04-8884 -- \
    bash -c '
        python scripts/run_p_true_for_model.py --model meta-llama/Llama-3.1-8B-Instruct &&
        python scripts/run_p_true_for_model.py --model Qwen/Qwen3-8B
    '
```

### Phase 2: bootstrap CIs + table integration (CPU, < 5 min)

1. Add P(true) to `scripts/assemble_baseline_table.py`:
   - Load `ptrue.jsonl` via `load_jsonl_by_row`.
   - Score column = `-p_true` (high p_true ⇒ low hallu likelihood; matches the table's "higher score = more hallucinated" convention).
   - Also write a `p_true_reversed` column and the per-cell max(AUROC, AUROC_reversed) into the table footer for the bias check.
2. Bootstrap 95% CIs on AUROC: 1000 resamples × 12 cells, save to `output/baseline_results/p_true_bootstrap.json`.

---

## 7. Acceptance Criteria

Mapped to the issue's bullet list:

| Issue criterion | Verified by |
|---|---|
| All 12 cells completed (no fallback models) | `len(ptrue.jsonl) == expected_test_count` for every (dataset, model). Audit script: `scripts/audit_p_true.py --check-counts` |
| Verified tokenization for `" A"` / `" B"` on both models | `PTrueScorer.__init__` asserts `len(tokenizer.encode(" A", add_special_tokens=False)) == 1` and same for `" B"`. Logged at startup. |
| Score direction explicitly documented | Module docstring of `scorer.py` + table column comment in `assemble_baseline_table.py` (`p_true_hallu_score = 1 - p_true`). |
| Spot-check on 20 random examples per combination | `scripts/audit_p_true.py --spot-check 20` prints `(question, answer, p_true, p_true_reversed, halu_label)` for 20 random rows per cell. |
| Bootstrap 95% CIs on AUROC | `scripts/assemble_baseline_table.py --with-ci` (re-used CI machinery from sampling baselines if present, else add a `bootstrap_auroc(scores, labels, n=1000)` helper). |
| Raw scores saved as `ptrue.jsonl` | Output convention §4. |
| Methods doc (~100 words) | Add a "P(true)" subsection to the methods part of the paper draft; reference Kadavath §3.2; describe forward+reversed and the max-AUROC convention. |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Chat-template default mismatch — if `get_model_and_tokenizer` returns a tokenizer with `chat_template` applied implicitly, raw prompts may decode differently | `PTrueScorer.__init__` explicitly tokenizes with `add_special_tokens=False` for the body and confirms the first token after `"The possible answer is:"` predicts to ` A` or ` B` in a 1-example dry run. Fail-fast. |
| Token-position bias — "(A) True" prior may dominate independent of answer quality | Reversed pass on every cell; report `max(AUROC_fwd, AUROC_rev)`. If reversed AUROC is systematically higher, flag in spec follow-up. |
| Generation text contains the literal token sequence `" A"` or `" B"` near the end (e.g. for MMLU letter answers), causing weird interactions | Generations are inserted in the *middle* of the prompt; the scored token is the first one after `"is:"`. MMLU answers like `" A"` are part of `{answer}`, not the scored position. Single-token-decode invariant on the scored position holds. |
| Resume after partial failure double-writes rows | `_load_done_rows()` (already used in `SamplingPass`) by `row_idx`. Identical pattern. |
| Halu labels misaligned with `row_idx` | `run_p_true_pass.py` reads `eval["halu_test_res"]` directly indexed by `row_idx` (no cap-based filtering needed since all rows are processed). Smoketest validates per-row alignment by printing 5 rows. |

---

## 9. Out of Scope

- Train-split P(true) — not needed for the main table.
- Few-shot variants — Kadavath §3.2 also reports few-shot; zero-shot is the simpler baseline and what reviewers expect for the comparison column.
- Calibration metrics (ECE etc.) — covered by [#59](https://github.com/hyang0129/HalluLens/issues/59) at table-assembly time.
- Sampling-based variants of P(true) (Kadavath §3.3 K-sample) — different baseline category, not in this issue.

---

## 10. Definition of Done

- [ ] `tasks/p_true/{paths.py, scorer.py}` implemented and unit-tested.
- [ ] `scripts/run_p_true_pass.py` works end-to-end on `sciq/Llama` smoketest.
- [ ] Either `fanout_p_true.py` or `run_p_true_for_model.py` dispatches all 12 cells.
- [ ] `output/p_true/<ds>/<model>/ptrue.jsonl` exists and is complete for all 12 cells.
- [ ] `assemble_baseline_table.py` includes a `p_true` column with bootstrap 95% CIs.
- [ ] `scripts/audit_p_true.py --spot-check 20` output reviewed for all 12 cells; saved to `reports/p_true_spotchecks/`.
- [ ] Methods paragraph (~100 words) added to paper draft section.
- [ ] PAPER_ROADMAP.md table updated: P(true) baseline status `❌ not started` → `✅ done`.
