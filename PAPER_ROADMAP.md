# Roadmap: From Notebook to Publishable Paper Results

This document is a **practical roadmap** for turning the current HalluLens codebase (activation logging + contrastive training + evaluation) into a **repeatable experiment pipeline** that produces **multi-benchmark, multi-seed, publishable results**.

## 0) North Star

### Primary claim we want to validate
A contrastive representation learning method trained on intermediate-layer activations yields **stronger hallucination detection** than a simple last-layer classifier, and generalizes across benchmarks.

### Current observed results (to formalize)
- TriviaQA: **80+ AUROC** (contrastive method)
- PreciseWikiShort / PreciseWikiQA-short: **70+ AUROC** (contrastive method)
- Baseline to beat: **classifying from last layer activations** (already supported in `activation_research/model.py` + `train_halu_classifier` in `activation_research/training.py`).

### Paper-style deliverables
- A unified experiment runner that can:
  - generate (prompt, answer) pairs,
  - log activations with stable IDs,
  - attach ground-truth hallucination labels,
  - train/evaluate over multiple seeds and hyperparameters,
  - emit tables/figures with confidence intervals.

## IDEAS_TO_TEST

This section summarizes concrete variations to explore (with brief theory justification) for the current contrastive-representation + distance-based hallucination detection pipeline.

### A) Scoring / OOD detection variations (inference-time)
1. **Replace Mahalanobis with KNN distance in embedding space**
  - Motivation: Mahalanobis assumes a unimodal Gaussian embedding distribution; contrastive embeddings can be multi-modal.
  - Test: score each example by its distance to the $k$-th nearest neighbor in the *training* embedding set (tune $k$ on dev).
2. **Relative Mahalanobis distance (control for “input difficulty”)**
  - Motivation: absolute distance can conflate generic “unusualness” with hallucination-specific deviation.
  - Test: fit (i) non-hallucination Gaussian and (ii) background Gaussian over all training points; score as $d(x;\mu_{id},\Sigma_{id}) - d(x;\mu_{bg},\Sigma_{bg})$.
3. **Multi-layer Mahalanobis fusion with learned weights**
  - Motivation: different layers carry different hallucination signal; uniform averaging assumes equal informativeness.
  - Test: compute per-layer distances and learn a linear combiner (logistic regression) on dev.
4. **Score fusion: Mahalanobis + inter-layer agreement**
  - Motivation: “distance from typical” and “layer-wise inconsistency” are distinct signals.
  - Test: combine Mahalanobis with negative cosine similarity between the two layer embeddings.

### B) Representation choices (what activations we embed)
5. **Answer-tokens-only pooling (response-only) vs prompt+response**
  - Motivation: hallucination signal should concentrate in generated tokens; prompt tokens add noise.
  - Test: use activation logging `sequence_mode='response'` and compare to `sequence_mode='all'`.
6. **Token selection ablations**
  - Motivation: mean pooling may dilute localized failure modes.
  - Test grid: last-token only; mean-pool; max-pool; attention-weighted pooling (learned weights over tokens).
7. **Residual-stream deltas instead of raw activations**
  - Motivation: $\Delta_l = h_l - h_{l-1}$ isolates what layer $l$ *adds*, potentially making hallucination-specific computation more separable.
  - Test: log/construct deltas and train the same compressor on $(\Delta_{l1}, \Delta_{l2})$ views.

### C) Contrastive objective / training variations
8. **Layer pair sweep and “best pair” discovery**
  - Motivation: the strongest signal may come from specific layer pairs (where internal representations diverge most under hallucination).
  - Test: sweep $(l_1, l_2)$ across early/mid/late ranges; report best AUROC and stability across seeds.
9. **Multi-view contrastive (K > 2 layers as views)**
  - Motivation: more views provide richer constraints than a single pair.
  - Test: use $K$ layers as $n_{views}=K$ in SupConLoss and compare to $K=2$.
10. **Temperature sweep / learnable temperature / alternative SSL losses**
  - Motivation: optimal temperature depends on embedding geometry; avoid brittle tuning.
  - Test: $\tau \in \{0.01, 0.05, 0.07, 0.1, 0.25, 0.5\}$; optionally learn $\tau$; consider VICReg as a no-negatives baseline.
11. **Asymmetric encoders (BYOL/SimSiam-style) across layers**
  - Motivation: different layers are not true “random augmentations”; separate encoders can reduce representational mismatch.
  - Test: online encoder + target encoder (EMA) with stop-gradient on target branch.
12. **Layer-conditioned encoder (single encoder + layer ID embedding / FiLM)**
  - Motivation: keep a single shared encoder for parameter efficiency, but let it adapt to systematic distribution shifts across layers.
  - Test: pass `layer_id` to the compressor and add a learned layer embedding to token states (or use FiLM-style scale/shift per block).
  - Expected outcome: approximates “separate encoders” while still sharing most parameters; should improve alignment when layers have different statistics.

### D) Analysis / interpretability (to understand *why* it works)
13. **Spectral analysis of covariance in embedding space**
  - Motivation: Mahalanobis emphasizes low-variance directions; identify whether separation lives in high- or low-variance subspaces.
  - Test: compute AUROC using distances restricted to top-$k$ vs bottom-$k$ eigen-directions; track which components dominate.

### Suggested priority order (best effortokto-impact first)
- KNN scoring; response-only pooling; relative Mahalanobis; learned layer-weight fusion.
- Then: residual deltas; cosine+Mahalanobis fusion; temperature sweep.
- Finally: K>2 views; asymmetric encoders; spectral analysis (interpretability-first).

## 1) What We Already Have (Repo Reality)

### Activation logging + inference (data generation)
- OpenAI-compatible server and activation logging infrastructure: `activation_logging/`.
- vLLM-compatible serving entry point: `activation_logging/vllm_serve.py` and docs in `activation_logging/README.md`.
- Task runners:
  - PreciseWikiQA: `tasks/shortform/precise_wikiqa.py`
  - TriviaQA: `tasks/triviaqa/triviaqa.py`
  - LongWiki: `tasks/longwiki/longwiki_main.py`
- Unified server-managed run script: `scripts/run_with_server.py` and helper bash wrappers in `scripts/`.

### Research models + training
- Contrastive training: `train_contrastive` in `activation_research/training.py`.
- Last-layer classifier baseline: `train_halu_classifier` in `activation_research/training.py`.
- Evaluation utilities: `activation_research/evaluation.py`, `activation_research/metrics.py`.

### External baseline reference suite (LLMsKnow)
- External repo vendored in: `external/LLMsKnow/`.
- Includes baseline methods and multi-dataset evaluation scripts:
  - `logprob_detection.py` (logprob baseline)
  - `p_true_detection.py` (p_true baseline)
  - probing scripts (`probe.py`, `probe_all_layers_and_tokens.py`, etc.)

## 2) Benchmarks: Breadth Target

We should target a benchmark breadth **similar to or greater than** LLMsKnow, while also including HalluLens-native benchmarks.

### A) HalluLens tasks already in this repo
1. **PreciseWikiQA (short-form factual QA)**
   - Task: `tasks/shortform/precise_wikiqa.py`
   - Planned reporting: AUROC/AUPRC on hallucination label; also calibration (ECE) if feasible.
2. **PreciseTruthfulQA**
   - Task: `tasks/shortform/precise_truthfulqa.py`
   - Planned reporting: AUROC and analysis by question category.
3. **TriviaQA (HalluLens wrapper)**
   - Task: `tasks/triviaqa/triviaqa.py`
   - Planned reporting: AUROC; and stratify by question difficulty/answer length.
4. **LongWiki / FactHalu / retrieval variants**
   - Task: `tasks/longwiki/longwiki_main.py`, `tasks/longwiki/facthalu.py`
   - Planned reporting: hallucination detection on long-form answers; consider chunk-level labels if available.

### B) LLMsKnow benchmark suite (integration target)
From `external/LLMsKnow/README.md`, the suite includes:
- TriviaQA
- Movies
- HotpotQA
- Winobias
- Winogrande
- NLI (MNLI)
- IMDB
- Math
- Natural Questions

Goal: run our method on as many of these as feasible, at least covering **QA + classification + reasoning** categories.

### C) Add-on benchmarks (paper-strengthening)
Not currently integrated, but worth adding for breadth:
- TruthfulQA (already partially present via `precise_truthfulqa.py`)
- HaluEval / FactScore-style evaluations (if we can define consistent labels)
- FEVER-like fact verification (if we can derive binary correctness signals)

## 3) Baselines + “Known SOTA” to Compare Against

### A) Baselines already in-repo (must include)
1. **Last-layer classifier baseline**
   - Train a classifier on last-layer activations only.
   - This is explicitly the baseline you mentioned.
2. **Simple heuristic baselines** (low-cost, strong sanity checks)
   - Response length, average token logprob (if available), entropy proxies.

### B) Baselines from LLMsKnow (strong must-include)
From `external/LLMsKnow/README.md`:
- **Logprob detection** (`logprob_detection.py`)
- **p_true detection** (`p_true_detection.py`)
- **Probing baselines** (layer/token probes; heatmaps; etc.)

### C) “Known SOTA” (to verify via web search)
We should list what we currently *know*, and explicitly note a required literature sweep.

Known relevant references already mentioned in the root README:
- “LLMs Know More Than They Show” (Orgad et al.)
- “Detecting LLM Hallucination Through Layer-wise Information Deficiency”

Action item:
- **Search the web for current SOTA hallucination detection** for each benchmark family (QA, long-form, open-ended), and for each detection setting (with/without access to internal activations).
  - Maintain a table of: paper, method, datasets, metrics, constraints (access to logits/activations), and reported numbers.

Important framing for a fair SOTA comparison:
- Many “SOTA” methods may assume access to:
  - multiple generations, self-consistency, retrieval, external verifiers, or proprietary models.
- Our comparisons should be categorized:
  - **Activation-based** (closest to our setup)
  - **Logit/uncertainty-based**
  - **External verifier / retrieval-based**

## 4) What “Publishable” Requires (Beyond a Single AUROC)

### A) Multi-seed reliability
- Run each training configuration across multiple seeds (e.g., 5–10).
- Report mean ± 95% CI, or bootstrap intervals.

### B) Proper splits and leakage control
- Define train/dev/test splits per benchmark.
- Ensure no overlap in prompts/questions across splits.
- Ensure activation datasets are built from the correct split only.

### C) Consistent labeling definition
- Define “hallucination” label per benchmark:
  - For QA: incorrect / unsupported answer (often derived from exact answer matching or annotation).
  - For long-form: need either sentence-level or response-level correctness.

### D) Calibration + thresholding
- AUROC is threshold-free; add at least one thresholded metric:
  - Accuracy/F1 at a chosen threshold (on dev, evaluated on test)
  - AUPRC if class imbalance is significant
  - Calibration (ECE) if we intend operational claims

### E) Ablations (minimum set)
- Which layers are used (early/middle/late)
- Token selection (last token, mean pooling, exact-answer token if available)
- Contrastive objective variants (temperature, positive/negative mining)
- Data size scaling (N = 1k, 10k, 50k)

## 5) “Notebook → Script” Engineering Plan

The goal is to replace notebook state with a **reproducible CLI**.

### A) Standardize experiment configuration
- Add a config system (YAML/JSON) or a CLI with argparse.
- Single experiment config should include:
  - benchmark/task name
  - model ID and inference backend (`vllm`)
  - activation logging settings (layers, tokens, storage format)
  - training hyperparameters (lr, temperature, batch sizes, epochs)
  - seed(s)
  - output directory structure

### B) Canonical artifacts (what each run must produce)
For each (benchmark, model, seed, method):
- `generations.jsonl` (prompt, response, metadata)
- activation store (LMDB/JSON+NPY/Zarr)
- evaluation results (per-example + aggregate)
- training logs (loss curves, metrics)
- a single “run manifest” with all parameters and git commit hash

### D) Results schema (contract)
To make experiments repeatable and aggregatable, every run should conform to a shared on-disk contract:
- Directory convention: `runs/{date}/{method}/{model}/{benchmark}/seed_{seed}/`
- Required files: `config.json`, `run_manifest.json`, `generations.jsonl`, `predictions.csv`, `eval_metrics.json`, `train_metrics.jsonl`
- Required invariants: stable per-example IDs, consistent split tags, and unambiguous linkage between
  `generations.jsonl` ↔ activations store ↔ labels.

The authoritative spec lives in `RESULTS_SCHEMA.md` (this is what the runner + aggregator should enforce).

### C) Aggregation and reporting
- A script that scans an experiment directory and produces:
  - a single `results.csv`
  - paper tables (LaTeX/Markdown)
  - plots (AUROC per benchmark, ablations)

## 6) Suggested Output Directory Convention

Adopt a stable naming scheme so you can rerun easily:

- `runs/{date}/{method}/{model}/{benchmark}/seed_{seed}/`
  - `config.json`
  - `train_metrics.jsonl`
  - `eval_metrics.json`
  - `predictions.csv`
  - `artifacts/` (checkpoints, plots)

This avoids accidental overwrites and makes aggregation trivial.

## 7) Experimental Matrix (What We Should Run)

### Stage 1: Lock in correctness + logging
- Run a small N (e.g., 100–500) for each benchmark to validate:
  - stable IDs
  - activations logged
  - labels correctly attached

### Stage 2: Establish baselines
Per benchmark (PreciseWikiQA-short, TriviaQA, …):
- Last-layer classifier baseline
- LLMsKnow logprob and p_true baselines (where applicable)

### Stage 3: Contrastive method sweep
- Start with a small grid:
  - temperature ∈ {0.07, 0.1, 0.25}
  - lr ∈ {1e-6, 1e-5}
  - representation choice: last-token vs mean-pool
- Seeds ∈ {0, 5, 26, 42, 63} (matches LLMsKnow convention; adjust as needed)

### Stage 4: Breadth expansion
- Scale to more benchmarks (LLMsKnow suite) once pipeline is stable.

### Stage 5: Ablations + analysis
- Layer sensitivity
- Token sensitivity
- Dataset size scaling
- Generalization (train on one benchmark, test on another)

## 8) Immediate Next Steps (Concrete)

1. **Freeze benchmark definitions**
   - Confirm exact benchmark names: “PreciseWikiShort” vs “PreciseWikiQA” variants.
   - Confirm label generation logic for each benchmark.
2. **Write a unified training script**
   - Turns notebook training into a CLI run with seeds + hyperparams.
3. **Write an experiment driver**
   - One command runs: inference → activation logging → dataset build → train → eval.
4. **Add an aggregator**
   - Produces the tables/plots needed for a paper.
5. **Literature sweep for SOTA**
   - Create `SOTA_TRACKER.md` / spreadsheet with verified numbers + settings.

## 9) Risks / Common Failure Modes (Plan for Them)

- **Data leakage** (train/test overlap) → enforce split-aware paths and hashes.
- **Non-determinism** → seed everything (PyTorch, numpy, dataloader workers).
- **Activation storage blow-up** → standardize on LMDB or JSON+NPY with caps on tokens/layers.
- **Benchmark mismatch** → document label definition and evaluation scripts per benchmark.

---

## Appendix: “Known SOTA” Placeholder Table (To Fill After Web Search)

| Benchmark | Best known method | Metric | Reported | Notes / constraints | Source |
|---|---:|---:|---:|---|---|
| TriviaQA | (TBD) | AUROC | (TBD) | access to logits/activations? | web search |
| PreciseWikiQA-short | (TBD) | AUROC | (TBD) | dataset definition clarity | web search |
| LongWiki | (TBD) | (TBD) | (TBD) | long-form labeling | web search |

