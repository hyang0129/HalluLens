# SOTA Tracker (Hallucination Detection)

This file is a **living tracker** for identifying and fairly comparing **state-of-the-art (SOTA)** hallucination detection methods against our approach.

Goal: for each benchmark family we evaluate on, record the **best known results** *and the experimental constraints* so we can make apples-to-apples comparisons.

---

## 0) How to Use This Tracker

For each benchmark:
1. Add candidate papers/methods to the **Candidate Methods** table.
2. For each method, fill in:
   - what access it assumes (activations, logits, black-box, retrieval, verifier, multiple samples)
   - what datasets and splits it uses
   - what metric(s) it reports
   - what model(s) it evaluates
3. Mark whether the comparison is:
   - **Comparable** (same access + same dataset definition + similar inference budget)
   - **Partially comparable** (differences exist; note them)
   - **Not comparable** (fundamentally different setting)

**Important:** Many “SOTA” claims depend on extra resources (retrieval, external verifiers, self-consistency, many samples). We should report these clearly rather than forcing a single leaderboard.

---

## 1) Comparison Buckets (Define Fair Groups)

We will group methods into buckets:

### A) Activation-based (white-box)
- Uses internal hidden states / layer activations.
- Closest to our setup.

### B) Logit/uncertainty-based (semi-white-box)
- Uses token logprobs, entropy, margins, etc.

### C) Black-box / text-only
- Uses only the generated text (and possibly the prompt).

### D) Retrieval/verifier/ensemble
- Uses external evidence sources, separate verifier models, or multiple generations.

When we present results, we should either:
- compare within the same bucket, or
- present bucketed leaderboards (recommended for paper clarity).

---

## 2) Benchmarks We Plan to Report

### HalluLens-native
- PreciseWikiQA (and/or PreciseWikiShort variant)
- PreciseTruthfulQA
- TriviaQA (HalluLens wrapper)
- LongWiki / FactHalu
- Refusal/nonsense stress tests (optional; mostly for analysis/robustness)

### LLMsKnow suite (integration target)
- TriviaQA
- Movies
- HotpotQA
- Winobias
- Winogrande
- NLI (MNLI)
- IMDB
- Math
- Natural Questions

---

## 3) Candidate Papers / Methods (Seed List)

These are already referenced in the repo and are likely key comparators:

| Method / Paper | Bucket | Notes | Link |
|---|---|---|---|
| LLMs Know More Than They Show (Orgad et al.) | Activations / Logits | Includes probing + baselines like `p_true` and logprob | https://arxiv.org/abs/2410.02707 |
| Layer-wise Information Deficiency (title in repo README) | (TBD) | Need to check setting + datasets | https://arxiv.org/html/2412.10246v1 |

Additional intrinsic/knowledge hallucination detection methods (representative papers):

| Method / Paper | Bucket | Notes | Link |
|---|---|---|---|
| SelfCheckGPT (Manakul et al., 2023) | Black-box / text-only | Sampling-based consistency checking; zero-resource | https://arxiv.org/abs/2303.08896 |
| Semantic Entropy Probes (Kossen et al., 2024) | Logits/uncertainty-based | “Semantic entropy” style UQ for hallucination detection; often requires multiple samples | https://arxiv.org/abs/2406.15927 |
| Kernel Language Entropy (Nikitin et al., 2024) | Logits/uncertainty-based | Semantic-similarity-based entropy variants | https://arxiv.org/abs/2405.20003 |
| Hallucination Detection on a Budget (Ciosek et al., 2025) | Logits/uncertainty-based | Efficient estimation of semantic entropy from fewer samples | https://arxiv.org/abs/2504.03579 |
| Beyond Semantic Entropy (Nguyen et al., 2025) | Logits/uncertainty-based | Pairwise semantic similarity to improve UQ | https://arxiv.org/abs/2506.00245 |
| Semantic Reformulation Entropy (Tong et al., 2025) | Logits/uncertainty-based | Reformulation-based semantic entropy for QA hallucination detection | https://arxiv.org/abs/2509.17445 |
| Semantic Energy (Ma et al., 2025) | Logits/uncertainty-based | Alternative UQ signal “beyond entropy” framing | https://arxiv.org/abs/2508.14496 |
| Real-Time Detection of Hallucinated Entities (Obeso et al., 2025) | Activations / Logits | Entity-level hallucination detection in long-form generation | https://arxiv.org/abs/2509.03531 |
| Geometry of Truth / Layer-wise Semantic Dynamics (Mir, 2025) | Activations / Logits | Single-pass, layer-wise geometric/representation dynamics approach (claims to beat SelfCheckGPT/semantic entropy baselines in its setting) | https://arxiv.org/abs/2510.04933 |
| Map of Misbelief (Hajji et al., 2025) | Activations / Logits | Traces intrinsic/extrinsic hallucinations using attention patterns | https://arxiv.org/abs/2511.10837 |
| Two Pathways to Truthfulness (Luo et al., 2026) | Activations / Logits | Mechanistic framing of internal truthfulness signals (intrinsic encoding) | https://arxiv.org/abs/2601.07422 |
| Can LLMs Predict Their Own Failures? (Ghasemabadi & Niu, 2025) | Activations / Logits | “Self-awareness via internal circuits” framing (needs setting verification) | https://arxiv.org/abs/2512.20578 |
| Neural Probe-Based Hallucination Detection (Liang & Wang, 2025) | Activations / Logits | Probe-based detector trained on internal representations (needs setting verification) | https://arxiv.org/abs/2512.20949 |

**TODO (web search):** add the latest best-performing hallucination detectors for QA and long-form factuality. Track whether they require retrieval/verifiers/multi-sampling.

### 3.1) Intrinsic hallucination detection: method taxonomy (what to track)

This project is focused on **intrinsic/knowledge hallucinations** (testing what the model *knows* and whether it can *signal uncertainty*), not primarily on retrieval grounding or multimodal object hallucination.

When adding SOTA methods, prefer methods that:
- do not require external evidence or retrieval (or at least clearly separate that setting),
- evaluate on knowledge-intensive QA / factual generation,
- report AUROC/AUPRC or abstention-risk metrics for “is this answer correct/faithful?”.

Recommended taxonomy for intrinsic hallucination detection methods:

1) **Uncertainty from probabilities/logits (white-box-ish)**
- Typical signals: token-level logprob, perplexity, entropy, margin, rank-calibration.
- “Semantic uncertainty” variants: cluster/merge paraphrased answers or sample multiple generations, then compute entropy in *semantic space*.
- Notes to track: number of samples required; whether semantic clustering uses another model.

2) **Sampling-based self-consistency (black-box)**
- Generate multiple answers; flag hallucination if answers vary or conflict.
- SelfCheckGPT-style methods sit here.
- Notes to track: compute cost; dependence on temperature and sampling.

3) **Representation/activation probes (white-box)**
- Train a classifier/probe on internal activations to predict correctness/hallucination.
- Includes token-selection insights (truthfulness concentrated in specific tokens) and layer/token probes.
- Notes to track: which layers/tokens, whether it is single-pass, whether it generalizes across datasets.

4) **Reasoning-process inspection (intrinsic, usually text-only)**
- Detect inconsistencies between answer and reasoning trace, or reasoning validity indicators.
- Notes to track: whether chain-of-thought is required; evaluation leakage risk.

Non-target (track separately, do not mix into intrinsic leaderboards):
- Retrieval/verifier-based (RAG) hallucination detection.
- Vision-language “object hallucination” detection.
- Domain-specific factuality checkers that rely on external KBs.

---

## 4) Overlap with HalluLens / “Our Approach”

HalluLens (this repo) is best understood as **infrastructure for white-box intrinsic hallucination research**:
- It runs OpenAI-protocol inference via a vLLM server and **logs internal activations + prompt/response** to disk.
- It is designed to link **(prompt, response, activations)** to **benchmark evaluation outcomes** (hallucinated vs not).

That means we overlap heavily with SOTA methods that rely on **internal states**, and we partially overlap with methods that rely on **sampling-based uncertainty**.

### 4.1) Where we directly overlap (strong alignment)

1) **Representation/activation-probe methods**
- SOTA pattern: train a classifier/probe over hidden states (often layer-wise, token-wise) to predict correctness/hallucination.
- Our overlap: we already log per-request activations (configurable across layers and across prompt/response tokens).
- What this enables: reproducing “probe over layer $\ell$ at token $t$” baselines; ablations over layers/tokens; single-pass detectors.

2) **Layer-wise dynamics / geometry-of-truth style methods**
- SOTA pattern: compute geometry/statistics on representations across layers/tokens (without necessarily training a heavy probe).
- Our overlap: we store the raw layer activations needed to compute these signals offline.

3) **Entity-level hallucination detection**
- SOTA pattern: detect hallucinated entities during generation.
- Our overlap: we log response text + (optionally response-token activations), so we can align activations to entity spans and train token/span-level detectors.

### 4.2) Where we partially overlap (needs small extensions or protocol choices)

1) **Semantic-entropy / uncertainty methods**
- SOTA pattern: generate multiple samples and compute uncertainty in semantic space.
- Current overlap: server supports temperature/top_p control; logging persists each request.
- Missing for “full parity”: a first-class multi-sample runner and standardized storage of “grouped samples per prompt” (plus any semantic clustering metadata).

2) **Logprob/logit-based detectors (cheap single-pass)**
- SOTA pattern: use token logprobs, entropy, margins, calibrated confidence.
- Current overlap: we run via an OpenAI-protocol endpoint; depending on server support, we may or may not be returning/storing token logprobs.
- Missing: if logprobs aren’t logged today, we’d need to extend the logging payload to include logprobs (and define schema).

### 4.3) Where we intentionally *do not* overlap (separate bucket)

- Retrieval/verifier based factuality detection: different resource assumptions; should be compared separately.
- Pure black-box text-only methods: can be baselined, but not the core claim of this repo.

### 4.4) Likely “novel contribution” framing relative to SOTA

Even if individual detectors exist in the literature, HalluLens can claim novelty in the **end-to-end, benchmark-integrated activation logging pipeline**:
- OpenAI-protocol inference → consistent activation capture → evaluation labels → disk artifacts that support reproducible probing.
- A research platform for studying *when/where* hallucination signals appear in internal states (layer/token granularity), rather than only reporting a detector number.

---

## 4) Per-Benchmark SOTA Table (Fill as We Search)

### 4.1 TriviaQA

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes (splits/labeling) | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.2 PreciseWikiQA / PreciseWikiShort

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes (dataset definition) | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | Must clarify “PreciseWikiShort” naming | (TBD) |

### 4.3 LongWiki / FactHalu

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes (label granularity) | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | sentence vs response-level labels | (TBD) |

### 4.4 Movies

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.5 HotpotQA

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.6 Winobias

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.7 Winogrande

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.8 NLI (MNLI)

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.9 IMDB

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.10 Math

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | answer extraction matters | (TBD) |

### 4.11 Natural Questions

| Method | Bucket | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

---

## 5) “Comparable?” Checklist (Fill Per Method)

For each method you add, answer:
- [ ] Same benchmark definition / preprocessing?
- [ ] Same split (train/test), or at least standard splits?
- [ ] Same model family/size, or do we need to rerun it?
- [ ] Same access constraints (activations vs logits vs black-box)?
- [ ] Similar inference budget (single sample vs multi-sample vs verifier)?
- [ ] Metric is AUROC (or convertible / comparable)?

If any box is unchecked, write a one-line caveat in the method’s Notes.

---

## 6) Where to Run Baselines in This Repo (Pointers)

- Last-layer baseline: `activation_research/training.py` (`train_halu_classifier`)
- Contrastive method: `activation_research/training.py` (`train_contrastive`)
- LLMsKnow baselines: `external/LLMsKnow/src/`
  - `logprob_detection.py`
  - `p_true_detection.py`
  - `probe.py`

