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

### 3.0) Viable Methods (near-term baselines)

These are the methods we can likely implement and compare in the current pipeline, given available artifacts and our **response-level binary classification** scope.

**Key: what the selfcheck pipeline now provides** (see `SELFCHECKGPT_INVESTIGATION.md`):
- `selfcheck_samples.jsonl`: k stochastic text responses per prompt, plus token-level logprobs (→ full sequence log-likelihood) for every sample, even text-only ones
- Zarr: activations for up to N stochastic samples per prompt
- The above is an append-only step on any existing run — no re-inference needed

| Method / Paper | Bucket | Implementation difficulty | Unblocked by selfcheck? | Notes | Link |
|---|---|---|---|---|---|
| LLMs Know More Than They Show (Orgad et al.) | Activations / Logits | NA | No (already viable, single-pass) | Includes probing + baselines like `p_true` and logprob | https://arxiv.org/abs/2410.02707 |
| SelfCheckGPT (Manakul et al., 2023) | Black-box / text-only | easy | **Yes — primary motivation** | Needs k samples ✓ + text ✓. Variants: n-gram (no extra model), BERTScore (needs `bert_score`), NLI (needs DeBERTa MNLI). | https://arxiv.org/abs/2303.08896 |
| Semantic Entropy (Farquhar et al., 2024) | Logits/uncertainty-based | medium | **Yes — now unblocked** | Needs: k samples ✓, sequence log-likelihoods ✓ (sum of `token_logprobs` from JSONL). **One missing piece: semantic equivalence clustering** (DeBERTa MNLI forward pass on sample pairs — offline, CPU-feasible). This is a minor post-processing step on `selfcheck_samples.jsonl`. Code: https://github.com/jlko/semantic_uncertainty | https://arxiv.org/abs/2402.09733 |
| Semantic Entropy Probes (Kossen et al., 2024) | Activations / Logits | medium | **Yes — unblocked once SE is computed** | SE labels (from above) serve as probe training targets; hidden states at last-token / pre-EOS positions already in Zarr. Same DeBERTa clustering step unlocks both SE and SEP. **SEP is then a linear probe on existing activations — no new inference.** Code: https://github.com/OATML/semantic-entropy-probes | https://arxiv.org/abs/2406.15927 |
| Semantic Energy (Ma et al., 2025) | Logits/uncertainty-based | medium | **Yes — unblocked** | Needs: k samples ✓, token-level logprobs ✓, semantic clustering via `TIGER-Lab/general-verifier`. Same structure as SE but uses a different verifier. Can share the DeBERTa clustering step or use their verifier. Repo is notebook-only; port to offline scorer over `selfcheck_samples.jsonl`. Code: https://github.com/MaHuanAAA/SemanticEnergy | https://arxiv.org/abs/2508.14496 |
| Geometry of Truth / Layer-wise Semantic Dynamics (Mir, 2025) | Activations / Logits | medium | No (already viable, single-pass) | Uses mean-pooled hidden states across layers vs a gold truth string. Feasible offline from existing activation logs for QA tasks with a reference answer. | https://arxiv.org/abs/2510.04933 |

### 3.0.1) The one minor additional change that unlocks a cluster of methods

**Add a semantic equivalence clustering step** (offline, run once after selfcheck sampling):

1. For each prompt, take the k=20 answer texts from `selfcheck_samples.jsonl`.
2. Run DeBERTa MNLI (or `cross-encoder/nli-deberta-v3-base`) on all O(k²) answer pairs to classify as entailing/contradicting/neutral.
3. Cluster entailing answers into semantic equivalence classes.
4. Store cluster IDs and sequence log-likelihoods per sample back into `selfcheck_samples.jsonl`.

This single CPU-feasible post-processing step (DeBERTa is ~400M params, no GPU needed for k=20) directly enables: **Semantic Entropy, SEP, Semantic Energy, Hallucination Detection on a Budget, Kernel Language Entropy, and Beyond Semantic Entropy** — all from the same already-generated samples.

### 3.1) Additional candidate methods — updated viability assessment

| Method / Paper | Bucket | Viability update | Notes | Link |
|---|---|---|---|---|
| Hallucination Detection on a Budget (Ciosek et al., 2025) | Logits/uncertainty-based | **Now viable** | Efficient SE estimation from fewer samples. With k=20 already generated, can run full SE and study the sample-efficiency curve at no extra inference cost. Same DeBERTa clustering step required. | https://arxiv.org/abs/2504.03579 |
| Beyond Semantic Entropy (Nguyen et al., 2025) | Logits/uncertainty-based | **Now viable** | Pairwise semantic similarity between samples to improve UQ. Needs k samples ✓, pairwise similarity (BERTScore or sentence embeddings — can reuse BERTScore from SelfCheckGPT-BERTScore variant). No new inference. | https://arxiv.org/abs/2506.00245 |
| Kernel Language Entropy (Nikitin et al., 2024) | Logits/uncertainty-based | **Likely viable** | Kernel-based entropy over sample distribution. Needs k samples ✓, logprobs for weighting ✓, semantic similarity kernel (BERTScore or embedding). Shares tooling with Beyond-SE. Need to verify exact inputs. | https://arxiv.org/abs/2405.20003 |
| Layer-wise Information Deficiency | (TBD) | Unchanged — needs investigation | Need to check whether multi-sampling or logprobs are required | https://arxiv.org/html/2412.10246v1 |
| Semantic Reformulation Entropy (Tong et al., 2025) | Logits/uncertainty-based | Not viable | Requires reformulating questions (different model calls), not just sampling answers to the same question | https://arxiv.org/abs/2509.17445 |
| Real-Time Detection of Hallucinated Entities (Obeso et al., 2025) | Activations / Logits | Out of scope | Focuses on token/span localization; our scope is binary response-level classification. | https://arxiv.org/abs/2509.03531 |
| Map of Misbelief (Hajji et al., 2025) | Activations / Logits | Not viable | Traces hallucinations using attention patterns; no public code compatible with our setup | https://arxiv.org/abs/2511.10837 |
| Two Pathways to Truthfulness (Luo et al., 2026) | Activations / Logits | Not viable | Mechanistic/intrinsic framing; not directly a detector | https://arxiv.org/abs/2601.07422 |
| Can LLMs Predict Their Own Failures? (Ghasemabadi & Niu, 2025) | Activations / Logits | Hard — unchanged | Internal circuits framing; setting verification still needed | https://arxiv.org/abs/2512.20578 |
| Neural Probe-Based Hallucination Detection (Liang & Wang, 2025) | Activations / Logits | Not viable | No public code found | https://arxiv.org/abs/2512.20949 |

**TODO (web search):** add the latest best-performing hallucination detectors for QA and long-form factuality. Track whether they require retrieval/verifiers/multi-sampling.

### 3.1) Intrinsic hallucination detection: method taxonomy (what to track)

This project is focused on **intrinsic/knowledge hallucinations** (testing what the model *knows* and whether it can *signal uncertainty*), not primarily on retrieval grounding or multimodal object hallucination.

When adding SOTA methods, prefer methods that:
- do not require external evidence or retrieval (or at least clearly separate that setting),
- evaluate on knowledge-intensive QA / factual generation,
- report AUROC/AUPRC for “is this answer correct/faithful?”.

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
- **Out of scope for SOTA tracking:** these methods localize hallucinated spans/entities during generation. Our scope is **binary response-level classification** (hallucinated vs not), not localization.

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

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes (splits/labeling) | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.2 PreciseWikiQA / PreciseWikiShort

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes (dataset definition) | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | Must clarify “PreciseWikiShort” naming | (TBD) |

### 4.3 LongWiki / FactHalu

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes (label granularity) | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | (TBD) | sentence vs response-level labels | (TBD) |

### 4.4 Movies

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.5 HotpotQA

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.6 Winobias

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.7 Winogrande

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.8 NLI (MNLI)

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.9 IMDB

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

### 4.10 Math

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | answer extraction matters | (TBD) |

### 4.11 Natural Questions

| Method | Bucket | Implementation difficulty | Model(s) | Metric(s) | Reported | Inference budget | Notes | Source |
|---|---|---|---|---|---:|---|---|---|
| (TBD) | (TBD) | (TBD) | (TBD) | AUROC | (TBD) | (TBD) | (TBD) | (TBD) |

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

