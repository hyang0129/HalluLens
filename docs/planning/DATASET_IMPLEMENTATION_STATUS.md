# Dataset Implementation Status for HalluLens
## QA-Style Datasets for Intrinsic Hallucination Evaluation

**Generated:** February 4, 2026  
**Purpose:** Track which QA-style datasets for evaluating intrinsic hallucinations are implemented vs. need implementation

---

## ✅ FULLY IMPLEMENTED DATASETS (Ready for Use)

### 1. PreciseWikiQA
- **Type:** Short-form factual QA
- **Status:** ✅ Production Ready
- **Implementation:** [tasks/shortform/precise_wikiqa.py](tasks/shortform/precise_wikiqa.py)
- **Data Location:** `data/precise_qa/save/`, `data/wiki_data/`
- **Evaluation:** Binary correctness (exact answer matching)
- **Activation Logging:** ✅ Fully integrated (JSON/NPY, Zarr, LMDB)
- **Results:** AUROC 70+ (contrastive method)
- **Usage:**
  ```bash
  python -m tasks.shortform.precise_wikiqa \
      --do_inference \
      --do_eval \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --wiki_src goodwiki \
      --mode dynamic \
      --inference_method vllm \
      --N 100
  ```

### 2. TriviaQA
- **Type:** Short-form factual QA (trivia)
- **Status:** ✅ Production Ready
- **Implementation:** [tasks/triviaqa/triviaqa.py](tasks/triviaqa/triviaqa.py)
- **Data Location:** `data/triviaqa-unfiltered/`
- **Data Source:** https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz (auto-download supported)
- **Evaluation:** Multiple answer alias matching
- **Activation Logging:** ✅ Fully integrated
- **Results:** AUROC 80+ (contrastive method)
- **Usage:**
  ```bash
  python -m tasks.triviaqa.triviaqa \
      --do_inference \
      --do_eval \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --inference_method vllm \
      --N 100
  ```

### 3. Natural Questions
- **Type:** Short-form factual QA (real-world queries)
- **Status:** ✅ Production Ready
- **Implementation:** [tasks/llmsknow/natural_questions.py](tasks/llmsknow/natural_questions.py)
- **Data Location:** `external/LLMsKnow/data/nq_wc_dataset.csv`
- **Data Source:** Included in LLMsKnow repository
- **Evaluation:** Substring matching (no LLM judge needed)
- **Activation Logging:** ✅ Fully integrated
- **Results:** TBD (newly implemented)
- **Usage:**
  ```bash
  python scripts/run_with_server.py \
      --step all \
      --task naturalquestions \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --N 1000
  ```

---

## ⚙️ PARTIALLY IMPLEMENTED DATASETS (Need Completion/Testing)

### 4. PreciseTruthfulQA
- **Type:** Short-form factual QA / adversarial truthfulness
- **Status:** ⚙️ Script exists, needs validation
- **Implementation:** [tasks/shortform/precise_truthfulqa.py](tasks/shortform/precise_truthfulqa.py)
- **Data Source:** HuggingFace `datasets` library (TruthfulQA)
- **Evaluation:** Binary correctness, truthfulness assessment
- **Activation Logging:** ⚠️ Needs testing
- **Next Steps:**
  - [ ] Verify data loading from HuggingFace
  - [ ] Test end-to-end pipeline with activation logging
  - [ ] Validate evaluation protocol
  - [ ] Run baseline experiments

### 5. LongWiki / FactHalu
- **Type:** Long-form generation
- **Status:** ⚙️ Scripts exist, evaluation protocol unclear
- **Implementation:** 
  - [tasks/longwiki/longwiki_main.py](tasks/longwiki/longwiki_main.py)
  - [tasks/longwiki/facthalu.py](tasks/longwiki/facthalu.py)
  - [tasks/longwiki/longwiki_retrieval.py](tasks/longwiki/longwiki_retrieval.py)
- **Data Source:** Wikipedia dump (shared with PreciseWikiQA)
- **Evaluation:** ⚠️ Needs definition (sentence-level vs response-level)
- **Activation Logging:** ⚠️ Long-form support TBD
- **Next Steps:**
  - [ ] Define evaluation protocol
  - [ ] Test activation logging for long-form responses
  - [ ] Determine viability for binary classification
  - [ ] Document usage

---

## 📋 NOT IMPLEMENTED - HIGH PRIORITY (LLMsKnow Suite)

The following datasets are available in `external/LLMsKnow/` but NOT yet integrated into HalluLens activation logging pipeline:

### 6. HotpotQA
- **Type:** Multi-hop reasoning QA
- **Status:** 📋 Not implemented
- **Data:** ✅ Available via HuggingFace `datasets` library
- **Priority:** 🔥 HIGH (multi-hop reasoning capability test)
- **Integration Steps:**
  1. Create `tasks/llmsknow/hotpotqa.py` wrapper
  2. Implement HuggingFace dataset loader
  3. Integrate with activation logging pipeline
  4. Test with vLLM server
  5. Compare multi-hop vs single-hop results

### 7. Math (AnswerableMath)
- **Type:** Mathematical reasoning QA
- **Status:** 📋 Not implemented
- **Data:** ✅ Available at `external/LLMsKnow/data/AnswerableMath*.csv`
- **Priority:** 🔥 HIGH (structured reasoning domain)
- **Integration Steps:**
  1. Create `tasks/llmsknow/math.py` wrapper
  2. Implement numerical evaluation protocol
  3. Integrate with activation logging
  4. Analyze activation patterns for mathematical reasoning

### 8. Movies
- **Type:** Short-form factual QA (domain-specific)
- **Status:** 📋 Not implemented
- **Data:** ✅ Available at `external/LLMsKnow/data/movie_qa_*.csv`
- **Priority:** 🔶 MEDIUM
- **Integration Steps:**
  1. Create `tasks/llmsknow/movies.py` wrapper
  2. Adapt data loader for HalluLens format
  3. Test inference + activation logging
  4. Run baseline experiments

### 9. Winobias
- **Type:** Coreference resolution / reasoning (with bias evaluation)
- **Status:** 📋 Not implemented
- **Data:** ✅ Available at `external/LLMsKnow/data/winobias_*.csv`
- **Priority:** 🔶 MEDIUM
- **Note:** Adds reasoning + fairness dimension
- **Integration Steps:**
  1. Create `tasks/llmsknow/winobias.py` wrapper
  2. Adapt evaluation for HalluLens
  3. Test with activation logging
  4. Optional: analyze bias-related patterns

### 10. Winogrande
- **Type:** Commonsense reasoning
- **Status:** 📋 Not implemented
- **Data:** ✅ Available at `external/LLMsKnow/data/winogrande_*.csv`
- **Priority:** 🔶 MEDIUM
- **Integration Steps:**
  1. Create `tasks/llmsknow/winogrande.py` wrapper
  2. Implement data loader
  3. Test inference pipeline
  4. Compare with Winobias

### 11. NLI / MNLI
- **Type:** Natural Language Inference (classification)
- **Status:** 📋 Not implemented
- **Data:** ✅ Available at `external/LLMsKnow/data/mnli_*.csv`
- **Priority:** 🔶 MEDIUM
- **Note:** 3-class prediction (entailment, contradiction, neutral)
- **Integration Steps:**
  1. Create `tasks/llmsknow/mnli.py` wrapper
  2. Handle 3-class prediction in HalluLens framework
  3. Test activation patterns for different entailment types

### 12. IMDB
- **Type:** Sentiment classification
- **Status:** 📋 Not implemented
- **Data:** ✅ Available via HuggingFace `datasets` library
- **Priority:** ⬇️ LOW (less directly related to factual hallucination)
- **Note:** Tests opinion/sentiment rather than factual correctness
- **Integration Steps:**
  1. Evaluate relevance to hallucination detection goals
  2. If included: create `tasks/llmsknow/imdb.py` wrapper
  3. Consider as control condition

---

## 🤔 UNDER CONSIDERATION (Future Extensions)

### 13. TruthfulQA (Original)
- **Type:** Short-form factual QA / adversarial truthfulness
- **Status:** 🤔 Under consideration
- **Data:** Available via HuggingFace
- **Note:** May overlap with PreciseTruthfulQA (already Tier 2)
- **Decision:** Evaluate whether PreciseTruthfulQA sufficiently covers this space

### 14. HaluEval
- **Type:** Multi-task hallucination detection
- **Status:** 🤔 Under consideration
- **Challenges:** Need to define consistent binary labels; task-specific evaluation
- **Next Steps:** Literature review + compatibility assessment

### 15. FactScore / FActScore
- **Type:** Long-form generation evaluation
- **Status:** 🤔 Under consideration
- **Challenges:** Requires atomic fact extraction; may not map to binary labels
- **Note:** Could enhance LongWiki evaluation

### 16. FEVER
- **Type:** Fact verification
- **Status:** 🤔 Under consideration
- **Data:** Available via HuggingFace
- **Challenges:** Requires evidence retrieval; needs adaptation for binary labels

---

## 📊 Implementation Summary Table

| Dataset | Type | Status | Script | Data Available | Logging Ready | Priority |
|---------|------|--------|--------|----------------|---------------|----------|
| **PreciseWikiQA** | Short QA | ✅ Production | ✅ | ✅ | ✅ | Critical |
| **TriviaQA** | Short QA | ✅ Production | ✅ | ✅ | ✅ | Critical |
| **PreciseTruthfulQA** | Short QA | ⚙️ Partial | ✅ | ⚠️ | ⚠️ | High |
| **LongWiki/FactHalu** | Long-form | ⚙️ Partial | ✅ | ✅ | ⚠️ | Medium |
| **Natural Questions** | Short QA | 📋 Planned | ❌ | ✅ | ❌ | High |
| **HotpotQA** | Reasoning | 📋 Planned | ❌ | ✅ (HF) | ❌ | High |
| **Math** | Reasoning | 📋 Planned | ❌ | ✅ | ❌ | High |
| **Movies** | Short QA | 📋 Planned | ❌ | ✅ | ❌ | Medium |
| **Winobias** | Reasoning | 📋 Planned | ❌ | ✅ | ❌ | Medium |
| **Winogrande** | Reasoning | 📋 Planned | ❌ | ✅ | ❌ | Medium |
| **NLI/MNLI** | Classification | 📋 Planned | ❌ | ✅ | ❌ | Medium |
| **IMDB** | Classification | 📋 Planned | ❌ | ✅ (HF) | ❌ | Low |
| **TruthfulQA (Original)** | Short QA | 🤔 Considering | ❌ | ✅ (HF) | ❌ | TBD |
| **HaluEval** | Multi-task | 🤔 Considering | ❌ | ⚠️ | ❌ | TBD |
| **Natural Questions** | Short QA | ✅ Production | ✅ | ✅ | ✅ | High |
| **PreciseTruthfulQA** | Short QA | ⚙️ Partial | ✅ | ⚠️ | ⚠️ | High |
| **LongWiki/FactHalu** | Long-form | ⚙️ Partial | ✅ | ✅ | ⚠️ | Medium
**Legend:**
- ✅ Complete / Available / Working
- ⚙️ Partial implementation / needs work
- ❌ Not implemented
- ⚠️ Needs verification / testing
- 🤔 Under evaluation
- 📋 Planned for implementation

---

## 🎯 Recommended Implementation Order

### Phase 1: Complete Partial Implementations (Immediate) - ✅ 1 of 3 Complete
1. ~~**Natural Questions**~~ - ✅ **COMPLETED** - Real-world queries implementation done!
2. **PreciseTruthfulQA** - Test and validate
3. **LongWiki/FactHalu** - Define evaluation protocol and test

### Phase 2: High-Priority LLMsKnow Integration (Short-term)
1. **HotpotQA** - Multi-hop reasoning
2. **Math** - Structured reasoning

### Phase 3: Breadth Expansion (Medium-term)
1. **Movies** - Domain-specific QA
2. **Winobias** - Reasoning + bias
3. **Winogrande** - Commonsense reasoning
4. **NLI/MNLI** - Textual entailment

### Phase 4: Optional Extensions (Long-term)
1. Evaluate IMDB relevance
2. Assess Tier 4 (consideration) datasets
3. Run comprehensive multi-benchmark experiments

---

## 📁 Required Directory Structure for New Implementations

For each new dataset, create:

```
tasks/llmsknow/               # Or appropriate subdirectory
├── {dataset_name}.py         # Main task script
├── README.md                 # Dataset-specific documentation
└── __init__.py

data/{dataset_name}/          # If local data needed
└── [data files]

scripts/
└── task_{dataset_name}.sh    # Optional: automation wrapper
```

---

## ✅ Integration Checklist Template

For each new dataset:

### Implementation
- [ ] Create task wrapper script (`tasks/llmsknow/{dataset}.py`)
- [ ] Implement data loader compatible with HalluLens format
- [ ] Add vLLM server integration
- [ ] Integrate activation logging (JSON/NPY or Zarr)
- [ ] Implement evaluation protocol with binary correctness labels
- [ ] Add script wrapper if needed

### Testing
- [ ] Test end-to-end run on small N (e.g., 100)
- [ ] Verify activation logging works correctly
- [ ] Validate evaluation correctness
- [ ] Check multi-seed compatibility
- [ ] Verify results schema compliance

### Documentation
- [ ] Add dataset description to DATASET_ROADMAP.md
- [ ] Document data requirements and download instructions
- [ ] Add usage examples
- [ ] Update this status document

### Validation
- [ ] Run baseline experiments
- [ ] Compare with LLMsKnow results (if applicable)
- [ ] Document results

---

## 📚 Related Documentation

- **[DATASET_ROADMAP.md](DATASET_ROADMAP.md)** - Comprehensive dataset planning and tracking
- **[../../PAPER_ROADMAP.md](../../PAPER_ROADMAP.md)** - Active EMNLP roadmap (see also [PAPER_ROADMAP_LEGACY.md](PAPER_ROADMAP_LEGACY.md) for the broader ideas pool)
- **[activation_logging/README.md](activation_logging/README.md)** - Activation logging infrastructure
- **[external/LLMsKnow/README.md](external/LLMsKnow/README.md)** - LLMsKnow benchmark suite details
- **[SOTA_TRACKER.md](SOTA_TRACKER.md)** - State-of-the-art methods tracking

---

## 🔄 Notes on Refusal Test Tasks

The `tasks/refusal_test/` directory contains:
- `nonsense_mixed_entities.py`
- `round_robin_nonsense_name.py`
- Other entity/nonsense generation tasks

These are **NOT QA-style datasets** but rather adversarial/refusal testing tasks. They are not included in this QA-focused status document but may be relevant for broader hallucination analysis.

---

**Last Updated:** February 4, 2026
