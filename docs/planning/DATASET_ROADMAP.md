# Dataset Roadmap for HalluLens Hallucination Detection

This document tracks all datasets currently used or under consideration for the HalluLens hallucination detection research project, organized by priority, implementation status, and integration requirements.

## Document Purpose

- **Track dataset availability**: What we have vs. what we need
- **Document data requirements**: Download scripts, preprocessing, storage locations
- **Plan integration**: What needs to be implemented vs. what's already working
- **Support paper goals**: Ensure breadth comparable to or exceeding baseline comparisons (LLMsKnow suite)

---

## Priority Classification

- **Tier 1 (Production)**: Currently working, actively used for experiments
- **Tier 2 (Implementation Ready)**: Partial implementation, needs completion
- **Tier 3 (Planned)**: No implementation yet, future integration target
- **Tier 4 (Consideration)**: Under evaluation for inclusion

---

## Dataset Categories

### A) Short-Form Factual QA (Question Answering)
Focus: Binary correctness evaluation based on exact answer matching or reference comparison

### B) Long-Form Generation
Focus: Multi-sentence responses, requires sentence-level or claim-level evaluation

### C) Classification & Reasoning
Focus: Multiple-choice, binary classification, or logical reasoning tasks

### D) Adversarial & Robustness
Focus: Datasets designed to test edge cases, biases, or adversarial prompts

---

## Tier 1: Production Datasets (Currently Working)

### 1.1 PreciseWikiQA (HalluLens Native)

**Status**: ✅ Fully Implemented and Working

**Description**: Short-form factual QA benchmark based on high-quality Wikipedia content. Questions are generated from Wikipedia articles at different hallucination risk levels (binned by h_score_cat).

**Task Type**: Short-form factual QA

**Implementation**:
- **Script**: `tasks/shortform/precise_wikiqa.py`
- **Data Generator**: `utils/generate_question.py` (WikiQA class, PRECISE_Q_GENERATION_PROMPT)
- **Automation**: `scripts/task1_precisewikiqa.sh`, `scripts/run_with_server.py`
- **Data Location**: `data/precise_qa/save/`
- **Wiki Source**: `data/wiki_data/` (Wikipedia dump)

**Data Requirements**:
- Wikipedia dump: `data/wiki_data/.cache/enwiki-20230401.db`
- Title database: `data/wiki_data/title_db.jsonl`
- Generated QA pairs stored per experiment

**Evaluation Method**:
- Exact answer matching
- Binary hallucination label (correct vs. incorrect)
- Support for dynamic vs. static modes
- Support for multiple wiki sources (goodwiki, goodwiki_v2)

**Integration Status**:
- ✅ Question generation
- ✅ Inference via vLLM server
- ✅ Activation logging (JSON/NPY, Zarr, LMDB)
- ✅ Correctness evaluation
- ✅ Multi-seed training support
- ✅ OOD evaluation support

**Current Results**:
- AUROC: 70+ (contrastive method on PreciseWikiShort variant)
- Primary working example for the project

**Next Steps**:
- Scale up N for production runs
- Document optimal hyperparameters
- Run multi-seed experiments for paper

---

### 1.2 TriviaQA (HalluLens Wrapper)

**Status**: ✅ Implemented, Actively Used

**Description**: Trivia questions from various sources with multiple acceptable answer aliases. Focused on factual knowledge across diverse topics.

**Task Type**: Short-form factual QA

**Implementation**:
- **Script**: `tasks/triviaqa/triviaqa.py`
- **README**: `tasks/triviaqa/README.md`
- **Data Location**: `data/triviaqa-unfiltered/`

**Data Requirements**:
- Unfiltered TriviaQA dataset (auto-download supported)
- Required files:
  - `data/triviaqa-unfiltered/unfiltered-web-train.json`
  - `data/triviaqa-unfiltered/unfiltered-web-dev.json`
- Download source: https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz

**Evaluation Method**:
- Multiple answer alias matching
- Binary correctness (any alias match = correct)

**Integration Status**:
- ✅ Dataset loading
- ✅ Inference via vLLM server
- ✅ Activation logging
- ✅ Correctness evaluation
- ✅ Compatible with LLMsKnow format

**Current Results**:
- AUROC: 80+ (contrastive method)
- Strong baseline performance

**Next Steps**:
- Multi-seed validation
- Stratified analysis by question difficulty
- Compare with LLMsKnow TriviaQA results

---

## Tier 2: Implementation Ready (Partial Support)

### 2.1 PreciseTruthfulQA (HalluLens Native)

**Status**: ⚙️ Implemented, Needs Validation

**Description**: Adaptation of TruthfulQA for the HalluLens framework, focusing on questions where models may generate plausible but incorrect answers.

**Task Type**: Short-form factual QA / adversarial truthfulness

**Implementation**:
- **Script**: `tasks/shortform/precise_truthfulqa.py`
- **Data Location**: TBD (likely `data/truthfulqa/`)

**Data Requirements**:
- Original TruthfulQA dataset
- Preprocessing for HalluLens format
- Ground truth answers / evaluation criteria

**Evaluation Method**:
- Similar to PreciseWikiQA (binary correctness)
- May include category-based stratification
- Truthfulness vs. informativeness tradeoffs

**Integration Status**:
- ✅ Script exists
- ⚠️ Needs validation and testing
- ⚠️ Data download/preprocessing unclear
- ⚠️ Integration with activation logging TBD

**Next Steps**:
- Verify data availability and download process
- Test end-to-end pipeline
- Document evaluation protocol
- Run baseline experiments

---

### 2.2 LongWiki / FactHalu (HalluLens Native)

**Status**: ⚙️ Implemented, Needs Production Testing

**Description**: Long-form generation task where models generate multi-sentence responses about Wikipedia topics. Focuses on factuality in extended generations.

**Task Type**: Long-form generation

**Implementation**:
- **Scripts**: 
  - `tasks/longwiki/longwiki_main.py`
  - `tasks/longwiki/facthalu.py`
  - `tasks/longwiki/longwiki_retrieval.py`
- **Utilities**: `tasks/longwiki/longwiki_utils.py`, `tasks/longwiki/prompt_templates.py`
- **Data Location**: `data/wiki_data/`

**Data Requirements**:
- Wikipedia dump (shared with PreciseWikiQA)
- Long-form question templates
- Retrieval support (optional)

**Evaluation Method**:
- Sentence-level or response-level hallucination detection
- May require claim extraction and verification
- Potential for chunk-level labeling

**Integration Status**:
- ✅ Scripts exist
- ⚠️ Evaluation protocol unclear
- ⚠️ Activation logging for long-form TBD
- ⚠️ No documented results yet

**Next Steps**:
- Define evaluation protocol (sentence vs. response level)
- Test with activation logging
- Determine if viable for binary response-level classification
- Compare with short-form QA results

---

## Tier 3: Planned (LLMsKnow Integration Targets)

The following datasets are part of the **LLMsKnow benchmark suite** and represent key integration targets for comprehensive evaluation. These datasets are well-documented in `external/LLMsKnow/` but are not yet integrated into the HalluLens activation logging pipeline.

### 3.1 Movies (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Movie-related QA dataset testing knowledge about films, actors, and production details.

**Task Type**: Short-form factual QA (domain-specific)

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data files**: 
  - `external/LLMsKnow/data/movie_qa_test.csv`
  - `external/LLMsKnow/data/movie_qa_train.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data files already in repo
- ⚠️ Need HalluLens-compatible loader
- ⚠️ Format conversion required

**Evaluation Method**:
- Binary correctness (exact match or semantic equivalence)
- Domain-specific evaluation criteria

**Integration Priority**: Medium

**Next Steps**:
1. Create `tasks/llmsknow/movies.py` wrapper
2. Adapt data loader for HalluLens format
3. Test inference + activation logging
4. Validate evaluation protocol
5. Run baseline experiments

---

### 3.2 HotpotQA (LLMsKnow)

**Status**: 📋 Planned, Uses HuggingFace

**Description**: Multi-hop reasoning QA dataset requiring models to combine information from multiple sources.

**Task Type**: Reasoning / multi-hop QA

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data source**: HuggingFace `datasets` library (auto-download)
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ⚠️ Loaded via HuggingFace (no local files needed)
- ⚠️ May require preprocessing for HalluLens format

**Evaluation Method**:
- Exact answer matching
- Binary correctness
- May consider reasoning chain evaluation (future)

**Integration Priority**: High (reasoning capability test)

**Next Steps**:
1. Create `tasks/llmsknow/hotpotqa.py` wrapper
2. Implement HuggingFace dataset loader
3. Test with activation logging pipeline
4. Compare multi-hop vs. single-hop results

---

### 3.3 Winobias (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Coreference resolution dataset designed to test gender bias in language models.

**Task Type**: Classification / reasoning (with bias evaluation)

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data files**: 
  - `external/LLMsKnow/data/winobias_dev.csv`
  - `external/LLMsKnow/data/winobias_test.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data files already in repo
- ⚠️ Format conversion required

**Evaluation Method**:
- Exact answer matching (during generation; no separate extraction step per LLMsKnow)
- Binary correctness
- Bias analysis (optional)

**Integration Priority**: Medium (adds reasoning + fairness dimension)

**Next Steps**:
1. Create `tasks/llmsknow/winobias.py` wrapper
2. Adapt evaluation for HalluLens
3. Test with activation logging
4. Document any bias-related patterns in activations

---

### 3.4 Winogrande (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Commonsense reasoning dataset based on the Winograd Schema Challenge.

**Task Type**: Classification / commonsense reasoning

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data files**: 
  - `external/LLMsKnow/data/winogrande_dev.csv`
  - `external/LLMsKnow/data/winogrande_test.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data files already in repo
- ⚠️ Format conversion required

**Evaluation Method**:
- Exact answer matching (during generation; no separate extraction step per LLMsKnow)
- Binary correctness

**Integration Priority**: Medium (commonsense reasoning test)

**Next Steps**:
1. Create `tasks/llmsknow/winogrande.py` wrapper
2. Implement data loader
3. Test inference pipeline
4. Compare with Winobias (similar task structure)

---

### 3.5 NLI / MNLI (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Natural Language Inference (Multi-Genre NLI) dataset for textual entailment classification.

**Task Type**: Classification (entailment, contradiction, neutral)

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/` (referred to as "mnli" in code)
- **Data files**: 
  - `external/LLMsKnow/data/mnli_train.csv`
  - `external/LLMsKnow/data/mnli_validation.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data files already in repo
- ⚠️ Format conversion required

**Evaluation Method**:
- Exact answer matching (during generation; no separate extraction step per LLMsKnow)
- Binary correctness (3-class → binary may need mapping)

**Integration Priority**: Medium (tests reasoning + semantic understanding)

**Next Steps**:
1. Create `tasks/llmsknow/mnli.py` wrapper
2. Handle 3-class prediction in HalluLens framework
3. Test activation patterns for different entailment types
4. Document evaluation protocol

---

### 3.6 IMDB (LLMsKnow)

**Status**: 📋 Planned, Uses HuggingFace

**Description**: Sentiment classification dataset based on movie reviews.

**Task Type**: Classification (sentiment analysis)

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data source**: HuggingFace `datasets` library (auto-download)
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ⚠️ Loaded via HuggingFace (no local files needed)

**Evaluation Method**:
- Exact answer matching (during generation; no separate extraction step per LLMsKnow)
- Binary correctness (positive/negative sentiment)

**Integration Priority**: Low (less directly related to factual hallucination)

**Note**: This dataset tests opinion/sentiment rather than factual correctness. May be useful for understanding activation patterns in subjective vs. factual tasks.

**Next Steps**:
1. Evaluate relevance to hallucination detection goals
2. If included: create `tasks/llmsknow/imdb.py` wrapper
3. Consider as ablation/control condition

---

### 3.7 Math (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Mathematical reasoning dataset testing arithmetic and problem-solving capabilities.

**Task Type**: Reasoning / mathematical QA

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data files**: 
  - `external/LLMsKnow/data/AnswerableMath_test.csv`
  - `external/LLMsKnow/data/AnswerableMath.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data files already in repo
- ⚠️ Format conversion required

**Evaluation Method**:
- Numerical answer matching
- Binary correctness

**Integration Priority**: High (tests reasoning in structured domain)

**Next Steps**:
1. Create `tasks/llmsknow/math.py` wrapper
2. Implement numerical evaluation protocol
3. Test with activation logging
4. Analyze activation patterns for mathematical reasoning

---

### 3.8 Natural Questions (LLMsKnow)

**Status**: 📋 Planned, Data Available

**Description**: Open-domain QA dataset based on real Google search queries with Wikipedia-based answers.

**Task Type**: Short-form factual QA (real-world queries)

**Implementation**:
- **LLMsKnow scripts**: Available in `external/LLMsKnow/`
- **Data files**: 
  - `external/LLMsKnow/data/nq_wc_dataset.csv`
- **HalluLens integration**: Not yet implemented

**Data Requirements**:
- ✅ Data file already in repo
- ⚠️ Format conversion required

**Evaluation Method**:
- Answer span matching
- Binary correctness

**Integration Priority**: High (real-world query distribution)

**Next Steps**:
1. Create `tasks/llmsknow/natural_questions.py` wrapper
2. Adapt answer extraction/matching for HalluLens
3. Test with activation logging
4. Compare with TriviaQA (similar QA format)

---

## Tier 4: Under Consideration (Future Extensions)

### 4.1 TruthfulQA (Original)

**Status**: 🤔 Under Consideration

**Description**: Benchmark measuring truthfulness in language models, focusing on questions where models may generate common but false answers.

**Task Type**: Short-form factual QA / adversarial truthfulness

**Relation to HalluLens**:
- Similar goals to PreciseTruthfulQA (already in Tier 2)
- May provide additional adversarial examples

**Data Requirements**:
- Original TruthfulQA dataset from HuggingFace
- May overlap with PreciseTruthfulQA

**Decision Point**:
- Evaluate whether PreciseTruthfulQA sufficiently covers this space
- Consider as extension if PreciseTruthfulQA proves valuable

---

### 4.2 HaluEval

**Status**: 🤔 Under Consideration

**Description**: Large-scale hallucination evaluation benchmark with diverse QA and generation tasks.

**Task Type**: Multi-task hallucination detection

**Challenges**:
- Need to define consistent binary labels
- May require task-specific evaluation protocols
- Large scale may require significant compute

**Next Steps**:
- Literature review of HaluEval methodology
- Assess compatibility with binary classification approach
- Determine if subset is viable for integration

---

### 4.3 FactScore / FActScore

**Status**: 🤔 Under Consideration

**Description**: Fact-checking framework for evaluating factuality in long-form generation.

**Task Type**: Long-form generation evaluation

**Challenges**:
- Requires atomic fact extraction
- Sentence-level or claim-level evaluation
- May not map directly to binary response-level labels

**Relation to HalluLens**:
- Could enhance LongWiki evaluation
- Provides fine-grained factuality assessment

**Next Steps**:
- Evaluate compatibility with activation logging approach
- Consider as evaluation metric rather than primary dataset
- Assess compute requirements

---

### 4.4 FEVER (Fact Extraction and VERification)

**Status**: 🤔 Under Consideration

**Description**: Fact verification dataset where models must verify claims against Wikipedia evidence.

**Task Type**: Classification / fact verification

**Challenges**:
- Requires evidence retrieval component
- May need adaptation for binary correctness labels

**Potential Benefits**:
- Tests evidence-based reasoning
- Well-established benchmark
- Large scale

**Next Steps**:
- Assess feasibility of deriving binary correctness signals
- Consider simplified version without retrieval
- Evaluate integration effort vs. value

---

## Dataset Integration Checklist

For each new dataset integration, ensure the following components are implemented:

### Implementation Requirements
- [ ] Data loader compatible with HalluLens format
- [ ] Inference script using vLLM server
- [ ] Activation logging integration (JSON/NPY or Zarr)
- [ ] Evaluation protocol with binary correctness labels
- [ ] Integration with `scripts/run_with_server.py`
- [ ] Wrapper script in `scripts/` directory

### Documentation Requirements
- [ ] Dataset description and source in this roadmap
- [ ] Data requirements and download instructions
- [ ] Evaluation method documentation
- [ ] Usage examples in README or script comments
- [ ] Expected data locations documented

### Testing Requirements
- [ ] Successful end-to-end run on small N (e.g., 100)
- [ ] Activation logging verification
- [ ] Evaluation correctness validation
- [ ] Multi-seed compatibility test
- [ ] Results schema compliance check

### Paper Requirements
- [ ] Baseline results documented
- [ ] SOTA comparison identified (see `SOTA_TRACKER.md`)
- [ ] Multi-seed experiments planned
- [ ] Statistical significance testing prepared

---

## Quick Reference: Dataset Matrix

| Dataset | Status | Type | Data Available | Script Ready | Logging Ready | Priority |
|---------|--------|------|---------------|--------------|---------------|----------|
| PreciseWikiQA | ✅ Production | Short QA | ✅ | ✅ | ✅ | Critical |
| TriviaQA | ✅ Production | Short QA | ✅ | ✅ | ✅ | Critical |
| PreciseTruthfulQA | ⚙️ Partial | Short QA | ⚠️ | ✅ | ⚠️ | High |
| LongWiki/FactHalu | ⚙️ Partial | Long-form | ✅ | ✅ | ⚠️ | Medium |
| Movies | 📋 Planned | Short QA | ✅ | ❌ | ❌ | Medium |
| HotpotQA | 📋 Planned | Reasoning | ✅ (HF) | ❌ | ❌ | High |
| Winobias | 📋 Planned | Reasoning | ✅ | ❌ | ❌ | Medium |
| Winogrande | 📋 Planned | Reasoning | ✅ | ❌ | ❌ | Medium |
| NLI/MNLI | 📋 Planned | Classification | ✅ | ❌ | ❌ | Medium |
| IMDB | 📋 Planned | Classification | ✅ (HF) | ❌ | ❌ | Low |
| Math | 📋 Planned | Reasoning | ✅ | ❌ | ❌ | High |
| Natural Questions | 📋 Planned | Short QA | ✅ | ❌ | ❌ | High |
| TruthfulQA | 🤔 Considering | Short QA | ✅ (HF) | ❌ | ❌ | TBD |
| HaluEval | 🤔 Considering | Multi-task | ⚠️ | ❌ | ❌ | TBD |
| FactScore | 🤔 Considering | Long-form | ⚠️ | ❌ | ❌ | TBD |
| FEVER | 🤔 Considering | Verification | ✅ (HF) | ❌ | ❌ | TBD |

**Legend**:
- ✅ Complete / Available
- ⚙️ Partial implementation
- ❌ Not implemented
- ⚠️ Needs verification
- 🤔 Under evaluation
- 📋 Planned

---

## Integration Strategy & Timeline

### Phase 1: Solidify Core (Current)
**Goal**: Ensure Tier 1 datasets are production-ready with multi-seed experiments

**Tasks**:
1. Complete multi-seed runs for PreciseWikiQA
2. Complete multi-seed runs for TriviaQA
3. Document optimal hyperparameters
4. Establish baseline results for paper

**Timeline**: Ongoing

---

### Phase 2: Complete Tier 2 (Short-term)
**Goal**: Validate and production-test partially implemented datasets

**Tasks**:
1. Verify PreciseTruthfulQA end-to-end
2. Test LongWiki/FactHalu evaluation protocol
3. Run baseline experiments
4. Document results

**Timeline**: 2-4 weeks

---

### Phase 3: High-Priority LLMsKnow Integration (Medium-term)
**Goal**: Integrate highest-value LLMsKnow benchmarks for broad evaluation

**Priorities**:
1. **HotpotQA** (multi-hop reasoning)
2. **Math** (structured reasoning)
3. **Natural Questions** (real-world queries)

**Tasks per dataset**:
1. Create task wrapper script
2. Implement data loader
3. Test inference + activation logging
4. Run baseline experiments
5. Compare with LLMsKnow results

**Timeline**: 4-8 weeks

---

### Phase 4: Breadth Expansion (Long-term)
**Goal**: Achieve comprehensive benchmark coverage comparable to LLMsKnow

**Tasks**:
1. Complete remaining LLMsKnow integrations (Movies, Winobias, Winogrande, NLI, IMDB)
2. Evaluate Tier 4 datasets for inclusion
3. Run full multi-seed, multi-benchmark experiments
4. Generate paper-ready result tables

**Timeline**: 8-12 weeks

---

## Data Storage & Management

### Current Storage Locations

```
HalluLens/
├── data/
│   ├── precise_qa/          # PreciseWikiQA generated data
│   ├── wiki_data/           # Wikipedia dump & preprocessing
│   ├── triviaqa-unfiltered/ # TriviaQA dataset (auto-downloaded)
│   └── (future datasets)    # TBD locations for new datasets
│
├── external/LLMsKnow/
│   └── data/                # LLMsKnow dataset files
│
├── test_output/             # Temporary test runs
├── output/                  # Experiment outputs
└── shared/                  # Shared storage for larger experiments (if applicable)
```

### Storage Considerations

**Activation Storage**:
- JSON + NPY: ~5-10 GB per 100 inferences
- Zarr: More efficient for large-scale (>10k samples)
- LMDB: Original format, consider phasing out

**Dataset Sizes** (estimated):
- PreciseWikiQA: Variable (generated on demand)
- TriviaQA: ~500 MB (unfiltered)
- LLMsKnow datasets: ~10-100 MB each
- Wikipedia dump: ~15 GB

**Recommendations**:
1. Use JSON + NPY for small-to-medium experiments (<10k)
2. Use Zarr for large-scale production runs (>10k)
3. Archive completed experiments to free space
4. Document storage requirements in each task README

---

## Related Documentation

- **[PAPER_ROADMAP.md](PAPER_ROADMAP.md)**: Overall research roadmap and experiment planning
- **[SOTA_TRACKER.md](SOTA_TRACKER.md)**: Tracking state-of-the-art methods and comparisons
- **[RESULTS_SCHEMA.md](RESULTS_SCHEMA.md)**: Output format and results structure
- **[activation_logging/README.md](activation_logging/README.md)**: Activation logging infrastructure
- **[external/LLMsKnow/README.md](external/LLMsKnow/README.md)**: LLMsKnow benchmark suite details

---

## Contributions & Updates

This roadmap should be updated as:
- New datasets are integrated
- Integration status changes
- New datasets are identified for consideration
- Storage requirements change
- Priority changes based on research progress

**Last Updated**: 2026-02-01
