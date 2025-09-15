# HalluLens: Mutual Information Analysis for LLM Hallucination Detection

[![arXiv](https://img.shields.io/badge/arXiv-2504.17550-b31b1b.svg)](https://arxiv.org/pdf/2504.17550)

## Overview

This repository contains research code for detecting hallucinations in Large Language Models (LLMs) through mutual information relationships in intermediate layers. Our approach leverages contrastive representation learning to identify patterns in neural activations that distinguish between hallucinated and factual responses.

## Research Purpose

We investigate **mutual information relationships in LLM intermediate layers** to detect hallucinations by:

1. **Capturing intermediate layer activations** during LLM inference
2. **Training contrastive representation models** that pair intermediate layers from the same answer together
3. **Analyzing activation patterns** to identify hallucination signatures
4. **Extending evaluation** to comprehensive benchmarks for robust validation

## Methodology

### Core Approach

Our methodology involves training a **contrastive representation sub-model** that learns to:
- Pair intermediate layer activations from the same model response
- Distinguish between activation patterns in hallucinated vs. factual responses
- Capture mutual information relationships across different layers
- Generalize across different types of questions and domains

### Architecture Components

- **Activation Logging**: Captures per-token, per-layer neural activations during inference
- **Contrastive Learning**: Trains models to identify relationships between layer representations
- **Hallucination Classification**: Uses learned representations to detect hallucination patterns

## Repository Structure

### 📊 Activation Logging (`activation_logging/`)
Core infrastructure for capturing and storing LLM intermediate layer activations:
- **`activations_logger.py`**: LMDB-based logger for storing activations, prompts, and responses
- **`server.py`**: OpenAI API-compatible server with activation logging capabilities
- **`vllm_serve.py`**: vLLM integration for scalable inference with activation capture
- **`zarr_activations_logger.py`**: Zarr-based storage for large-scale activation datasets

### 🧠 Activation Research (`activation_research/`)
Sub-model training and evaluation for hallucination detection:
- **`model.py`**: Transformer-based classifiers for hallucination detection
- **`training.py`**: Contrastive learning training procedures
- **`evaluation.py`**: Evaluation metrics and validation procedures
- **`metrics.py`**: Performance measurement utilities

### 🎯 Benchmarks

#### HalluLens PreciseWikiQA (Working Example)
Our primary working example uses the **HalluLens PreciseWikiQA benchmark**:
- **Location**: `tasks/shortform/precise_wikiqa.py`
- **Purpose**: Evaluates hallucination on short, fact-seeking queries
- **Data**: Based on high-quality Wikipedia content
- **Script**: `scripts/task1_precisewikiqa.sh`

#### LLMsKnow Integration (Extension Target)
We aim to extend our method to work on benchmarks from the **LLMsKnow repository**:
- **Location**: `external/LLMsKnow/`
- **Benchmarks**: TriviaQA, Movies, HotpotQA, Winobias, Winogrande, NLI, IMDB, Math, Natural Questions
- **Purpose**: Comprehensive evaluation across diverse question types and domains
- **Reference**: [LLMs Know More Than They Show](https://arxiv.org/abs/2410.02707)

### 🛠 Utilities
- **`utils/`**: Core utilities for evaluation, caching, and model interfaces
- **`scripts/`**: Automation scripts for running experiments
- **`data/`**: Data management and download utilities

## Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/HalluLens.git
cd HalluLens
conda create --name hallulens python==3.12
conda activate hallulens
pip install -r requirements.txt
```

### 2. Data Setup
```bash
bash scripts/download_data.sh
```

### 3. Run PreciseWikiQA Benchmark
```bash
# Start activation logging server
python -m activation_logging.vllm_serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# Run benchmark with activation logging
bash scripts/task1_precisewikiqa.sh
```

### 4. Train Hallucination Classifier
```python
from activation_research.training import train_halu_classifier
from activation_research.model import LastLayerHaluClassifier

# Load activation data and train classifier
model = LastLayerHaluClassifier(input_dim=4096)
train_halu_classifier(model, train_dataset, test_dataset)
```

## Key Features

### 🔍 Comprehensive Activation Logging
- **Multi-format storage**: LMDB, JSON, Zarr, and NPY formats
- **Flexible targeting**: Configurable layer and token selection
- **Scalable infrastructure**: Supports large-scale experiments
- **vLLM integration**: High-performance inference with logging

### 🤖 Advanced Model Architecture
- **Transformer-based classifiers**: Sophisticated attention mechanisms for activation analysis
- **Contrastive learning**: Learns relationships between layer representations
- **Multiple architectures**: From simple feed-forward to complex transformer models

### 📈 Robust Evaluation
- **Multiple benchmarks**: HalluLens tasks + LLMsKnow datasets
- **Cross-domain validation**: Ensures generalization across question types
- **Comprehensive metrics**: Accuracy, precision, recall, and specialized hallucination metrics

## Research Extensions

### Current Status
- ✅ **PreciseWikiQA**: Fully implemented and validated
- ✅ **Activation logging infrastructure**: Complete and scalable
- ✅ **Basic classifier models**: Implemented and tested

### Planned Extensions
- 🔄 **LLMsKnow integration**: Extend methodology to all LLMsKnow benchmarks
- 🔄 **Advanced contrastive models**: Improve mutual information capture
- 🔄 **Cross-benchmark validation**: Ensure robustness across domains
- 🔄 **Real-time detection**: Deploy for live hallucination detection

## Citation

```bibtex
@article{bang2025hallulens,
    title={HalluLens: LLM Hallucination Benchmark}, 
    author={Yejin Bang and Ziwei Ji and Alan Schelten and Anthony Hartshorn and Tara Fowler and Cheng Zhang and Nicola Cancedda and Pascale Fung},
    year={2025},
    eprint={2504.17550},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.17550}, 
}
```

## Related Work

- [LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](https://arxiv.org/abs/2410.02707)
- [Detecting LLM Hallucination Through Layer-wise Information Deficiency](https://arxiv.org/html/2412.10246v1)

## License

The majority of HalluLens is licensed under CC-BY-NC. Portions of the project are available under separate license terms.
