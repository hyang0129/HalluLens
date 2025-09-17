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

### 🌐 Remote Development
- **`REMOTE_DEV_SETUP.md`**: Comprehensive guide for GPU-accelerated remote development
- **`connect_gpu.sh`**: Automated script for connecting to remote GPU nodes
- **`test_gpu_connection.sh`**: Environment verification script for remote setups

## Quick Start

### 1. Installation

#### Local Development
```bash
git clone https://github.com/your-repo/HalluLens.git
cd HalluLens
conda create --name hallulens python==3.12
conda activate hallulens
pip install -r requirements.txt
```

#### Remote Development Environment
For GPU-accelerated development on remote servers, see our comprehensive remote development setup guide:

📖 **[Remote Development Setup Guide](REMOTE_DEV_SETUP.md)**

Key features:
- 🚀 **One-command GPU connection** via `./connect_gpu.sh`
- 🔐 **Automated SSH agent setup** (no repeated passphrase prompts)
- 🎯 **Dynamic job discovery** (finds running Slurm jobs automatically)
- 🖥️ **Direct GPU node access** for ML training and inference
- 📁 **Automatic project navigation** and environment activation

### 2. Data Setup
```bash
bash scripts/download_data.sh
```

### 3. Run PreciseWikiQA Benchmark

#### Option A: Unified Script (Recommended)
```bash
# Run complete experiment with automatic server management
python scripts/run_with_server.py \
    --step all \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 100

# Or use enhanced bash script
bash scripts/task1_precisewikiqa_with_server.sh --N 100
```

#### Option B: Manual Server Management
```bash
# Start activation logging server (in one terminal)
python -m activation_logging.vllm_serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# Run benchmark (in another terminal)
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

## Testing Inference with Activation Logging

### JSON Activation Logging (NPY Format)

For testing inference with efficient binary tensor storage while maintaining JSON metadata readability:

```bash
# Test with JSON logger using NPY binary format (recommended for large-scale experiments)
python scripts/run_with_server.py \
  --step inference \
  --task precisewikiqa \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --wiki_src goodwiki \
  --mode gguf_big \
  --inference_method vllm \
  --logger-type json \
  --activations-path test_output/activations.json \
  --max_inference_tokens 256 \
  --N 100 \
  --qa_output_path data/precise_qa/save/qa_goodwiki_Llama-3.1-8B-Instruct_gguf_big.jsonl \
  --generations_file_path test_output/generation.jsonl
```

**Key Parameters for JSON Logging:**
- `--logger-type json`: Enables JSON activation logging with NPY binary tensor storage
- `--activations-path`: Directory path for JSON metadata and NPY activation files
- `--max_inference_tokens 256`: Maximum tokens per response (controls generation length and storage size)
- `--N 100`: Number of inference samples (adjust based on testing needs)

**Storage Efficiency:**
- **NPY Binary Format**: ~95% storage reduction compared to JSON text tensors
- **Expected Size**: ~5-10GB for 100 inferences (vs 106GB for pure JSON)
- **Format**: JSON metadata + separate `.npy` files for activation tensors
- **Backward Compatibility**: Supports both new NPY and legacy JSON activation files

### LMDB Activation Logging (Alternative)

For maximum storage efficiency:

```bash
# Test with LMDB logger (most compact storage)
python scripts/run_with_server.py \
  --step inference \
  --task precisewikiqa \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --wiki_src goodwiki \
  --mode gguf_big \
  --inference_method vllm \
  --logger-type lmdb \
  --activations-path test_output/activations.lmdb \
  --max_inference_tokens 256 \
  --N 100 \
  --qa_output_path data/precise_qa/save/qa_goodwiki_Llama-3.1-8B-Instruct_gguf_big.jsonl \
  --generations_file_path test_output/generation.jsonl
```

### Remote Testing

For comprehensive remote GPU testing with JSON logging, see:

📖 **[Remote Testing Guide](REMOTE_TESTING_GUIDE.md)**

This guide includes step-by-step instructions for testing JSON activation logging in remote GPU environments with automated scripts and verification procedures.

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
