# TriviaQA Task

This folder contains code for generating answers and evaluating correctness for the TriviaQA dataset, following a similar approach to the `precise_wikiqa` process.

## Overview

Unlike the `precise_wikiqa` task which generates both questions and answers, the TriviaQA task focuses on the **inference** and **evaluation** steps since the questions are already provided in the original TriviaQA dataset.

## Process Flow

The TriviaQA task follows this simplified pipeline:

1. **Load Dataset**: Load pre-existing TriviaQA questions and ground truth answers
2. **Server-Based Inference**: Generate model responses via activation logging server
3. **Evaluation (Correctness)**: Assess answer correctness against ground truth

## Server-Based Architecture

Following the `precise_wikiqa` pattern, TriviaQA inference is performed against a server that handles:
- **Model Loading**: Server manages model weights and GPU memory
- **Activation Logging**: Automatic capture of internal representations during inference
- **Batch Processing**: Efficient handling of multiple questions
- **API Interface**: RESTful API for question answering requests

## Key Differences from PreciseWikiQA

- **No Question Generation**: TriviaQA questions are pre-existing, so we skip the question generation step
- **Focus on Inference**: Primary task is generating high-quality answers to trivia questions
- **Evaluation-Centric**: Emphasis on robust correctness evaluation against multiple answer aliases

## Dataset Structure

TriviaQA provides:
- **Questions**: Trivia questions from various sources
- **Answer Aliases**: Multiple acceptable answer variations for each question
- **Evidence**: Supporting documents (optional, depending on variant used)

## Data Requirements

The TriviaQA task requires the TriviaQA unfiltered dataset:
- **Automatic Download**: The script will automatically download and extract the dataset on first run
- **Manual Download**: If needed, download from: https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
- Required files:
  - `data/triviaqa-unfiltered/unfiltered-web-train.json`
  - `data/triviaqa-unfiltered/unfiltered-web-dev.json`
- Use `--no_auto_download` to disable automatic downloading

## Planned Components

### Core Scripts
- `triviaqa_inference.py`: Send questions to activation logging server and collect responses
- `triviaqa_evaluation.py`: Evaluate answer correctness against ground truth
- `triviaqa_utils.py`: Utility functions for data loading, server communication, and processing

### Configuration
- Support for different TriviaQA variants (filtered/unfiltered, with/without evidence)
- Server endpoint configuration for activation logging
- Configurable inference parameters (temperature, max_tokens, etc.)
- Flexible evaluation metrics and thresholds

### Integration
- **Server Communication**: Uses existing activation logging server infrastructure
- **Activation Collection**: Automatic capture of internal representations during inference
- **Batch Processing**: Efficient server-based inference for multiple questions
- **Output Format**: Results compatible with hallucination detection research pipeline

## Usage Pattern

Similar to `precise_wikiqa`, the TriviaQA task will support server-based inference:

```bash
# Start activation logging server (in separate terminal)
python -m activation_logging.vllm_serve --model [MODEL] --host 0.0.0.0 --port 8000

# Generate answers via server
python -m tasks.triviaqa.triviaqa_inference --server_url http://localhost:8000 --dataset_variant [VARIANT]

# Evaluate answer correctness
python -m tasks.triviaqa.triviaqa_evaluation --results_file [RESULTS] --eval_method [METHOD]

# Combined inference and evaluation with server
python -m tasks.triviaqa.run_triviaqa --server_url [URL] --do_inference --do_evaluation
```

## Research Applications

This task is designed to support:
- **Hallucination Detection**: Generate training data for hallucination classifiers
- **Model Evaluation**: Assess factual accuracy on trivia knowledge
- **Activation Analysis**: Collect internal representations during question answering
- **Comparative Studies**: Compare performance across different model architectures

## Future Extensions

- Support for different answer extraction methods
- Integration with retrieval-augmented generation (RAG)
- Multi-turn question answering scenarios
- Cross-dataset evaluation and transfer learning
