# File Generation Guide

This document explains how the three key files in the HalluLens project are generated:

1. `output/test_llama.json` - Inference results file
2. `output/test_llama_eval2.json` - Evaluation results file  
3. `lmdb_data/gguf/activations.lmdb` - Neural activation data in LMDB format

## Overview

The HalluLens project is an LLM hallucination benchmark that captures neural activations during model inference to study hallucination patterns. The data pipeline consists of four main stages:

1. **Question Generation Stage**: Generate test questions/prompts from source data
2. **Inference Stage**: Generate model responses and capture activations
3. **Evaluation Stage**: Evaluate responses for hallucination
4. **Storage Stage**: Store activations in LMDB format for analysis

## File Generation Process

### 0. Question Generation (Prerequisite Step)

Before generating the main files, test questions/prompts must be created from source data.

**Generation Process:**
- **Script**: Task scripts with `--do_generate_prompt` flag
- **Source Data**: Wikipedia articles, entity databases, or other reference materials
- **Question Generator**: GGUF models (e.g., `Llama-3.3-70B-Instruct-IQ3_M.gguf`)

**Command Example:**
```bash
source halu/bin/activate && \
python -m tasks.shortform.precise_wikiqa \
    --do_generate_prompt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --wiki_src goodwiki \
    --mode gguf \
    --inference_method vllm \
    --N 5000 \
    --q_generator Llama-3.3-70B-Instruct-IQ3_M.gguf
```

**Key Parameters:**
- `--do_generate_prompt`: Enables question generation stage
- `--wiki_src goodwiki`: Uses "good" Wikipedia articles as source
- `--mode gguf`: Specifies GGUF model format for generation
- `--N 5000`: Generates 5000 test questions
- `--q_generator`: Specifies the GGUF model for question generation

**Output:**
- Generates prompt files in `data/` directory
- Creates question-answer pairs from source documents
- Prepares test dataset for subsequent inference and evaluation

### 1. Inference Results (`output/test_llama.json`)

This file contains the raw inference results from running LLM models on test prompts.

**Generation Process:**
- **Script**: Various task scripts in `tasks/` directory (e.g., `tasks/shortform/precise_wikiqa.py`, `tasks/refusal_test/nonsense_mixed_entities.py`)
- **Command Example**:
  ```bash
  python -m tasks.shortform.precise_wikiqa \
      --do_inference \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --wiki_src goodwiki \
      --mode dynamic \
      --inference_method vllm \
      --N 1
  ```

**Note**: This step requires questions to be generated first using `--do_generate_prompt` (see Step 0).

**File Structure:**
- Contains prompt-response pairs
- Includes model metadata
- Stores generation parameters
- Format: JSONL (one JSON object per line)

**Key Components:**
- `prompt`: Input text sent to the model
- `generation`: Model's response text
- `model`: Model identifier used for inference
- `timestamp`: When the inference was performed

### 2. Evaluation Results (`output/test_llama_eval2.json`)

This file contains evaluation results determining whether model responses contain hallucinations.

**Generation Process:**
- **Script**: Same task scripts with `--do_eval` flag
- **Evaluator Models**: 
  - Default: `meta-llama/Llama-3.1-8B-Instruct`
  - GGUF: `Llama-3.3-70B-Instruct-IQ3_M.gguf`
- **Command Example**:
  ```bash
  python -m tasks.shortform.precise_wikiqa \
      --do_eval \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --eval_results_path output/test_llama_eval2.json
  ```

**Evaluation Types:**
1. **Abstention Evaluation**: Determines if model appropriately refuses to answer
2. **Hallucination Evaluation**: Identifies factual errors in responses
3. **Automatic Scoring**: Uses LLM evaluators to score responses

**File Structure:**
- Links to original inference results
- Contains evaluation scores/labels
- Includes evaluator metadata
- Format: JSON with evaluation metrics

### 3. LMDB Activations (`lmdb_data/gguf/activations.lmdb`)

This database stores neural network activations captured during model inference.

**Generation Process:**
- **Server**: Activation logging server (`activation_logging/server.py`)
- **Storage**: LMDB (Lightning Memory-Mapped Database)
- **Format**: GGUF (GPT-Generated Unified Format) compatible

**Setup Steps:**

1. **Start Activation Logging Server:**
   ```bash
   # Set LMDB path (optional)
   export ACTIVATION_LMDB_PATH=lmdb_data/gguf/activations.lmdb
   
   # Start server with vLLM
   python -m activation_logging.vllm_serve \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --host 0.0.0.0 \
       --port 8000
   ```

2. **Run Inference with Activation Logging:**
   ```bash
   python -m activation_logging.run_benchmark \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --lmdb_path lmdb_data/gguf/activations.lmdb \
       --exp test_experiment
   ```

**LMDB Structure:**
- **Key**: SHA256 hash of the input prompt
- **Value**: Pickled dictionary containing:
  - `prompt`: Original input text
  - `response`: Generated response
  - `activations`: NumPy array of neural activations
  - `all_layers_activations`: Activations from all transformer layers
  - `model`: Model identifier
  - `timestamp`: Generation timestamp

**Activation Data:**
- **Layers**: All transformer layers (not just last layer)
- **Tokens**: Per-token activations for generated text
- **Format**: NumPy arrays stored as pickled objects
- **Size**: Default LMDB map size is 16GB

## GGUF Format

GGUF (GPT-Generated Unified Format) is a file format for storing large language models:

- **Purpose**: Efficient storage and loading of quantized models
- **Benefits**: Reduced memory usage, faster inference
- **Usage**: Compatible with llama.cpp and similar inference engines
- **Integration**: The activation logging system supports both HuggingFace and GGUF models

## Usage Examples

### Complete Pipeline Example:
```bash
# Step 0: Generate test questions/prompts (prerequisite)
source halu/bin/activate && \
python -m tasks.shortform.precise_wikiqa \
    --do_generate_prompt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --wiki_src goodwiki \
    --mode gguf \
    --inference_method vllm \
    --N 5000 \
    --q_generator Llama-3.3-70B-Instruct-IQ3_M.gguf

# Step 1-3: Run complete pipeline (inference + evaluation + activation logging)
python -m tasks.refusal_test.nonsense_mixed_entities \
    --exp nonsense_all \
    --do_inference \
    --do_eval \
    --tested_model meta-llama/Llama-3.1-8B-Instruct \
    --N 10 \
    --seed 0

# Step 4: Check LMDB contents
python activation_logging/test_check_lmdb.py <prompt_hash> lmdb_data/gguf/activations.lmdb
```

### Testing Activation Logging:
```bash
# Test basic functionality
python activation_logging/test_lmdb_logging.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lmdb_path lmdb_data/test_activations.lmdb

# Test GGUF model inference
python activation_logging/test_gguf_inference.py \
    --model-path Llama-3.3-70B-Instruct-IQ3_M.gguf \
    --test-type chat
```

## File Locations

- **Inference Results**: `output/{task_name}/{model_name}/generation.jsonl`
- **Evaluation Results**: `output/{task_name}/{model_name}/eval_results.json`
- **LMDB Activations**: `lmdb_data/gguf/activations.lmdb` (configurable)
- **Raw Evaluation**: `output/{task_name}/{model_name}/raw_eval_res.jsonl`

## Dependencies

- **vLLM**: For model serving and inference
- **LMDB**: For activation storage
- **transformers**: For HuggingFace model support
- **llama-cpp-python**: For GGUF model support
- **numpy**: For activation array handling
- **pickle**: For data serialization

## Notes

- The LMDB database can grow large (10k+ activation samples)
- Prompt hashes ensure unique identification of inference sessions
- All three files are linked by prompt content and timestamps
- The system supports both local and remote inference setups
