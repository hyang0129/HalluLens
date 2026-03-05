# JSON Activation Logging

This document describes the JSON-based activation logging functionality that provides an alternative to LMDB storage for LLM activations.

## Overview

The JSON activation logger stores activations in a human-readable, file-based format that's easier to inspect and debug compared to LMDB. Each activation is stored as a separate JSON file, with metadata maintained in a central index.

## File Structure

```
output_directory/
├── metadata.json          # Central metadata file
└── activations/
    ├── <hash1>.json       # Activation data for entry 1
    ├── <hash2>.json       # Activation data for entry 2
    └── ...
```

### Metadata File (`metadata.json`)

Contains configuration and metadata for all logged entries:

```json
{
  "logger_config": {
    "target_layers": "all",
    "sequence_mode": "all", 
    "version": "1.0"
  },
  "entries": {
    "prompt_hash_1": {
      "prompt": "What is the capital of France?",
      "response": "The capital of France is Paris.",
      "model": "test-model",
      "input_length": 10,
      "timestamp": 1234567890.0,
      "has_activations": true,
      "activation_file": "activations/prompt_hash_1.json",
      "logging_config": {
        "target_layers": "all",
        "sequence_mode": "all"
      }
    }
  }
}
```

### Activation Files (`activations/<hash>.json`)

Contains the actual activation tensors in JSON format:

```json
{
  "all_layers_activations": [
    {
      "_type": "torch.Tensor",
      "data": [[[...]]],
      "shape": [1, 15, 4096],
      "dtype": "torch.float32"
    },
    null,
    ...
  ]
}
```

## Usage

### Command Line

Start the server with JSON logging:

```bash
# Basic JSON logging
python -m activation_logging.vllm_serve \
  --logger-type json \
  --activations-path json_data/activations

# With specific layer and sequence settings
python -m activation_logging.vllm_serve \
  --logger-type json \
  --activations-path json_data/activations \
  --target-layers all \
  --sequence-mode response
```

### Environment Variables

```bash
export ACTIVATION_STORAGE_PATH="json_data/activations"
export ACTIVATION_LOGGER_TYPE="json"
export ACTIVATION_TARGET_LAYERS="all"
export ACTIVATION_SEQUENCE_MODE="all"
```

### Python API

```python
from activation_logging.activations_logger import JsonActivationsLogger

# Initialize logger
logger = JsonActivationsLogger(
    output_dir="json_data/activations",
    target_layers="all",
    sequence_mode="all",
    read_only=False
)

# Log an entry
logger.log_entry(key, {
    "prompt": "What is 2+2?",
    "response": "4",
    "model_outputs": model_outputs,
    "input_length": 5,
    "model": "test-model"
})

# Retrieve entries
entry = logger.get_entry(key)
metadata_only = logger.get_entry(key, metadata_only=True)

# List all entries
keys = logger.list_entries()

# Search entries
results = logger.search_metadata(lambda meta: "France" in meta.get("prompt", ""))

logger.close()
```

## Compatibility

The JsonActivationsLogger is designed to be compatible with the existing activation parser:

```python
from activation_logging.activation_parser import ActivationParser

# Use with JSON logger
parser = ActivationParser(
    inference_json="inference.jsonl",
    eval_json="eval.json", 
    activations_path="json_data/activations",
    logger_type="json"
)

# Get dataset (same API as LMDB)
dataset = parser.get_dataset("train")
```

## Advantages of JSON Logging

1. **Human Readable**: JSON files can be inspected directly
2. **Debugging Friendly**: Easy to examine individual activations
3. **No Dependencies**: No LMDB library required
4. **Portable**: Files can be easily copied and shared
5. **Incremental**: Each activation is a separate file

## Disadvantages

1. **Storage Overhead**: JSON format is less space-efficient than LMDB
2. **Performance**: Slower for large-scale operations
3. **File System Limits**: Many small files can impact filesystem performance

## Migration

### From LMDB to JSON

```python
from activation_logging.activations_logger import ActivationsLogger, JsonActivationsLogger

# Load from LMDB
lmdb_logger = ActivationsLogger(lmdb_path="lmdb_data/activations.lmdb", read_only=True)

# Create JSON logger
json_logger = JsonActivationsLogger(output_dir="json_data/activations")

# Migrate entries
for key in lmdb_logger.list_entries():
    entry = lmdb_logger.get_entry_by_key(key)
    json_logger.log_entry(key, entry)

lmdb_logger.close()
json_logger.close()
```

### From JSON to LMDB

```python
from activation_logging.activations_logger import ActivationsLogger, JsonActivationsLogger

# Load from JSON
json_logger = JsonActivationsLogger(output_dir="json_data/activations", read_only=True)

# Create LMDB logger
lmdb_logger = ActivationsLogger(lmdb_path="lmdb_data/activations.lmdb")

# Migrate entries
for key in json_logger.list_entries():
    entry = json_logger.get_entry(key)
    lmdb_logger.log_entry(key, entry)

json_logger.close()
lmdb_logger.close()
```

## Testing

Test the JSON activation logging:

```bash
# Start server with JSON logging
python -m activation_logging.vllm_serve --logger-type json --activations-path json_test_data

# Run tests
python test_json_server.py

# Show usage examples
python test_json_server.py --examples
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Directory for JSON files | `"json_data/"` |
| `target_layers` | Which layers to extract | `"all"` |
| `sequence_mode` | Which tokens to extract | `"all"` |
| `read_only` | Read-only mode | `False` |

### Target Layers
- `"all"`: Extract all layers
- `"first_half"`: Extract first half of layers
- `"second_half"`: Extract second half of layers

### Sequence Mode
- `"all"`: Extract activations for full sequence
- `"prompt"`: Extract activations for prompt tokens only
- `"response"`: Extract activations for response tokens only
