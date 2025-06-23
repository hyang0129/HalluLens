# Inference JSONL Reconstruction

This directory contains scripts to reconstruct inference JSONL files from activations logger metadata and original QA data.

## Overview

When running inference with activation logging enabled, the original inference JSONL file (containing prompts, responses, and metadata) may not be saved directly. However, we can reconstruct it by combining:

1. **Original QA data**: The questions and answers used for inference (from `QA_OUTPUT_PATH`)
2. **Activations logger metadata**: The generated responses and metadata stored in the LMDB file

## Important Note: Prompt Formatting

The script handles the difference between:
- **QA data format**: Raw questions like `"What is the name of the actress who played the role of Sarah Packard?"`
- **Model input format**: Formatted prompts like `"user: Answer in one sentence. Q:What is the name of the actress who played the role of Sarah Packard?\n A:"`

The reconstruction script automatically formats the raw questions to match what was actually sent to the model for proper matching.

## Files

- `reconstruct_inference.py`: Main reconstruction script
- `example_reconstruct.py`: Example usage script
- `README_reconstruction.md`: This documentation file

## Usage

### Basic Usage

```bash
python activation_logging/reconstruct_inference.py \
    --qa_output_path /path/to/qa_data.jsonl \
    --lmdb_path /path/to/activations.lmdb \
    --output_path /path/to/reconstructed_inference.jsonl
```

### Parameters

- `--qa_output_path`: Path to the original QA JSONL file (required)
- `--lmdb_path`: Path to the activations logger LMDB file (required)
- `--output_path`: Path to save the reconstructed inference JSONL file (required)
- `--verbose`: Enable verbose logging (optional)

### Example

```bash
# Reconstruct inference JSONL for precise_wikiqa experiment
python activation_logging/reconstruct_inference.py \
    --qa_output_path data/precise_qa/save/qa_goodwiki_Llama-3.1-8B-Instruct_dynamic.jsonl \
    --lmdb_path lmdb_data/precise_wikiqa_activations.lmdb \
    --output_path output/precise_wikiqa_goodwiki_dynamic/Llama-3.1-8B-Instruct/reconstructed_generation.jsonl \
    --verbose
```

## How It Works

1. **Load QA Data**: Reads the original questions and answers from the JSONL file
2. **Load Activations Metadata**: Extracts metadata from the LMDB file (prompts, responses, model info, etc.)
3. **Format Prompts**: Converts raw questions to the formatted prompt format sent to the model
4. **Match by Prompt Hash**: Uses SHA-256 hashing to match formatted prompts between QA data and activations
5. **Combine Data**: Merges original QA fields with generated responses and metadata
6. **Save Result**: Writes the reconstructed inference JSONL file

## Output Format

The reconstructed JSONL file contains all original QA fields plus:

- `generation`: The model's generated response
- `model`: The model used for generation
- `input_length`: Number of input tokens
- `timestamp`: When the generation was logged
- `messages`: Chat messages (if applicable)
- `trim_position`: Response trim position (if applicable)

## Example Output Entry

```json
{
  "index": 0,
  "title": "Example Article",
  "h_score_cat": 8,
  "pageid": 12345,
  "revid": 67890,
  "description": "Example description",
  "categories": ["Category1", "Category2"],
  "reference": "Example reference text...",
  "prompt": "What is the name of the actress who played the role of Sarah Packard?",
  "answer": "Sarah Jessica Parker",
  "generation": "Sarah Jessica Parker played the role of Sarah Packard.",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "input_length": 15,
  "timestamp": 1703123456.789
}
```

## Troubleshooting

### Common Issues

1. **No matches found**: 
   - Ensure the prompt format in QA data matches the expected raw question format
   - Check that the activations LMDB contains entries for the same model and experiment
   - Use `--verbose` to see detailed matching information

2. **LMDB not found**: Check that the LMDB path is correct and the file exists

3. **QA file not found**: Verify the path to the original QA data file

### Debugging

Use the `--verbose` flag to see detailed logging information:

```bash
python activation_logging/reconstruct_inference.py \
    --qa_output_path /path/to/qa_data.jsonl \
    --lmdb_path /path/to/activations.lmdb \
    --output_path /path/to/output.jsonl \
    --verbose
```

This will show:
- Number of QA entries loaded
- Number of activation entries found
- Formatted prompt hashes for debugging
- Matching statistics

### Verification

After reconstruction, you can verify the output by:

1. Checking the number of entries matches expectations
2. Examining a few sample entries to ensure data integrity
3. Using the reconstructed file for evaluation (e.g., with `PreciseQAEval`)

## Integration with Evaluation

The reconstructed inference JSONL file can be used directly with the evaluation scripts:

```python
from tasks.shortform.precise_wikiqa import PreciseQAEval

# Use the reconstructed file for evaluation
eval = PreciseQAEval(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    TASKNAME="precise_wikiqa_goodwiki_dynamic",
    generations_file_path="output/reconstructed_inference.jsonl"
)
eval.run_eval()
```

## Notes

- The script preserves all original QA data fields
- Only prompts that have corresponding activations in the LMDB will be included
- The script handles missing or corrupted entries gracefully
- Large LMDB files may take some time to process
- Prompt formatting is automatically handled to match the model input format 