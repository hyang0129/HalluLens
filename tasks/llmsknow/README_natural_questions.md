# Natural Questions (NQ) - HalluLens Implementation

Natural Questions is a question answering dataset featuring real queries from Google Search paired with answers from Wikipedia. This implementation provides activation logging compatible inference and evaluation for the HalluLens framework.

## Dataset Source

- **Original Paper**: [Natural Questions: A Benchmark for Question Answering Research](https://ai.google/research/NaturalQuestions/)
- **Data Location**: `external/LLMsKnow/data/nq_wc_dataset.csv`
- **Format**: CSV with columns: Question, Answer, Context
- **Size**: ~3,000+ real-world questions

## Key Features

- ✅ **No Question Generation Needed**: Uses pre-existing real-world questions
- ✅ **Automatic Evaluation**: String matching (no LLM judge required)
- ✅ **Activation Logging**: Fully compatible with vLLM activation logging
- ✅ **Resume Support**: Can resume interrupted inference runs
- ✅ **Real-World Queries**: Based on actual Google Search queries

## Data Format

Each sample contains:
- `question`: Real user query from Google Search
- `answer`: Ground truth answer from Wikipedia
- `context`: Supporting Wikipedia passage (optional)

Example:
```
Question: where was miss marple with joan hickson filmed?
Answer: Norfolk
Context: <P> BBC producer Guy Slater cast Joan Hickson as Miss Marple...
```

## Usage

### Basic Inference and Evaluation

```bash
# Run both inference and evaluation (recommended)
python scripts/run_with_server.py \
    --step all \
    --task naturalquestions \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 1000
```

### Inference Only

```bash
python scripts/run_with_server.py \
    --step inference \
    --task naturalquestions \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --N 500 \
    --max_tokens 64 \
    --temperature 0.0
```

### Evaluation Only

```bash
python scripts/run_with_server.py \
    --step eval \
    --task naturalquestions \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Direct Script Usage

```bash
# Inference
python -m tasks.llmsknow.natural_questions \
    --do_inference \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --N 1000 \
    --max_tokens 64

# Evaluation
python -m tasks.llmsknow.natural_questions \
    --do_eval \
    --model mistralai/Mistral-7B-Instruct-v0.2

# Both
python -m tasks.llmsknow.natural_questions \
    --do_inference \
    --do_eval \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --N 1000
```

## Output Structure

```
output/natural_questions/{model_name}/
├── generation.jsonl          # Model generations with prompts and answers
├── eval_results.json         # Summary metrics
└── raw_eval_res.jsonl        # Detailed per-sample results
```

### generation.jsonl Format

```json
{
    "question": "where was miss marple with joan hickson filmed?",
    "answer": "Norfolk",
    "context": "<P> BBC producer Guy Slater cast Joan Hickson...",
    "prompt": "Answer the question concisely.\n\nQuestion: where was...",
    "generation": "Norfolk",
    "model": "mistralai/Mistral-7B-Instruct-v0.2"
}
```

### eval_results.json Format

```json
{
    "model": "Mistral-7B-Instruct-v0.2",
    "halu_Rate": 0.25,
    "refusal_Rate": 0.0,
    "correct_rate": 0.75,
    "accurate_count": 750,
    "hallu_count": 250,
    "total_count": 1000,
    "refusal_count": 0,
    "hallucination_evaluation": "string_matching"
}
```

## Evaluation Method

Natural Questions uses **substring matching** for evaluation:
- Answer is considered **correct** if the ground truth answer appears in the model's response (case-insensitive)
- No LLM judge needed - fast and deterministic
- Compatible with PreciseWikiQA result format for unified analysis

## Comparison with TriviaQA

| Feature | Natural Questions | TriviaQA |
|---------|------------------|----------|
| **Question Source** | Real Google queries | Trivia competitions |
| **Answer Format** | Single answer | Multiple aliases |
| **Evaluation** | Substring match | Multi-alias substring match |
| **Context** | Wikipedia passages | Multiple sources |
| **Difficulty** | Real-world queries | Trivia knowledge |

## Parameters

### Inference Parameters

- `--model`: Model name or path (required)
- `--N`: Number of samples to process (default: all samples)
- `--max_tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 0.0)
- `--inference_method`: Inference method (default: vllm)

### Data Parameters

- `--data_dir`: Directory containing nq_wc_dataset.csv (default: external/LLMsKnow/data)
- `--output_dir`: Base output directory (default: output)

### File Parameters

- `--generations_file_path`: Custom path for generations file
- `--eval_results_path`: Custom path for evaluation results
- `--log_file`: Log file path

### Debug Parameters

- `--quick_debug_mode`: Use only first 50 samples for testing

## Integration with Activation Logging

Natural Questions is fully integrated with the HalluLens activation logging system:

1. **Automatic Activation Capture**: When using `run_with_server.py`, activations are automatically logged
2. **Per-Token Activations**: Last-layer activations captured for each generated token
3. **Binary Labels**: Each sample gets a binary hallucination label (correct/incorrect)
4. **LMDB Storage**: Activations stored in LMDB with prompt hash as key

## Expected Results

Based on LLMsKnow benchmarks (approximate):

| Model | Accuracy | Error Rate |
|-------|----------|------------|
| Llama-3-8B-Instruct | ~65-70% | ~30-35% |
| Mistral-7B-Instruct | ~60-65% | ~35-40% |

*Note: Results may vary based on prompt format and generation parameters*

## Troubleshooting

### Data File Not Found

```
FileNotFoundError: Natural Questions data file not found
```

**Solution**: Ensure `external/LLMsKnow/data/nq_wc_dataset.csv` exists in your repository.

### Resume Not Working

If inference doesn't resume properly:
1. Check that `generation.jsonl` exists in the output directory
2. Verify the file is valid JSONL (one JSON object per line)
3. Corrupted file? Delete it and restart

### Evaluation Accuracy Seems Low

Natural Questions has challenging real-world queries. Consider:
- Increase `max_tokens` if answers are being cut off
- Try temperature 0.0 for deterministic outputs
- Check if model is generating explanations instead of direct answers

## Citation

If you use Natural Questions in your research, please cite:

```bibtex
@article{kwiatkowski2019natural,
  title={Natural questions: a benchmark for question answering research},
  author={Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others},
  journal={Transactions of the Association for Computational Linguistics},
  volume={7},
  pages={453--466},
  year={2019},
  publisher={MIT Press}
}
```

## Related Documentation

- [DATASET_ROADMAP.md](../../DATASET_ROADMAP.md) - Complete dataset planning
- [DATASET_IMPLEMENTATION_STATUS.md](../../DATASET_IMPLEMENTATION_STATUS.md) - Implementation status
- [activation_logging/README.md](../../activation_logging/README.md) - Activation logging infrastructure
- [external/LLMsKnow/README.md](../../external/LLMsKnow/README.md) - Original LLMsKnow benchmark
