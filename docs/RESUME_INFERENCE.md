# Resumable Inference Feature

## Overview

The inference pipeline now supports automatic resume functionality with **incremental saving**. If your inference job is interrupted (e.g., due to cluster time limits, crashes, or manual stops), you can simply re-run the same command and it will automatically resume from where it left off.

## How It Works

1. **Single-Threaded Sequential Processing**: Inference runs in a single thread, processing one prompt at a time. This is optimal because:
   - The vLLM server handles parallelism internally
   - Client-side parallelism adds complexity without benefit
   - Sequential processing makes incremental saving simpler and safer

2. **Incremental Saving**: **Each sample is saved to disk immediately after generation** (with `flush()` to ensure it's written). This means no progress is lost if the job crashes - every completed sample is already safely on disk.

3. **Deterministic Ordering**: Questions are now selected deterministically (first N after filtering) instead of randomly, ensuring the same set of questions is selected each time.

4. **Automatic Detection**: When starting inference, the system checks if a generations file already exists at the target path.

5. **Smart Filtering**: If existing generations are found, the system:
   - Loads the existing generations
   - Identifies which prompts have already been processed (by matching prompt text)
   - Filters them out from the current batch
   - Only processes the remaining prompts

6. **Append Mode**: When resuming, new generations are appended to the existing file, preserving all previous work.

## Usage

### Basic Usage (Resume Enabled by Default)

```bash
# Run inference with 60k samples
python scripts/run_with_server.py \
  --step inference \
  --task precisewikiqa \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --wiki_src goodwiki \
  --mode gguf_big \
  --inference_method vllm \
  --logger-type json \
  --activations-path goodwiki_json_2/activations.json \
  --max_inference_tokens 64 \
  --N 60000 \
  --qa_output_path data/precise_qa/save/qa_goodwiki_Llama-3.1-8B-Instruct_gguf_big.jsonl \
  --generations_file_path goodwiki_json_2/generation.jsonl

# If interrupted, simply run the EXACT SAME command again
# It will automatically resume from where it left off
```

### Disable Resume (Start Fresh)

If you want to start from scratch and ignore existing generations:

```bash
python scripts/run_with_server.py \
  --step inference \
  --task precisewikiqa \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --N 60000 \
  --no-resume \
  ...
```

## Example Output

When resuming, you'll see output like:

```
📂 Found existing generations file: goodwiki_json_2/generation.jsonl
✅ Loaded 15000 existing generations
📊 Resume statistics:
   - Total prompts: 60000
   - Already completed: 15000
   - Remaining to process: 45000
   - Progress: 25.0%
Client logging initialized for 45000 remaining requests
📊 Starting inference: 45000 remaining requests to process (total: 60000, completed: 15000)
```

And in the client logs, you'll see progress continue from where it left off:

```
2025-11-28 16:06:06 | INFO | [CLIENT 307ba024] Starting API call - Model: meta-llama/Llama-3.1-8B-Instruct, Prompt length: 103 chars
2025-11-28 16:06:06 | INFO | [CLIENT 307ba024] Progress: 15000/60000 completed, 45000 remaining
```

When all prompts are already processed:

```
📂 Found existing generations file: goodwiki_json_2/generation.jsonl
✅ Loaded 60000 existing generations
✅ All 60000 prompts already processed! Nothing to do.
```

## Important Notes

1. **Deterministic Selection**: The change from random sampling to deterministic selection means you'll always get the same questions for a given N value. This is necessary for resume to work correctly.

2. **Matching by Prompt**: The system matches prompts by their exact text content. Make sure you use the same preprocessing/formatting when resuming.

3. **Supported Tasks**: Currently implemented for:
   - PreciseWikiQA
   - TriviaQA
   - (Can be extended to other tasks as needed)

4. **Activation Logging**: When resuming, only the new prompts will have their activations logged. Existing prompts won't be re-processed.

## Benefits for Large-Scale Experiments

- **Cluster-Friendly**: Perfect for shared research clusters with time limits
- **Fault-Tolerant**: Automatically recovers from crashes or interruptions
- **No Duplicate Work**: Avoids re-processing already completed samples
- **Progress Tracking**: Clear visibility into what's been completed and what remains

## Technical Details

### Modified Files

1. **utils/exp.py**: Added resume logic to `run_exp()` function
2. **tasks/shortform/precise_wikiqa.py**: Changed from random to deterministic sampling, added `--no-resume` flag
3. **tasks/triviaqa/triviaqa.py**: Added `--no-resume` flag support
4. **scripts/run_with_server.py**: Added `--no-resume` flag and parameter passing

### Resume Logic

The resume logic in `utils/exp.py`:
- Loads existing generations file (if it exists and resume=True)
- Creates a set of already-processed prompts
- Filters the input DataFrame to exclude processed prompts
- Processes only remaining prompts
- Merges new and existing generations before saving

