# Integrated Inference Pipeline Specification

## 1) Purpose
Provide a single, unified inference run that produces **all artifacts required to train/evaluate viable hallucination detectors**, so downstream methods can be run without re-generating model outputs.

## 2) Scope
**In scope**
- Response-level **binary classification** (hallucinated vs not) for QA-style benchmarks.
- One inference run that supports multiple methods in parallel.
- OpenAI API–compatible inference via vLLM, with activation logging enabled.

**Out of scope**
- Real-time entity/span localization during generation.
- Retrieval-augmented verification pipelines and external KB checks.

## 3) Target Methods (Viable)
- LLMs Know More Than They Show baselines (logprob/probing).
- SelfCheckGPT (multi-sample, text-only consistency).
- Semantic Entropy Probes (SE target + SEP probe).
- Semantic Energy (multi-sample + verifier clustering).
- Geometry of Truth / Layer-wise Semantic Dynamics (layerwise features vs reference answer).

## 4) Design Goals
- **Single pass per prompt group** with all artifacts logged once.
- **Stable IDs** to link prompt, generations, activations, and labels.
- **Configurable sampling** (primary + K stochastic samples) without changing dataset files.
- **Minimal changes to existing task runners** while replacing the old inference path.

## 5) High-Level Pipeline Flow
1. Load dataset split and construct prompts (existing task code).
2. Create a **Prompt Group** for each item:
   - `primary` generation (deterministic).
   - `k_samples` generations (stochastic).
3. Send requests to vLLM server (OpenAI protocol) with activation logging enabled.
4. Persist **generation outputs** + **request metadata** + **activation references**.
5. Persist **evaluation labels** for response-level correctness (ground truth).
6. Export standardized artifacts for each method’s training/evaluation scripts.

## 6) Required Artifacts (Single Run Output)
All artifacts must be linked by a stable `prompt_id` and a `sample_id`.

**Prompt ID Generation**
- `prompt_id`: Deterministic hash of `(dataset, split, dataset_index)` using SHA256, truncated to 16 hex chars.
- `sample_id`: Sequential integer (0 for primary, 1-K for stochastic samples).
- `activation_key`: `{prompt_id}_{sample_id}` for LMDB/storage lookups.

### 6.1 Manifest
A `run_manifest.json` that records:
- `run_id` (UUID v4), `model`, `dataset`, `split`, `timestamp` (ISO 8601), `prompt_template_hash` (SHA256).
- Sampling config: `k_samples` (int), `primary_temperature` (0.0), `sample_temperature` (float), `sample_top_p` (float), `max_tokens` (int).
- Logging config: `logging_profile` (string), `target_layers` (all/first_half/second_half), `sequence_mode` (all/prompt/response), `storage_format` (npy/npz), `dtype` (float32/float16).
- Storage paths: `artifact_root`, `activations_subdir` (default: 'activations/'), `worker_count` (int).
- vLLM server info: `model_revision`, `tensor_parallel_size`.
- Resume info: `resumable` (bool), `completed_prompt_ids` (list).

### 6.2 Prompt Index
A `prompt_index.jsonl` with one row per prompt:
- `prompt_id` (string, 16-char hex), `dataset_id` (original dataset index/identifier), `prompt` (string), `reference_answer` (string or null), `label` (int: 0=correct, 1=hallucinated), `meta` (dict with dataset-specific fields: e.g., `entity_type`, `question_type`).
- Indexed by `prompt_id` for O(1) lookup during consolidation and evaluation.

### 6.3 Generations
A `generations.jsonl` with one row per generation:
- `prompt_id`, `sample_id`, `role` (primary|sample),
- `text`, `finish_reason`, `usage` (dict: `prompt_tokens`, `completion_tokens`, `total_tokens`),
- `logprobs` (list of dicts, one per token: `{"token": str, "logprob": float, "top_logprobs": [...]}`). **Required for all generations** to support logit-based baselines.
- `request_params` (dict: `temperature`, `top_p`, `max_tokens`, `model`, `timestamp`).
- `generation_time_ms` (int): Time taken for generation in milliseconds.

### 6.4 Activations
Stored as **JSON metadata + NPY arrays** with **per-worker isolation** (see section 11.4).

**Storage Format**
- Metadata: `activation_index.jsonl` with one entry per generation.
- Arrays: `activations/{activation_key}.npy` - NumPy arrays saved with `np.save()`.
- Structure: `{artifact_root}/worker_{worker_id}/activations/{prompt_id}_{sample_id}.npy`

**Activation Index Entry**
- `prompt_id`, `sample_id`, `activation_key` ({prompt_id}_{sample_id}),
- `layers` (list of ints or "all"), `sequence_mode` (all/prompt/response),
- `prompt_token_count` (int): Number of tokens in prompt (for slicing activations),
- `response_token_count` (int): Number of tokens in response,
- `shape` (list: [num_layers, num_tokens, hidden_dim]),
- `dtype` (string: float32/float16), `size_bytes` (int),
- `file_path` (relative path: `activations/{activation_key}.npy`),
- `worker_id` (int): Which worker wrote this entry.

**Advantages over LMDB**
- Simple file operations (no transaction management).
- Easy merging (copy files + concatenate JSONL).
- Standard NumPy format (any tool can read).
- No map_size limits or resize issues.
- Optional: Apply compression with `np.savez_compressed()` if needed.

### 6.5 Method-Specific Exports (Derived)
Created by post-processing from the above artifacts:
- **SelfCheckGPT:** `selfcheck_samples.jsonl` (primary + k_samples texts).
- **Semantic Entropy/SEP:** `semantic_entropy_inputs.jsonl` (primary + samples + logprobs).
- **Semantic Energy:** `semantic_energy_inputs.jsonl` (samples + logprobs).
- **Geometry of Truth:** `geometry_inputs.jsonl` (primary + reference answer + activation_key).
- **LLMsKnow baselines:** `logprob_inputs.jsonl` (primary + logprobs).

## 7) Logging Profiles
Provide named presets for common use cases:

### Profile: `integrated` (default)
- **Layers**: All layers (`target_layers='all'`)
- **Sequence**: Response tokens only (`sequence_mode='response'`)
- **Logprobs**: All tokens (`logprobs=5` top alternatives)
- **Storage format**: NPY files (float32), optional compression with `np.savez_compressed()`
- **Samples**: K stochastic samples (configurable, default K=5)
- **Storage estimate**: ~500MB per 1K prompts for 7B model (model-dependent)
- **Supports**: All viable methods (probes, geometry, semantic entropy, SelfCheckGPT)

### Profile: `light`
- **Layers**: Second half only (`target_layers='second_half'`)
- **Sequence**: Last token only (custom slice)
- **Logprobs**: All tokens (`logprobs=5`)
- **Storage format**: NPY files (float16 to reduce size)
- **Samples**: K stochastic samples
- **Storage estimate**: ~50MB per 1K prompts
- **Supports**: Logit baselines, semantic methods (NOT geometry/full probing)

### Profile: `full`
- **Layers**: All layers
- **Sequence**: Prompt + response (`sequence_mode='all'`)
- **Logprobs**: All tokens
- **Storage format**: NPY files (float32)
- **Samples**: K stochastic samples
- **Storage estimate**: ~2GB per 1K prompts for 7B model
- **Supports**: All methods + prompt analysis

## 8) Sampling Policy
- **Primary generation**: deterministic (temperature=0, top_p=1).
- **Samples**: stochastic (temperature configurable, e.g., 0.7; top_p configurable).
- All samples share the same `prompt_id` and a unique `sample_id`.

## 9) Dataset/Task Compatibility
The pipeline should be invoked similarly to `run_with_server.py` but with a new inference entry point that:
- Accepts all current task options (precisewikiqa, longwiki, mixedentities, triviaqa).
- Produces the unified artifacts regardless of task.
- **Supports resume behavior** via:
  - Global `completed_prompts.json` file with completed `prompt_id` set (atomic updates via file lock).
  - Workers check this file before processing each prompt.
  - On completion, worker atomically adds `prompt_id` to completed set.
  - If process crashes, next run reads completed set and skips those prompts.
  - Validation step checks artifact completeness (all K+1 samples exist for each completed prompt_id).

## 10) Interface (Proposed CLI)
New script: `run_integrated_inference.py` mirrors the old interface and adds:
- `--integrated` (enable the new pipeline, default: True).
- `--k-samples` (int, default: 5): Number of stochastic samples per prompt.
- `--primary-temperature` (float, default: 0.0): Temperature for primary generation.
- `--sample-temperature` (float, default: 0.7): Temperature for stochastic samples.
- `--sample-top-p` (float, default: 0.95): Top-p for stochastic samples.
- `--logging-profile` (str, default: 'integrated'): One of [integrated, light, full].
- `--artifact-root` (path): Output directory for all artifacts.
- `--num-workers` (int, default: 1): Number of parallel workers.
- `--gpus` (str): Comma-separated GPU IDs to use (e.g., '0,1,2'). Orchestrator spawns one server per GPU.
- `--manage-servers` (bool, default: True): If True, orchestrator spawns/manages vLLM servers. If False, connect to existing servers.
- `--server-base-port` (int, default: 8000): Base port for servers (GPU i uses port base_port + i).
- `--resume` (bool, default: True): Enable resume from completed_prompts.json.
- `--validate-only` (bool): Validate existing artifacts without running inference.
- `--consolidate-only` (bool): Only run consolidation step on existing worker outputs.

## 11) Multi-GPU Parallelization
Support parallel inference across multiple GPUs using a **single orchestrator** architecture.

### Architecture
**Single Orchestrator with Multiple Workers**
- One orchestrator process manages N worker threads/processes.
- **Ideal setup**: Orchestrator spawns vLLM server processes (one per GPU), then spawns workers to consume from those servers.
- Orchestrator manages full lifecycle: server startup → worker assignment → consolidation → server shutdown.
- Each worker connects to its assigned vLLM server via HTTP.
- **Coordination**: In-memory state (orchestrator tracks completion, no external DB needed).
- **Resume**: Orchestrator writes `completed_prompts.json` periodically; on restart, loads and skips completed.

**Server Management**
- Orchestrator spawns one `vllm serve` process per GPU:
  ```python
  # Example: Start servers on GPUs 0, 1, 2
  servers = []
  for gpu_id in [0, 1, 2]:
      cmd = ['vllm', 'serve', model, '--host', '0.0.0.0', '--port', str(8000 + gpu_id)]
      env = os.environ.copy()
      env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
      servers.append(subprocess.Popen(cmd, env=env))
  ```
- Wait for all servers to be healthy before starting workers.
- On completion or error, orchestrator gracefully shuts down all servers.

**Worker Coordination**
- Orchestrator maintains in-memory state:
  ```python
  prompt_queue = Queue(all_prompt_ids)
  completed_prompts = set()  # Loaded from completed_prompts.json if resuming
  worker_assignments = {}  # worker_id -> (server_url, current_prompt_id)
  ```
- Each worker:
  1. Requests next `prompt_id` from orchestrator (thread-safe queue pop).
  2. Generates primary + K samples via assigned server.
  3. Saves outputs to isolated directory.
  4. Notifies orchestrator of completion → orchestrator updates completed set.
- Periodic checkpointing: orchestrator writes `completed_prompts.json` every N prompts.

**Activation Storage Isolation**
- Each worker writes to separate directory: `{artifact_root}/worker_{worker_id}/activations/`
- Activations saved as individual NPY files: `{activation_key}.npy`
- Metadata in worker's `activation_index.jsonl`
- No concurrent write conflicts (one writer per directory).
- Consolidation step (see 11.2) copies NPY files and merges metadata.

**Requirements**
- Per-worker output directories: `{artifact_root}/worker_{worker_id}/`
- GPU assignments: orchestrator knows which GPUs to use (e.g., `--gpus 0,1,2`).
- Validation: Ensure all prompts completed exactly once with K+1 samples.

### 11.2) Consolidation
Merge outputs from all workers into a single consolidated artifact set.

**Per-Worker Artifacts**
Each worker writes to isolated directories:
```
{artifact_root}/
├── run_manifest.json                  # Global manifest (created by orchestrator at start)
├── completed_prompts.json             # Completed prompt_ids (maintained by orchestrator)
├── worker_0/
│   ├── prompt_index.jsonl            # Worker 0's prompts
│   ├── generations.jsonl             # Worker 0's generations
│   ├── activation_index.jsonl        # Worker 0's activation metadata
│   └── activations/                  # Worker 0's activation arrays
│       ├── {prompt_id}_0.npy         # Primary generation activations
│       ├── {prompt_id}_1.npy         # Sample 1 activations
│       └── ...
├── worker_1/
│   └── ...
└── consolidated/                      # Created by consolidation step
    ├── prompt_index.jsonl            # Merged from all workers
    ├── generations.jsonl             # Merged from all workers
    ├── activation_index.jsonl        # Merged from all workers
    └── activations/                  # Merged activation arrays
        ├── {prompt_id}_0.npy
        ├── {prompt_id}_1.npy
        └── ...
```

**Consolidation Steps**
1. **Validate worker artifacts**: Check each worker's outputs for completeness (no partial prompts, verify NPY files exist).
2. **Merge JSONL files**: Concatenate `prompt_index.jsonl`, `generations.jsonl`, `activation_index.jsonl` from all workers.
3. **Deduplicate**: Remove any duplicate `prompt_id` entries (shouldn't exist if coordination works).
4. **Copy activation arrays**: Copy all NPY files from worker directories to consolidated:
   ```python
   import shutil
   from pathlib import Path
   
   consolidated_activations = Path(artifact_root) / 'consolidated' / 'activations'
   consolidated_activations.mkdir(parents=True, exist_ok=True)
   
   for worker_id in range(num_workers):
       worker_activations = Path(artifact_root) / f'worker_{worker_id}' / 'activations'
       for npy_file in worker_activations.glob('*.npy'):
           shutil.copy2(npy_file, consolidated_activations / npy_file.name)
   ```
5. **Update manifest**: Add consolidation metadata (timestamp, num_prompts, total_size_bytes, num_activations).
6. **Generate method exports**: Create method-specific files (section 6.5) from consolidated artifacts.

**Resume Support (Orchestrator Mode)**
- Orchestrator p**
- Orchestrator periodically writes `completed_prompts.json` (e.g., every 100 prompts).
- On restart, orchestrator loads completed set and removes from work queue.
- Each worker's partial outputs remain intact; consolidation handles deduplication.

**Conflict Prevention**
- In-memory coordination by orchestrator = no conflicts between workers.
- NPY file writes are isolated per worker (unique activation_keys = no filename collisions).
- JSONL files are append-only within a worker (no conflicts).
- Simple filesystem operations (no transaction management needed)
**Conflict Prevention**
- **Orchestrator mode**: In-memory coordination = no conflicts.
- **Independent mode**: File locks on `completed_prompts.json` ensure atomic updatesas src_env:
               with dst_env.begin(write=True) as dst_txn:
                   with src_env.begin() as src_txn:
                       cursor = src_txn.cursor()
                       for key, value in cursor:
                           dst_txn.put(key, value)
   ```
5. **Update manifest**: Add consolidation metadata (timestamp, num_prompts, total_size).
6. **Generate method exports**: Create method-specific files (section 6.5) from consolidated artifacts.

**Resume Support**
- Orchestrator periodically writes `completed_prompts.json` (e.g., every 100 prompts).
- On restart, orchestrator loads completed set and removes from work queue.
- Each worker's partial outputs remain intact; consolidation handles deduplication.

**Conflict Prevention**
- In-memory coordination by orchestrator = no conflicts between workers.
- NPY file writes are isolated per worker (unique activation_keys = no filename collisions).
- JSONL files are append-only within a worker (no conflicts).
- Simple filesystem operations (no transaction management needed).

## 12) Error Handling

### Server Errors
- **vLLM timeout (>60s)**: Retry up to 3 times with exponential backoff (1s, 2s, 4s).
- **vLLM crash**: Worker logs error, writes to `errors.jsonl`, continues to next prompt.
- **Activation logging failure**: Log warning, store generation without activations, continue (activations are optional for some methods).

### Worker Errors
- **Worker crash**: On restart, orchestrator resumes from `completed_prompts.json` (skips completed, retries unclaimed).
- **Storage full**: Worker detects NPY save failure (disk full), logs error, gracefully shuts down. Admin must expand storage and restart.
- **File write permission errors**: Worker logs error with file path, skips prompt, continues to next.

### Validation Errors
- **Missing samples**: If prompt has <K+1 samples in consolidated output, mark as incomplete in validation report.
- **Orphaned activations**: Activation entries without corresponding generation records (logged as warning).
- **Schema violations**: JSONL entries that don't match expected schema (logged as error with line number).

### Recovery Strategy
- All errors logged to `{artifact_root}/worker_{worker_id}/errors.jsonl` with:
  - `timestamp`, `error_type`, `prompt_id`, `sample_id`, `error_message`, `stack_trace`.
- Validation script (`validate_artifacts.py`) checks for incomplete prompts and generates repair commands.
- Repair mode: `--retry-failed` flag reprocesses failed prompts from errors.jsonl.

## 13) Acceptance Criteria
- One inference run produces all required artifacts.
- The pipeline **supports resume**: it can continue from partially completed runs without duplicating or corrupting artifacts.
- Each viable method can run **without re-contacting the model**.
- Artifacts include a stable mapping of `prompt_id` → `sample_id` → activations/logprobs.
- Binary response-level labels are stored in the prompt index.
- Multi-worker runs complete without write conflicts or data corruption.
- Consolidation step produces valid, de-duplicated artifacts.
- Validation script confirms artifact integrity and completeness.

## 14) Future Extensions (Non-Blocking)
- Add optional prompt-token activations for prompt-engineering analysis.
- Add per-token alignment metadata for entity/span methods (not required now).
- Implement streaming consolidation to reduce peak memory usage.
- Support for distributed storage backends (S3, GCS) in addition to POSIX filesystems.
- Real-time monitoring dashboard for multi-worker progress.
- Automatic dataset sharding optimization based on prompt length distribution.
