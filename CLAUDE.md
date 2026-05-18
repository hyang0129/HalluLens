# HalluLens — Claude Code Guidelines

## Paper Writing

The EMNLP 2026 paper draft lives in `paper/`. **When writing the paper, only reference files inside `paper/`, unless the user explicitly names a source outside it.** This applies to drafting, editing, or extending sections, outline, abstract, claims, tables, figures, and bibliography — and covers reading, citing, comparing against, or otherwise consulting external files, not just copying prose from them.

Concretely: do not read `PAPER_ROADMAP.md`, `THEORETICAL_JUSTIFICATION.md`, `docs/planning/`, `results/`, the SOTA tracker, `README.md`, or any other location outside `paper/` when working on paper content. If a fact, number, or framing from outside `paper/` needs to land in the draft, the user will name the source explicitly.

Rationale: the roadmap, theory doc, and SOTA tracker are *planning material* written for collaborators, not reviewers. The user controls when and how content bridges from planning into the paper — auto-importing or even tacitly cross-referencing that material loses voice and short-circuits the editorial pass.

The single source of truth for numbers cited in the paper is `paper/data/*.csv` and `paper/generated/figures/*.numbers.csv`, accessed through the `\result`/`\resdelta`/`\resratio`/`\resultCI`/`\resultPM` macros defined in `paper/macros.tex`. See `paper/README.md` for the build pipeline.

## Project Overview

Research codebase for detecting hallucinations in LLMs via mutual information analysis of intermediate layer activations. Pairs contrastive representation learning with benchmark evaluation (PreciseWikiQA, LLMsKnow).

Paper: [arXiv:2504.17550](https://arxiv.org/abs/2504.17550)
License: CC-BY-NC

## Environment

- Python 3.12 (conda env: `hallulens`)
- Install: `pip install -r requirements.txt`
- Key deps: PyTorch, vLLM, Transformers, FastAPI, LMDB, Zarr

## Project Structure

```
activation_logging/     # Activation capture infrastructure
  server.py             # OpenAI-compatible API server with activation hooks
  vllm_serve.py         # vLLM-based inference server
  activations_logger.py # LMDB-based storage
  zarr_activations_logger.py # Zarr-based storage (preferred for large-scale)
  activation_parser.py  # Parse/load stored activations

activation_research/    # Sub-model training and evaluation
  model.py              # Classifier architectures
  training.py           # Contrastive learning training loop
  trainer.py            # New trainer implementation
  evaluation.py         # Evaluation procedures
  metrics.py            # Performance metrics

tasks/                  # Benchmark task definitions
  shortform/precise_wikiqa.py  # Primary working benchmark

scripts/                # Experiment automation
  run_with_server.py    # Unified experiment runner (recommended)
  train_activation_model.py  # CLI training script
  task1_precisewikiqa.sh

utils/                  # Shared utilities
external/LLMsKnow/      # External benchmark suite (extension target)
```

## Supported Models

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | Primary baseline |
| Qwen3 8B | `Qwen/Qwen3-8B` | Thinking mode disabled automatically |

Both models work with all LLMsKnow tasks via the default batch inference path.

## Key Commands

### LLMsKnow batch inference (default — no server needed)

Tasks: `hotpotqa`, `mmlu`, `natural_questions`, `popqa`, `sciq`, `searchqa`

> **Note:** `movies` is excluded — it has no train split so it cannot be used for classifier training.

```bash
python scripts/run_with_server.py \
    --step inference \
    --task hotpotqa \
    --model Qwen/Qwen3-8B \
    --activations-path shared/hotpotqa_qwen3/activations.zarr
```

- `--batch-size` defaults to 32 — no need to pass it explicitly
- No server is started; inference runs via `HFTransformersAdapter` directly
- Expected throughput: **~16 samp/s** after model load (steady-state, H200)
- For eval after inference: `--step eval` (same command, replace `inference` with `eval`)
- For both in one go: `--step all`

To use Llama instead, swap `--model meta-llama/Llama-3.1-8B-Instruct`.
To override batch size: `--batch-size 16`. To force vLLM server: `--batch-size 0`.

### PreciseWikiQA (requires vLLM server + question generator)
```bash
python scripts/run_with_server.py \
    --step all \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
  --N 100 \
  --logger-type lmdb \
  --activations-path shared/goodwiki.zarr/activations.zarr \
  --log-file shared/goodwiki.zarr/server.log
```

### Train hallucination classifier
```bash
python scripts/train_activation_model.py \
  --inference-json shared/goodwiki_jsonv2/generation.jsonl \
  --eval-json shared/goodwiki_jsonv2/eval_results.json \
  --activations-path shared/goodwiki_jsonv2/activations.json \
  --logger-type json \
  --routine contrastive \
  --model progressive_compressor \
  --train-layers 14-29 \
  --eval-layers 22,26 \
  --epochs 150
```

### Data setup
```bash
bash scripts/download_data.sh
```

## Storage Formats

| Format | Use case | Notes |
|--------|----------|-------|
| Zarr   | Large-scale experiments (preferred) | Best compression |
| JSON+NPY | Medium scale, human-readable metadata | ~5-10GB per 100 inferences |
| LMDB   | Maximum storage efficiency | |

**Important**: Use full-precision models (e.g., `meta-llama/Llama-3.1-8B-Instruct`) for activation logging. GGUF models skip activation logging automatically.

## Notebooks

**IMPORTANT: Before doing ANY work with notebooks, read [`NOTEBOOK_WORKFLOW.md`](NOTEBOOK_WORKFLOW.md) first.** It defines the master/working-copy workflow, required cells, and rules for syncing changes.

- **Master notebooks** live in `notebooks/` — these are git-tracked reference copies. Do not run them.
- **Working notebooks** are copies in the repo root for active execution. Root-level `*.ipynb` files are gitignored.


## Compute Contexts

**At the start of any session, check `.env` in the repo root for `COMPUTE_CONTEXT`:**
```bash
# .env is gitignored — set it manually on each machine
COMPUTE_CONTEXT=LOCAL_CPU   # or REMOTE_GPU
```

### Context: LOCAL_CPU (default if `.env` absent or unset)
- No GPU available; avoid launching vLLM servers or GPU-heavy training
- Can edit code, run lightweight scripts, inspect data, analyze results
- GPU tasks must be deferred or routed through the Jupyter server

### Context: REMOTE_GPU
- H200 GPU accessible via Jupyter notebook server at `http://alphagpu23:8889` (password: `123`)
- GPU-intensive work (inference, activation logging, model training) can run here via notebooks or CLI
- Preferred workflow: use notebooks for interactive GPU work, CLI scripts for batch jobs

### Never run compute on the login node

The Empire AI login host (`alpha1.empire-ai.org`, reachable via `ssh empire-ai`)
is for orchestration only — `git`, `gh`, `gpu_dispatch.py`, file inspection.
**Never run CPU- or GPU-intensive work on the login node**, including pytest
suites that load real models (even small ones like `sshleifer/tiny-gpt2`),
training scripts, inference loops, or anything that pegs cores for more than
a few seconds. The login node is shared across users; bursting CPU there
degrades the cluster for everyone and violates Empire AI usage policy.

For test and verification runs, prefer **`utils/jupyter_exec.py`** against an
already-allocated Jupyter-only node (e.g. `alphagpu23:8889`). It needs no new
SLURM allocation and runs inside an existing kernel. For long-running jobs,
dispatch through `gpu_dispatch.py run` after explicit user approval. Either
way, redirect pytest / script output to a log file on the remote node and
poll the log via subsequent `jupyter_exec` or `tail` calls rather than
holding an SSH stdout stream open.

### Claude Code GPU execution (REMOTE_GPU only)

Claude Code can execute code directly on the GPU node without user intervention using `utils/jupyter_exec.py`, which connects via the Jupyter REST + WebSocket API.

**Quick CLI check:**
```bash
python utils/jupyter_exec.py "import torch; print(torch.cuda.get_device_name(0))"
```

**In Python / agent scripts:**
```python
from utils.jupyter_exec import JupyterExecutor

with JupyterExecutor() as jup:
    result = jup.run("import torch; print(torch.cuda.is_available())")
    print(result.stdout)   # stdout text
    print(result.status)   # "ok" | "error" | "timeout"
```

**How it works:**
1. Password-login to `http://alphagpu23:8889` → session cookie
2. `POST /api/kernels` → start a fresh `p311` kernel (micromamba venv); auto-deleted after `run()` returns
3. WebSocket to `/api/kernels/{id}/channels` → send `execute_request`, stream back `stream`/`display_data`/`execute_reply` messages

**Dependency:** `websocket-client` (install with `pip install websocket-client` on alphacpu if missing).

**When to use:** Any time `COMPUTE_CONTEXT=REMOTE_GPU` and code needs to run on GPU (model inference, training, activation logging). Prefer this over asking the user to manually run notebook cells.

## Dataset & Experiment Status

### Canonical results view across all baselines

`scripts/results_table.py` is the **single source of truth** for what experiments have produced numbers. It walks every cell across training runs (`configs/experiments/baseline_comparison_*.json`), sampling baselines (SE variants, SelfCheckGPT), SEP probes, and P(true), and emits one row per (dataset × model × method × seed) with status (`complete` / `pending` / `running` / `missing` / `failed` / `partial`) and metrics. Prefer it over `audit_coverage.py` for any "what do we have" question.

```bash
# Writes output/results_table/results_table.{json,csv}
python scripts/results_table.py

# Custom output dir
python scripts/results_table.py --out-dir /tmp/results
```

The JSON form (`results_table.json`) is agent-friendly — one entry per cell with `kind`, `key`, `status`, `metrics`, `expected_rows`, `actual_rows`, `paths`. The CSV form is long-form (one row per cell × metric) and convenient for pandas / spreadsheet inspection.

### Checking inference/data generation status

**Always run the audit script rather than guessing whether data is present.** Do not assume a dataset or its activations exist (or don't) based on prior context — verify with:

```bash
python scripts/audit_datasets.py --model Qwen/Qwen3-8B
python scripts/audit_datasets.py --model meta-llama/Llama-3.1-8B-Instruct

# Also check zarr sample counts (fast — metadata only):
python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr

# Also check zarr disk usage (slow — runs du on every store):
python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr-size
```

Each dataset has train and test splits. The model name in output paths is the **last component of the HuggingFace model ID** (e.g., `Llama-3.1-8B-Instruct` for `meta-llama/Llama-3.1-8B-Instruct`, `Qwen3-8B` for `Qwen/Qwen3-8B`).

Data lives at:
- Generations: `output/{dataset}[_train]/{model_name}/generation.jsonl`
- Eval results: `output/{dataset}[_train]/{model_name}/eval_results.json`
- Activations: `shared/{dataset}[_train]_{model_slug}/activations.zarr/`

A dataset split is **complete** when generation.jsonl line count matches the expected split size AND eval_results.json exists.

### Checking experiment/training status

Use the existing `scripts/experiment_status.py` to check training run progress:
```bash
# Check all runs for an experiment config
python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_mmlu.json

# Scan a runs directory directly
python scripts/experiment_status.py --runs-dir runs/baseline_comparison_mmlu

# Machine-readable JSON
python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_mmlu.json --json

# Verbose: list every run
python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_mmlu.json --verbose
```

Related utilities in `scripts/experiment_utils.py` provide `RunStatus`, `RunSpec`, and `load_experiment_config()` for programmatic access to run enumeration and classification.

### Dataset configs convention

Each dataset has separate configs for train and test splits:
- `configs/datasets/{name}_test.json` — test split (used for evaluation)
- `configs/datasets/{name}_train.json` — train split (used for training)
- `configs/datasets/{name}.json` — legacy single-split config (points to test)

Experiment configs live at `configs/experiments/baseline_comparison_{name}.json`.

### Running on GPU nodes

**Never submit or kill SLURM jobs** without explicit user approval. Do not run `sbatch`, `srun`, `scancel`, `gpu_dispatch.py run/kill`, or kill remote processes via SSH. This includes killing orphaned or stale processes on GPU nodes.

Approval must be obtained with a message of the form:
> "I want to [specific action, e.g. kill PID 12345 on alphagpu22]. Yes/No?"

Wait for the user to respond with **Yes** before proceeding. A general instruction to "check the situation" or "investigate" does not constitute approval to kill or submit jobs.

**Avoid duplicate dispatches.** GPU dispatch via SSH can appear to fail (timeout) while the remote job actually launched successfully. Before re-dispatching:
1. Check `ps aux | grep <script>` on the target node via SSH to confirm whether the process is running.
2. Check `python scripts/gpu_dispatch.py jobs --all` for recent job records.
3. Wait at least **2 minutes** after a failed/uncertain dispatch before concluding it did not start.
Dispatching duplicates wastes GPU memory and causes jobs to compete for the same output files.

For GPU job dispatch, use `scripts/gpu_dispatch.py` instead of raw SSH:

```bash
# Check which nodes are available and their GPU utilization
python scripts/gpu_dispatch.py status

# Run a training script on the best available node
python scripts/gpu_dispatch.py run bash run_training.sh

# Run on a specific node
python scripts/gpu_dispatch.py run --node alphagpu04 bash run_training.sh

# Run with minimum VRAM requirement
python scripts/gpu_dispatch.py run --min-vram 30 bash run_training.sh

# Check running jobs
python scripts/gpu_dispatch.py jobs

# Show all jobs (including completed/killed)
python scripts/gpu_dispatch.py jobs --all

# Kill a job
python scripts/gpu_dispatch.py kill JOB_ID
```

Node registry is in `configs/nodes.json`. Job manifest is at `shared/gpu_jobs.json`.

### Logical node names (Jupyter-only nodes)

Node identity in the registry is the `name` field (defaults to `hostname` when absent). A single physical host can appear as multiple logical nodes when it hosts multiple Jupyter-bound GPU allocations on different ports. Example: `alphagpu01` runs two SLURM-allocated Jupyter servers (ports 8888 and 8889), registered as:

- `alphagpu01-8888` → `http://alphagpu01:8888` (GPU 0 of that SLURM allocation)
- `alphagpu01-8889` → `http://alphagpu01:8889` (a different physical GPU)

Each logical node has its own `jupyter_url` and is dispatched to explicitly:

```bash
python scripts/gpu_dispatch.py run --jupyter --node alphagpu01-8888 -- bash my_job.sh
```

Notes:
- `--node` matches the logical `name`, not `hostname`.
- These Jupyter-only entries have no SSH path enabled, so auto-selection (no `--node`) will not pick them — they must be addressed explicitly via `--jupyter --node <name>`.
- `gpu_dispatch.py status` will probe each logical node via its own Jupyter URL and report the GPU it actually sees through its `CUDA_VISIBLE_DEVICES` mask.

### Syncing Jupyter-only nodes from squeue

The Jupyter-only logical nodes in `configs/nodes.json` are managed by `gpu_dispatch.py sync-jupyter`. SLURM allocations for Jupyter servers come and go (they expire after `TIME_LIMI` or are cancelled), so the registry should be reconciled with `squeue --me` before relying on it.

```bash
# Preview which Jupyter nodes squeue says are live, without writing the file
python scripts/gpu_dispatch.py sync-jupyter --dry-run

# Apply: rewrite all entries with source=squeue from current SLURM state
python scripts/gpu_dispatch.py sync-jupyter
```

How it works:
- `sync-jupyter` runs `squeue --me` and matches RUNNING jobs whose name follows `jupyter_<word>_<port>` (e.g. `jupyter_empire_8889`). Each match becomes a logical node `<hostname>-<port>` with `source: "squeue"`, `slurm_job_id`, and `slurm_time_left` recorded.
- Only entries with `"source": "squeue"` are touched — SSH-only nodes and hybrid SSH+Jupyter nodes (e.g. `alphagpu16`) are left untouched.
- Multi-node SLURM allocations (NODELIST containing `[`) are skipped because the Jupyter server only runs on one host within the allocation.
- Run this any time after starting/stopping a `jupyter_empire_<port>` SLURM job, or whenever `gpu_dispatch.py status` shows stale `unreachable` entries.

The GPU Python env is: `/mnt/home/hyang1/.local/share/mamba/envs/p311/bin/python`

> **Note:** `.env` `GPUNODE` is still used by `utils/jupyter_exec.py` for interactive notebook work. `gpu_dispatch.py` is preferred for batch job dispatch.

Always use `resume=True` for long jobs in case of node reclamation.

## Development Notes

- Remote GPU development: see `REMOTE_DEV_SETUP.md` and `./connect_gpu.sh`
- Outputs written to `output/activation_training/<run-name>/`
- Tests in `tests/` directory
- Don't commit large activation files (LMDB, Zarr, NPY) to git
