# HalluLens — Claude Code Guidelines

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

## Key Commands

### Run inference + activation logging
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
- H200 GPU accessible via Jupyter notebook server at `http://alphagpu24:8889` (password: `123`)
- GPU-intensive work (inference, activation logging, model training) can run here via notebooks or CLI
- Preferred workflow: use notebooks for interactive GPU work, CLI scripts for batch jobs

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
1. Password-login to `http://alphagpu24:8889` → session cookie
2. `GET /api/kernels` → pick an idle `p311` kernel (micromamba venv for this project); falls back to any idle kernel, then least-recently-used
3. WebSocket to `/api/kernels/{id}/channels` → send `execute_request`, stream back `stream`/`display_data`/`execute_reply` messages

**Dependency:** `websocket-client` (install with `pip install websocket-client` on alphacpu if missing).

**When to use:** Any time `COMPUTE_CONTEXT=REMOTE_GPU` and code needs to run on GPU (model inference, training, activation logging). Prefer this over asking the user to manually run notebook cells.

## Development Notes

- Remote GPU development: see `REMOTE_DEV_SETUP.md` and `./connect_gpu.sh`
- Outputs written to `output/activation_training/<run-name>/`
- Tests in `tests/` directory
- Don't commit large activation files (LMDB, Zarr, NPY) to git
