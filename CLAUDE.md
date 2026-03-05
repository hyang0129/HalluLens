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

- `b_contrastive_training_with_new_trainer.ipynb` — contrastive model training
- `c_layeraware_training_with_new_trainer.ipynb` — layer-aware training
- `k_view_loader_profile.ipynb` — K-view loader profiling

Prototype notebooks are migrated to `scripts/train_activation_model.py` for reproducible CLI runs.

## Development Notes

- Remote GPU development: see `REMOTE_DEV_SETUP.md` and `./connect_gpu.sh`
- Outputs written to `output/activation_training/<run-name>/`
- Tests in `tests/` directory
- Don't commit large activation files (LMDB, Zarr, NPY) to git
