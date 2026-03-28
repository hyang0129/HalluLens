# Integration Tests

These tests require a GPU environment (H100 or similar) with the full model stack:

- `test_hf_adapter_gpu.py` — HFTransformersAdapter inference + activation extraction
- `test_server_async_gpu.py` — Server with async writer, end-to-end inference + Zarr persistence
- `test_batch_inference_gpu.py` — Batch vs sequential activation consistency
- `test_resume_zarr_gpu.py` — Resume from Zarr index after interruption

## How to run

```bash
# On GPU machine, with hallulens conda env active:
pip install pytest
cd /workspaces/HalluLens

# All integration tests
python -m pytest tests/integration/ -v --tb=short

# Specific suite
python -m pytest tests/integration/test_hf_adapter_gpu.py -v

# Slow tests (throughput benchmarks) are marked with @pytest.mark.slow — skip with:
python -m pytest tests/integration/ -v -m "not slow"
```

## Prerequisites

```bash
# Download a small model for testing (Llama 3.1 8B or similar):
python scripts/download_hf_model.py --model meta-llama/Llama-3.1-8B-Instruct

# Or use any HuggingFace causal LM available on the machine.
# Set TEST_MODEL_NAME env var to override the default:
export TEST_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```
