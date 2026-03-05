# Requirements Update Summary

## Overview
Updated both `requirements.txt` files to include all dependencies found in the codebase with proper version constraints.

## Changes Made

### Root `requirements.txt`
**Added/Updated packages:**
- Added `python>=3.9` specification
- Added version constraints to all packages (previously many had no versions)
- Added `scipy>=1.11.0` (used in activation_research)
- Added `seaborn>=0.12.0` (used in LSD code)
- Added `llama-cpp-python>=0.2.0` (used in server.py)
- Added `jupyter>=1.0.0` and `websocket-client>=1.6.0` (used in jupyter_api_executor.py)
- Added `matplotlib>=3.7.0` (used for visualization)
- Added `psutil>=5.9.0` (used for monitoring)

**Key dependencies organized by purpose:**
1. **Core ML Frameworks**: torch, transformers, sentence-transformers, numpy, scikit-learn, scipy
2. **LM Inference**: vllm, openai
3. **Web Server**: fastapi, uvicorn, pydantic
4. **Data Storage**: lmdb, zarr, webdataset
5. **Data Processing**: pandas, jsonlines, datasets
6. **Text Processing**: segtok (for longwiki)
7. **API Rate Limiting**: ratelimit, retry (for refusal tests)

### `activation_logging/requirements.txt`
**Added/Updated packages:**
- Added `scipy>=1.11.0` (needed for activation_research integration)
- Added version constraints to all packages
- Better organization with comments

**Key features:**
- All packages needed for the activation logging server
- Includes optional dependencies (llama-cpp-python, vllm) clearly marked
- Focused on core activation logging functionality

## Dependencies by Module

### Core System (`utils/`)
- openai, requests, loguru, pandas, transformers, jsonlines

### Activation Logging (`activation_logging/`)
- fastapi, uvicorn, pydantic, lmdb, zarr, webdataset
- torch, transformers, numpy, scipy
- psutil, loguru, tqdm
- Optional: llama-cpp-python, vllm

### Activation Research (`activation_research/`)
- torch, numpy, scipy, scikit-learn
- Uses datasets from activation_logging

### Tasks
- **triviaqa**: pandas, jsonlines, scikit-learn
- **shortform**: pandas, jsonlines, tqdm, loguru, transformers
- **longwiki**: segtok, sentence-transformers, transformers, pandas
- **refusal_test**: pandas, tqdm, ratelimit, retry

### External (in `external/` - not included in main requirements)
- Various research codebases with their own dependencies
- Should not be edited per instructions

## Installation Instructions

### For Full Project:
```bash
pip install -r requirements.txt
```

### For Activation Logging Only:
```bash
pip install -r activation_logging/requirements.txt
```

### Recommended: Use Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Using conda
conda create -n hallulens python=3.9
conda activate hallulens
pip install -r requirements.txt
```

## Version Constraints Rationale

- **Minimum versions specified**: Ensures compatibility with features used in code
- **sentence-transformers pinned**: Pinned to 3.3.1 as specified in original requirements
- **torch>=2.1.2**: Required for latest transformer features
- **transformers>=4.35.0**: Required for model compatibility
- **openai>=1.0.0**: New API structure
- **Other packages**: Set to recent stable versions compatible with the codebase

## Testing Recommendations

After installing, verify with:
```bash
python -c "import torch; import transformers; import openai; print('Core imports OK')"
python -c "from activation_logging.server import *; print('Activation logging OK')"
python -c "from activation_research.evaluation import *; print('Activation research OK')"
```

## Notes

- Some packages like `llama-cpp-python` are optional and only needed for GGUF model support
- `vllm` is optional but recommended for optimized inference
- `jupyter` and `websocket-client` are optional, only needed for notebook-based workflows
- External dependencies in `external/` folder have their own setup.py files and should be installed separately if needed
