# tqdm Progress Adapter (Notebook + CLI Compatible)

This project now provides a shared progress adapter at `utils/progress.py`.

The adapter standardizes progress bars across environments:
- Jupyter notebooks: notebook-compatible bars via `tqdm.auto` / `tqdm.notebook`
- Terminal/CLI scripts: normal terminal bars via `tqdm.std`

## Why this exists

Mixing direct imports like `from tqdm import tqdm` with notebook execution can cause rendering/memory issues.
Using one adapter avoids backend drift and keeps behavior consistent.

## Recommended import pattern (new code)

```python
from utils.progress import tqdm, trange
```

This uses `tqdm.auto` by default, so it adapts to notebook vs terminal automatically.

## Notebook usage (force notebook backend)

Run this in the first notebook cell **before importing project modules that use tqdm**:

```python
from utils.progress import set_tqdm_backend, install_tqdm_global

set_tqdm_backend("notebook")
install_tqdm_global()
```

Why both calls:
- `set_tqdm_backend("notebook")` sets backend preference for adapter imports.
- `install_tqdm_global()` patches `tqdm.tqdm` for legacy code that still does `from tqdm import tqdm`.

## Script/CLI usage

Default behavior is `auto` (recommended):

```bash
# Linux/macOS
export HALLULENS_TQDM_BACKEND=auto

# Windows (cmd)
set HALLULENS_TQDM_BACKEND=auto
```

Optional forced backends:
- `HALLULENS_TQDM_BACKEND=notebook`
- `HALLULENS_TQDM_BACKEND=std`

`TQDM_BACKEND` is also supported for compatibility, but `HALLULENS_TQDM_BACKEND` takes precedence.

## Entry-point behavior

`scripts/run_with_server.py` now calls `install_tqdm_global()` during startup so legacy task imports are normalized when launched through this runner.

## Migration guidance

- Prefer replacing direct imports:
  - `from tqdm import tqdm` → `from utils.progress import tqdm`
  - `from tqdm.autonotebook import tqdm` → `from utils.progress import tqdm`
- For files you cannot modify, use notebook first-cell patching shown above.

## Troubleshooting

If notebook bars do not render as widgets:
1. Install/upgrade `ipywidgets` in the active environment.
2. Restart the kernel.
3. Keep backend as `auto` unless you explicitly need `notebook`.
