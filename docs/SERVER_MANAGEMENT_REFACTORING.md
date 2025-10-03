# Server Management Refactoring

## Date: 2025-10-03

## Problem

The error "Cannot restart server - no ServerManager and REST endpoint failed" occurred because:

1. **Process Isolation**: `run_with_server.py` launched task scripts as subprocesses using `subprocess.run()`
2. **Global Variable Scope**: The `_global_server_manager` variable was set in the parent process but not accessible in child processes
3. **Module Import Isolation**: Each subprocess got its own fresh copy of imported modules, so `get_server_manager()` always returned `None`

## Solution

**Moved server management to where API calls are made** - the `run_exp()` function in `utils/exp.py`.

### Architecture Change

**Before:**
```
run_with_server.py (parent process)
  ├─ Creates ServerManager
  ├─ Sets _global_server_manager
  └─ Launches subprocess: tasks/shortform/precise_wikiqa.py
       └─ Calls utils/exp.py → run_exp()
            └─ Calls utils/lm.py → call_vllm_api()
                 └─ On timeout: tries to get_server_manager() → Returns None ❌
```

**After:**
```
run_with_server.py (parent process)
  └─ Launches subprocess: tasks/shortform/precise_wikiqa.py
       └─ Calls utils/exp.py → run_exp()
            ├─ Creates ServerManager (same process!)
            ├─ Sets _global_server_manager
            └─ Calls utils/lm.py → call_vllm_api()
                 └─ On timeout: get_server_manager() → Returns manager ✅
```

## Changes Made

### 1. Moved `ServerManager` to `utils/lm.py`

**File**: `utils/lm.py`

- Added `ServerManager` class (moved from `scripts/run_with_server.py`)
- Added global `_global_server_manager` variable
- Added `get_server_manager()` and `set_server_manager()` functions
- Updated `restart_server()` to use the local `get_server_manager()`

**Benefits**:
- ServerManager is in the same module where API calls are made
- No cross-process communication needed
- Direct access to server process handle

### 2. Updated `run_exp()` to Manage Server

**File**: `utils/exp.py`

Added parameters:
- `manage_server=True` - Whether to manage server lifecycle
- `server_host="0.0.0.0"` - Server host
- `server_port=8000` - Server port
- `logger_type="lmdb"` - Activation logger type
- `activations_path=None` - Path for activation storage
- `log_file_path=None` - Path for server logs

**Behavior**:
```python
def run_exp(..., manage_server=True, ...):
    server_manager = None
    if inference_method == "vllm" and manage_server:
        # Check if server already running
        if not lm.check_server_health(f"http://{server_host}:{server_port}"):
            # Start server
            server_manager = lm.ServerManager(...)
            server_manager.start_server()
            lm.set_server_manager(server_manager)
    
    try:
        # Run inference...
        all_prompts["generation"] = thread_map(...)
    finally:
        # Stop server if we started it
        if server_manager:
            server_manager.stop_server()
            lm.set_server_manager(None)
```

### 3. Simplified `run_with_server.py`

**File**: `scripts/run_with_server.py`

- Removed duplicate `ServerManager` class definition
- Imported `ServerManager`, `get_server_manager`, `set_server_manager` from `utils.lm`
- Removed unused imports (`time`, `requests`)
- Server management now delegated to `run_exp()`

## Usage

### Option 1: Automatic Server Management (Recommended)

The server is automatically started and stopped by `run_exp()`:

```python
from utils import exp
import pandas as pd

prompts_df = pd.DataFrame({"prompt": ["What is AI?", "Explain ML"]})

# Server automatically started, used, and stopped
exp.run_exp(
    task="my_task",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    all_prompts=prompts_df,
    inference_method="vllm",
    manage_server=True  # Default
)
```

### Option 2: Manual Server Management

If you want to manage the server yourself (e.g., for multiple experiments):

```python
from utils import lm, exp
import pandas as pd

# Start server once
server_manager = lm.ServerManager(
    model="meta-llama/Llama-3.1-8B-Instruct",
    port=8000
)
server_manager.start_server()
lm.set_server_manager(server_manager)

try:
    # Run multiple experiments
    for task in ["task1", "task2", "task3"]:
        prompts_df = load_prompts(task)
        exp.run_exp(
            task=task,
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            all_prompts=prompts_df,
            inference_method="vllm",
            manage_server=False  # Don't start/stop server
        )
finally:
    # Stop server when done
    server_manager.stop_server()
    lm.set_server_manager(None)
```

### Option 3: Using `run_with_server.py`

The script still works but now delegates server management to `run_exp()`:

```bash
python scripts/run_with_server.py \
    --step inference \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 100
```

## Benefits

1. **✅ Server Restart Works**: `get_server_manager()` now returns the actual manager
2. **✅ Simpler Architecture**: Server managed where it's used
3. **✅ No Process Boundaries**: Everything in the same process
4. **✅ Flexible**: Can manage server automatically or manually
5. **✅ Backward Compatible**: Existing scripts still work

## Testing

Test the refactoring:

```bash
# Test automatic server management
python -c "
from utils import exp
import pandas as pd

prompts = pd.DataFrame({'prompt': ['Test prompt']})
exp.run_exp(
    task='test',
    model_path='meta-llama/Llama-3.1-8B-Instruct',
    all_prompts=prompts,
    inference_method='vllm',
    manage_server=True,
    max_workers=1
)
"

# Test with run_with_server.py
python scripts/run_with_server.py \
    --step inference \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 10
```

## Migration Guide

### For Direct `run_exp()` Calls

No changes needed! The function is backward compatible:

```python
# Old code still works
exp.run_exp(task, model, prompts, inference_method="vllm")

# New code with explicit server management
exp.run_exp(
    task, model, prompts,
    inference_method="vllm",
    manage_server=True,  # Optional, default is True
    server_port=8000,    # Optional
    log_file_path="custom.log"  # Optional
)
```

### For Custom Scripts Using ServerManager

Update imports:

```python
# Before
from scripts.run_with_server import ServerManager

# After
from utils.lm import ServerManager
```

## Files Modified

1. `utils/lm.py` - Added ServerManager class and global manager functions
2. `utils/exp.py` - Added server management to run_exp()
3. `scripts/run_with_server.py` - Simplified to use ServerManager from utils.lm
4. `docs/SERVER_MANAGEMENT_REFACTORING.md` - This document

## Related Documentation

- `docs/SERVER_RESTART_PROTOCOL.md` - Server restart protocol
- `docs/SERVER_RESTART_SOLUTION.md` - Original server restart solution
- `CHANGELOG_SERVER_RESTART.md` - Server restart changelog

