# Server Management Fix - Final Solution

## Date: 2025-10-03

## Problem Recap

After the initial refactoring, we still got the error:
```
2025-10-03 15:11:56 | WARNING | ServerManager not available, falling back to REST endpoint method
2025-10-03 15:12:01 | ERROR | Cannot restart server - no ServerManager and REST endpoint failed
```

## Root Cause

The issue was a **process boundary problem**:

1. `run_with_server.py` (parent process) started the server and set `_global_server_manager`
2. `run_with_server.py` launched task scripts as **subprocesses** using `subprocess.run()`
3. Subprocesses called `run_exp()` which detected server was running, so didn't start a new one
4. But `run_exp()` also didn't set `server_manager` variable (because it didn't start the server)
5. When timeout occurred in subprocess, `get_server_manager()` returned `None`

**Key insight**: Even though we moved `ServerManager` to `utils/lm.py`, the parent process and subprocess have **separate memory spaces**. Global variables are not shared across process boundaries.

## Solution

**Remove server management from `run_with_server.py` entirely**. Let each subprocess manage its own server via `run_exp()`.

### Changes Made

#### 1. Updated `scripts/run_with_server.py`

**Removed**:
- Server creation and startup code
- `server_manager.start_server()` call
- `set_server_manager()` call  
- `server_manager.stop_server()` in finally block
- `server_manager` parameter to `run_task_step()`

**Added**:
- Info message that server management is handled by `run_exp()`

**Before**:
```python
# Create server manager
server_manager = ServerManager(...)
server_manager.start_server()
set_server_manager(server_manager)

try:
    run_task_step(step, task, model, server_manager=server_manager, **kwargs)
finally:
    server_manager.stop_server()
```

**After**:
```python
logger.info("Server management is now handled by run_exp()")

try:
    run_task_step(step, task, model, **kwargs)
except:
    ...
# No server cleanup needed
```

#### 2. Updated `utils/exp.py`

Added warning when server is already running:

```python
if server_was_running:
    print(f"Server already running at http://{server_host}:{server_port}")
    print(f"⚠️  Note: Server restart will not be available (server not managed by this process)")
```

This warns users that if they manually start a server, restarts won't work.

## How It Works Now

### Architecture

```
run_with_server.py (parent process)
  └─ Launches subprocess: tasks/shortform/precise_wikiqa.py
       └─ Calls utils/exp.py → run_exp()
            ├─ Creates ServerManager (in subprocess!)
            ├─ Starts server
            ├─ Sets _global_server_manager (in subprocess!)
            └─ Calls utils/lm.py → call_vllm_api()
                 └─ On timeout: get_server_manager() → Returns manager ✅
```

### Flow

1. **User runs**: `python scripts/run_with_server.py --step inference --task precisewikiqa --model ...`
2. **run_with_server.py**: Launches subprocess for the task
3. **Subprocess**: Imports and calls `run_exp()`
4. **run_exp()**: 
   - Checks if server is running (it's not)
   - Creates `ServerManager`
   - Starts server
   - Sets `_global_server_manager` (in subprocess memory)
   - Runs inference with `thread_map()`
5. **Worker threads**: Call `lm.call_vllm_api()`
   - Threads share memory with subprocess ✅
   - On timeout: `get_server_manager()` returns the manager ✅
   - Server restart works! ✅
6. **run_exp() cleanup**: Stops server in finally block

## Benefits

✅ **Server restart works** - Manager accessible in same process  
✅ **Simpler** - No cross-process communication needed  
✅ **Each subprocess manages its own server** - Clean isolation  
✅ **Automatic cleanup** - Server stopped when `run_exp()` exits  
✅ **Works with threads** - Worker threads share memory with parent process  

## Usage

### Option 1: Using `run_with_server.py` (Recommended)

Server is automatically managed:

```bash
python scripts/run_with_server.py \
    --step inference \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 100
```

The subprocess will:
- Start the server
- Run inference
- Stop the server on exit

### Option 2: Direct Task Script

Call task scripts directly (server still auto-managed):

```bash
python -m tasks.shortform.precise_wikiqa \
    --do_inference \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 100
```

### Option 3: Direct `run_exp()` Call

For custom scripts:

```python
from utils import exp
import pandas as pd

prompts_df = pd.DataFrame({"prompt": ["What is AI?"]})

# Server automatically started and stopped
exp.run_exp(
    task="my_task",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    all_prompts=prompts_df,
    inference_method="vllm",
    manage_server=True  # Default
)
```

### Option 4: Manual Server Management

If you want to reuse a server across multiple experiments:

```python
from utils import lm, exp

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
        prompts = load_prompts(task)
        exp.run_exp(
            task=task,
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            all_prompts=prompts,
            inference_method="vllm",
            manage_server=False  # Don't auto-manage
        )
finally:
    server_manager.stop_server()
```

## Testing

The refactoring has been tested:

```bash
# Unit tests
python tests/test_server_management_refactoring.py
# Result: ✅ All 5 tests passed

# Integration test
python scripts/run_with_server.py \
    --step inference \
    --task precisewikiqa \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --N 10
```

## Migration Guide

### For Users of `run_with_server.py`

**No changes needed!** The script works the same way, just with better server management.

### For Custom Scripts

If you were manually starting servers before calling `run_exp()`:

**Before**:
```python
# Manually start server
os.system("python -m activation_logging.vllm_serve --model ... &")
time.sleep(30)  # Wait for server

# Run experiment
exp.run_exp(task, model, prompts, inference_method="vllm")
```

**After**:
```python
# Just run experiment - server auto-managed
exp.run_exp(
    task, model, prompts,
    inference_method="vllm",
    manage_server=True  # Default
)
```

## Files Modified

1. `scripts/run_with_server.py` - Removed server management code
2. `utils/exp.py` - Added warning for pre-existing servers
3. `docs/SERVER_MANAGEMENT_FIX.md` - This document

## Related Documentation

- `docs/SERVER_MANAGEMENT_REFACTORING.md` - Initial refactoring
- `docs/SERVER_RESTART_PROTOCOL.md` - Server restart protocol
- `CHANGELOG_SERVER_RESTART.md` - Server restart changelog

## Summary

The fix was simple: **Don't manage the server in the parent process**. Let each subprocess manage its own server via `run_exp()`. This ensures the server manager is accessible where it's needed (in the same process making API calls), enabling server restarts to work correctly.

