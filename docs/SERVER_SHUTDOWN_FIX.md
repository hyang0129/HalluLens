# Server Shutdown Fix

## Problem

When running the inference pipeline with automatic server management, the server was not properly shutting down after task completion. This caused issues when starting new runs:

1. Server appeared to stop (logs showed "Server stopped")
2. But the actual uvicorn/vLLM processes remained running
3. Subsequent runs detected an "already running" server
4. This orphaned server wasn't managed by the new script instance

## Root Cause

The `vllm_serve.py` script uses `subprocess.run(uvicorn_cmd)` which starts uvicorn, which may spawn multiple child processes (workers, GPU processes, etc.). When the parent Python process was terminated, the child processes continued running.

The original `stop_server()` method only called `terminate()` on the parent process, which didn't propagate to the entire process tree.

## Solution

Implemented comprehensive process cleanup in [utils/lm.py](../utils/lm.py):

### 1. Process Group Termination (Unix/Linux)

- Start server process with `preexec_fn=os.setsid` to create a new process group
- Use `os.killpg()` to terminate the entire process group, not just the parent
- This ensures all child processes (uvicorn workers, vLLM processes) are killed

### 2. Port-Based Cleanup

Added `kill_process_on_port()` helper function that:
- Uses `lsof` (Unix) or `netstat` (Windows) to find processes using a port
- Forcefully kills any processes bound to the server port
- Called before starting server (cleanup orphans) and after stopping (final cleanup)

### 3. Robust Error Handling

- Catches `ProcessLookupError` if process already terminated
- Falls back to force kill if graceful shutdown times out
- Provides clear logging at each step

## Changes Made

### Modified Functions

1. **`ServerManager.start_server()`**
   - Checks for orphaned processes before starting
   - Creates process group on Unix systems
   - Cleans port before launching server

2. **`ServerManager.stop_server()`**
   - Kills entire process group (Unix) instead of just parent
   - Performs final port cleanup after process termination
   - More robust error handling

3. **`ServerManager.restart_server()`**
   - Updated to use new process group termination logic
   - Consistent with `stop_server()` implementation

### New Functions

4. **`kill_process_on_port(port)`**
   - Finds and kills any process using specified port
   - Platform-aware (Unix/Windows)
   - Used for both pre-start cleanup and post-stop cleanup

## Testing

To verify the fix works:

```bash
# Run a task with N=100 (should complete quickly)
python scripts/run_with_server.py --step generate --task precisewikiqa \
  --model "models/Llama-3.3-70B-Instruct-Q6_K_L" --N 100

# Check if server is actually stopped
lsof -i :8000  # Should return nothing

# Run another task - server should start fresh
python scripts/run_with_server.py --step generate --task precisewikiqa \
  --model "models/Llama-3.3-70B-Instruct-Q6_K_L" --N 100

# Should see "Starting vLLM server" not "Server already running"
```

## Platform Compatibility

- **Linux/Unix**: Uses process groups with `os.setsid()` and `os.killpg()`
- **Windows**: Falls back to standard `terminate()`/`kill()` methods
- Port cleanup works on both platforms using appropriate tools

## Related Files

- [utils/lm.py](../utils/lm.py) - Server manager implementation
- [scripts/run_with_server.py](../scripts/run_with_server.py) - Main task runner
- [activation_logging/vllm_serve.py](../activation_logging/vllm_serve.py) - vLLM server wrapper
