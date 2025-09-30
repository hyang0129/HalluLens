# Server Restart Protocol

## Overview

The HalluLens activation logging system now includes an automatic server restart protocol that detects when the server becomes unresponsive and automatically restarts it to recover from hung states.

## Problem Statement

During long-running inference tasks, the vLLM server can occasionally become unresponsive due to:
- GPU memory issues
- Model state corruption
- Specific inputs that cause the model to hang
- Resource exhaustion

Previously, when this happened, all subsequent requests would timeout (taking ~15 minutes each), wasting significant time and resources.

## Solution

The new restart protocol automatically:
1. **Detects timeouts** - When a request times out (now after 60 seconds instead of 300 seconds)
2. **Triggers server restart** - Calls the `/restart` endpoint to gracefully restart the server
3. **Waits for recovery** - Monitors the `/health` endpoint until the server is back online
4. **Retries the request** - Automatically retries the failed request after restart

## Configuration

### Environment Variables

- **`ENABLE_SERVER_RESTART`** (default: `"true"`)
  - Set to `"true"` to enable automatic server restart on timeout
  - Set to `"false"` to disable (uses old retry-only behavior)

- **`SERVER_RESTART_WAIT_TIME`** (default: `30`)
  - Maximum time (in seconds) to wait for server to restart
  - Recommended: 30-60 seconds depending on model size

### Example Usage

```bash
# Enable server restart with 45-second wait time
export ENABLE_SERVER_RESTART=true
export SERVER_RESTART_WAIT_TIME=45

# Run inference
python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct
```

## How It Works

### 1. Timeout Detection

When a request times out (after 60 seconds), the client detects the `APITimeoutError`:

```python
except APITimeoutError as e:
    request_time = time.time() - start_time
    if attempt < max_retries:
        # First timeout triggers restart
        if attempt == 0 and ENABLE_SERVER_RESTART:
            restart_server(port, wait_time=SERVER_RESTART_WAIT_TIME)
```

### 2. Server Restart Endpoint

The server provides a `/restart` endpoint that:
- Logs current state and active requests
- Clears GPU cache (`torch.cuda.empty_cache()`)
- Clears model caches
- Gracefully exits (supervisor/systemd should restart)

```python
@app.post("/restart")
async def restart_server():
    # Clear caches
    torch.cuda.empty_cache()
    _model_cache.clear()
    _tokenizer_cache.clear()
    
    # Schedule restart
    os._exit(0)  # Force exit
```

### 3. Health Check Monitoring

The client monitors the `/health` endpoint to detect when the server is back online:

```python
def check_server_health(port, timeout=5):
    try:
        response = requests.get(f"{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False
```

### 4. Automatic Retry

After successful restart, the client automatically retries the failed request without additional delay.

## Timeout Configuration Changes

### Previous Behavior
- **Client timeout**: 300 seconds (5 minutes)
- **Actual timeout observed**: ~901 seconds (15 minutes)
- **Total time per failed sample**: ~1 hour (4 retries × 15 minutes)

### New Behavior
- **Client timeout**: 60 seconds (1 minute)
- **Retry attempts**: 4 (initial + 3 retries)
- **Server restart**: Triggered on first timeout
- **Total time per failed sample**: ~2-3 minutes (restart + retry)

## Benefits

1. **Faster failure detection**: 60s timeout vs 900s timeout
2. **Automatic recovery**: Server restarts instead of continuing in hung state
3. **Reduced wasted time**: ~2-3 minutes per failure vs ~1 hour
4. **Higher success rate**: Fresh server state increases chance of success on retry
5. **Resource cleanup**: GPU cache and model caches cleared on restart

## Monitoring and Logging

### Client Logs

The client logs detailed information about the restart process:

```
2025-09-30 15:59:44 | WARNING | [CLIENT 9cbe6af9] API timeout on attempt 1/4 after 60.1s
2025-09-30 15:59:44 | WARNING | [CLIENT 9cbe6af9] First timeout detected - triggering server restart
2025-09-30 15:59:44 | WARNING | Attempting to restart server at http://0.0.0.0:8000/v1...
2025-09-30 15:59:44 | INFO | Restart request sent successfully
2025-09-30 15:59:49 | INFO | Waiting up to 30s for server to restart...
2025-09-30 16:00:14 | SUCCESS | Server restarted successfully after 25.3s
2025-09-30 16:00:14 | SUCCESS | [CLIENT 9cbe6af9] Server restarted successfully, will retry request
```

### Server Logs

The server logs the restart event:

```
2025-09-30 15:59:44 | WARNING | ================================================================================
2025-09-30 15:59:44 | WARNING | SERVER RESTART REQUESTED
2025-09-30 15:59:44 | WARNING | ================================================================================
2025-09-30 15:59:44 | WARNING | Restarting with 1 active requests:
2025-09-30 15:59:44 | WARNING |   [9cbe6af9] /v1/completions - 60.2s
2025-09-30 15:59:44 | INFO | Clearing CUDA cache...
2025-09-30 15:59:44 | INFO | CUDA cache cleared
2025-09-30 15:59:44 | INFO | Clearing model caches...
2025-09-30 15:59:44 | INFO | Model caches cleared
2025-09-30 15:59:45 | WARNING | Initiating server restart...
```

## Deployment Considerations

### Using with run_with_server.py

The `run_with_server.py` script manages the server lifecycle. When the server exits (via `/restart`), the script will detect the exit and can restart it:

**Current behavior**: Server exits, script stops
**Recommended**: Wrap in a restart loop or use systemd/supervisor

### Using with systemd (Recommended)

Create a systemd service that automatically restarts the server:

```ini
[Unit]
Description=HalluLens Activation Logging Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/HalluLens
ExecStart=/path/to/python -m activation_logging.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct
Restart=always
RestartSec=5
Environment="ENABLE_SERVER_RESTART=true"
Environment="SERVER_RESTART_WAIT_TIME=30"

[Install]
WantedBy=multi-user.target
```

### Using with Docker

Add restart policy to your docker-compose.yml:

```yaml
services:
  activation-server:
    image: hallulens:latest
    restart: always
    environment:
      - ENABLE_SERVER_RESTART=true
      - SERVER_RESTART_WAIT_TIME=30
```

## Troubleshooting

### Server doesn't restart

**Symptom**: Client logs show "Server restart failed"

**Possible causes**:
1. Server is managed by `run_with_server.py` which doesn't auto-restart
2. No supervisor/systemd configured
3. Server process is truly hung and can't respond to restart request

**Solution**:
- Use systemd or supervisor for automatic restart
- Increase `SERVER_RESTART_WAIT_TIME`
- Check server logs for errors

### Restart loop

**Symptom**: Server keeps restarting repeatedly

**Possible causes**:
1. Persistent issue with model or data
2. Insufficient GPU memory
3. Corrupted model files

**Solution**:
- Check server logs for error patterns
- Verify GPU memory availability
- Test with smaller model or batch size
- Disable auto-restart: `export ENABLE_SERVER_RESTART=false`

### Slow restarts

**Symptom**: Server takes longer than expected to restart

**Possible causes**:
1. Large model takes time to reload
2. GPU memory cleanup is slow
3. Network latency

**Solution**:
- Increase `SERVER_RESTART_WAIT_TIME`
- Monitor GPU memory usage
- Check server startup logs

## Testing

### Manual Test

```bash
# Start server
python -m activation_logging.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct

# In another terminal, trigger restart
curl -X POST http://localhost:8000/restart

# Check health
curl http://localhost:8000/health
```

### Automated Test

```python
import requests
import time

# Trigger restart
response = requests.post("http://localhost:8000/restart")
print(f"Restart triggered: {response.json()}")

# Wait and check health
time.sleep(10)
for i in range(30):
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            print(f"Server healthy after {i*2}s")
            break
    except:
        pass
    time.sleep(2)
```

## Future Enhancements

1. **Smarter restart triggers**: Restart based on multiple timeouts, not just first
2. **Graceful request handling**: Save in-flight requests and replay after restart
3. **Health metrics**: Track restart frequency and success rate
4. **Automatic scaling**: Spin up backup server during restart
5. **Request queuing**: Queue requests during restart instead of failing

## References

- Client code: `utils/lm.py`
- Server code: `activation_logging/server.py`
- Server manager: `scripts/run_with_server.py`

