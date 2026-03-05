# Changelog: Server Restart Protocol Implementation

## Date: 2025-09-30

## Summary

Implemented automatic server restart protocol to handle server unresponsiveness during long-running inference tasks. This reduces timeout-related failures from ~1 hour per sample to ~2-3 minutes per sample.

## Changes

### 1. Reduced Timeout Duration

**File**: `utils/lm.py`  
**Line**: 185

**Before**:
```python
timeout=300.0  # 5 minute timeout instead of default 30 minutes
```

**After**:
```python
timeout=60.0  # 1 minute timeout
```

**Impact**: Faster detection of server unresponsiveness (60s vs 300s)

---

### 2. Added Server Restart Endpoint

**File**: `activation_logging/server.py`  
**Lines**: 778-829

**New endpoint**: `POST /restart`

**Features**:
- Logs current server state and active requests
- Clears GPU cache (`torch.cuda.empty_cache()`)
- Clears model caches (`_model_cache`, `_tokenizer_cache`, `_llamacpp_model_cache`)
- Gracefully exits with `os._exit(0)` to trigger supervisor restart

**Example**:
```bash
curl -X POST http://localhost:8000/restart
```

---

### 3. Added Client-Side Restart Logic

**File**: `utils/lm.py`

#### 3a. Added imports
**Lines**: 8-15

```python
import requests  # Added for health checks and restart calls
```

#### 3b. Added configuration variables
**Lines**: 29-31

```python
# Global flag to enable/disable server restart on timeout
ENABLE_SERVER_RESTART = os.environ.get("ENABLE_SERVER_RESTART", "true").lower() == "true"
SERVER_RESTART_WAIT_TIME = int(os.environ.get("SERVER_RESTART_WAIT_TIME", "30"))
```

#### 3c. Added helper functions
**Lines**: 55-121

**New functions**:
- `check_server_health(port, timeout=5)` - Checks if server is responsive
- `restart_server(port, wait_time=30)` - Triggers restart and waits for recovery

#### 3d. Modified timeout handling
**Lines**: 280-301

**New behavior**:
- On first timeout, triggers server restart if `ENABLE_SERVER_RESTART=true`
- Waits for server to come back online (up to `SERVER_RESTART_WAIT_TIME` seconds)
- Automatically retries the failed request after successful restart
- Falls back to normal retry behavior if restart fails

---

### 4. Added Documentation

**File**: `docs/SERVER_RESTART_PROTOCOL.md`

Comprehensive documentation covering:
- Problem statement and solution overview
- Configuration options
- How it works (step-by-step)
- Deployment considerations (systemd, Docker)
- Monitoring and logging
- Troubleshooting guide
- Testing procedures

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SERVER_RESTART` | `"true"` | Enable/disable automatic server restart on timeout |
| `SERVER_RESTART_WAIT_TIME` | `30` | Maximum seconds to wait for server restart |

### Usage Example

```bash
# Enable server restart with custom wait time
export ENABLE_SERVER_RESTART=true
export SERVER_RESTART_WAIT_TIME=45

# Run inference
python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct
```

---

## Performance Impact

### Before Changes

- **Timeout duration**: 300-900 seconds per attempt
- **Retry attempts**: 4 (initial + 3 retries)
- **Total time per failed sample**: ~1 hour (4 × 15 minutes)
- **Behavior on hang**: All subsequent requests fail with timeouts

### After Changes

- **Timeout duration**: 60 seconds per attempt
- **Retry attempts**: 4 (initial + 3 retries)
- **Server restart**: Triggered on first timeout
- **Total time per failed sample**: ~2-3 minutes (restart + retry)
- **Behavior on hang**: Server restarts, subsequent requests succeed

### Example Scenario

**Scenario**: Server hangs on sample #897 out of 940 samples

**Before**:
- Sample #897: 1 hour timeout → skip
- Sample #898: 1 hour timeout → skip
- Sample #899: 1 hour timeout → skip
- ...
- **Total wasted time**: 44 hours (44 samples × 1 hour)

**After**:
- Sample #897: 60s timeout → restart (30s) → retry (2s) → success
- Sample #898: 2s → success
- Sample #899: 2s → success
- ...
- **Total time**: ~2 minutes for recovery

---

## Testing

### Manual Testing

1. **Start server**:
   ```bash
   python -m activation_logging.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct
   ```

2. **Trigger restart**:
   ```bash
   curl -X POST http://localhost:8000/restart
   ```

3. **Check health**:
   ```bash
   curl http://localhost:8000/health
   ```

### Integration Testing

Run inference with restart enabled:
```bash
export ENABLE_SERVER_RESTART=true
export SERVER_RESTART_WAIT_TIME=30

python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100
```

Monitor logs for restart events:
```bash
tail -f goodwiki_json/client.log | grep -E "(timeout|restart|RESTART)"
```

---

## Deployment Recommendations

### For Development

Use `run_with_server.py` with manual restart:
```bash
python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct
```

### For Production

Use systemd for automatic restart:

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

---

## Backward Compatibility

### Disabling the Feature

To use the old behavior (retry without restart):
```bash
export ENABLE_SERVER_RESTART=false
```

### Default Behavior

By default, the restart protocol is **enabled** (`ENABLE_SERVER_RESTART=true`).

---

## Known Limitations

1. **Requires supervisor**: Server must be managed by systemd/supervisor/Docker to auto-restart after `os._exit(0)`
2. **In-flight requests lost**: Active requests are terminated during restart
3. **Model reload time**: Large models may take 10-30 seconds to reload
4. **Single restart attempt**: Only attempts restart once per timeout sequence

---

## Future Improvements

1. **Graceful request handling**: Save and replay in-flight requests
2. **Smarter restart triggers**: Multiple timeouts before restart
3. **Health metrics**: Track restart frequency and patterns
4. **Request queuing**: Queue requests during restart
5. **Backup server**: Spin up secondary server during restart

---

## Files Modified

1. `utils/lm.py` - Client-side timeout and restart logic
2. `activation_logging/server.py` - Server-side restart endpoint
3. `docs/SERVER_RESTART_PROTOCOL.md` - Documentation (new)
4. `CHANGELOG_SERVER_RESTART.md` - This changelog (new)

---

## Migration Guide

### For Existing Deployments

1. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

2. **Set environment variables** (optional, defaults are reasonable):
   ```bash
   export ENABLE_SERVER_RESTART=true
   export SERVER_RESTART_WAIT_TIME=30
   ```

3. **Update deployment scripts** to use systemd/supervisor if not already

4. **Test with small dataset**:
   ```bash
   python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 10
   ```

5. **Monitor logs** for restart events

### For New Deployments

The restart protocol is enabled by default. Just ensure your deployment uses systemd/supervisor for automatic process restart.

---

## Support

For issues or questions:
1. Check `docs/SERVER_RESTART_PROTOCOL.md` for detailed documentation
2. Review server and client logs for error messages
3. Test with `ENABLE_SERVER_RESTART=false` to isolate restart-related issues

