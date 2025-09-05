# Remote JSON Activation Logging Testing Guide

This guide walks you through testing the new JSON activation logging functionality in your remote GPU environment.

## Prerequisites

1. **Remote GPU Access**: Your `connect_gpu.sh` script should work
2. **Active Slurm Job**: Job named "nb8887new" should be running
3. **HalluLens Environment**: Project should be set up in `notebook_llm/HalluLens`

## Step-by-Step Testing Process

### 1. Connect to Remote GPU Environment

```bash
# From your local machine, run the connection script
./connect_gpu.sh
```

This will:
- Find your running Slurm job "nb8887new"
- Connect to the GPU node
- Navigate to `notebook_llm/HalluLens`
- Activate the `halu` virtual environment

### 2. Verify Environment Setup

Once connected to the remote GPU node:

```bash
# Check you're in the right directory
pwd
# Should show: /path/to/notebook_llm/HalluLens

# Check virtual environment is active
echo $VIRTUAL_ENV
# Should show: /path/to/notebook_llm/HalluLens/halu

# Check if the new JSON logger is available
python -c "from activation_logging.activations_logger import JsonActivationsLogger; print('✅ JSON logger available')"
```

### 3. Run the Automated Test Script

```bash
# Make the test script executable
chmod +x test_remote_json_logging.sh

# Run the comprehensive test
./test_remote_json_logging.sh
```

This script will:
- ✅ Check environment setup
- 📁 Create test directory
- 🚀 Start server with JSON logging
- 🧪 Test server endpoints
- 📂 Verify JSON file creation
- 🧹 Clean up resources

### 4. Manual Testing (Alternative)

If you prefer manual testing:

#### 4.1 Start Server with JSON Logging

```bash
# Start server with JSON activation logging
python -m activation_logging.vllm_serve \
  --logger-type json \
  --activations-path json_test_data \
  --target-layers all \
  --sequence-mode all \
  --port 8000 \
  --log-file json_server.log &

# Wait for server to start
sleep 10
```

#### 4.2 Test Server Endpoints

```bash
# Test health check
curl http://localhost:8000/health

# Test completion endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.7,
    "model": "test-model"
  }'

# Test chat completion endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 30,
    "temperature": 0.5
  }'
```

#### 4.3 Verify JSON Files

```bash
# Check if JSON files were created
ls -la json_test_data/

# Should see:
# metadata.json
# activations/

# Check metadata file
cat json_test_data/metadata.json | python -m json.tool

# Check activations directory
ls -la json_test_data/activations/

# Check file sizes
du -h json_test_data/activations/*.json
```

### 5. Compare with LMDB Logging

To verify the JSON logger works the same as LMDB:

```bash
# Start server with LMDB logging
python -m activation_logging.vllm_serve \
  --logger-type lmdb \
  --activations-path lmdb_test_data/activations.lmdb \
  --port 8001 &

# Test the same requests on port 8001
# Compare outputs and verify they're similar
```

### 6. Test with Activation Parser

```bash
# Create a simple test script
cat > test_parser.py << 'EOF'
from activation_logging.activation_parser import ActivationParser
import pandas as pd

# Create dummy metadata for testing
dummy_data = {
    'prompt': ['What is the capital of France?', 'What is 2+2?'],
    'prompt_hash': ['hash1', 'hash2'],
    'halu': [False, False],
    'abstain': [False, False],
    'split': ['train', 'test']
}
df = pd.DataFrame(dummy_data)

# Test JSON parser
try:
    parser = ActivationParser(
        inference_json="dummy.jsonl",  # Won't be used since we pass df
        eval_json="dummy.json",        # Won't be used since we pass df
        activations_path="json_test_data",
        df=df,
        logger_type="json"
    )
    
    entries = parser.activation_logger.list_entries()
    print(f"✅ JSON Parser found {len(entries)} entries")
    
    if entries:
        entry = parser.activation_logger.get_entry(entries[0])
        print(f"✅ Successfully retrieved entry: {entries[0][:8]}...")
        
except Exception as e:
    print(f"❌ Error: {e}")
EOF

python test_parser.py
```

## Expected Results

### Successful Test Indicators

1. **Server Startup**: Server starts without errors on specified port
2. **API Responses**: All endpoints return valid JSON responses
3. **File Creation**: 
   - `metadata.json` file created with proper structure
   - `activations/` directory created with `.json` files
4. **File Content**: 
   - Metadata contains logger config and entry information
   - Activation files contain tensor data in JSON format
5. **Parser Compatibility**: Activation parser can read JSON files

### File Structure Verification

```
json_test_data/
├── metadata.json          # Central metadata file
└── activations/
    ├── <hash1>.json       # Activation data for request 1
    ├── <hash2>.json       # Activation data for request 2
    └── ...
```

### Performance Considerations

- **File Sizes**: JSON files will be larger than LMDB equivalents
- **Speed**: JSON logging may be slower for large-scale operations
- **Memory**: Each activation file loads entirely into memory

## Troubleshooting

### Common Issues

1. **Server Won't Start**
   ```bash
   # Check if port is already in use
   netstat -tlnp | grep :8000
   
   # Check server logs
   tail -f json_server.log
   ```

2. **No JSON Files Created**
   ```bash
   # Check permissions
   ls -la json_test_data/
   
   # Check server logs for errors
   grep -i error json_server.log
   ```

3. **Import Errors**
   ```bash
   # Verify environment
   python -c "import activation_logging.activations_logger"
   
   # Check if changes were saved
   grep -n "JsonActivationsLogger" activation_logging/activations_logger.py
   ```

4. **Connection Issues**
   ```bash
   # Test local connection
   curl -v http://localhost:8000/health
   
   # Check if server is listening
   ss -tlnp | grep :8000
   ```

## Cleanup

```bash
# Stop server
pkill -f "uvicorn.*activation_logging.server"

# Remove test data
rm -rf json_test_data lmdb_test_data
rm -f json_server.log test_parser.py
```

## Next Steps

After successful testing:

1. **Production Use**: Use JSON logging for actual inference runs
2. **Data Analysis**: Use the activation parser with JSON data
3. **Migration**: Convert existing LMDB data to JSON if needed
4. **Monitoring**: Set up log monitoring for production deployments

## Environment Variables Reference

```bash
# For JSON logging
export ACTIVATION_STORAGE_PATH="json_data/activations"
export ACTIVATION_LOGGER_TYPE="json"
export ACTIVATION_TARGET_LAYERS="all"
export ACTIVATION_SEQUENCE_MODE="all"

# For LMDB logging (default)
export ACTIVATION_STORAGE_PATH="lmdb_data/activations.lmdb"
export ACTIVATION_LOGGER_TYPE="lmdb"
export ACTIVATION_LMDB_MAP_SIZE="68719476736"  # 64GB in bytes
```
