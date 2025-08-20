#!/bin/bash

# Test script to verify GPU connection and environment
# Run this after connecting to the GPU node

echo "🧪 HalluLens GPU Environment Test"
echo "================================="
echo

# Test 1: Check hostname
echo "1️⃣ Testing hostname..."
HOSTNAME=$(hostname)
echo "   Hostname: $HOSTNAME"
if [[ $HOSTNAME == *"skl-a-"* ]]; then
    echo "   ✅ Connected to GPU node"
else
    echo "   ❌ Not on a GPU node"
fi
echo

# Test 2: Check GPU availability
echo "2️⃣ Testing GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi available"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    echo "   GPU: $GPU_INFO"
else
    echo "   ❌ nvidia-smi not found"
fi
echo

# Test 3: Check Python environment
echo "3️⃣ Testing Python environment..."
PYTHON_VERSION=$(python --version 2>&1)
echo "   Python: $PYTHON_VERSION"
PYTHON_PATH=$(which python)
echo "   Path: $PYTHON_PATH"
echo

# Test 4: Check current directory
echo "4️⃣ Testing current directory..."
CURRENT_DIR=$(pwd)
echo "   Current: $CURRENT_DIR"
if [[ $CURRENT_DIR == *"HalluLens"* ]]; then
    echo "   ✅ In HalluLens directory"
else
    echo "   ⚠️  Not in HalluLens directory"
    echo "   💡 Run: cd notebook_llm/HalluLens"
fi
echo

# Test 5: Check HalluLens files
echo "5️⃣ Testing HalluLens files..."
if [ -f "requirements.txt" ]; then
    echo "   ✅ requirements.txt found"
else
    echo "   ❌ requirements.txt not found"
fi

if [ -d "utils" ]; then
    echo "   ✅ utils directory found"
else
    echo "   ❌ utils directory not found"
fi

if [ -f "utils/lm.py" ]; then
    echo "   ✅ utils/lm.py found"
else
    echo "   ❌ utils/lm.py not found"
fi
echo

# Test 6: Test Python imports
echo "6️⃣ Testing Python imports..."
python -c "
try:
    import torch
    print('   ✅ PyTorch available:', torch.__version__)
    if torch.cuda.is_available():
        print('   ✅ CUDA available:', torch.version.cuda)
        print('   ✅ GPU count:', torch.cuda.device_count())
    else:
        print('   ❌ CUDA not available')
except ImportError:
    print('   ⚠️  PyTorch not installed')

try:
    import pandas
    print('   ✅ Pandas available:', pandas.__version__)
except ImportError:
    print('   ⚠️  Pandas not installed')

try:
    import openai
    print('   ✅ OpenAI available:', openai.__version__)
except ImportError:
    print('   ⚠️  OpenAI not installed')
"
echo

# Test 7: Check storage space
echo "7️⃣ Testing storage space..."
STORAGE=$(df -h /home/hy3134 | tail -1 | awk '{print $3 " used, " $4 " available (" $5 " full)"}')
echo "   Storage: $STORAGE"
echo

echo "🎯 Test Summary"
echo "==============="
echo "If all tests show ✅, your environment is ready for HalluLens development!"
echo "If you see ❌ or ⚠️, please check the setup or install missing dependencies."
echo
