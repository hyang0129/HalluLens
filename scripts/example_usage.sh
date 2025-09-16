#!/bin/bash
# Example usage of the unified server management scripts

echo "=== HalluLens Unified Server Management Examples ==="
echo ""

echo "Step 0: Check setup and download required data"
echo "python scripts/check_setup.py"
echo "python data/download_data.py --all"
echo ""

# Example 1: Run only prompt generation
echo "Example 1: Generate prompts only"
echo "python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 10"
echo ""

# Example 2: Run only inference
echo "Example 2: Run inference only (requires prompts to exist)"
echo "python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct"
echo ""

# Example 3: Run only evaluation
echo "Example 3: Run evaluation only (requires inference results to exist)"
echo "python scripts/run_with_server.py --step eval --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct"
echo ""

# Example 4: Run all steps in sequence
echo "Example 4: Run all steps in sequence"
echo "python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 10"
echo ""

# Example 5: Use the enhanced bash script
echo "Example 5: Use enhanced bash script (equivalent to original task1_precisewikiqa.sh)"
echo "bash scripts/task1_precisewikiqa_with_server.sh --N 10 --step all"
echo ""

# Example 6: Custom activation storage
echo "Example 6: Custom activation storage path"
echo "python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --activations-path custom_data/my_experiment.lmdb"
echo ""

# Example 7: JSON activation logging
echo "Example 7: Use JSON activation logging instead of LMDB"
echo "python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --logger-type json --activations-path json_data/activations"
echo ""

echo "=== Key Benefits ==="
echo "✅ No need to manually start/stop server"
echo "✅ Automatic server health checking"
echo "✅ Proper cleanup on interruption"
echo "✅ Configurable activation storage"
echo "✅ Support for all existing task parameters"
echo ""

echo "=== Quick Start ==="
echo "To run a complete PreciseWikiQA experiment with 10 samples:"
echo "python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 10"
