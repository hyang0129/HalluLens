#!/usr/bin/env python3
"""
Test script to demonstrate the fixed layer functionality in ActivationDataset.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from activation_logging.activation_parser import ActivationParser


def test_fixed_layer_dataset():
    """Test the fixed layer functionality."""
    
    # Example paths - you'll need to adjust these to your actual data
    inference_json = "path/to/your/inference.json"
    eval_json = "path/to/your/eval.json"
    activations_path = "path/to/your/activations"
    
    try:
        # Initialize the parser
        parser = ActivationParser(
            inference_json=inference_json,
            eval_json=eval_json,
            activations_path=activations_path,
            logger_type="lmdb"  # or "json"
        )
        
        # Test 1: Original behavior (random layer selection)
        print("=== Test 1: Original behavior (random layer selection) ===")
        dataset_random = parser.get_dataset(split='train')
        
        if len(dataset_random) > 0:
            sample = dataset_random[0]
            print(f"Layer 1 index: {sample['layer1_idx']}")
            print(f"Layer 2 index: {sample['layer2_idx']}")
            print(f"Layer 1 activations shape: {sample['layer1_activations'].shape}")
            print(f"Layer 2 activations shape: {sample['layer2_activations'].shape}")
        
        # Test 2: Fixed layer behavior
        print("\n=== Test 2: Fixed layer behavior (layer 0 always selected) ===")
        fixed_layer_idx = 0  # This corresponds to the first layer in relevant_layers
        dataset_fixed = parser.get_dataset(split='train', fixed_layer=fixed_layer_idx)
        
        if len(dataset_fixed) > 0:
            # Test multiple samples to verify layer 1 is always the fixed layer
            for i in range(min(3, len(dataset_fixed))):
                sample = dataset_fixed[i]
                print(f"Sample {i}: Layer 1 index: {sample['layer1_idx']}, Layer 2 index: {sample['layer2_idx']}")
                assert sample['layer1_idx'] == fixed_layer_idx, f"Expected layer1_idx to be {fixed_layer_idx}, got {sample['layer1_idx']}"
                assert sample['layer2_idx'] != fixed_layer_idx, f"Layer 2 should be different from fixed layer {fixed_layer_idx}"
        
        # Test 3: Different fixed layer
        print("\n=== Test 3: Different fixed layer (layer 5 always selected) ===")
        fixed_layer_idx = 5
        dataset_fixed2 = parser.get_dataset(split='train', fixed_layer=fixed_layer_idx)
        
        if len(dataset_fixed2) > 0:
            sample = dataset_fixed2[0]
            print(f"Layer 1 index: {sample['layer1_idx']}")
            print(f"Layer 2 index: {sample['layer2_idx']}")
            assert sample['layer1_idx'] == fixed_layer_idx, f"Expected layer1_idx to be {fixed_layer_idx}, got {sample['layer1_idx']}"
            assert sample['layer2_idx'] != fixed_layer_idx, f"Layer 2 should be different from fixed layer {fixed_layer_idx}"
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please update the file paths in this script to point to your actual data files.")
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_usage():
    """Demonstrate how to use the new functionality."""
    
    print("=== Usage Examples ===")
    print("""
# Example 1: Original behavior (random layer selection)
parser = ActivationParser(inference_json, eval_json, activations_path)
dataset = parser.get_dataset(split='train')

# Example 2: Fixed layer behavior
# One activation will always be from layer index 0 (first layer in relevant_layers)
# The other activation will be from a random different layer
dataset_fixed = parser.get_dataset(split='train', fixed_layer=0)

# Example 3: Custom relevant layers with fixed layer
# Use layers 10-20, with layer index 5 (corresponds to actual layer 15) always selected
dataset_custom = parser.get_dataset(
    split='train', 
    relevant_layers=list(range(10, 21)),  # layers 10-20
    fixed_layer=5  # index 5 in the relevant_layers list (actual layer 15)
)

# When you iterate through the dataset:
for sample in dataset_fixed:
    layer1_activations = sample['layer1_activations']  # Always from fixed layer
    layer2_activations = sample['layer2_activations']  # From random different layer
    layer1_idx = sample['layer1_idx']  # Will always be the fixed_layer value
    layer2_idx = sample['layer2_idx']  # Will be different from layer1_idx
    break
""")


if __name__ == "__main__":
    demonstrate_usage()
    test_fixed_layer_dataset()
