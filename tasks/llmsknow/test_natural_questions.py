"""
Quick test script for Natural Questions implementation.
Tests data loading and basic functionality without running full inference.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from tasks.llmsknow.natural_questions import load_nq_data, compute_correctness_nq

def test_data_loading():
    """Test that Natural Questions data can be loaded."""
    print("=" * 60)
    print("Testing Natural Questions Data Loading")
    print("=" * 60)
    
    try:
        # Load first 10 samples
        data = load_nq_data(split="test", n_samples=10)
        
        print(f"\n✅ Successfully loaded {len(data)} samples")
        
        # Display first sample
        print("\nFirst sample:")
        sample = data[0]
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Context (first 100 chars): {sample['context'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return False


def test_correctness_evaluation():
    """Test correctness evaluation function."""
    print("\n" + "=" * 60)
    print("Testing Correctness Evaluation")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "model_answer": "The answer is Norfolk",
            "correct_answer": "Norfolk",
            "expected": 1,
            "description": "Exact match in sentence"
        },
        {
            "model_answer": "It was filmed in norfolk",
            "correct_answer": "Norfolk",
            "expected": 1,
            "description": "Case-insensitive match"
        },
        {
            "model_answer": "The show was filmed in London",
            "correct_answer": "Norfolk",
            "expected": 0,
            "description": "No match"
        },
        {
            "model_answer": "Adam Vinatieri is the player",
            "correct_answer": "Adam Vinatieri",
            "expected": 1,
            "description": "Answer in context"
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        model_answers = [test["model_answer"]]
        correct_answers = [test["correct_answer"]]
        
        result = compute_correctness_nq(model_answers, correct_answers)
        
        passed = result[0] == test["expected"]
        status = "✅" if passed else "❌"
        
        print(f"\nTest {i}: {test['description']} {status}")
        print(f"  Model: '{test['model_answer']}'")
        print(f"  Correct: '{test['correct_answer']}'")
        print(f"  Expected: {test['expected']}, Got: {result[0]}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_file_structure():
    """Test that expected files and directories exist."""
    print("\n" + "=" * 60)
    print("Testing File Structure")
    print("=" * 60)
    
    # Check data file
    data_file = project_root / "external" / "LLMsKnow" / "data" / "nq_wc_dataset.csv"
    
    if data_file.exists():
        print(f"✅ Data file exists: {data_file}")
        
        # Check file size
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        
        return True
    else:
        print(f"❌ Data file not found: {data_file}")
        print("   Make sure external/LLMsKnow/ submodule is properly initialized")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NATURAL QUESTIONS IMPLEMENTATION TEST")
    print("=" * 60)
    
    results = {
        "File Structure": test_file_structure(),
        "Data Loading": test_data_loading(),
        "Correctness Evaluation": test_correctness_evaluation()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nNatural Questions is ready to use. Try:")
        print("\npython scripts/run_with_server.py \\")
        print("    --step all \\")
        print("    --task naturalquestions \\")
        print("    --model mistralai/Mistral-7B-Instruct-v0.2 \\")
        print("    --N 100")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("\nPlease fix the issues before using Natural Questions.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
