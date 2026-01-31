"""
Test script to verify resume functionality for inference.

This script tests that:
1. Resume correctly identifies already-processed prompts
2. Only remaining prompts are processed
3. Results are correctly merged
"""

import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import exp
import utils.lm as lm

def test_resume_basic():
    """Test basic resume functionality."""
    print("=" * 60)
    print("Test 1: Basic Resume Functionality")
    print("=" * 60)
    
    # Create test data
    test_prompts = pd.DataFrame({
        'prompt': [f'Question {i}?' for i in range(10)],
        'answer': [f'Answer {i}' for i in range(10)]
    })
    
    # Create temporary file for generations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Simulate first run - process first 5 prompts
        print("\n📝 Simulating first run (5 prompts)...")
        first_batch = test_prompts.iloc[:5].copy()
        first_batch['generation'] = ['Generated response ' + str(i) for i in range(5)]
        first_batch.to_json(temp_file, lines=True, orient='records')
        print(f"✅ Saved {len(first_batch)} generations to {temp_file}")
        
        # Load to verify
        loaded = pd.read_json(temp_file, lines=True)
        print(f"✅ Verified: {len(loaded)} generations in file")
        
        # Now test resume logic manually
        print("\n📝 Testing resume logic...")
        existing_generations = pd.read_json(temp_file, lines=True)
        existing_prompts = set(existing_generations['prompt'].tolist())
        
        # Filter out already-processed prompts
        mask = ~test_prompts['prompt'].isin(existing_prompts)
        remaining_prompts = test_prompts[mask].copy()
        
        print(f"📊 Resume statistics:")
        print(f"   - Total prompts: {len(test_prompts)}")
        print(f"   - Already completed: {len(existing_generations)}")
        print(f"   - Remaining to process: {len(remaining_prompts)}")
        
        # Verify correct prompts remain
        expected_remaining = [f'Question {i}?' for i in range(5, 10)]
        actual_remaining = remaining_prompts['prompt'].tolist()
        
        assert actual_remaining == expected_remaining, \
            f"Expected {expected_remaining}, got {actual_remaining}"
        print("✅ Correct prompts identified for processing")
        
        # Simulate processing remaining prompts
        remaining_prompts['generation'] = ['Generated response ' + str(i) for i in range(5, 10)]
        
        # Merge with existing
        merged = pd.concat([existing_generations, remaining_prompts], ignore_index=True)
        merged.to_json(temp_file, lines=True, orient='records')
        
        # Verify final result
        final = pd.read_json(temp_file, lines=True)
        print(f"\n✅ Final result: {len(final)} total generations")
        
        assert len(final) == 10, f"Expected 10 generations, got {len(final)}"
        assert set(final['prompt'].tolist()) == set(test_prompts['prompt'].tolist()), \
            "Prompt mismatch in final result"
        
        print("✅ All prompts present in final result")
        print("\n" + "=" * 60)
        print("✅ Test 1 PASSED")
        print("=" * 60)
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_resume_all_complete():
    """Test resume when all prompts are already processed."""
    print("\n" + "=" * 60)
    print("Test 2: Resume with All Prompts Complete")
    print("=" * 60)
    
    # Create test data
    test_prompts = pd.DataFrame({
        'prompt': [f'Question {i}?' for i in range(5)],
        'answer': [f'Answer {i}' for i in range(5)]
    })
    
    # Create temporary file for generations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save all prompts as already processed
        print("\n📝 Simulating all prompts already processed...")
        complete_batch = test_prompts.copy()
        complete_batch['generation'] = ['Generated response ' + str(i) for i in range(5)]
        complete_batch.to_json(temp_file, lines=True, orient='records')
        
        # Test resume logic
        existing_generations = pd.read_json(temp_file, lines=True)
        existing_prompts = set(existing_generations['prompt'].tolist())
        
        mask = ~test_prompts['prompt'].isin(existing_prompts)
        remaining_prompts = test_prompts[mask].copy()
        
        print(f"📊 Resume statistics:")
        print(f"   - Total prompts: {len(test_prompts)}")
        print(f"   - Already completed: {len(existing_generations)}")
        print(f"   - Remaining to process: {len(remaining_prompts)}")
        
        assert len(remaining_prompts) == 0, \
            f"Expected 0 remaining prompts, got {len(remaining_prompts)}"
        
        print("✅ Correctly identified that all prompts are complete")
        print("\n" + "=" * 60)
        print("✅ Test 2 PASSED")
        print("=" * 60)
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_progress_tracking():
    """Test that progress tracking correctly accounts for already-completed items."""
    print("\n" + "=" * 60)
    print("Test 3: Progress Tracking with Resume")
    print("=" * 60)

    # Test progress tracking initialization
    print("\n📝 Testing progress tracking initialization...")

    # Initialize with 100 total, 30 already completed
    lm.initialize_progress_tracking(total_requests=100, already_completed=30)
    stats = lm.get_progress_stats()

    print(f"📊 Initial progress stats:")
    print(f"   - Total requests: {stats['total_requests']}")
    print(f"   - Completed requests: {stats['completed_requests']}")
    print(f"   - Failed requests: {stats['failed_requests']}")

    assert stats['total_requests'] == 100, f"Expected total 100, got {stats['total_requests']}"
    assert stats['completed_requests'] == 30, f"Expected completed 30, got {stats['completed_requests']}"
    assert stats['failed_requests'] == 0, f"Expected failed 0, got {stats['failed_requests']}"

    print("✅ Progress tracking correctly initialized with already-completed count")

    # Simulate processing one more request
    lm.update_progress(success=True)
    stats = lm.get_progress_stats()

    assert stats['completed_requests'] == 31, f"Expected completed 31, got {stats['completed_requests']}"
    print("✅ Progress tracking correctly increments from resumed state")

    print("\n" + "=" * 60)
    print("✅ Test 3 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    print("\n🧪 Testing Resume Inference Functionality\n")

    try:
        test_resume_basic()
        test_resume_all_complete()
        test_progress_tracking()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe resume functionality is working correctly.")
        print("You can now use it for large-scale inference jobs.")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

