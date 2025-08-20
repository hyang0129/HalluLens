#!/usr/bin/env python3
"""
Simple script to generate inference JSON from LMDB activations data.

This script reads from the LMDB activations database and generates a JSONL file
compatible with the HalluLens inference format.

Usage:
    python generate_inference_from_lmdb.py --lmdb_path lmdb_data/gguf/activations.lmdb --output_path output/test_llama.json
"""

import argparse
import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the path so we can import from activation_logging
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from activation_logging.activations_logger import ActivationsLogger
except ImportError as e:
    print(f"Error importing ActivationsLogger: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def load_lmdb_entries(lmdb_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all entries from the LMDB activations database.
    
    Args:
        lmdb_path: Path to the LMDB file
        
    Returns:
        Dictionary mapping prompt hashes to metadata entries
    """
    print(f"Loading LMDB entries from {lmdb_path}")
    
    # Initialize activations logger in read-only mode
    try:
        activation_logger = ActivationsLogger(lmdb_path=lmdb_path, read_only=True)
    except Exception as e:
        print(f"Error opening LMDB file: {e}")
        return {}
    
    # Get all keys from the metadata store
    try:
        keys = activation_logger.list_entries()
        print(f"Found {len(keys)} entries in LMDB")
    except Exception as e:
        print(f"Error listing entries: {e}")
        activation_logger.close()
        return {}
    
    # Load metadata for each key
    metadata_dict = {}
    for i, key in enumerate(keys):
        try:
            metadata = activation_logger.get_entry_by_key(key, metadata_only=True)
            if metadata:
                metadata_dict[key] = metadata
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(keys)} entries...")
        except Exception as e:
            print(f"Warning: Failed to load metadata for key {key[:8]}: {e}")
    
    activation_logger.close()
    print(f"Successfully loaded metadata for {len(metadata_dict)} entries")
    return metadata_dict


def extract_question_from_prompt(formatted_prompt: str) -> str:
    """
    Extract the original question from a formatted prompt.
    
    Args:
        formatted_prompt: The formatted prompt sent to the model
        
    Returns:
        The extracted question or the original prompt if format doesn't match
    """
    # Check for the expected format: "user: Answer in one sentence. Q:<question>\n A:"
    if formatted_prompt.startswith('user: Answer in one sentence. Q:') and formatted_prompt.endswith('\n A:'):
        question = formatted_prompt[len('user: Answer in one sentence. Q:'):-len('\n A:')]
        return question.strip()
    
    # Check for other common formats
    if formatted_prompt.startswith('user: ') and '\n' in formatted_prompt:
        # Extract everything after "user: " and before the first newline
        question = formatted_prompt[6:].split('\n')[0].strip()
        return question
    
    # If no recognized format, return the full prompt
    return formatted_prompt


def generate_inference_jsonl(lmdb_path: str, output_path: str) -> int:
    """
    Generate inference JSONL file from LMDB activations data.
    
    Args:
        lmdb_path: Path to the LMDB file
        output_path: Path to save the inference JSONL file
        
    Returns:
        Number of entries written
    """
    # Validate input
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
    
    # Load LMDB entries
    lmdb_entries = load_lmdb_entries(lmdb_path)
    
    if not lmdb_entries:
        print("No entries found in LMDB file")
        return 0
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert LMDB entries to inference format
    inference_entries = []
    for prompt_hash, metadata in lmdb_entries.items():
        # Extract the original prompt
        original_prompt = metadata.get('prompt', '')
        question = extract_question_from_prompt(original_prompt)
        
        # Create inference entry in the expected format
        inference_entry = {
            'prompt': question,
            'generation': metadata.get('response', ''),
            'model': metadata.get('model', ''),
            'input_length': metadata.get('input_length', 0),
            'timestamp': metadata.get('timestamp', 0)
        }
        
        # Add optional fields if they exist
        if 'messages' in metadata:
            inference_entry['messages'] = metadata['messages']
        if 'trim_position' in metadata:
            inference_entry['trim_position'] = metadata['trim_position']
        
        inference_entries.append(inference_entry)
    
    # Write to JSONL file (one JSON object per line)
    print(f"Writing {len(inference_entries)} entries to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in inference_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully generated inference JSONL with {len(inference_entries)} entries")
    print(f"Output saved to: {output_path}")
    return len(inference_entries)


def main():
    parser = argparse.ArgumentParser(description="Generate inference JSONL from LMDB activations data")
    parser.add_argument("--lmdb_path", type=str, required=True,
                       help="Path to the LMDB activations file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the inference JSONL file")
    parser.add_argument("--check_files", action="store_true",
                       help="Check if input files exist before processing")
    
    args = parser.parse_args()
    
    if args.check_files:
        print(f"Checking if LMDB file exists: {args.lmdb_path}")
        if Path(args.lmdb_path).exists():
            print("✅ LMDB file found")
        else:
            print("❌ LMDB file not found")
            return 1
    
    try:
        count = generate_inference_jsonl(args.lmdb_path, args.output_path)
        print(f"✅ Successfully generated {count} inference entries")
        return 0
    except Exception as e:
        print(f"❌ Error generating inference JSONL: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
