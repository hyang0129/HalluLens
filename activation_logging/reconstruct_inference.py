#!/usr/bin/env python3
"""
Script to reconstruct inference JSONL file from activations logger metadata and original QA data.

This script takes:
1. QA_OUTPUT_PATH: Path to the original questions/answers JSONL file
2. LMDB_PATH: Path to the activations logger LMDB file
3. OUTPUT_PATH: Path to save the reconstructed inference JSONL file

The script matches prompts between the QA data and activations metadata using prompt hashing,
then reconstructs the inference JSONL with all the original fields plus the generated responses.

Note: The QA data contains raw questions, but the model receives formatted prompts with
"user: Answer in one sentence. Q:" prefix and "\n A:" suffix.
"""

import argparse
import json
import hashlib
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from activation_logging.activations_logger import ActivationsLogger


def hash_prompt(prompt: str) -> str:
    """
    Hash a prompt string to match the format used in ActivationsLogger.
    
    Args:
        prompt: The prompt string to hash
        
    Returns:
        SHA-256 hash of the prompt
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def format_prompt_for_model(raw_question: str) -> str:
    """
    Format a raw question into the prompt format sent to the model.
    
    Args:
        raw_question: The raw question from QA data
        
    Returns:
        Formatted prompt as sent to the model
    """
    return f"user: Answer in one sentence. Q:{raw_question}\n A:"


def load_qa_data(qa_path: str) -> List[Dict[str, Any]]:
    """
    Load the original QA data from JSONL file.
    
    Args:
        qa_path: Path to the QA JSONL file
        
    Returns:
        List of QA dictionaries
    """
    logger.info(f"Loading QA data from {qa_path}")
    qa_data = []
    with jsonlines.open(qa_path, 'r') as reader:
        for item in reader:
            qa_data.append(item)
    logger.info(f"Loaded {len(qa_data)} QA pairs")
    return qa_data


def get_activations_metadata(lmdb_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from the activations logger LMDB.
    
    Args:
        lmdb_path: Path to the LMDB file
        
    Returns:
        Dictionary mapping prompt hashes to metadata entries
    """
    logger.info(f"Loading activations metadata from {lmdb_path}")
    
    # Initialize activations logger in read-only mode
    activation_logger = ActivationsLogger(lmdb_path=lmdb_path, read_only=True)
    
    # Get all keys from the metadata store
    keys = activation_logger.list_entries()
    logger.info(f"Found {len(keys)} entries in LMDB")
    
    # Load metadata for each key
    metadata_dict = {}
    for key in keys:
        try:
            metadata = activation_logger.get_entry_by_key(key, metadata_only=True)
            metadata_dict[key] = metadata
        except Exception as e:
            logger.warning(f"Failed to load metadata for key {key[:8]}: {e}")
    
    activation_logger.close()
    logger.info(f"Successfully loaded metadata for {len(metadata_dict)} entries")
    return metadata_dict


def match_qa_with_activations(qa_data: List[Dict[str, Any]], 
                            activations_metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Match QA data with activations metadata using prompt hashing.
    
    Args:
        qa_data: List of QA dictionaries
        activations_metadata: Dictionary mapping prompt hashes to metadata
        
    Returns:
        List of matched dictionaries containing QA data plus generated responses
    """
    logger.info("Matching QA data with activations metadata...")
    
    matched_data = []
    matched_count = 0
    unmatched_count = 0
    
    for qa_item in qa_data:
        # Get the raw question from QA data
        raw_question = qa_item.get('prompt', '')
        
        # Format the question to match what was sent to the model
        formatted_prompt = format_prompt_for_model(raw_question)
        
        # Hash the formatted prompt to match with activations
        prompt_hash = hash_prompt(formatted_prompt)
        
        # Look for matching activation metadata
        if prompt_hash in activations_metadata:
            activation_item = activations_metadata[prompt_hash]
            
            # Create combined entry
            combined_item = qa_item.copy()
            combined_item['generation'] = activation_item.get('response', '')
            combined_item['model'] = activation_item.get('model', '')
            combined_item['input_length'] = activation_item.get('input_length', 0)
            combined_item['timestamp'] = activation_item.get('timestamp', 0)
            
            # Add any additional fields from activation metadata
            if 'messages' in activation_item:
                combined_item['messages'] = activation_item['messages']
            if 'trim_position' in activation_item:
                combined_item['trim_position'] = activation_item['trim_position']
            
            matched_data.append(combined_item)
            matched_count += 1
        else:
            logger.warning(f"No activation found for question: {raw_question[:50]}...")
            logger.debug(f"Formatted prompt hash: {prompt_hash}")
            unmatched_count += 1
    
    logger.info(f"Matched {matched_count} entries, {unmatched_count} unmatched")
    return matched_data


def save_reconstructed_jsonl(data: List[Dict[str, Any]], output_path: str):
    """
    Save the reconstructed data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path to save the JSONL file
    """
    logger.info(f"Saving reconstructed data to {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, 'w') as writer:
        for item in data:
            writer.write(item)
    
    logger.info(f"Saved {len(data)} entries to {output_path}")


def reconstruct_inference(
    qa_output_path: str,
    lmdb_path: str,
    output_path: str,
    verbose: bool = True
) -> int:
    """
    Reconstruct inference JSONL from activations logger and QA data.

    Args:
        qa_output_path: Path to the original QA JSONL file
        lmdb_path: Path to the activations logger LMDB file
        output_path: Path to save the reconstructed inference JSONL file
        verbose: Enable verbose logging
    Returns:
        Number of matched entries written to output_path
    Raises:
        FileNotFoundError: If input files are missing
        Exception: For other errors during reconstruction
    """
    # Set up logging
    if verbose:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # Validate input files
    if not Path(qa_output_path).exists():
        raise FileNotFoundError(f"QA file not found: {qa_output_path}")
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")

    # Load QA data
    qa_data = load_qa_data(qa_output_path)
    # Load activations metadata
    activations_metadata = get_activations_metadata(lmdb_path)
    # Match QA data with activations
    matched_data = match_qa_with_activations(qa_data, activations_metadata)
    # Save reconstructed JSONL
    save_reconstructed_jsonl(matched_data, output_path)
    logger.info(f"Successfully reconstructed inference JSONL with {len(matched_data)} entries")
    logger.info(f"Output saved to: {output_path}")
    return len(matched_data)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct inference JSONL from activations logger")
    parser.add_argument("--qa_output_path", type=str, required=True,
                       help="Path to the original QA JSONL file")
    parser.add_argument("--lmdb_path", type=str, required=True,
                       help="Path to the activations logger LMDB file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the reconstructed inference JSONL file")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose logging")
    args = parser.parse_args()
    try:
        reconstruct_inference(
            qa_output_path=args.qa_output_path,
            lmdb_path=args.lmdb_path,
            output_path=args.output_path,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Error reconstructing inference JSONL: {e}")
        raise


if __name__ == "__main__":
    main() 