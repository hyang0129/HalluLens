#!/usr/bin/env python3
"""
Setup checker for HalluLens project.

This script checks if all required data files and dependencies are available
for running the different tasks and provides instructions for downloading missing data.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_file_exists(file_path, description):
    """Check if a file exists and log the result."""
    if file_path.exists():
        logger.success(f"✅ {description}: {file_path}")
        return True
    else:
        logger.error(f"❌ {description}: {file_path}")
        return False

def check_directory_exists(dir_path, description):
    """Check if a directory exists and log the result."""
    if dir_path.exists() and dir_path.is_dir():
        logger.success(f"✅ {description}: {dir_path}")
        return True
    else:
        logger.error(f"❌ {description}: {dir_path}")
        return False

def check_precisewikiqa_setup():
    """Check setup for PreciseWikiQA task."""
    logger.info("🔍 Checking PreciseWikiQA setup...")
    
    all_good = True
    
    # Required data files
    wiki_data_file = project_root / "data" / "wiki_data" / "doc_goodwiki_h_score.jsonl"
    all_good &= check_file_exists(wiki_data_file, "Wiki data file")
    
    # Required directories (will be created if missing)
    qa_save_dir = project_root / "data" / "precise_qa" / "save"
    qa_save_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(qa_save_dir, "QA save directory")
    
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(output_dir, "Output directory")
    
    return all_good

def check_longwiki_setup():
    """Check setup for LongWiki task."""
    logger.info("🔍 Checking LongWiki setup...")
    
    all_good = True
    
    # Required data files
    wiki_db = project_root / "data" / "wiki_data" / ".cache" / "enwiki-20230401.db"
    all_good &= check_file_exists(wiki_db, "Wikipedia database")
    
    # Required directories (will be created if missing)
    longwiki_save_dir = project_root / "data" / "longwiki" / "save"
    longwiki_save_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(longwiki_save_dir, "LongWiki save directory")
    
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(output_dir, "Output directory")
    
    return all_good

def check_mixedentities_setup():
    """Check setup for Mixed Entities task."""
    logger.info("🔍 Checking Mixed Entities setup...")
    
    all_good = True
    
    # Required data directories
    refusal_dir = project_root / "data" / "refusal_test"
    all_good &= check_directory_exists(refusal_dir, "Refusal test data directory")
    
    # Check for specific data files if directory exists
    if refusal_dir.exists():
        medicine_file = refusal_dir / "medicine_data.csv"  # Example file
        # Note: Actual file names may vary, this is just an example
        if not medicine_file.exists():
            logger.warning(f"⚠️  Medicine data may be missing in {refusal_dir}")
    
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(output_dir, "Output directory")
    
    return all_good

def check_activation_logging_setup():
    """Check activation logging setup."""
    logger.info("🔍 Checking activation logging setup...")
    
    all_good = True
    
    # Check if activation logging module exists
    activation_logging_dir = project_root / "activation_logging"
    all_good &= check_directory_exists(activation_logging_dir, "Activation logging module")
    
    # Check key files
    if activation_logging_dir.exists():
        server_file = activation_logging_dir / "server.py"
        all_good &= check_file_exists(server_file, "Activation logging server")
        
        vllm_serve_file = activation_logging_dir / "vllm_serve.py"
        all_good &= check_file_exists(vllm_serve_file, "vLLM serve wrapper")
    
    # Create LMDB data directory
    lmdb_dir = project_root / "lmdb_data"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    check_directory_exists(lmdb_dir, "LMDB data directory")
    
    return all_good

def print_download_instructions():
    """Print instructions for downloading missing data."""
    logger.info("\n📥 To download missing data, run:")
    logger.info("  python data/download_data.py --all")
    logger.info("\nOr for specific tasks:")
    logger.info("  python data/download_data.py --precisewikiqa")
    logger.info("  python data/download_data.py --longwiki")
    logger.info("  python data/download_data.py --nonexistent_refusal")

def main():
    """Main setup checker."""
    logger.info("🚀 HalluLens Setup Checker")
    logger.info("=" * 50)
    
    all_tasks_ready = True
    
    # Check each task setup
    precisewikiqa_ready = check_precisewikiqa_setup()
    longwiki_ready = check_longwiki_setup()
    mixedentities_ready = check_mixedentities_setup()
    activation_ready = check_activation_logging_setup()
    
    all_tasks_ready = precisewikiqa_ready and longwiki_ready and mixedentities_ready and activation_ready
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 Setup Summary:")
    
    status_icon = "✅" if precisewikiqa_ready else "❌"
    logger.info(f"  {status_icon} PreciseWikiQA: {'Ready' if precisewikiqa_ready else 'Missing data'}")
    
    status_icon = "✅" if longwiki_ready else "❌"
    logger.info(f"  {status_icon} LongWiki: {'Ready' if longwiki_ready else 'Missing data'}")
    
    status_icon = "✅" if mixedentities_ready else "❌"
    logger.info(f"  {status_icon} Mixed Entities: {'Ready' if mixedentities_ready else 'Missing data'}")
    
    status_icon = "✅" if activation_ready else "❌"
    logger.info(f"  {status_icon} Activation Logging: {'Ready' if activation_ready else 'Missing components'}")
    
    if all_tasks_ready:
        logger.success("\n🎉 All tasks are ready! You can now run:")
        logger.success("  python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 10")
    else:
        logger.warning("\n⚠️  Some tasks are missing required data.")
        print_download_instructions()
    
    logger.info("\n📖 For more information, see:")
    logger.info("  - README.md")
    logger.info("  - README_file_generation.md")
    logger.info("  - scripts/example_usage.sh")

if __name__ == "__main__":
    main()
