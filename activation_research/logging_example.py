"""
Example usage showing how to suppress loguru messages during data loading.
"""

from activation_research.hallucination_evaluator import HallucinationEvaluator, suppress_dataloader_logs, set_logging_level
from activation_logging.activation_parser import ActivationParser

def example_with_suppressed_logs():
    """Example showing how to suppress noisy log messages."""

    # Option 1: Suppress all info-level logs (recommended for data loading)
    suppress_dataloader_logs()

    # Option 2: Set specific logging level
    # set_logging_level("ERROR")  # Only show errors
    # set_logging_level("WARNING")  # Show warnings and errors
    # set_logging_level("INFO")  # Show all messages (default)

    # Note: verbose=False will suppress:
    # - ActivationParser metadata loading messages ("Found X prompts...")
    # - ActivationsLogger initialization messages
    # - JsonActivationsLogger detailed processing messages
    # - Activation extraction progress messages
    
    # Create ActivationParser with verbose=False to reduce messages
    ap = ActivationParser(
        inference_json="path/to/inference.jsonl",
        eval_json="path/to/eval.json",
        activations_path="path/to/activations.lmdb",
        verbose=False  # This suppresses both metadata loading AND activation logger messages
    )
    
    # Get datasets (these will now be much quieter)
    train_dataset = ap.get_dataset('train')
    eval_dataset = ap.get_dataset('test')
    
    # Create evaluator and run evaluation
    evaluator = HallucinationEvaluator(
        model=your_model,
        activation_parser_df=ap.df,
        layers=[10, 15, 20, 25, 30]
    )
    
    # This will now run with minimal logging output
    stats = evaluator.evaluate_detection(train_dataset, eval_dataset)
    
    print("Results:", stats)

def example_with_different_log_levels():
    """Example showing different logging levels."""
    
    print("=== With INFO level (default) ===")
    set_logging_level("INFO")
    # Your code here - will show all messages
    
    print("\n=== With WARNING level (quieter) ===") 
    set_logging_level("WARNING")
    # Your code here - will only show warnings and errors
    
    print("\n=== With ERROR level (very quiet) ===")
    set_logging_level("ERROR") 
    # Your code here - will only show errors

if __name__ == "__main__":
    example_with_suppressed_logs()
