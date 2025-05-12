#!/usr/bin/env python
"""
Utility to analyze the activations logged from LLM responses.
Specifically designed to compare activations between hallucinated and non-hallucinated responses.

This tool reads from LMDB and correlates with benchmark evaluation results to identify
patterns in neural activations that distinguish hallucination from factual responses.
"""
import argparse
import os
import json
import numpy as np
import lmdb
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from pathlib import Path


def load_lmdb_entries(lmdb_path: str) -> Dict[str, Any]:
    """
    Load all entries from an LMDB file.
    
    Args:
        lmdb_path: Path to the LMDB file
        
    Returns:
        Dictionary of key -> entry mappings
    """
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
    
    entries = {}
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            entries[key.decode("utf-8")] = pickle.loads(value)
    
    env.close()
    return entries


def load_benchmark_results(results_dir: str) -> Dict[str, bool]:
    """
    Load benchmark results from the output directory.
    
    Args:
        results_dir: Path to the directory containing eval_results.json
        
    Returns:
        Dictionary mapping prompt hashes to hallucination status
    """
    results_file = Path(results_dir) / "eval_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Create a mapping from prompt hash to hallucination status
    hallucination_map = {}
    if "abstractions" in results:
        # Each entry in abstractions represents whether the model 
        # correctly abstained (didn't hallucinate) on that example
        for i, abstained in enumerate(results["abstractions"]):
            # We need to retrieve the corresponding prompt from the raw eval results
            # This is a simplified example - you might need to adjust based on the actual data structure
            prompt_hash = f"prompt_{i}"  # This is a placeholder - need to get actual hash
            hallucination_map[prompt_hash] = not abstained  # True if hallucinated, False if abstained
    
    return hallucination_map


def analyze_activations(
    entries: Dict[str, Any],
    hallucination_map: Dict[str, bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze activations, separating hallucinated from non-hallucinated responses.
    
    Args:
        entries: Dictionary of LMDB entries
        hallucination_map: Dictionary mapping prompt hashes to hallucination status
        
    Returns:
        Tuple of (hallucinated_activations, non_hallucinated_activations)
    """
    hallucinated_activations = []
    non_hallucinated_activations = []
    
    for key, entry in entries.items():
        if key in hallucination_map:
            activations = entry.get("activations")
            if activations is not None:
                # Calculate the mean activation across tokens
                mean_activation = np.mean(activations, axis=0)
                
                if hallucination_map[key]:
                    hallucinated_activations.append(mean_activation)
                else:
                    non_hallucinated_activations.append(mean_activation)
    
    return (
        np.array(hallucinated_activations) if hallucinated_activations else np.array([]),
        np.array(non_hallucinated_activations) if non_hallucinated_activations else np.array([])
    )


def plot_activation_comparison(
    hallucinated_activations: np.ndarray,
    non_hallucinated_activations: np.ndarray,
    output_path: str = "activation_comparison.png"
):
    """
    Plot a comparison of hallucinated vs non-hallucinated activations.
    
    Args:
        hallucinated_activations: Array of activations from hallucinated responses
        non_hallucinated_activations: Array of activations from non-hallucinated responses
        output_path: Path to save the plot
    """
    if hallucinated_activations.size == 0 or non_hallucinated_activations.size == 0:
        print("Not enough data to create a meaningful plot.")
        return
    
    # Reduce dimensionality for visualization if needed
    # Here we're simply calculating the mean across all examples
    mean_hallucinated = np.mean(hallucinated_activations, axis=0)
    mean_non_hallucinated = np.mean(non_hallucinated_activations, axis=0)
    
    # Take the first 100 dimensions for visualization (or fewer if not available)
    dim_count = min(100, mean_hallucinated.shape[0])
    x = np.arange(dim_count)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, mean_hallucinated[:dim_count], label="Hallucinated", color="red", alpha=0.7)
    plt.plot(x, mean_non_hallucinated[:dim_count], label="Non-hallucinated", color="blue", alpha=0.7)
    plt.title("Comparison of Mean Activation Patterns")
    plt.xlabel("Activation Dimension")
    plt.ylabel("Mean Activation Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Calculate and print the most divergent dimensions
    diff = np.abs(mean_hallucinated - mean_non_hallucinated)
    top_dims = np.argsort(diff)[-10:][::-1]  # Top 10 most different dimensions
    
    print("\nTop 10 most divergent activation dimensions:")
    for i, dim in enumerate(top_dims):
        print(f"{i+1}. Dimension {dim}: Difference = {diff[dim]:.4f}")
        print(f"   Hallucinated: {mean_hallucinated[dim]:.4f}, Non-hallucinated: {mean_non_hallucinated[dim]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM activations to distinguish hallucination patterns")
    parser.add_argument("--lmdb_path", type=str, required=True,
                      help="Path to the LMDB file containing logged activations")
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Directory containing benchmark evaluation results")
    parser.add_argument("--output_dir", type=str, default="activation_analysis",
                      help="Directory to save analysis outputs (default: activation_analysis)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading LMDB entries from {args.lmdb_path}")
    entries = load_lmdb_entries(args.lmdb_path)
    print(f"Loaded {len(entries)} entries")
    
    print(f"Loading benchmark results from {args.results_dir}")
    hallucination_map = load_benchmark_results(args.results_dir)
    print(f"Loaded hallucination data for {len(hallucination_map)} prompts")
    
    print("Analyzing activations...")
    hallucinated_activations, non_hallucinated_activations = analyze_activations(entries, hallucination_map)
    print(f"Found {len(hallucinated_activations)} hallucinated and {len(non_hallucinated_activations)} non-hallucinated examples")
    
    output_plot_path = os.path.join(args.output_dir, "activation_comparison.png")
    plot_activation_comparison(hallucinated_activations, non_hallucinated_activations, output_plot_path)
    
    # Save metadata about the analysis
    metadata = {
        "lmdb_path": args.lmdb_path,
        "results_dir": args.results_dir,
        "hallucinated_count": len(hallucinated_activations),
        "non_hallucinated_count": len(non_hallucinated_activations),
        "total_entries": len(entries),
    }
    
    with open(os.path.join(args.output_dir, "analysis_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 