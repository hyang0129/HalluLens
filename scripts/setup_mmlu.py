#!/usr/bin/env python3
"""
Standalone CLI script for MMLU inference + evaluation.

Usage:
    # Run inference only
    python scripts/setup_mmlu.py --model meta-llama/Llama-3.1-8B-Instruct --step inference

    # Run eval only (after inference)
    python scripts/setup_mmlu.py --model meta-llama/Llama-3.1-8B-Instruct --step eval

    # Run both inference and eval
    python scripts/setup_mmlu.py --model meta-llama/Llama-3.1-8B-Instruct --step all

    # Batched inference (no server needed)
    python scripts/setup_mmlu.py --model meta-llama/Llama-3.1-8B-Instruct --step all --batch-size 8
"""

import argparse
import io
import os
import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)


def main():
    parser = argparse.ArgumentParser(
        description="MMLU setup: inference + evaluation for hallucination detection"
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--step", type=str, required=True,
                        choices=["inference", "eval", "eval_llm", "all"],
                        help="Which step to run")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "validation", "auxiliary_train"],
                        help="MMLU split (default: test)")
    parser.add_argument("--N", type=int, default=None,
                        help="Number of samples (None = entire filtered split)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for direct batched inference (bypasses HTTP server)")
    parser.add_argument("--logger-type", type=str, default="zarr",
                        choices=["lmdb", "json", "zarr"],
                        help="Activation logger type")
    parser.add_argument("--activations-path", type=str, default=None,
                        help="Path for storing activations (.zarr)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Base output directory")
    parser.add_argument("--generations-file-path", type=str, default=None,
                        help="Path for generations JSONL file")
    parser.add_argument("--eval-results-path", type=str, default=None,
                        help="Path for evaluation results JSON")
    parser.add_argument("--llm-evaluator", type=str, default=None,
                        help="Model to use as LLM judge for eval_llm step")
    parser.add_argument("--quick-debug-mode", action="store_true",
                        help="Use first 50 samples only")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume from existing generations")
    parser.add_argument("--progress-log", type=str, default=None,
                        help="Path to progress log file (tee stdout+stderr for monitoring)")

    args = parser.parse_args()

    # --- Tee stdout/stderr to progress log file if requested ---
    if args.progress_log:
        Path(args.progress_log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.progress_log, "a", encoding="utf-8")

        class Tee(io.TextIOBase):
            """Write to both the original stream and a log file."""
            def __init__(self, stream, log_file):
                self._stream = stream
                self._log = log_file
            def write(self, data):
                self._stream.write(data)
                self._log.write(data)
                self._log.flush()
                return len(data)
            def flush(self):
                self._stream.flush()
                self._log.flush()

        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    from scripts.run_with_server import run_experiment

    model_name = args.model.split("/")[-1]

    # Default activations path
    activations_path = args.activations_path
    if activations_path is None:
        activations_path = f"shared/mmlu/activations.zarr"

    # Default paths
    generations_file = args.generations_file_path
    if generations_file is None:
        task_dir = Path(args.output_dir) / "mmlu" / model_name
        task_dir.mkdir(parents=True, exist_ok=True)
        generations_file = str(task_dir / "generation.jsonl")

    if args.step == "all":
        steps = ["inference", "eval"]
    else:
        steps = [args.step]

    for step in steps:
        print(f"\n{'='*60}")
        print(f"Running MMLU step: {step}")
        print(f"{'='*60}\n")

        run_experiment(
            step=step,
            task="mmlu",
            model=args.model,
            split=args.split,
            N=args.N,
            logger_type=args.logger_type,
            activations_path=activations_path,
            output_dir=args.output_dir,
            generations_file_path=generations_file,
            eval_results_path=args.eval_results_path,
            max_inference_tokens=args.max_tokens,
            temperature=args.temperature,
            resume=not args.no_resume,
            batch_size=args.batch_size,
            quick_debug_mode=args.quick_debug_mode,
            llm_evaluator=args.llm_evaluator,
        )

    print(f"\nMMLU setup complete for {model_name}.")
    print(f"Generations: {generations_file}")
    print(f"Activations: {activations_path}")


if __name__ == "__main__":
    main()
