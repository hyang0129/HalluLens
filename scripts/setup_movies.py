#!/usr/bin/env python3
"""
Standalone Movies QA setup: inference + evaluation.

Reproduces the Movies QA notebook pipeline from the command line.

Usage:
  python scripts/setup_movies.py --model meta-llama/Llama-3.1-8B-Instruct [--split test] [--N 100] [--batch-size 8]
"""

import argparse
import os
import sys
import io
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(description="Movies QA: inference + evaluation pipeline")

    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="Dataset split (default: test = 7,856 questions)")
    parser.add_argument("--N", type=int, default=None, help="Number of samples (None = all)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for batched inference (None = sequential server path)")
    parser.add_argument("--activations-path", type=str, default=None,
                        help="Path to .zarr store for activations")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Base output directory")
    parser.add_argument("--logger-type", type=str, default="zarr",
                        choices=["lmdb", "json", "zarr"],
                        help="Activation logger type (default: zarr)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Server log file path")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference step (only run eval)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip eval step (only run inference)")
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

    model_name = args.model.split("/")[-1]
    task_dir = Path(args.output_dir) / "movies" / model_name
    task_dir.mkdir(parents=True, exist_ok=True)

    generations_file = str(task_dir / "generation.jsonl")
    eval_results_file = str(task_dir / "eval_results.json")

    print("=" * 60)
    print("Movies QA Setup Pipeline")
    print("=" * 60)
    print(f"Model          : {args.model}")
    print(f"Split          : {args.split}  (N={args.N or 'all'})")
    print(f"Batch size     : {args.batch_size or 'disabled (sequential server path)'}")
    print(f"Generations    : {generations_file}")
    print(f"Eval results   : {eval_results_file}")
    if args.activations_path:
        print(f"Activations    : {args.activations_path}")
    print("=" * 60)

    from tasks.llmsknow.movies import run_step

    # --- Inference ---
    if not args.skip_inference:
        print("\n>>> Step 1: Inference")
        run_step(
            step="inference",
            model=args.model,
            output_dir=args.output_dir,
            split=args.split,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            N=args.N,
            generations_file_path=generations_file,
            log_file=args.log_file,
            logger_type=args.logger_type,
            activations_path=args.activations_path,
            resume=True,
            batch_size=args.batch_size,
        )

    # --- Evaluation ---
    if not args.skip_eval:
        print("\n>>> Step 2: Evaluation (substring match)")
        run_step(
            step="eval",
            model=args.model,
            output_dir=args.output_dir,
            generations_file_path=generations_file,
            eval_results_path=eval_results_file,
        )

    print("\nDone! All steps completed.")


if __name__ == "__main__":
    main()
