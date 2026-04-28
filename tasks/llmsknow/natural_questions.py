# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Natural Questions (NQ) dataset wrapper for HalluLens.
Adapted from LLMsKnow benchmark suite.
"""

import hashlib
import json
import pandas as pd
import jsonlines
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import exp


def load_nq_data(split="test", n_samples=None, data_dir="external/LLMsKnow/data", split_seed=42):
    """
    Load Natural Questions dataset from LLMsKnow format.

    Since NQ comes as a single CSV file, a reproducible 80/20 train/test partition
    is created using split_seed (same approach as PopQA).

    Args:
        split: "test" (~20%, ~4155 samples) or "train" (~80%, ~16617 samples)
        n_samples: Number of samples to load (None for all in split)
        data_dir: Directory containing nq_wc_dataset.csv
        split_seed: Random seed for train/test partition (default: 42)

    Returns:
        List of dicts with 'question', 'answer', and 'context' fields
    """
    data_file = Path(data_dir) / "nq_wc_dataset.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Natural Questions data file not found: {data_file}\n"
            f"Expected location: {data_file}\n"
            f"This file should be available in the external/LLMsKnow/data/ directory."
        )

    print(f"Loading Natural Questions data from: {data_file}")

    # Load CSV data
    df = pd.read_csv(data_file)

    # Rename columns for consistency
    df = df.rename(columns={
        'Question': 'question',
        'Answer': 'answer',
        'Context': 'context'
    })

    # Create reproducible 80/20 train/test partition
    n_total = len(df)
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n_total)
    train_size = int(0.8 * n_total)
    if split == "train":
        selected_indices = sorted(indices[:train_size])
    elif split == "test":
        selected_indices = sorted(indices[train_size:])
    else:
        raise ValueError(f"Unknown split '{split}'. NQ supports: train, test")

    data = df.iloc[selected_indices].to_dict('records')

    # Subsample if requested
    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} samples from Natural Questions '{split}' partition "
          f"(split_seed={split_seed}, total={n_total})")
    return data


def compute_correctness_nq(model_answers, correct_answers):
    """
    Compute correctness for Natural Questions using substring matching.
    Similar to TriviaQA evaluation method.

    Args:
        model_answers: List of model-generated answers
        correct_answers: List of correct answers (strings)

    Returns:
        List of binary correctness labels (1 for correct, 0 for incorrect)
    """
    correctness = []

    for idx in range(len(model_answers)):
        model_answer = model_answers[idx]
        correct_answer = correct_answers[idx]

        # Check if correct answer appears in model response (case-insensitive)
        # NQ has single answers, not lists like TriviaQA
        if correct_answer.lower() in model_answer.lower():
            correctness.append(1)
        else:
            correctness.append(0)

    return correctness


class NaturalQuestionsEval:
    """Evaluation class for Natural Questions dataset."""

    def __init__(self, model_path, TASKNAME, generations_file_path=None, quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]
        self.TASKNAME = TASKNAME
        self.quick_debug_mode = quick_debug_mode

        # Set up paths
        self.output_path = f'output/{TASKNAME}/{self.model_name}'
        os.makedirs(self.output_path, exist_ok=True)

        # Load generations
        if generations_file_path:
            self.generations_file_path = generations_file_path
        else:
            self.generations_file_path = f'{self.output_path}/generation.jsonl'

        # Load test data
        self.test_df = self.load_test_data()

        # NQ uses automatic string matching evaluation (no LLM judge needed)

    def load_test_data(self):
        """Load generated responses and prepare for evaluation"""
        print(f"Loading test data from: {self.generations_file_path}")

        if not os.path.exists(self.generations_file_path):
            raise FileNotFoundError(f"Generations file not found: {self.generations_file_path}")

        # Load generations
        generations = []
        with open(self.generations_file_path, 'r') as f:
            for line in f:
                generations.append(json.loads(line))

        test_df = pd.DataFrame(generations)

        if self.quick_debug_mode:
            print("Quick debug mode: Using first 50 samples")
            test_df = test_df.head(50)

        print(f"Loaded {len(test_df)} test samples")
        return test_df

    def evaluate_correctness(self):
        """Evaluate correctness using NQ string matching"""
        print("Evaluating correctness using Natural Questions string matching...")

        # Extract model answers and correct answers from the test dataframe
        model_answers = self.test_df['generation'].tolist()
        correct_answers = self.test_df['answer'].tolist()

        # Use the NQ-specific correctness function
        binary_correctness = compute_correctness_nq(model_answers, correct_answers)

        print(f"Evaluated {len(binary_correctness)} samples")
        return binary_correctness

    def process_correctness_results(self, correctness_results):
        """Process binary correctness results into categorical labels"""
        processed_results = []

        for result in correctness_results:
            if result == 1:
                processed_results.append("CORRECT")
            else:
                processed_results.append("INCORRECT")

        return processed_results

    def compute_metrics(self, correctness_results):
        """Compute evaluation metrics"""
        total_samples = len(correctness_results)
        correct_count = sum(1 for result in correctness_results if result == "CORRECT")
        incorrect_count = sum(1 for result in correctness_results if result == "INCORRECT")

        accuracy = correct_count / total_samples if total_samples > 0 else 0
        error_rate = incorrect_count / total_samples if total_samples > 0 else 0

        return {
            "total_samples": total_samples,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy": accuracy,
            "error_rate": error_rate
        }

    def run_eval(self, eval_results_path=None, log_file=None, resume=True):
        """Run complete evaluation pipeline"""
        # Set up basic logging for NQ evaluation
        if log_file:
            client_log_file = log_file.replace(".log", "_eval_client.log") if log_file.endswith(".log") else f"{log_file}_eval_client.log"
        else:
            # Default to same directory as generations file
            generations_dir = os.path.dirname(self.generations_file_path) if self.generations_file_path != f'{self.output_path}/generation.jsonl' else self.output_path
            client_log_file = f"{generations_dir}/eval_client.log"

        print(f"Starting Natural Questions evaluation for model: {self.model_name}")
        print(f" Evaluation logs: {client_log_file}")
        print(f" Resume mode: {'enabled' if resume else 'disabled'}")

        # Evaluate correctness using NQ string matching
        binary_correctness = self.evaluate_correctness()
        correctness_results = self.process_correctness_results(binary_correctness)

        # Add correctness results to dataframe
        self.test_df['correctness_judgment'] = correctness_results
        self.test_df['is_correct'] = binary_correctness

        # Convert to PreciseWikiQA-compatible format
        # For NQ: no abstentions, incorrect answers are treated as hallucinations
        total_samples = len(binary_correctness)

        # Abstention arrays (all False for NQ)
        abstention_res = [False] * total_samples
        abstention_raw_gen = ['{"is_abstaining":false}'] * total_samples

        # Hallucination arrays (True = hallucinated/incorrect, False = correct)
        halu_test_res = [not bool(correct) for correct in binary_correctness]  # Invert: 0->True, 1->False
        halu_raw_gen = ["INCORRECT" if not bool(correct) else "CORRECT" for correct in binary_correctness]

        # Compute PreciseWikiQA-style metrics
        correct_count = sum(binary_correctness)
        incorrect_count = total_samples - correct_count

        # Since no abstentions in NQ, all samples are evaluated
        halu_rate = incorrect_count / total_samples if total_samples > 0 else 0
        refusal_rate = 0.0  # No refusals in NQ
        correct_rate = correct_count / total_samples if total_samples > 0 else 0

        # Prepare PreciseWikiQA-compatible results
        res = {
            'model': self.model_name,
            'halu_Rate': halu_rate,
            'refusal_Rate': refusal_rate,
            'correct_rate': correct_rate,
            'accurate_count': correct_count,
            'hallu_count': incorrect_count,
            'total_count': total_samples,
            'refusal_count': 0,
            'hallucination_evaluation': 'string_matching'
        }

        # Save evaluation results to JSON
        if eval_results_path is None:
            eval_results_path = f'{self.output_path}/eval_results.json'

        print(f" Saving evaluation results to: {eval_results_path}")

        with open(eval_results_path, 'w') as f:
            json.dump(res, f, indent=4)

        # Save detailed results to JSONL
        eval_raw_path = f'{self.output_path}/raw_eval_res.jsonl'
        print(f" Saving detailed results to: {eval_raw_path}")

        with jsonlines.open(eval_raw_path, mode='w') as writer:
            for idx in range(total_samples):
                writer.write({
                    'question': self.test_df.iloc[idx]['question'],
                    'answer': self.test_df.iloc[idx]['answer'],
                    'generation': self.test_df.iloc[idx]['generation'],
                    'correctness_judgment': correctness_results[idx],
                    'is_correct': binary_correctness[idx],
                    'is_hallucination': halu_test_res[idx],
                    'is_abstaining': abstention_res[idx]
                })

        # Print summary
        print("\n" + "="*60)
        print(" EVALUATION SUMMARY - Natural Questions")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Total Samples: {total_samples}")
        print(f"Correct: {correct_count} ({correct_rate:.2%})")
        print(f"Incorrect (Hallucinations): {incorrect_count} ({halu_rate:.2%})")
        print(f"Accuracy: {correct_rate:.2%}")
        print("="*60)

        return res


class NaturalQuestionsInference:
    """Inference class for Natural Questions dataset."""

    def __init__(self, model_path, TASKNAME, output_base_dir="output", data_dir="external/LLMsKnow/data",
                 inference_method="vllm", n_samples=None, generations_file_path=None,
                 split="test", split_seed=42):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.TASKNAME = TASKNAME
        self.output_base_dir = output_base_dir
        self.data_dir = data_dir
        self.inference_method = inference_method
        self.n_samples = n_samples
        self.split = split

        # Set up output paths
        self.task_output_dir = f"{output_base_dir}/{TASKNAME}/{self.model_name}"
        os.makedirs(self.task_output_dir, exist_ok=True)

        if generations_file_path:
            self.generations_file_path = generations_file_path
        else:
            self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'

        # Load dataset
        self.data = load_nq_data(split=split, n_samples=n_samples, data_dir=data_dir, split_seed=split_seed)

    def format_prompt(self, question):
        return f"Answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"

    def _build_prompts_df(self):
        rows = []
        for item in self.data:
            rows.append({
                'prompt': self.format_prompt(item['question']),
                'question': item['question'],
                'answer': item['answer'],
                'context': item.get('context', ''),
            })
        return pd.DataFrame(rows)

    def run_inference(self, max_tokens=64, temperature=0.0, logger_type="zarr",
                      activations_path=None, log_file=None, resume=True,
                      max_retries=3, base_delay=1.0):
        """Run inference on Natural Questions dataset using exp.run_exp."""
        from utils import exp

        prompts_df = self._build_prompts_df()
        print(f"Starting NQ inference | model={self.model_path} | split={self.split} | N={len(prompts_df)}")

        exp.run_exp(
            task="natural_questions",
            model_path=self.model_path,
            all_prompts=prompts_df,
            generations_file_path=self.generations_file_path,
            inference_method=self.inference_method,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            base_delay=base_delay,
            logger_type=logger_type,
            activations_path=activations_path,
            log_file_path=log_file,
            resume=resume,
        )
        print(f"Inference complete -> {self.generations_file_path}")

    def run_inference_batched(self, batch_size=8, max_tokens=64, temperature=0.0,
                              activations_path=None, resume=True):
        """Run batched inference using HFTransformersAdapter directly (no server needed)."""
        from activation_logging.model_adapter import HFTransformersAdapter
        from activation_logging.zarr_activations_logger import ZarrActivationsLogger
        from activation_logging.server import AsyncActivationWriter

        prompts_df = self._build_prompts_df()
        total = len(prompts_df)

        # Resume: skip already-processed prompts
        already_done = 0
        if resume and os.path.exists(self.generations_file_path):
            try:
                existing = pd.read_json(self.generations_file_path, lines=True)
                existing_prompts = set(existing["prompt"].tolist())
                mask = ~prompts_df["prompt"].isin(existing_prompts)
                prompts_df = prompts_df[mask].reset_index(drop=True)
                already_done = total - len(prompts_df)
                if len(prompts_df) == 0:
                    print(f"All {total} prompts already processed -- nothing to do.")
                    return
                print(f"Resuming: {already_done}/{total} done, {len(prompts_df)} remaining")
            except Exception as e:
                print(f"Warning: could not load existing generations ({e}), starting fresh.")
                already_done = 0

        print(f"Starting batched NQ inference | model={self.model_path} | "
              f"split={self.split} | remaining={len(prompts_df)} | batch_size={batch_size}")

        adapter = HFTransformersAdapter(
            model_name=self.model_path,
            target_layers="all",
            sequence_mode="all",
            enable_logprobs=True,
        )

        writer = None
        zarr_logger = None
        if activations_path:
            zarr_logger = ZarrActivationsLogger(
                zarr_path=activations_path, read_only=False,
                expected_samples=total,
                activation_chunk_shape=(batch_size, 1, max_tokens, -1),
                response_max_tokens=max_tokens,
            )
            writer = AsyncActivationWriter(zarr_logger)

        file_mode = "a" if already_done > 0 else "w"
        prompts = prompts_df["prompt"].tolist()
        n_batches = (len(prompts) + batch_size - 1) // batch_size

        try:
            with open(self.generations_file_path, file_mode, encoding="utf-8") as f:
                pbar = tqdm(total=len(prompts), desc="Batched NQ inference")
                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(prompts))
                    batch_prompts = prompts[start:end]

                    results = adapter.infer_batch(
                        batch_prompts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    for i, result in enumerate(results):
                        row_idx = start + i
                        record = prompts_df.iloc[row_idx].to_dict()
                        record["generation"] = result.response_text
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                    if writer is not None:
                        batch_entries = []
                        for result in results:
                            if result.activations is None:
                                continue
                            prompt_key = hashlib.sha256(
                                result.prompt.encode("utf-8")
                            ).hexdigest()
                            log_entry = {
                                "all_layers_activations": result.activations,
                                "input_length": result.input_length,
                                "prompt": result.prompt,
                                "response": result.response_text,
                                "model": self.model_path,
                                "prompt_hash": prompt_key,
                                "trim_position": result.trim_position,
                            }
                            if result.logprobs is not None:
                                log_entry.update(result.logprobs)
                            batch_entries.append((prompt_key, log_entry))
                        writer.enqueue_batch(batch_entries)

                    pbar.update(len(batch_prompts))
                pbar.close()
        finally:
            if writer is not None:
                print("Draining activation writer...")
                writer.shutdown(timeout=60.0)
            if zarr_logger is not None:
                zarr_logger.close()

        print(f"Batched inference complete -> {self.generations_file_path}")
        if activations_path:
            print(f"Activations saved -> {activations_path}")


def run_step(step, model, data_dir="external/LLMsKnow/data", output_dir="output",
             inference_method="vllm", max_tokens=64, temperature=0.0, N=None,
             generations_file_path=None, eval_results_path=None, log_file=None,
             quick_debug_mode=False, split="test", split_seed=42,
             logger_type="zarr", activations_path=None, resume=True, batch_size=32):
    """Run a single step of the Natural Questions task. Callable from Python directly."""
    TASKNAME = "natural_questions_train" if split == "train" else "natural_questions"

    if step == "inference":
        inferencer = NaturalQuestionsInference(
            model_path=model,
            TASKNAME=TASKNAME,
            output_base_dir=output_dir,
            data_dir=data_dir,
            inference_method=inference_method,
            n_samples=N,
            generations_file_path=generations_file_path,
            split=split,
            split_seed=split_seed)
        if batch_size:
            inferencer.run_inference_batched(
                batch_size=batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                activations_path=activations_path,
                resume=resume,
            )
        else:
            inferencer.run_inference(
                max_tokens=max_tokens,
                temperature=temperature,
                logger_type=logger_type,
                activations_path=activations_path,
                log_file=log_file,
                resume=resume,
            )

    elif step == "eval":
        evaluator = NaturalQuestionsEval(
            model_path=model,
            TASKNAME=TASKNAME,
            generations_file_path=generations_file_path,
            quick_debug_mode=quick_debug_mode)
        evaluator.run_eval(eval_results_path=eval_results_path, log_file=log_file)

    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Natural Questions (NQ) inference and evaluation')

    # Task selection
    parser.add_argument('--do_inference', action='store_true', help='Run inference on NQ dataset')
    parser.add_argument('--do_eval', action='store_true', help='Run evaluation on generated answers')

    # Model and data settings
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--data_dir', type=str, default='external/LLMsKnow/data', help='Directory containing nq_wc_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output directory')

    # Inference settings
    parser.add_argument('--inference_method', type=str, default='vllm', choices=['vllm'], help='Inference method')
    parser.add_argument('--max_tokens', type=int, default=64, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--N', type=int, default=None, help='Number of samples to process (None for all)')

    # File paths
    parser.add_argument('--generations_file_path', type=str, default=None, help='Path to generations file')
    parser.add_argument('--eval_results_path', type=str, default=None, help='Path to save evaluation results')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')

    # Debug mode
    parser.add_argument('--quick_debug_mode', action='store_true', help='Use only first 50 samples for quick testing')

    args = parser.parse_args()

    if args.do_inference:
        run_step("inference", args.model, data_dir=args.data_dir, output_dir=args.output_dir,
                 inference_method=args.inference_method, max_tokens=args.max_tokens,
                 temperature=args.temperature, N=args.N,
                 generations_file_path=args.generations_file_path,
                 log_file=args.log_file, quick_debug_mode=args.quick_debug_mode)
    if args.do_eval:
        run_step("eval", args.model, generations_file_path=args.generations_file_path,
                 eval_results_path=args.eval_results_path, log_file=args.log_file,
                 quick_debug_mode=args.quick_debug_mode)
    if not args.do_inference and not args.do_eval:
        print("No action specified. Use --do_inference and/or --do_eval")
        parser.print_help()
