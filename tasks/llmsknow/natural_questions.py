# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Natural Questions (NQ) dataset wrapper for HalluLens.
Adapted from LLMsKnow benchmark suite.
"""

import json
import pandas as pd
import jsonlines
import os
import argparse
from pathlib import Path

from utils import exp


def load_nq_data(split="test", n_samples=None, data_dir="external/LLMsKnow/data"):
    """
    Load Natural Questions dataset from LLMsKnow format.

    Args:
        split: Dataset split (only "test" available, NQ from LLMsKnow uses single file)
        n_samples: Number of samples to load (None for all)
        data_dir: Directory containing nq_wc_dataset.csv

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

    # Convert to list of dicts
    data = df.to_dict('records')

    # Subsample if requested
    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} samples from Natural Questions")
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
        print(f"📝 Evaluation logs: {client_log_file}")
        print(f"🔄 Resume mode: {'enabled' if resume else 'disabled'}")

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

        print(f"💾 Saving evaluation results to: {eval_results_path}")

        with open(eval_results_path, 'w') as f:
            json.dump(res, f, indent=4)

        # Save detailed results to JSONL
        eval_raw_path = f'{self.output_path}/raw_eval_res.jsonl'
        print(f"💾 Saving detailed results to: {eval_raw_path}")

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
        print("📊 EVALUATION SUMMARY - Natural Questions")
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
                 inference_method="vllm", n_samples=None, generations_file_path=None):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.TASKNAME = TASKNAME
        self.output_base_dir = output_base_dir
        self.data_dir = data_dir
        self.inference_method = inference_method
        self.n_samples = n_samples

        # Set up output paths
        self.task_output_dir = f"{output_base_dir}/{TASKNAME}/{self.model_name}"
        os.makedirs(self.task_output_dir, exist_ok=True)

        if generations_file_path:
            self.generations_file_path = generations_file_path
        else:
            self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'

        # Load dataset
        self.data = load_nq_data(split="test", n_samples=n_samples, data_dir=data_dir)

    def format_prompt(self, question):
        """Format a Natural Questions question into a prompt."""
        # Simple prompt format similar to TriviaQA
        return f"Answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"

    def run_inference(self, max_tokens=64, temperature=0.0):
        """Run inference on Natural Questions dataset."""
        from utils import lm

        print(f"Starting Natural Questions inference with {self.inference_method}")
        print(f"Processing {len(self.data)} samples")
        print(f"Saving to: {self.generations_file_path}")

        # Check if generations already exist (for resume capability)
        existing_questions = set()
        if os.path.exists(self.generations_file_path):
            print(f"Found existing generations file. Checking for completed questions...")
            with open(self.generations_file_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        existing_questions.add(entry.get('question', ''))
                    except:
                        pass
            print(f"Found {len(existing_questions)} existing generations")

        # Open file in append mode
        with jsonlines.open(self.generations_file_path, mode='a') as writer:
            for idx, item in enumerate(self.data):
                question = item['question']
                answer = item['answer']
                context = item.get('context', '')

                # Skip if already processed
                if question in existing_questions:
                    print(f"[{idx+1}/{len(self.data)}] Skipping (already processed): {question[:60]}...")
                    continue

                # Format prompt
                prompt = self.format_prompt(question)

                # Generate response
                print(f"[{idx+1}/{len(self.data)}] Generating for: {question[:60]}...")

                try:
                    if self.inference_method == "vllm":
                        response = lm.generate(
                            prompt,
                            self.model_path,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    else:
                        raise ValueError(f"Unsupported inference method: {self.inference_method}")

                    # Save to file
                    result = {
                        'question': question,
                        'answer': answer,
                        'context': context,
                        'prompt': prompt,
                        'generation': response,
                        'model': self.model_path
                    }
                    writer.write(result)

                except Exception as e:
                    print(f"Error generating for question {idx}: {e}")
                    # Write error entry
                    result = {
                        'question': question,
                        'answer': answer,
                        'context': context,
                        'prompt': prompt,
                        'generation': f"ERROR: {str(e)}",
                        'model': self.model_path
                    }
                    writer.write(result)

        print(f"\n✅ Inference complete! Results saved to: {self.generations_file_path}")


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

    # Generate task name
    TASKNAME = "natural_questions"

    if args.do_inference:
        print("="*60)
        print("NATURAL QUESTIONS - INFERENCE")
        print("="*60)

        inferencer = NaturalQuestionsInference(
            model_path=args.model,
            TASKNAME=TASKNAME,
            output_base_dir=args.output_dir,
            data_dir=args.data_dir,
            inference_method=args.inference_method,
            n_samples=args.N,
            generations_file_path=args.generations_file_path
        )

        inferencer.run_inference(
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

    if args.do_eval:
        print("="*60)
        print("NATURAL QUESTIONS - EVALUATION")
        print("="*60)

        evaluator = NaturalQuestionsEval(
            model_path=args.model,
            TASKNAME=TASKNAME,
            generations_file_path=args.generations_file_path,
            quick_debug_mode=args.quick_debug_mode
        )

        evaluator.run_eval(
            eval_results_path=args.eval_results_path,
            log_file=args.log_file
        )

    if not args.do_inference and not args.do_eval:
        print("No action specified. Use --do_inference and/or --do_eval")
        parser.print_help()
