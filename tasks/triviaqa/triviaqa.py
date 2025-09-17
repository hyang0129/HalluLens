# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas as pd
import jsonlines
import os
import argparse
import tarfile
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import exp


def download_triviaqa_unfiltered(data_dir="data"):
    """
    Download and extract TriviaQA unfiltered dataset.

    Args:
        data_dir: Directory to download and extract data to
    """
    triviaqa_url = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"
    download_dir = Path(data_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    tar_path = download_dir / "triviaqa-unfiltered.tar.gz"
    extract_dir = download_dir / "triviaqa-unfiltered"

    # Check if already extracted
    dev_file = extract_dir / "unfiltered-web-dev.json"
    train_file = extract_dir / "unfiltered-web-train.json"

    if dev_file.exists() and train_file.exists():
        print(f"TriviaQA unfiltered data already exists at {extract_dir}")
        return str(extract_dir)

    print(f"Downloading TriviaQA unfiltered dataset from {triviaqa_url}")
    print(f"This may take a few minutes...")

    try:
        # Download the tar.gz file
        urllib.request.urlretrieve(triviaqa_url, tar_path)
        print(f"Downloaded to {tar_path}")

        # Extract the tar.gz file
        print(f"Extracting to {extract_dir}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=download_dir)

        # Clean up the tar file
        tar_path.unlink()
        print(f"Extraction complete. Data available at {extract_dir}")

        # Verify the expected files exist
        if not (dev_file.exists() and train_file.exists()):
            raise FileNotFoundError(
                f"Expected files not found after extraction:\n"
                f"  - {dev_file}\n"
                f"  - {train_file}"
            )

        return str(extract_dir)

    except Exception as e:
        print(f"Error downloading TriviaQA data: {e}")
        if tar_path.exists():
            tar_path.unlink()  # Clean up partial download
        raise


def compute_correctness_triviaqa(model_answers, correct_answers_list):
    """
    Compute correctness for TriviaQA using substring matching.
    Based on the evaluation method from LLMsKnow repository.

    Args:
        model_answers: List of model-generated answers
        correct_answers_list: List of lists containing acceptable answer aliases

    Returns:
        List of binary correctness labels (1 for correct, 0 for incorrect)
    """
    correctness = []

    for idx in range(len(model_answers)):
        model_answer = model_answers[idx]
        correct = 0

        # Handle both string and list formats for correct answers
        if isinstance(correct_answers_list[idx], str):
            try:
                # Try to parse as list if it's a string representation
                labels_ = eval(correct_answers_list[idx])
            except:
                # If parsing fails, treat as single answer
                labels_ = [correct_answers_list[idx]]
        else:
            labels_ = correct_answers_list[idx]

        # Check if any acceptable answer appears in model response (case-insensitive)
        for ans in labels_:
            if ans.lower() in model_answer.lower():
                correct = 1
                break

        correctness.append(correct)

    return correctness


class TriviaQAEval:
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
        
        # TriviaQA uses automatic string matching evaluation (no LLM judge needed)
        
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
    
    def run_eval(self, eval_results_path=None):
        """Run complete evaluation pipeline"""
        print(f"Starting TriviaQA evaluation for model: {self.model_name}")

        # Evaluate correctness using TriviaQA string matching
        binary_correctness = self.evaluate_correctness()
        correctness_results = self.process_correctness_results(binary_correctness)

        # Compute metrics
        metrics = self.compute_metrics(correctness_results)

        # Add correctness results to dataframe
        self.test_df['correctness_judgment'] = correctness_results
        self.test_df['is_correct'] = binary_correctness

        # Prepare final results
        res = {
            "model": self.model_name,
            "task": self.TASKNAME,
            "evaluation_method": "triviaqa_string_matching",
            "metrics": metrics,
            "sample_results": self.test_df.to_dict('records')
        }

        # Save results
        if eval_results_path:
            res_path = eval_results_path
        else:
            res_path = f'{self.output_path}/eval_results.json'

        with open(res_path, 'w') as f:
            json.dump(res, f, indent=4)

        # Print results
        print("=" * 80)
        print(f" TriviaQA Evaluation Results for: <<{self.model_name}>>")
        print("=" * 80)
        print(f"  >> Results saved to: {res_path}")
        print("-" * 80)
        print(f"  Evaluation Method: TriviaQA String Matching")
        print("-" * 80)
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Error Rate: {metrics['error_rate']:.3f}")
        print("-" * 80)


def load_triviaqa_data(dataset_variant="unfiltered", split="dev", n_samples=1000, data_dir="data", auto_download=True):
    """
    Load TriviaQA dataset from JSON files.

    Args:
        dataset_variant: "filtered" or "unfiltered"
        split: "train" or "dev"
        n_samples: number of samples to load (will be subsampled if dataset is larger)
        data_dir: directory containing TriviaQA data files
        auto_download: whether to automatically download data if not found

    Returns:
        dict with 'questions', 'correct_answers', 'question_ids'
    """


    print(f"Loading TriviaQA {dataset_variant} {split} split with {n_samples} samples")

    # Determine file path based on variant and split
    if dataset_variant == "unfiltered":
        if split == "dev":
            file_path = f'{data_dir}/triviaqa-unfiltered/unfiltered-web-dev.json'
        elif split == "train":
            file_path = f'{data_dir}/triviaqa-unfiltered/unfiltered-web-train.json'
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'")
    else:
        raise NotImplementedError(f"Dataset variant '{dataset_variant}' not implemented yet")

    # Check if file exists, download if needed
    if not os.path.exists(file_path):
        if auto_download and dataset_variant == "unfiltered":
            print(f"TriviaQA data not found at {file_path}")
            print("Attempting automatic download...")
            try:
                download_triviaqa_unfiltered(data_dir)
                print("Download completed successfully!")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to automatically download TriviaQA data: {e}\n"
                    f"Please manually download from https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz\n"
                    f"and extract to {data_dir}/triviaqa-unfiltered/"
                )
        else:
            raise FileNotFoundError(
                f"TriviaQA data file not found: {file_path}\n"
                f"Please download TriviaQA data from https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz\n"
                f"and extract to {data_dir}/triviaqa-unfiltered/"
            )

    # Load JSON data
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
        data = data['Data']  # Extract the 'Data' field

    print(f"Loaded {len(data)} total samples from TriviaQA")

    # Subsample if requested (following LLMsKnow approach)
    if n_samples < len(data):
        print(f"Subsampling {n_samples} samples from {len(data)} total samples")
        data, _ = train_test_split(data, train_size=n_samples, random_state=42)

    # Extract questions, answers, and IDs
    questions = []
    correct_answers = []
    question_ids = []

    for idx, item in enumerate(data):
        questions.append(item['Question'])
        correct_answers.append(item['Answer']['Aliases'])  # List of acceptable answers

        # Use QuestionId if available, otherwise create one
        if 'QuestionId' in item:
            question_ids.append(item['QuestionId'])
        else:
            question_ids.append(f"triviaqa_{split}_{idx}")

    print(f"Successfully loaded {len(questions)} TriviaQA samples")
    print(f"Sample question: {questions[0]}")
    print(f"Sample answers: {correct_answers[0]}")

    return {
        'questions': questions,
        'correct_answers': correct_answers,
        'question_ids': question_ids
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_inference', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='model to use for generation')
    parser.add_argument('--inference_method', type=str, default='vllm', help='check and customize util/lm.py')
    parser.add_argument('--max_inference_tokens', type=int, default=256)
    parser.add_argument('--inf_batch_size', type=int, default=64)
    
    parser.add_argument('--dataset_variant', type=str, default='unfiltered', help='TriviaQA variant: filtered/unfiltered')
    parser.add_argument('--split', type=str, default='dev', help='TriviaQA split: train/dev')
    parser.add_argument('--data_dir', type=str, default='', help='directory containing TriviaQA data files')
    parser.add_argument('--auto_download', action='store_true', default=True, help='automatically download TriviaQA data if not found')
    parser.add_argument('--no_auto_download', dest='auto_download', action='store_false', help='disable automatic download')
    parser.add_argument('--generations_file_path', type=str, default='', help='path to save model generations')
    parser.add_argument('--eval_results_path', type=str, default='', help='path to save evaluation results')
    parser.add_argument('--N', type=int, default=1000, help='number of samples to evaluate')
    parser.add_argument('--quick_debug_mode', action='store_true', default=False, help='if True, only evaluate first 50 questions')
    
    args = parser.parse_args()
    
    # Set up task name and paths
    base_path = os.path.dirname(os.path.abspath(__name__))
    TASKNAME = f'triviaqa_{args.dataset_variant}_{args.split}'
    model_name = args.model.split("/")[-1]
    
    print(f"Running {TASKNAME} with model {model_name}")
    
    if args.do_inference:
        # Load TriviaQA data
        data_dir = args.data_dir if args.data_dir else f"{base_path}/data"
        triviaqa_data = load_triviaqa_data(
            dataset_variant=args.dataset_variant,
            split=args.split,
            n_samples=args.N,
            data_dir=data_dir,
            auto_download=args.auto_download
        )

        # Prepare prompts for inference (following TriviaQA format from LLMsKnow)
        prompts_df = pd.DataFrame({
            'prompt': [f"Q: {q}\nA:" for q in triviaqa_data['questions']],  # Simple Q&A format
            'question': triviaqa_data['questions'],
            'correct_answers': triviaqa_data['correct_answers'],
            'question_id': triviaqa_data['question_ids']
        })
        
        print(f"Starting Inference for [{args.model}], Testset_N: {prompts_df.shape}")
        exp.run_exp(
            task=TASKNAME,
            model_path=args.model,
            all_prompts=prompts_df,
            generations_file_path=args.generations_file_path if args.generations_file_path else None,
            inference_method=args.inference_method,
            max_tokens=args.max_inference_tokens,
            max_workers=1
        )
        print('Inference completed')
    
    if args.do_eval:
        print(f"Starting Evaluation for {args.model}")
        TriviaQAEval(
            model_path=args.model,
            TASKNAME=TASKNAME,
            generations_file_path=args.generations_file_path,
            quick_debug_mode=args.quick_debug_mode
        ).run_eval(eval_results_path=args.eval_results_path)
        print(f'{TASKNAME} Evaluation completed')
