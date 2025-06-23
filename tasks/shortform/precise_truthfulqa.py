# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Scaffold for TruthfulQA response generation and ground truth determination.
References: hal_det_llama.py, precise_wikiqa.py
"""

import argparse
import os
import jsonlines
from tqdm import tqdm

class TruthfulQAGenerator:
    """
    Handles generation of model responses for the TruthfulQA dataset.
    Allows tuning of LLM arguments such as temperature and max_tokens.
    """
    def __init__(self, model_path, output_dir, num_samples=1, most_likely=True, temperature=1.0, max_tokens=64):
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.most_likely = most_likely
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_name = "truthful_qa"
        self.split = "validation"
        self.generations_file_path = os.path.join(self.output_dir, "generation.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_truthfulqa(self):
        """
        Loads the TruthfulQA validation set using the datasets library.
        """
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, 'generation')[self.split]
        return dataset

    def _call_chatgpt_api(self, prompt):
        """
        Calls the ChatGPT API to generate a response for the given prompt.
        Passes temperature and max_tokens as arguments.
        """
        from utils import lm
        return lm.generate(prompt, self.model_path, temperature=self.temperature, max_tokens=self.max_tokens)

    def generate_responses(self):
        """
        Generate responses for each question in the TruthfulQA dataset and save to JSONL.
        """
        dataset = self._load_truthfulqa()
        results = []
        for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Generating TruthfulQA responses"):
            question = item["question"]
            prompt = f"Answer the question concisely. Q: {question} A:"
            generations = []
            for _ in range(self.num_samples):
                response = self._call_chatgpt_api(prompt)
                generations.append(response)
            result = {
                "question": question,
                "generations": generations,
                "best_answer": item.get("best_answer", ""),
                "correct_answers": item.get("correct_answers", []),
                "id": item.get("id", i)
            }
            results.append(result)
            with jsonlines.open(self.generations_file_path, mode='a') as writer:
                writer.write(result)
        print(f"Saved generations to {self.generations_file_path}")

class TruthfulQAGroundTruth:
    """
    Handles ground truth generation for model responses using ROUGE or BleuRT.
    """
    def __init__(self, model_path, output_dir, use_rouge=False):
        pass

    def compute_ground_truth(self):
        """
        Compute ground truth scores for generated responses.
        """
        pass

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Generation and Ground Truth Scaffold")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per question')
    parser.add_argument('--most_likely', action='store_true', help='Generate most likely answers')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=64, help='Maximum number of tokens to generate')
    parser.add_argument('--use_rouge', action='store_true', help='Use ROUGE for ground truth scoring')
    parser.add_argument('--do_generate', action='store_true', help='Run response generation')
    parser.add_argument('--do_ground_truth', action='store_true', help='Run ground truth computation')
    args = parser.parse_args()

    if args.do_generate:
        generator = TruthfulQAGenerator(
            model_path=args.model_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            most_likely=args.most_likely,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        generator.generate_responses()

    if args.do_ground_truth:
        gt = TruthfulQAGroundTruth(
            model_path=args.model_path,
            output_dir=args.output_dir,
            use_rouge=args.use_rouge
        )
        gt.compute_ground_truth()

if __name__ == '__main__':
    main() 