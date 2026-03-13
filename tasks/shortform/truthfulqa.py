# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TruthfulQA task for HalluLens.

Dataset: https://huggingface.co/datasets/truthful_qa (generation config, validation split)
817 questions targeting misconception-driven hallucinations across 38 categories.

⚠️  EVAL-ONLY DATASET — DO NOT USE FOR CONTRASTIVE TRAINING
    TruthfulQA has only 817 total samples (~400 per class after label split), which is
    far too small for contrastive representation learning to converge. Using it for
    training will produce unreliable embeddings and misleading AUROC numbers.

    Intended use: cross-dataset generalization testing.
    Workflow: train contrastive model on PreciseWikiQA or TriviaQA, then run inference +
    eval here to measure how well the learned representations transfer to a different
    hallucination regime (misconception-driven vs. factual-recall).

Pipeline:
  inference  → generate model responses for each TruthfulQA question
  eval       → score responses against correct_answers; produce ActivationParser-compatible JSON

No "generate" step: TruthfulQA is a fixed dataset loaded from HuggingFace.

Evaluation method:
  Correctness = case-insensitive substring match of any correct_answer in the model response.
  Hallucination = response does not contain any correct answer (binary label).

Output eval_results.json schema (ActivationParser-compatible):
  {
    "evaluator_abstantion": "truthfulqa_no_abstain",
    "evaluator_hallucination": "truthfulqa_substring_match",
    "abstantion": [false, ...],          # always False — TruthfulQA has no abstain labels
    "halu_test_res": [true, false, ...], # True = hallucinated (incorrect)
    "total_count": N,
    "accurate_count": K,
    "hallu_count": N-K,
    "refusal_count": 0,
    "correct_rate": float,
    "halu_Rate": float,
    "refusal_Rate": 0.0
  }
"""

import json
import os
import argparse
from pathlib import Path

import pandas as pd
import jsonlines

from utils import exp


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_truthfulqa(n_samples=None):
    """
    Load TruthfulQA validation split from HuggingFace.

    Returns a list of dicts with keys:
        question, best_answer, correct_answers, incorrect_answers, category, source
    """
    from datasets import load_dataset

    print("Loading TruthfulQA from HuggingFace (truthful_qa / generation / validation)...")
    dataset = load_dataset("truthful_qa", "generation", trust_remote_code=True)["validation"]

    data = []
    for item in dataset:
        data.append({
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item.get("incorrect_answers", []),
            "category": item.get("category", ""),
            "source": item.get("source", ""),
        })

    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} TruthfulQA samples")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def is_correct_substring(generation: str, correct_answers: list[str]) -> bool:
    """Return True if any correct answer appears (case-insensitive) in the generation."""
    gen_lower = generation.lower().strip()
    for ans in correct_answers:
        if ans.lower().strip() in gen_lower:
            return True
    return False


def compute_correctness(generations: list[str], correct_answers_list: list[list[str]]) -> list[bool]:
    """Return per-sample correctness (True = correct)."""
    return [
        is_correct_substring(gen, correct_answers)
        for gen, correct_answers in zip(generations, correct_answers_list)
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQ: {question}\nA:"


class TruthfulQAInference:
    """Run model inference on TruthfulQA questions."""

    def __init__(self, model_path, output_base_dir="output", n_samples=None,
                 generations_file_path=None):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.n_samples = n_samples

        self.task_output_dir = f"{output_base_dir}/truthfulqa/{self.model_name}"
        os.makedirs(self.task_output_dir, exist_ok=True)

        self.generations_file_path = (
            generations_file_path or f"{self.task_output_dir}/generation.jsonl"
        )

        self.data = load_truthfulqa(n_samples=n_samples)

    def run_inference(self, inference_method="vllm", max_tokens=128, temperature=0.0,
                      logger_type="lmdb", activations_path=None, log_file=None,
                      resume=True, max_retries=3, base_delay=1.0):
        """Generate responses and log activations via exp.run_exp()."""
        # Build prompts DataFrame (exp.run_exp expects a DataFrame with 'prompt' column)
        rows = []
        for item in self.data:
            rows.append({
                "prompt": format_prompt(item["question"]),
                "question": item["question"],
                "best_answer": item["best_answer"],
                # Store correct_answers as JSON string so it survives JSONL round-trip
                "correct_answers": json.dumps(item["correct_answers"]),
                "incorrect_answers": json.dumps(item.get("incorrect_answers", [])),
                "category": item.get("category", ""),
                "answer": item["best_answer"],  # canonical single answer for compatibility
            })
        prompts_df = pd.DataFrame(rows)

        print(f"Starting TruthfulQA inference | model={self.model_path} | N={len(prompts_df)}")

        exp.run_exp(
            task="truthfulqa",
            model_path=self.model_path,
            all_prompts=prompts_df,
            generations_file_path=self.generations_file_path,
            inference_method=inference_method,
            max_tokens=max_tokens,
            max_workers=1,
            max_retries=max_retries,
            base_delay=base_delay,
            logger_type=logger_type,
            activations_path=activations_path,
            log_file_path=log_file,
            resume=resume,
        )
        print(f"Inference complete → {self.generations_file_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TruthfulQAEval:
    """Score model generations against TruthfulQA correct_answers."""

    def __init__(self, model_path, generations_file_path=None, output_base_dir="output",
                 quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]

        if generations_file_path:
            self.generations_file_path = generations_file_path
            self.output_path = os.path.dirname(generations_file_path)
        else:
            self.output_path = f"{output_base_dir}/truthfulqa/{self.model_name}"
            self.generations_file_path = f"{self.output_path}/generation.jsonl"

        if not os.path.exists(self.generations_file_path):
            raise FileNotFoundError(
                f"Generations file not found: {self.generations_file_path}\n"
                "Run inference first."
            )

        self.df = pd.read_json(self.generations_file_path, lines=True)

        if quick_debug_mode:
            print("Quick debug mode: using first 50 samples")
            self.df = self.df.head(50)

        print(f"Loaded {len(self.df)} generations for evaluation")

    def _parse_correct_answers(self, row) -> list[str]:
        """Extract correct_answers from row; handles JSON-string or list."""
        val = row.get("correct_answers", row.get("answer", ""))
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Fallback: treat as single answer string
            return [val] if val else []
        return []

    def run_eval(self, eval_results_path=None, log_file=None):
        """Score generations and write ActivationParser-compatible eval JSON."""
        generations = self.df["generation"].tolist()
        correct_answers_list = [self._parse_correct_answers(row) for _, row in self.df.iterrows()]

        correctness = compute_correctness(generations, correct_answers_list)  # True = correct

        n = len(correctness)
        correct_count = sum(correctness)
        incorrect_count = n - correct_count

        # ActivationParser requires: halu_test_res (True=hallucinated), abstantion (all False)
        halu_test_res = [not c for c in correctness]
        abstantion = [False] * n

        res = {
            "evaluator_abstantion": "truthfulqa_no_abstain",
            "evaluator_hallucination": "truthfulqa_substring_match",
            "abstantion": abstantion,
            "halu_test_res": halu_test_res,
            "total_count": n,
            "accurate_count": correct_count,
            "hallu_count": incorrect_count,
            "refusal_count": 0,
            "correct_rate": correct_count / max(1, n),
            "halu_Rate": incorrect_count / max(1, n),
            "refusal_Rate": 0.0,
            "prompt": self.df["prompt"].tolist() if "prompt" in self.df.columns else [""] * n,
            "generation": generations,
            "answer": self.df["answer"].tolist() if "answer" in self.df.columns else [""] * n,
        }

        # Save aggregate eval_results.json
        out_path = eval_results_path or f"{self.output_path}/eval_results.json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Eval results saved → {out_path}")

        # Save per-sample raw_eval_res.jsonl
        raw_path = f"{self.output_path}/raw_eval_res.jsonl"
        with jsonlines.open(raw_path, mode="w") as writer:
            for i, (_, row) in enumerate(self.df.iterrows()):
                writer.write({
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "correct_answers": correct_answers_list[i],
                    "generation": generations[i],
                    "category": row.get("category", ""),
                    "is_correct": correctness[i],
                    "is_hallucination": halu_test_res[i],
                    "is_abstaining": False,
                })
        print(f"Raw eval results saved → {raw_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY — TruthfulQA")
        print("=" * 60)
        print(f"Model          : {self.model_name}")
        print(f"Total samples  : {n}")
        print(f"Correct        : {correct_count}  ({correct_count/max(1,n):.1%})")
        print(f"Hallucinated   : {incorrect_count}  ({incorrect_count/max(1,n):.1%})")
        print("=" * 60)

        return res


# ---------------------------------------------------------------------------
# Unified run_step interface (matches run_with_server.py contract)
# ---------------------------------------------------------------------------

def run_step(step, model, output_dir="output", inference_method="vllm",
             max_tokens=128, temperature=0.0, N=None,
             generations_file_path=None, eval_results_path=None, log_file=None,
             logger_type="lmdb", activations_path=None,
             quick_debug_mode=False, resume=True, max_retries=3, base_delay=1.0):
    """Run a single step of the TruthfulQA task. Callable from Python directly."""

    if step == "inference":
        runner = TruthfulQAInference(
            model_path=model,
            output_base_dir=output_dir,
            n_samples=N,
            generations_file_path=generations_file_path,
        )
        runner.run_inference(
            inference_method=inference_method,
            max_tokens=max_tokens,
            temperature=temperature,
            logger_type=logger_type,
            activations_path=activations_path,
            log_file=log_file,
            resume=resume,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    elif step == "eval":
        evaluator = TruthfulQAEval(
            model_path=model,
            generations_file_path=generations_file_path,
            output_base_dir=output_dir,
            quick_debug_mode=quick_debug_mode,
        )
        evaluator.run_eval(eval_results_path=eval_results_path, log_file=log_file)

    else:
        raise ValueError(f"Unknown step '{step}'. TruthfulQA supports: inference, eval")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthfulQA inference and evaluation")

    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--N", type=int, default=None, help="Number of samples (None = all 817)")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--inference_method", type=str, default="vllm")
    parser.add_argument("--logger_type", type=str, default="lmdb", choices=["lmdb", "json", "zarr"])
    parser.add_argument("--activations_path", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--generations_file_path", type=str, default=None)
    parser.add_argument("--eval_results_path", type=str, default=None)
    parser.add_argument("--quick_debug_mode", action="store_true")
    parser.add_argument("--no_resume", action="store_true")

    args = parser.parse_args()

    if args.do_inference:
        run_step("inference", args.model,
                 output_dir=args.output_dir,
                 inference_method=args.inference_method,
                 max_tokens=args.max_tokens,
                 temperature=args.temperature,
                 N=args.N,
                 generations_file_path=args.generations_file_path,
                 log_file=args.log_file,
                 logger_type=args.logger_type,
                 activations_path=args.activations_path,
                 resume=not args.no_resume)

    if args.do_eval:
        run_step("eval", args.model,
                 output_dir=args.output_dir,
                 generations_file_path=args.generations_file_path,
                 eval_results_path=args.eval_results_path,
                 quick_debug_mode=args.quick_debug_mode)

    if not args.do_inference and not args.do_eval:
        print("No action specified. Use --do_inference and/or --do_eval")
        parser.print_help()
