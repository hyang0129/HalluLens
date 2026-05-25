"""
TriviaQA task for HalluLens (closed-book, rc.nocontext config).

Dataset: https://huggingface.co/datasets/mandarjoshi/trivia_qa (rc.nocontext)
  - train:      138,384 questions (capped via n_samples in practice)
  - validation:  17,944 questions  (used as "test" split in our pipeline)
  - test:        17,210 questions  (no gold answers — not used)

Evaluation mode: CLOSED-BOOK (no Wikipedia passage or web snippets provided).
The model must rely on parametric memory; incorrect answers are treated as hallucinations.

Why TriviaQA?
  Quiz-style factual recall spanning diverse domains. Like PopQA, hallucination is
  driven by knowledge gaps rather than reasoning failures: the model either knows the
  trivia fact or confabulates. Short entity-level answers and a natural difficulty
  gradient (quiz question difficulty tiers) create tight, separable activation clusters —
  the same structural property that makes contrastive learning dominate on PopQA.

Evaluation method:
  Correctness = case-insensitive substring match of ANY alias in answer['aliases'].
  Hallucination = response contains none of the valid aliases.

Output eval_results.json schema (ActivationParser-compatible):
  {
    "evaluator_abstantion": "triviaqa_no_abstain",
    "evaluator_hallucination": "triviaqa_alias_match",
    "abstantion": [false, ...],
    "halu_test_res": [true, false, ...],   # True = hallucinated
    "total_count": N, "accurate_count": K, "hallu_count": N-K,
    "refusal_count": 0,
    "correct_rate": float, "halu_Rate": float, "refusal_Rate": 0.0
  }
"""

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_triviaqa_data(split="validation", n_samples=None):
    """
    Load TriviaQA rc.nocontext (closed-book) from HuggingFace.

    Args:
        split: "train" (~138k, cap with n_samples) or "validation" (~17.9k, used as test)
        n_samples: cap on number of samples; None = entire split

    Returns:
        List of dicts with keys: id, question, possible_answers
    """
    from datasets import load_dataset

    print(f"Loading TriviaQA rc.nocontext ({split}) from HuggingFace...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)

    data = []
    for item in dataset:
        aliases = item["answer"].get("aliases", [])
        if not aliases:
            # Fall back to the canonical value
            val = item["answer"].get("value", "")
            aliases = [val] if val else []

        data.append({
            "id": item.get("question_id", f"triviaqa_{split}_{len(data)}"),
            "question": item["question"],
            "possible_answers": aliases,
        })

        if n_samples is not None and len(data) >= n_samples:
            break

    print(f"Loaded {len(data)} TriviaQA samples from '{split}' split")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def is_correct(generation: str, possible_answers: list) -> bool:
    """Case-insensitive substring match against any alias."""
    gen_lower = generation.lower().strip()
    for answer in possible_answers:
        if answer.lower().strip() in gen_lower:
            return True
    return False


def compute_correctness(generations: list, possible_answers_list: list) -> list:
    return [is_correct(gen, ans) for gen, ans in zip(generations, possible_answers_list)]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQ: {question}\nA:"
