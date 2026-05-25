"""
SimpleQA task for HalluLens.

Dataset: https://huggingface.co/datasets/basicv8vc/SimpleQA
  - Single "test" split: 4,441 questions
  - Train/test partition created internally using split_seed (80/20)
    train: ~3,553  |  test: ~888

Evaluation mode: CLOSED-BOOK.
The model must rely on parametric memory; incorrect answers are treated as hallucinations.

Why SimpleQA?
  Curated specifically for factual questions where frontier models frequently fail.
  Questions are short, single-hop, and have unambiguously verified answers (1–3 words).
  The high error rate (~60% for strong models) ensures a balanced train set.
  Like PopQA, each hallucination stems from a single cause — parametric knowledge gap —
  producing consistent activation signatures that contrastive learning can leverage.

Evaluation method:
  Correctness = case-insensitive substring match of the gold answer in the response.
  Hallucination = response does not contain the gold answer.

Output eval_results.json schema (ActivationParser-compatible):
  {
    "evaluator_abstantion": "simpleqa_no_abstain",
    "evaluator_hallucination": "simpleqa_substring_match",
    "abstantion": [false, ...],
    "halu_test_res": [true, false, ...],   # True = hallucinated
    "total_count": N, "accurate_count": K, "hallu_count": N-K,
    "refusal_count": 0,
    "correct_rate": float, "halu_Rate": float, "refusal_Rate": 0.0
  }
"""

import ast
import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_simpleqa_data(split="test", n_samples=None, split_seed=42):
    """
    Load SimpleQA from HuggingFace (basicv8vc/SimpleQA).

    Since SimpleQA has only a single "test" split, we create an internal
    train/test partition (80/20) using split_seed for reproducibility.

    Args:
        split: "test" (20% = ~888) or "train" (80% = ~3,553)
        n_samples: cap on number of samples; None = entire partition
        split_seed: random seed for train/test partition (default: 42)

    Returns:
        List of dicts with keys: id, question, answer, topic, answer_type
    """
    from datasets import load_dataset
    import numpy as np

    print(f"Loading SimpleQA from HuggingFace (basicv8vc/SimpleQA)...")
    dataset = load_dataset("basicv8vc/SimpleQA", split="test")

    n_total = len(dataset)
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n_total)
    train_size = int(0.8 * n_total)

    if split == "train":
        selected_indices = sorted(indices[:train_size])
    elif split == "test":
        selected_indices = sorted(indices[train_size:])
    else:
        raise ValueError(f"Unknown split '{split}'. SimpleQA supports: train, test")

    data = []
    for idx in selected_indices:
        item = dataset[int(idx)]
        raw_meta = item.get("metadata", {})
        if isinstance(raw_meta, str):
            try:
                raw_meta = ast.literal_eval(raw_meta)
            except Exception:
                raw_meta = {}
        data.append({
            "id": f"simpleqa_{split}_{idx}",
            "question": item["problem"],
            "answer": item["answer"],
            "topic": raw_meta.get("topic", ""),
            "answer_type": raw_meta.get("answer_type", ""),
        })

    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} SimpleQA samples from '{split}' partition "
          f"(split_seed={split_seed}, total={n_total})")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def is_correct(generation: str, answer: str) -> bool:
    """Case-insensitive substring match of the gold answer."""
    return answer.lower().strip() in generation.lower().strip()


def compute_correctness(generations: list, answers: list) -> list:
    return [is_correct(gen, ans) for gen, ans in zip(generations, answers)]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQ: {question}\nA:"
