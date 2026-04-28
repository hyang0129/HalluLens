# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
PopQA task for HalluLens.

Dataset: https://huggingface.co/datasets/akariasai/PopQA
  - Single "test" split: 14,267 entity-centric factual QA pairs
  - Train/test partition created from "test" split using split_seed (80/20)

Evaluation mode: CLOSED-BOOK (no supporting passages provided in the prompt).
The model must rely on parametric memory; incorrect answers are treated as hallucinations.

Why PopQA?
  Entity-centric questions with Wikidata-backed popularity scores (s_pop) enable
  popularity-stratified hallucination analysis.  This reveals whether models hallucinate
  more on obscure entities -- a key insight for understanding parametric knowledge gaps.
  Questions span diverse Wikidata relations (prop), complementing Wikipedia-passage-based
  benchmarks (HotpotQA, NQ) with structured knowledge graph coverage.

Evaluation method:
  Correctness = case-insensitive substring match of ANY answer in possible_answers list.
  Hallucination = response does not contain any valid answer.

Output eval_results.json schema (ActivationParser-compatible):
  {
    "evaluator_abstantion": "popqa_no_abstain",
    "evaluator_hallucination": "popqa_substring_match",
    "abstantion": [false, ...],
    "halu_test_res": [true, false, ...],   # True = hallucinated
    "total_count": N, "accurate_count": K, "hallu_count": N-K,
    "refusal_count": 0,
    "correct_rate": float, "halu_Rate": float, "refusal_Rate": 0.0
  }
"""

import hashlib
import json
import os
import argparse
from pathlib import Path

import pandas as pd
import jsonlines
from tqdm import tqdm

from utils import exp, lm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_popqa_data(split="test", n_samples=None, split_seed=42):
    """
    Load PopQA from HuggingFace (akariasai/PopQA).

    Since PopQA has only a single "test" split, we create a train/test
    partition (80/20) using split_seed for reproducibility.

    Args:
        split: "test" (20% = ~2,853) or "train" (80% = ~11,414)
        n_samples: cap on number of samples; None = entire partition
        split_seed: random seed for train/test partition (default: 42)

    Returns:
        List of dicts with keys: question, possible_answers, subj, obj, prop, s_pop
    """
    from datasets import load_dataset
    import numpy as np

    print(f"Loading PopQA from HuggingFace (akariasai/PopQA)...")
    dataset = load_dataset("akariasai/PopQA", trust_remote_code=True)["test"]

    # Create reproducible train/test partition (80/20)
    n_total = len(dataset)
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(n_total)
    train_size = int(0.8 * n_total)

    if split == "train":
        selected_indices = sorted(indices[:train_size])
    elif split == "test":
        selected_indices = sorted(indices[train_size:])
    else:
        raise ValueError(f"Unknown split '{split}'. PopQA supports: train, test")

    data = []
    for idx in selected_indices:
        item = dataset[int(idx)]
        # possible_answers may be stored as a list or as a string representation
        possible_answers = item.get("possible_answers", [])
        if isinstance(possible_answers, str):
            # Handle string-encoded lists (e.g. '["ans1", "ans2"]')
            try:
                possible_answers = json.loads(possible_answers)
            except (json.JSONDecodeError, TypeError):
                possible_answers = [possible_answers]

        data.append({
            "question": item["question"],
            "possible_answers": possible_answers,
            "subj": item.get("subj", ""),
            "obj": item.get("obj", ""),
            "prop": item.get("prop", ""),
            "s_pop": item.get("s_pop", 0),
        })

    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} PopQA samples from '{split}' partition "
          f"(split_seed={split_seed}, total={n_total})")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def is_correct(generation: str, possible_answers: list) -> bool:
    """Check if ANY answer in possible_answers appears in generation (case-insensitive substring)."""
    gen_lower = generation.lower().strip()
    for answer in possible_answers:
        if answer.lower().strip() in gen_lower:
            return True
    return False


def compute_correctness(generations: list[str], possible_answers_list: list[list]) -> list[bool]:
    return [is_correct(gen, ans) for gen, ans in zip(generations, possible_answers_list)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQ: {question}\nA:"


class PopQAInference:
    """Run model inference on PopQA (closed-book, no context provided)."""

    def __init__(self, model_path, output_base_dir="output", split="test",
                 n_samples=None, generations_file_path=None, split_seed=42):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.split = split
        self.split_seed = split_seed

        self.task_output_dir = f"{output_base_dir}/popqa/{self.model_name}"
        os.makedirs(self.task_output_dir, exist_ok=True)

        self.generations_file_path = (
            generations_file_path or f"{self.task_output_dir}/generation.jsonl"
        )

        self.data = load_popqa_data(split=split, n_samples=n_samples, split_seed=split_seed)

    def _build_prompts_df(self):
        """Build a DataFrame of prompts from the loaded dataset."""
        rows = []
        for item in self.data:
            rows.append({
                "question": item["question"],
                "possible_answers": json.dumps(item["possible_answers"]),
                "subj": item["subj"],
                "obj": item["obj"],
                "prop": item["prop"],
                "s_pop": item["s_pop"],
                "prompt": format_prompt(item["question"]),
            })
        return pd.DataFrame(rows)

    def run_inference(self, inference_method="vllm", max_tokens=128, temperature=0.0,
                      logger_type="lmdb", activations_path=None, log_file=None,
                      resume=True, max_retries=3, base_delay=1.0):
        prompts_df = self._build_prompts_df()

        print(f"Starting PopQA inference | model={self.model_path} | split={self.split} | N={len(prompts_df)}")

        exp.run_exp(
            task="popqa",
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
        print(f"Inference complete -> {self.generations_file_path}")

    def run_inference_batched(self, batch_size=8, max_tokens=128, temperature=0.0,
                              activations_path=None, resume=True):
        """Run batched inference using HFTransformersAdapter directly."""
        from activation_logging.model_adapter import HFTransformersAdapter
        from activation_logging.zarr_activations_logger import ZarrActivationsLogger
        from activation_logging.server import AsyncActivationWriter

        prompts_df = self._build_prompts_df()
        total = len(prompts_df)

        # --- Resume: skip already-processed prompts ---
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

        print(f"Starting batched PopQA inference | model={self.model_path} | "
              f"split={self.split} | remaining={len(prompts_df)} | batch_size={batch_size}")

        # --- Set up adapter and activation logging ---
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
                pbar = tqdm(total=len(prompts), desc="Batched inference")
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class PopQAEval:
    """Score model generations against PopQA gold answers with popularity-stratified analysis."""

    def __init__(self, model_path, generations_file_path=None, output_base_dir="output",
                 quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]

        if generations_file_path:
            self.generations_file_path = generations_file_path
            self.output_path = os.path.dirname(generations_file_path)
        else:
            self.output_path = f"{output_base_dir}/popqa/{self.model_name}"
            self.generations_file_path = f"{self.output_path}/generation.jsonl"

        if not os.path.exists(self.generations_file_path):
            raise FileNotFoundError(
                f"Generations file not found: {self.generations_file_path}\n"
                "Run inference first."
            )

        self.df = pd.read_json(self.generations_file_path, lines=True)

        # Parse possible_answers from JSON string if needed
        if "possible_answers" in self.df.columns:
            self.df["possible_answers"] = self.df["possible_answers"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        if quick_debug_mode:
            print("Quick debug mode: using first 50 samples")
            self.df = self.df.head(50)

        print(f"Loaded {len(self.df)} generations for evaluation")

    def _popularity_stratified_summary(self, correctness):
        """Print popularity-stratified accuracy breakdown."""
        if "s_pop" not in self.df.columns:
            return

        df = self.df.copy()
        df["is_correct"] = correctness

        # Define popularity buckets based on log-scale s_pop
        bins = [0, 100, 1000, 10000, 100000, float("inf")]
        labels = ["0-100", "100-1K", "1K-10K", "10K-100K", "100K+"]
        df["pop_bucket"] = pd.cut(df["s_pop"], bins=bins, labels=labels, right=False)

        print("\n--- Popularity-Stratified Accuracy ---")
        print(f"{'Popularity':<15} {'Count':>8} {'Correct':>8} {'Accuracy':>10}")
        print("-" * 45)
        for label in labels:
            bucket = df[df["pop_bucket"] == label]
            if len(bucket) == 0:
                continue
            correct = bucket["is_correct"].sum()
            total = len(bucket)
            print(f"{label:<15} {total:>8} {correct:>8} {correct/max(1,total):>10.1%}")
        print("-" * 45)

        # Also break down by relation type (prop)
        if "prop" in self.df.columns:
            print("\n--- Per-Relation Accuracy (top 10) ---")
            prop_stats = df.groupby("prop").agg(
                count=("is_correct", "size"),
                correct=("is_correct", "sum"),
            )
            prop_stats["accuracy"] = prop_stats["correct"] / prop_stats["count"]
            prop_stats = prop_stats.sort_values("count", ascending=False).head(10)
            print(f"{'Relation':<25} {'Count':>8} {'Correct':>8} {'Accuracy':>10}")
            print("-" * 55)
            for prop, row in prop_stats.iterrows():
                print(f"{str(prop):<25} {int(row['count']):>8} {int(row['correct']):>8} {row['accuracy']:>10.1%}")

    def run_eval(self, eval_results_path=None, log_file=None):
        generations = self.df["generation"].tolist()
        possible_answers_list = self.df["possible_answers"].tolist()

        correctness = compute_correctness(generations, possible_answers_list)

        n = len(correctness)
        correct_count = sum(correctness)
        incorrect_count = n - correct_count

        halu_test_res = [not c for c in correctness]
        abstantion = [False] * n

        res = {
            "evaluator_abstantion": "popqa_no_abstain",
            "evaluator_hallucination": "popqa_substring_match",
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
            "possible_answers": [json.dumps(pa) for pa in possible_answers_list],
        }

        out_path = eval_results_path or f"{self.output_path}/eval_results.json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Eval results saved -> {out_path}")

        # Per-sample raw_eval_res.jsonl
        raw_path = f"{self.output_path}/raw_eval_res.jsonl"
        with jsonlines.open(raw_path, mode="w") as writer:
            for i, (_, row) in enumerate(self.df.iterrows()):
                writer.write({
                    "question": row.get("question", ""),
                    "possible_answers": possible_answers_list[i],
                    "generation": generations[i],
                    "is_correct": correctness[i],
                    "is_hallucination": halu_test_res[i],
                    "is_abstaining": False,
                    "subj": row.get("subj", ""),
                    "prop": row.get("prop", ""),
                    "s_pop": row.get("s_pop", 0),
                })
        print(f"Raw eval results saved -> {raw_path}")

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY -- PopQA")
        print("=" * 60)
        print(f"Model          : {self.model_name}")
        print(f"Total samples  : {n}")
        print(f"Correct        : {correct_count}  ({correct_count/max(1,n):.1%})")
        print(f"Hallucinated   : {incorrect_count}  ({incorrect_count/max(1,n):.1%})")
        print("=" * 60)

        self._popularity_stratified_summary(correctness)

        return res


# ---------------------------------------------------------------------------
# LLM-judge evaluation
# ---------------------------------------------------------------------------

IS_CORRECT_RESPONSE = """You are given a question, a model response, and a list of correct answers.
Your task is to determine whether the model response conveys any of the correct answers.

The question is entity-centric factual QA. Accept semantically equivalent answers even
if phrased differently (e.g., full name vs. shortened name, alternative spellings).

Answer with exactly one word: CORRECT or INCORRECT.

Question: {prompt}
Response: {generation}
Correct Answers: {gold_answers}

YOUR JUDGEMENT:"""


class PopQALLMEval:
    """Score PopQA generations with an LLM judge, alongside substring match for comparison."""

    DEFAULT_EVALUATOR = "neuralmagic-ent/Llama-3.3-70B-Instruct-quantized.w8a8"

    def __init__(self, model_path, generations_file_path=None, output_base_dir="output",
                 evaluator=None, quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]
        self.evaluator = evaluator or self.DEFAULT_EVALUATOR

        if generations_file_path:
            self.generations_file_path = generations_file_path
            self.output_path = os.path.dirname(generations_file_path)
        else:
            self.output_path = f"{output_base_dir}/popqa/{self.model_name}"
            self.generations_file_path = f"{self.output_path}/generation.jsonl"

        if not os.path.exists(self.generations_file_path):
            raise FileNotFoundError(
                f"Generations file not found: {self.generations_file_path}\n"
                "Run inference first."
            )

        self.df = pd.read_json(self.generations_file_path, lines=True)

        # Parse possible_answers from JSON string if needed
        if "possible_answers" in self.df.columns:
            self.df["possible_answers"] = self.df["possible_answers"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        if quick_debug_mode:
            print("Quick debug mode: using first 50 samples")
            self.df = self.df.head(50)

        print(f"Loaded {len(self.df)} generations for LLM evaluation")

    def _run_llm_judge(self, resume=True):
        """Call the LLM judge for each sample; return list of raw string responses."""
        raw_path = f"{self.output_path}/llm_halu_eval_raw.jsonl"

        judge_prompts = [
            IS_CORRECT_RESPONSE.format(
                prompt=row["prompt"],
                generation=row["generation"],
                gold_answers=", ".join(row["possible_answers"]),
            )
            for _, row in self.df.iterrows()
        ]

        existing = []
        prompts_to_run = judge_prompts

        if resume and os.path.exists(raw_path):
            try:
                with open(raw_path) as f:
                    existing = [json.loads(l)["eval_res"] for l in f]
                if len(existing) >= len(judge_prompts):
                    print(f"All {len(judge_prompts)} LLM evaluations already cached.")
                    return existing
                prompts_to_run = judge_prompts[len(existing):]
                print(f"Resuming LLM eval: {len(existing)} done, {len(prompts_to_run)} remaining.")
            except Exception as e:
                print(f"Warning: could not load cached results ({e}), starting fresh.")
                existing = []
                prompts_to_run = judge_prompts

        server_was_running = lm.check_server_health("http://0.0.0.0:8000")
        server_manager = None
        if not server_was_running:
            print(f"Starting evaluation server for {self.evaluator}...")
            server_manager = lm.ServerManager(
                model=self.evaluator,
                host="0.0.0.0",
                port=8000,
                logger_type="lmdb",
                activations_path=None,
            )
            server_manager.start_server()
            lm.set_server_manager(server_manager)

        lm.initialize_progress_tracking(len(judge_prompts), already_completed=len(existing))

        try:
            file_mode = "a" if existing else "w"
            new_results = []
            with open(raw_path, file_mode, encoding="utf-8") as f:
                for prompt in tqdm(prompts_to_run, desc=f"LLM eval ({self.evaluator})"):
                    result = lm.generate(prompt, self.evaluator)
                    new_results.append(result)
                    f.write(json.dumps({"eval_res": result}, ensure_ascii=False) + "\n")
                    f.flush()
        finally:
            if server_manager and not server_was_running:
                server_manager.stop_server()
                lm.set_server_manager(None)

        print(f"LLM eval saved -> {raw_path}")
        return existing + new_results

    def run_eval(self, eval_results_path=None, resume=True):
        raw_responses = self._run_llm_judge(resume=resume)

        generations = self.df["generation"].tolist()
        possible_answers_list = self.df["possible_answers"].tolist()

        # LLM judge verdict
        llm_correct = []
        for txt in raw_responses:
            first = txt.strip().split()[0].lower().rstrip(".,;:") if txt.strip() else ""
            llm_correct.append(first == "correct")

        # Substring match for comparison
        substring_correct = compute_correctness(generations, possible_answers_list)

        n = len(llm_correct)
        llm_halu = [not c for c in llm_correct]
        substring_halu = [not c for c in substring_correct]

        llm_correct_count = sum(llm_correct)
        substring_correct_count = sum(substring_correct)

        # Agreement between the two methods
        agree_count = sum(a == b for a, b in zip(llm_halu, substring_halu))

        res = {
            "evaluator_abstantion": "popqa_no_abstain",
            "evaluator_hallucination": f"popqa_llm_judge:{self.evaluator}",
            "abstantion": [False] * n,
            # Primary results from LLM judge
            "halu_test_res": llm_halu,
            "total_count": n,
            "accurate_count": llm_correct_count,
            "hallu_count": n - llm_correct_count,
            "refusal_count": 0,
            "correct_rate": llm_correct_count / max(1, n),
            "halu_Rate": (n - llm_correct_count) / max(1, n),
            "refusal_Rate": 0.0,
            # Substring match for comparison
            "substring_halu_test_res": substring_halu,
            "substring_correct_rate": substring_correct_count / max(1, n),
            "substring_halu_Rate": (n - substring_correct_count) / max(1, n),
            "evaluator_agreement_rate": agree_count / max(1, n),
            # Raw data
            "llm_judge_raw": raw_responses,
            "prompt": self.df["prompt"].tolist() if "prompt" in self.df.columns else [""] * n,
            "generation": generations,
            "possible_answers": [json.dumps(pa) for pa in possible_answers_list],
        }

        out_path = eval_results_path or f"{self.output_path}/eval_results_llm.json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY -- PopQA (LLM Judge vs Substring)")
        print("=" * 60)
        print(f"Model          : {self.model_name}")
        print(f"Evaluator      : {self.evaluator}")
        print(f"Total samples  : {n}")
        print(f"LLM   correct  : {llm_correct_count}  ({llm_correct_count/max(1,n):.1%})")
        print(f"Substr correct : {substring_correct_count}  ({substring_correct_count/max(1,n):.1%})")
        print(f"Agreement      : {agree_count}  ({agree_count/max(1,n):.1%})")
        print(f"Results saved  : {out_path}")
        print("=" * 60)

        return res


# ---------------------------------------------------------------------------
# Unified run_step interface
# ---------------------------------------------------------------------------

def run_step(step, model, output_dir="output",
             split="test", inference_method="vllm", max_tokens=128, temperature=0.0,
             N=None, generations_file_path=None, eval_results_path=None, log_file=None,
             logger_type="lmdb", activations_path=None,
             quick_debug_mode=False, resume=True, max_retries=3, base_delay=1.0,
             llm_evaluator=None, batch_size=32, split_seed=42):
    """Run a single step of the PopQA task. Callable from Python directly."""

    if step == "inference":
        runner = PopQAInference(
            model_path=model,
            output_base_dir=output_dir,
            split=split,
            n_samples=N,
            generations_file_path=generations_file_path,
            split_seed=split_seed,
        )
        if batch_size:
            runner.run_inference_batched(
                batch_size=batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                activations_path=activations_path,
                resume=resume,
            )
        else:
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
        evaluator = PopQAEval(
            model_path=model,
            generations_file_path=generations_file_path,
            output_base_dir=output_dir,
            quick_debug_mode=quick_debug_mode,
        )
        evaluator.run_eval(eval_results_path=eval_results_path, log_file=log_file)

    elif step == "eval_llm":
        evaluator = PopQALLMEval(
            model_path=model,
            generations_file_path=generations_file_path,
            output_base_dir=output_dir,
            evaluator=llm_evaluator,
            quick_debug_mode=quick_debug_mode,
        )
        evaluator.run_eval(eval_results_path=eval_results_path, resume=resume)

    else:
        raise ValueError(f"Unknown step '{step}'. PopQA supports: inference, eval, eval_llm")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PopQA inference and evaluation (closed-book)")

    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_eval_llm", action="store_true",
                        help="Run LLM-judge evaluation (outputs both LLM and substring results)")
    parser.add_argument("--llm_evaluator", type=str, default=None,
                        help=f"Model to use as LLM judge (default: {PopQALLMEval.DEFAULT_EVALUATOR})")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="PopQA partition to use (default: test = ~2,853 questions)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for train/test partition (default: 42)")
    parser.add_argument("--N", type=int, default=None,
                        help="Number of samples (None = entire partition)")
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
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for batched inference (None = sequential server path)")

    args = parser.parse_args()

    if args.do_inference:
        run_step("inference", args.model,
                 output_dir=args.output_dir,
                 split=args.split,
                 inference_method=args.inference_method,
                 max_tokens=args.max_tokens,
                 temperature=args.temperature,
                 N=args.N,
                 generations_file_path=args.generations_file_path,
                 log_file=args.log_file,
                 logger_type=args.logger_type,
                 activations_path=args.activations_path,
                 resume=not args.no_resume,
                 batch_size=args.batch_size,
                 split_seed=args.split_seed)

    if args.do_eval:
        run_step("eval", args.model,
                 output_dir=args.output_dir,
                 generations_file_path=args.generations_file_path,
                 eval_results_path=args.eval_results_path,
                 quick_debug_mode=args.quick_debug_mode)

    if args.do_eval_llm:
        run_step("eval_llm", args.model,
                 output_dir=args.output_dir,
                 generations_file_path=args.generations_file_path,
                 eval_results_path=args.eval_results_path,
                 quick_debug_mode=args.quick_debug_mode,
                 llm_evaluator=args.llm_evaluator,
                 resume=not args.no_resume)

    if not args.do_inference and not args.do_eval and not args.do_eval_llm:
        print("No action specified. Use --do_inference, --do_eval, and/or --do_eval_llm")
        parser.print_help()
