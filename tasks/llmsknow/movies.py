# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Movies QA task for HalluLens.

Dataset: external/LLMsKnow/data/movie_qa_{train,test}.csv
  - train: 10,000 questions
  - test:   7,856 questions  <- default for inference / eval

Evaluation mode: CLOSED-BOOK (no supporting passages provided in the prompt).
The model must rely on parametric memory; incorrect answers are treated as hallucinations.

Why Movies QA?
  Actor-role-movie trivia questions test factual recall of pop-culture knowledge.
  This complements Wikipedia-based benchmarks (HotpotQA, NQ) with a different knowledge
  domain and question style, probing whether hallucination detection generalises across
  diverse factual categories.

Evaluation method:
  Correctness = case-insensitive substring match of the gold answer in the model response.
  Hallucination = response does not contain the gold answer.

Output eval_results.json schema (ActivationParser-compatible):
  {
    "evaluator_abstantion": "movies_no_abstain",
    "evaluator_hallucination": "movies_substring_match",
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
import time
import argparse
from pathlib import Path

import pandas as pd
import jsonlines
from tqdm import tqdm

from utils import exp, lm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_movies_data(split="test", n_samples=None, data_dir="external/LLMsKnow/data"):
    """
    Load Movies QA from LLMsKnow CSV files (closed-book: no context provided).

    Args:
        split: "test" (7,856) or "train" (10,000)
        n_samples: cap on number of samples; None = entire split
        data_dir: directory containing movie_qa_{split}.csv

    Returns:
        List of dicts with keys: question, answer
    """
    data_file = Path(data_dir) / f"movie_qa_{split}.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Movies QA data file not found: {data_file}\n"
            f"Expected location: {data_file}\n"
            f"This file should be available in external/LLMsKnow/data/"
        )

    print(f"Loading Movies QA data from: {data_file}")

    df = pd.read_csv(data_file)
    df = df.rename(columns={"Question": "question", "Answer": "answer"})

    data = df.to_dict("records")

    if n_samples is not None and n_samples < len(data):
        data = data[:n_samples]

    print(f"Loaded {len(data)} Movies QA samples from '{split}' split")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def is_correct(generation: str, answer: str) -> bool:
    """Case-insensitive substring check."""
    return answer.lower().strip() in generation.lower().strip()


def compute_correctness(generations: list[str], answers: list[str]) -> list[bool]:
    return [is_correct(gen, ans) for gen, ans in zip(generations, answers)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQ: {question}\nA:"


class MoviesInference:
    """Run model inference on Movies QA (closed-book, no context provided)."""

    def __init__(self, model_path, output_base_dir="output", split="test",
                 n_samples=None, generations_file_path=None,
                 data_dir="external/LLMsKnow/data"):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.split = split

        self.task_output_dir = f"{output_base_dir}/movies/{self.model_name}"
        os.makedirs(self.task_output_dir, exist_ok=True)

        self.generations_file_path = (
            generations_file_path or f"{self.task_output_dir}/generation.jsonl"
        )

        self.data = load_movies_data(split=split, n_samples=n_samples, data_dir=data_dir)

    def _build_prompts_df(self):
        """Build a DataFrame of prompts from the loaded dataset."""
        rows = []
        for item in self.data:
            rows.append({
                "question": item["question"],
                "answer": item["answer"],
                "prompt": format_prompt(item["question"]),
            })
        return pd.DataFrame(rows)

    def run_inference(self, inference_method="vllm", max_tokens=128, temperature=0.0,
                      logger_type="lmdb", activations_path=None, log_file=None,
                      resume=True, max_retries=3, base_delay=1.0):
        prompts_df = self._build_prompts_df()

        print(f"Starting Movies QA inference | model={self.model_path} | split={self.split} | N={len(prompts_df)}")

        exp.run_exp(
            task="movies",
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
        """Run batched inference using HFTransformersAdapter directly.

        Bypasses the HTTP server and calls model.generate() on batches of
        prompts for higher throughput.  Activations are written to Zarr via
        AsyncActivationWriter, matching the server's storage format.

        Args:
            batch_size: Number of prompts per model.generate() call.
            max_tokens: Maximum new tokens per response.
            temperature: Sampling temperature (0.0 = greedy).
            activations_path: Path to .zarr store for activations.
            resume: Skip prompts already present in the generations file.
        """
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

        print(f"Starting batched Movies QA inference | model={self.model_path} | "
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
        t0 = time.time()
        completed = 0

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

                    completed += len(batch_prompts)
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(prompts) - completed) / rate if rate > 0 else 0
                    pbar.set_postfix_str(f"{rate:.1f} samples/s | ETA {remaining/60:.1f}m")
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

class MoviesEval:
    """Score model generations against Movies QA gold answers."""

    def __init__(self, model_path, generations_file_path=None, output_base_dir="output",
                 quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]

        if generations_file_path:
            self.generations_file_path = generations_file_path
            self.output_path = os.path.dirname(generations_file_path)
        else:
            self.output_path = f"{output_base_dir}/movies/{self.model_name}"
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

    def run_eval(self, eval_results_path=None, log_file=None):
        generations = self.df["generation"].tolist()
        answers = self.df["answer"].tolist()

        correctness = compute_correctness(generations, answers)  # True = correct

        n = len(correctness)
        correct_count = sum(correctness)
        incorrect_count = n - correct_count

        halu_test_res = [not c for c in correctness]
        abstantion = [False] * n

        res = {
            "evaluator_abstantion": "movies_no_abstain",
            "evaluator_hallucination": "movies_substring_match",
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
            "answer": answers,
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
                    "answer": answers[i],
                    "generation": generations[i],
                    "is_correct": correctness[i],
                    "is_hallucination": halu_test_res[i],
                    "is_abstaining": False,
                })
        print(f"Raw eval results saved -> {raw_path}")

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY -- Movies QA")
        print("=" * 60)
        print(f"Model          : {self.model_name}")
        print(f"Total samples  : {n}")
        print(f"Correct        : {correct_count}  ({correct_count/max(1,n):.1%})")
        print(f"Hallucinated   : {incorrect_count}  ({incorrect_count/max(1,n):.1%})")
        print("=" * 60)

        return res


# ---------------------------------------------------------------------------
# LLM-judge evaluation
# ---------------------------------------------------------------------------

IS_CORRECT_RESPONSE = """You are given a question, a model response, and the correct answer.
Your task is to determine whether the model response conveys the correct answer.

The question asks about an actor or role in a specific movie.
Accept semantically equivalent answers even if phrased differently from the gold answer
(e.g., full name vs. shortened name, alternative spellings).

Answer with exactly one word: CORRECT or INCORRECT.

Question: {prompt}
Response: {generation}
Correct Answer: {gold_answer}

YOUR JUDGEMENT:"""


class MoviesLLMEval:
    """Score Movies QA generations with an LLM judge, alongside substring match for comparison."""

    DEFAULT_EVALUATOR = "neuralmagic-ent/Llama-3.3-70B-Instruct-quantized.w8a8"

    def __init__(self, model_path, generations_file_path=None, output_base_dir="output",
                 evaluator=None, quick_debug_mode=False):
        self.model_name = model_path.split("/")[-1]
        self.evaluator = evaluator or self.DEFAULT_EVALUATOR

        if generations_file_path:
            self.generations_file_path = generations_file_path
            self.output_path = os.path.dirname(generations_file_path)
        else:
            self.output_path = f"{output_base_dir}/movies/{self.model_name}"
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

        print(f"Loaded {len(self.df)} generations for LLM evaluation")

    def _run_llm_judge(self, resume=True):
        """Call the LLM judge for each sample; return list of raw string responses."""
        raw_path = f"{self.output_path}/llm_halu_eval_raw.jsonl"

        judge_prompts = [
            IS_CORRECT_RESPONSE.format(
                prompt=row["prompt"],
                generation=row["generation"],
                gold_answer=row["answer"],
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
        answers = self.df["answer"].tolist()

        # LLM judge verdict
        llm_correct = []
        for txt in raw_responses:
            first = txt.strip().split()[0].lower().rstrip(".,;:") if txt.strip() else ""
            llm_correct.append(first == "correct")

        # Substring match for comparison
        substring_correct = compute_correctness(generations, answers)

        n = len(llm_correct)
        llm_halu = [not c for c in llm_correct]
        substring_halu = [not c for c in substring_correct]

        llm_correct_count = sum(llm_correct)
        substring_correct_count = sum(substring_correct)

        # Agreement between the two methods
        agree_count = sum(a == b for a, b in zip(llm_halu, substring_halu))

        res = {
            "evaluator_abstantion": "movies_no_abstain",
            "evaluator_hallucination": f"movies_llm_judge:{self.evaluator}",
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
            "answer": answers,
        }

        out_path = eval_results_path or f"{self.output_path}/eval_results_llm.json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY -- Movies QA (LLM Judge vs Substring)")
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

def run_step(step, model, data_dir="external/LLMsKnow/data", output_dir="output",
             split="test", inference_method="vllm", max_tokens=128, temperature=0.0,
             N=None, generations_file_path=None, eval_results_path=None, log_file=None,
             logger_type="lmdb", activations_path=None,
             quick_debug_mode=False, resume=True, max_retries=3, base_delay=1.0,
             llm_evaluator=None, batch_size=32):
    """Run a single step of the Movies QA task. Callable from Python directly.

    Args:
        batch_size: If set, use direct batched inference via HFTransformersAdapter
                    instead of the HTTP server path.  Recommended for throughput.
    """

    if step == "inference":
        runner = MoviesInference(
            model_path=model,
            output_base_dir=output_dir,
            split=split,
            n_samples=N,
            generations_file_path=generations_file_path,
            data_dir=data_dir,
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
        evaluator = MoviesEval(
            model_path=model,
            generations_file_path=generations_file_path,
            output_base_dir=output_dir,
            quick_debug_mode=quick_debug_mode,
        )
        evaluator.run_eval(eval_results_path=eval_results_path, log_file=log_file)

    elif step == "eval_llm":
        evaluator = MoviesLLMEval(
            model_path=model,
            generations_file_path=generations_file_path,
            output_base_dir=output_dir,
            evaluator=llm_evaluator,
            quick_debug_mode=quick_debug_mode,
        )
        evaluator.run_eval(eval_results_path=eval_results_path, resume=resume)

    else:
        raise ValueError(f"Unknown step '{step}'. Movies QA supports: inference, eval, eval_llm")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movies QA inference and evaluation (closed-book)")

    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_eval_llm", action="store_true",
                        help="Run LLM-judge evaluation (outputs both LLM and substring results)")
    parser.add_argument("--llm_evaluator", type=str, default=None,
                        help=f"Model to use as LLM judge (default: {MoviesLLMEval.DEFAULT_EVALUATOR})")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="external/LLMsKnow/data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="Movies QA split to use (default: test = 7,856 questions)")
    parser.add_argument("--N", type=int, default=None,
                        help="Number of samples (None = entire split)")
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
                 data_dir=args.data_dir,
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
                 batch_size=args.batch_size)

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
