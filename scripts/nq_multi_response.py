#!/usr/bin/env python3
"""
nq_multi_response.py

Multi-response NQ inference for hallucination correlation measurement.

For each of N_QUESTIONS questions, generates:
  - 1 greedy response   (temperature=0.0, key suffix _r0)
  - K stochastic responses (temperature=TEMPERATURE, key suffix _r1..rK)

All responses are activation-logged to shared/nq_test_hallu_cor/activations.zarr
via the activation server's request_id / multi_sample mechanism.

After inference, evaluates each response with NQ string matching and computes
cross-response hallucination agreement. If mean disagreement < DISAGREEMENT_TARGET,
the script raises TEMPERATURE and re-runs the stochastic rounds only (once).

Usage (run on GPU node via jupyter_exec, or directly):
    python scripts/nq_multi_response.py [--temperature T] [--N N] [--port PORT]

Outputs under OUTPUT_DIR:
    generation_r0.jsonl   - greedy responses
    generation_r1.jsonl   - stochastic response 1
    ...
    generation_r4.jsonl   - stochastic response 4
    eval_results.json     - per-question labels + correlation metrics
    server.log            - vLLM server log
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import Counter

import openai

# ── project root ──────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from utils.log_config import configure_logging
configure_logging("INFO")

from loguru import logger
from utils import lm

# ── constants (overridable via CLI) ───────────────────────────────────────────
MODEL              = "meta-llama/Llama-3.1-8B-Instruct"
N_QUESTIONS        = 1000
N_RESPONSES        = 5          # 1 greedy + 4 stochastic
TEMPERATURE_INIT   = 0.7        # stochastic temperature (first pass)
MAX_TOKENS         = 64
DATA_DIR           = "external/LLMsKnow/data"
OUTPUT_DIR         = "shared/nq_test_hallu_cor"
PORT               = 8000
HOST               = "0.0.0.0"
DISAGREEMENT_TARGET = 0.20     # re-run at higher temp if below this
TEMPERATURE_HIGH   = 1.0       # temperature for second pass


# ── helpers ───────────────────────────────────────────────────────────────────

def format_prompt(question: str) -> str:
    return f"Answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"


def nq_is_correct(generation: str, answer: str) -> bool:
    """NQ string-matching correctness (case-insensitive substring)."""
    if not isinstance(generation, str) or not isinstance(answer, str):
        return False
    return answer.strip().lower() in generation.strip().lower()


def _jsonl_existing_questions(path: Path) -> set:
    """Return the set of questions already in a JSONL file."""
    if not path.exists():
        return set()
    existing = set()
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                existing.add(rec.get("question", ""))
            except Exception:
                pass
    return existing


def run_inference_round(
    questions: list,
    response_idx: int,
    temperature: float,
    out_path: Path,
    model: str,
    port: int,
    max_tokens: int,
) -> None:
    """Run one response-round (response_idx) for all questions not yet done."""
    existing = _jsonl_existing_questions(out_path)
    todo = [q for q in questions if q["question"] not in existing]

    if not todo:
        logger.info(f"[r{response_idx}] All {len(questions)} questions already done — skipping")
        return

    logger.info(
        f"[r{response_idx}] Running inference for {len(todo)}/{len(questions)} questions "
        f"(temperature={temperature})"
    )

    server_url = f"http://localhost:{port}/v1"
    client = openai.OpenAI(base_url=server_url, api_key="NOT_A_REAL_KEY", timeout=90.0)

    with open(out_path, "a") as f:
        for i, item in enumerate(todo):
            question = item["question"]
            answer   = item["answer"]
            prompt   = format_prompt(question)

            # request_id encodes the round index so the server assigns a unique zarr key
            extra = {"request_id": f"r{response_idx}", "multi_sample": True, "sample_index": response_idx}

            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body=extra,
                    )
                    generation = resp.choices[0].message.content or ""
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"[r{response_idx}] attempt {attempt+1}/4 failed for q{i}: {e} — retry in {wait}s")
                    time.sleep(wait)
            else:
                logger.error(f"[r{response_idx}] All retries failed for question {i}; writing ERROR")
                generation = ""

            rec = {
                "question":     question,
                "answer":       answer,
                "prompt":       prompt,
                "generation":   generation,
                "response_idx": response_idx,
                "temperature":  temperature,
                "model":        model,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            if (i + 1) % 50 == 0 or i == len(todo) - 1:
                logger.info(f"[r{response_idx}] {i+1}/{len(todo)} done")


def evaluate_and_correlate(questions: list, output_dir: Path, n_responses: int):
    """
    Load all response rounds, evaluate correctness, compute per-question
    label vectors, and return correlation metrics.

    Returns a dict with:
        per_question   : list of {question, answer, labels, majority, disagree_frac}
        mean_disagree  : float  (0..1)
        exact_agreement: float  (fraction of questions where all 5 labels match)
        halu_rates     : list[float]  per-response hallucination rate
    """
    # Load all rounds
    # generations[q_idx][r_idx] = generation string
    q_map = {q["question"]: {"answer": q["answer"], "gens": {}} for q in questions}

    for r in range(n_responses):
        path = output_dir / f"generation_r{r}.jsonl"
        if not path.exists():
            logger.warning(f"Missing {path} — treating as all empty")
            continue
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    q = rec["question"]
                    if q in q_map:
                        q_map[q]["gens"][r] = rec.get("generation", "")
                except Exception:
                    pass

    per_question = []
    halu_per_response = [0] * n_responses
    n_complete = 0

    for q, info in q_map.items():
        answer = info["answer"]
        gens = info["gens"]

        labels = []
        for r in range(n_responses):
            gen = gens.get(r, "")
            is_halu = not nq_is_correct(gen, answer)
            labels.append(int(is_halu))  # 1 = hallucinated

        if len(labels) < n_responses:
            continue  # skip incomplete

        n_complete += 1
        counts = Counter(labels)
        majority = counts.most_common(1)[0][0]
        disagree_count = sum(1 for l in labels if l != majority)
        disagree_frac = disagree_count / n_responses

        per_question.append({
            "question":     q,
            "answer":       answer,
            "labels":       labels,
            "majority":     majority,
            "disagree_frac": disagree_frac,
        })

        for r, lbl in enumerate(labels):
            if lbl == 1:
                halu_per_response[r] += 1

    if n_complete == 0:
        return {"error": "no complete questions"}

    mean_disagree  = sum(pq["disagree_frac"] for pq in per_question) / len(per_question)
    exact_agreement = sum(1 for pq in per_question if pq["disagree_frac"] == 0.0) / len(per_question)
    halu_rates = [c / n_complete for c in halu_per_response]

    return {
        "per_question":    per_question,
        "mean_disagree":   mean_disagree,
        "exact_agreement": exact_agreement,
        "halu_rates":      halu_rates,
        "n_questions":     n_complete,
        "n_responses":     n_responses,
    }


def print_correlation_summary(metrics: dict, temperature: float):
    logger.info("=" * 70)
    logger.info("HALLUCINATION CORRELATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Questions evaluated : {metrics['n_questions']}")
    logger.info(f"  Responses per Q     : {metrics['n_responses']}")
    logger.info(f"  Stochastic temp     : {temperature}")
    logger.info(f"  Mean disagreement   : {metrics['mean_disagree']:.3f}  (target ≥ {DISAGREEMENT_TARGET})")
    logger.info(f"  Exact agreement     : {metrics['exact_agreement']:.3f}")
    logger.info("  Per-response halu rates:")
    for r, rate in enumerate(metrics["halu_rates"]):
        tag = "greedy" if r == 0 else f"T={temperature}"
        logger.info(f"    r{r} ({tag}): {rate:.3f}")
    logger.info("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-response NQ correlation experiment")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_INIT)
    parser.add_argument("--N", type=int, default=N_QUESTIONS)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-server", action="store_true",
                        help="Skip server startup (use already-running server)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_path = str(output_dir / "activations.zarr")
    log_file         = str(output_dir / "server.log")

    # ── load data ─────────────────────────────────────────────────────────────
    from tasks.llmsknow.natural_questions import load_nq_data
    questions = load_nq_data(split="test", n_samples=args.N, data_dir=DATA_DIR)
    logger.info(f"Loaded {len(questions)} NQ questions")

    # ── start server ──────────────────────────────────────────────────────────
    server_manager = None
    if not args.no_server:
        if lm.check_server_health(f"http://localhost:{args.port}"):
            logger.info(f"Server already running on port {args.port} — using it")
        else:
            logger.info("Starting vLLM activation server …")
            server_manager = lm.ServerManager(
                model=MODEL,
                host=HOST,
                port=args.port,
                logger_type="zarr",
                activations_path=activations_path,
                log_file_path=log_file,
            )
            server_manager.start_server()
            lm.set_server_manager(server_manager)
            logger.success(f"Server started — activations → {activations_path}")

    try:
        stochastic_temperature = args.temperature

        # ── Phase 1: Run all 5 response rounds ──────────────────────────────
        for r in range(N_RESPONSES):
            temp = 0.0 if r == 0 else stochastic_temperature
            out_path = output_dir / f"generation_r{r}.jsonl"
            run_inference_round(
                questions=questions,
                response_idx=r,
                temperature=temp,
                out_path=out_path,
                model=MODEL,
                port=args.port,
                max_tokens=MAX_TOKENS,
            )

        # ── Phase 2: Evaluate + correlation ────────────────────────────────
        logger.info("Evaluating hallucination labels and computing correlation …")
        metrics = evaluate_and_correlate(questions, output_dir, N_RESPONSES)
        print_correlation_summary(metrics, stochastic_temperature)

        # Save results
        results = {
            "phase": 1,
            "stochastic_temperature": stochastic_temperature,
            **{k: v for k, v in metrics.items() if k != "per_question"},
        }

        # ── Phase 3: Temperature adjustment (one-shot) ─────────────────────
        if metrics.get("mean_disagree", 1.0) < DISAGREEMENT_TARGET:
            stochastic_temperature = TEMPERATURE_HIGH
            logger.warning(
                f"Mean disagreement {metrics['mean_disagree']:.3f} < {DISAGREEMENT_TARGET} target. "
                f"Re-running stochastic rounds at T={TEMPERATURE_HIGH} …"
            )

            # Clear stochastic rounds and rerun
            for r in range(1, N_RESPONSES):
                out_path = output_dir / f"generation_r{r}.jsonl"
                # Back up old file
                backup = output_dir / f"generation_r{r}_T{stochastic_temperature:.1f}_old.jsonl"
                if out_path.exists():
                    out_path.rename(backup)
                    logger.info(f"Backed up {out_path.name} → {backup.name}")

                run_inference_round(
                    questions=questions,
                    response_idx=r,
                    temperature=stochastic_temperature,
                    out_path=out_path,
                    model=MODEL,
                    port=args.port,
                    max_tokens=MAX_TOKENS,
                )

            metrics2 = evaluate_and_correlate(questions, output_dir, N_RESPONSES)
            logger.info("\n=== RESULTS AFTER TEMPERATURE INCREASE ===")
            print_correlation_summary(metrics2, stochastic_temperature)

            results["phase2_stochastic_temperature"] = stochastic_temperature
            results.update({
                f"phase2_{k}": v
                for k, v in metrics2.items()
                if k != "per_question"
            })

            # Append detailed per-question data for phase 2
            per_q_path = output_dir / "per_question_phase2.jsonl"
            with open(per_q_path, "w") as f:
                for pq in metrics2.get("per_question", []):
                    f.write(json.dumps(pq) + "\n")
            logger.info(f"Per-question phase-2 data → {per_q_path}")
        else:
            logger.success(
                f"Disagreement {metrics['mean_disagree']:.3f} ≥ {DISAGREEMENT_TARGET} target — "
                "no temperature adjustment needed."
            )

        # Save final per-question data (phase 1)
        per_q_path1 = output_dir / "per_question_phase1.jsonl"
        with open(per_q_path1, "w") as f:
            for pq in metrics.get("per_question", []):
                f.write(json.dumps(pq) + "\n")

        # Save summary JSON
        eval_path = output_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Results saved → {eval_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if server_manager:
            logger.info("Stopping server …")
            server_manager.stop_server()
            lm.set_server_manager(None)
            logger.success("Server stopped")


if __name__ == "__main__":
    main()
