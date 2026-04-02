#!/usr/bin/env python3
"""
Unified script to run HalluLens tasks with automatic server management.

This script automatically starts the activation logging server and runs one of the three
generation steps, eliminating the need for manual server startup.

Note:
    In this runner, setting inference_method="vllm" starts the OpenAI-compatible
    FastAPI server via uvicorn (activation_logging.server:app, launched through
    activation_logging.vllm_serve), rather than invoking raw `vllm serve` directly.

Can be used as a CLI script or imported as a Python module (e.g., from a Jupyter notebook).
When used as a Python module, the client (inference/task) always runs in the calling thread,
which avoids tqdm RAM issues in Jupyter.

Usage (CLI):
    python scripts/run_with_server.py --step [generate|inference|eval] [task_options...]

Usage (Python):
    from scripts.run_with_server import run_experiment
    run_experiment(step="all", task="precisewikiqa", model="meta-llama/Llama-3.1-8B-Instruct", N=100)

Examples:
    # Generate prompts (PreciseWikiQA) - requires --q_generator for question generation
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 100

    # Generate with different model for questions vs inference
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 100

    # Run inference (PreciseWikiQA)
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --logger-type lmdb --activations-path shared/goodwiki.zarr/activations.zarr --log-file shared/goodwiki.zarr/server.log

    # Run evaluation (PreciseWikiQA)
    python scripts/run_with_server.py --step eval --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct

    # Run all steps in sequence (PreciseWikiQA)
    python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 100 --logger-type lmdb --activations-path shared/goodwiki.zarr/activations.zarr --log-file shared/goodwiki.zarr/server.log

    # TriviaQA inference and evaluation
    python scripts/run_with_server.py --step all --task triviaqa --model meta-llama/Llama-3.1-8B-Instruct --N 1000 --logger-type lmdb --activations-path shared/triviaqa_unfiltered_dev/activations.lmdb --log-file shared/triviaqa_unfiltered_dev/server.log

    # TriviaQA with custom dataset variant
    python scripts/run_with_server.py --step inference --task triviaqa --model meta-llama/Llama-3.1-8B-Instruct --dataset_variant unfiltered --split dev --logger-type lmdb --activations-path shared/triviaqa_unfiltered_dev/activations.lmdb --log-file shared/triviaqa_unfiltered_dev/server.log

    # Natural Questions inference and evaluation
    python scripts/run_with_server.py --step all --task naturalquestions --model meta-llama/Llama-3.1-8B-Instruct --N 1000 --logger-type lmdb --activations-path shared/natural_questions_dev/activations.lmdb --log-file shared/natural_questions_dev/server.log

    # Natural Questions with custom settings
    python scripts/run_with_server.py --step inference --task naturalquestions --model meta-llama/Llama-3.1-8B-Instruct --max_tokens 64 --temperature 0.0 --logger-type lmdb --activations-path shared/natural_questions_dev/activations.lmdb --log-file shared/natural_questions_dev/server.log

    # Custom server log file
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --logger-type lmdb --activations-path custom/activations.lmdb --log-file custom/server.log

    # Question generation with increased concurrency (8 parallel requests)
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 100 --max-workers-qgen 8

    # LongWiki with concurrent question generation
    python scripts/run_with_server.py --step generate --task longwiki --model meta-llama/Llama-3.1-70B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 50 --max-workers-qgen 4

    # Supported q_generator models (both served via vLLM, no activation logging):
    #   neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8  (W8A8, ~70GB, routed via vLLM w8a8 path)
    #   Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8                (GPTQ-Int8, ~72GB, routed via vLLM GPTQ path)
    """

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging BEFORE any logger usage.
# When imported as a library, callers should invoke configure_logging() themselves;
# when run via CLI, main() handles it via --log-level.
from utils.log_config import configure_logging  # noqa: E402

configure_logging()  # respects HALLULENS_LOG_LEVEL env var

from loguru import logger  # noqa: E402

from utils.progress import install_tqdm_global  # noqa: E402

_TQDM_BACKEND = install_tqdm_global()
logger.debug(f"tqdm backend configured: {_TQDM_BACKEND}")

# Import ServerManager from utils.lm
from utils import lm

# Use ServerManager from utils.lm
ServerManager = lm.ServerManager
get_server_manager = lm.get_server_manager
set_server_manager = lm.set_server_manager


Q_GENERATOR_REQUIRED_TASKS = {"precisewikiqa", "longwiki"}


def requires_q_generator(step, task):
    """Return True when the requested run includes generation for a task that needs q_generator."""
    return task in Q_GENERATOR_REQUIRED_TASKS and step in {"generate", "all"}


def check_dependencies(task, step):
    """Check if required data files and directories exist for the task and step."""
    missing_deps = []

    if task == "precisewikiqa":
        # Check for wiki data file (required for generate step)
        if step in ["generate", "all"]:
            wiki_data_file = project_root / "data" / "wiki_data" / "doc_goodwiki_h_score.jsonl"
            if not wiki_data_file.exists():
                missing_deps.append(f"Wiki data file: {wiki_data_file}")
                missing_deps.append("Run: python data/download_data.py --precisewikiqa")

        # Create required directories
        qa_save_dir = project_root / "data" / "precise_qa" / "save"
        qa_save_dir.mkdir(parents=True, exist_ok=True)

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    elif task == "longwiki":
        # Check for wiki database (required for longwiki)
        if step in ["generate", "inference", "eval", "all"]:
            wiki_db = project_root / "data" / "wiki_data" / ".cache" / "enwiki-20230401.db"
            if not wiki_db.exists():
                missing_deps.append(f"Wikipedia database: {wiki_db}")
                missing_deps.append("Run: python data/download_data.py --longwiki")

        # Create required directories
        longwiki_save_dir = project_root / "data" / "longwiki" / "save"
        longwiki_save_dir.mkdir(parents=True, exist_ok=True)

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    elif task == "mixedentities":
        # Check for refusal test data
        if step in ["generate", "inference", "eval", "all"]:
            refusal_dir = project_root / "data" / "refusal_test"
            if not refusal_dir.exists():
                missing_deps.append(f"Refusal test data directory: {refusal_dir}")
                missing_deps.append("Run: python data/download_data.py --nonexistent_refusal")

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    elif task == "triviaqa":
        # TriviaQA has auto-download capability, so no strict dependency checks needed
        # Just ensure output directory exists
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    elif task == "naturalquestions":
        # Check for Natural Questions data file
        nq_data_file = project_root / "external" / "LLMsKnow" / "data" / "nq_wc_dataset.csv"
        if not nq_data_file.exists():
            missing_deps.append(f"Natural Questions data file: {nq_data_file}")
            missing_deps.append("This file should be available in external/LLMsKnow/data/")

        # Ensure output directory exists
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    elif task == "movies":
        # Check for Movies QA data file
        movies_data_file = project_root / "external" / "LLMsKnow" / "data" / "movie_qa_test.csv"
        if not movies_data_file.exists():
            missing_deps.append(f"Movies QA data file: {movies_data_file}")
            missing_deps.append("This file should be available in external/LLMsKnow/data/")

        # Ensure output directory exists
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    if missing_deps:
        logger.error("Missing required dependencies:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        logger.error("\nPlease run the data download script first:")
        logger.error("  python data/download_data.py --all")
        logger.error("Or for specific tasks:")
        logger.error("  python data/download_data.py --precisewikiqa")
        logger.error("  python data/download_data.py --longwiki")
        logger.error("  python data/download_data.py --nonexistent_refusal")
        return False

    return True


def get_task_name(task, **kwargs):
    """Generate the task name based on task type and parameters."""
    if task == "precisewikiqa":
        wiki_src = kwargs.get("wiki_src", "goodwiki")
        mode = kwargs.get("mode", "dynamic")
        return f"precise_wikiqa_{wiki_src}_{mode}"
    elif task == "longwiki":
        return "longwiki"
    elif task == "mixedentities":
        exp = kwargs.get("exp", "nonsense_all")
        return f"refusal_test_{exp}"
    elif task == "triviaqa":
        dataset_variant = kwargs.get("dataset_variant", "unfiltered")
        split = kwargs.get("split", "dev")
        return f"triviaqa_{dataset_variant}_{split}"
    elif task == "naturalquestions":
        return "natural_questions"
    elif task == "truthfulqa":
        return "truthfulqa"
    elif task == "hotpotqa":
        return "hotpotqa"
    elif task == "movies":
        return "movies"
    else:
        return task


def determine_generations_file_path(task_name, model, generations_file_path=None):
    """Determine the generations file path based on task and model."""
    if generations_file_path:
        return generations_file_path

    # Use the standard pattern: output/{task_name}/{model_name}/generation.jsonl
    model_name = model.split("/")[-1]
    return f"output/{task_name}/{model_name}/generation.jsonl"


def _is_null_like(value):
    """Return True for JSON nulls and NaN-like numeric values."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def sanitize_naturalquestions_eval_input(generations_file_path):
    """Create a type-safe JSONL copy for Natural Questions eval.

    The original evaluator assumes `answer` and `generation` are strings and calls
    `.lower()` on both. Some generation files contain null/NaN/non-string values,
    which causes runtime failures. Rows with null-like gold answers are excluded
    from evaluation, and a sanitized sidecar JSONL file is written and returned.
    """
    source_path = Path(generations_file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Generations file not found: {source_path}")

    sanitized_path = source_path.with_name(
        f"{source_path.stem}.sanitized_for_eval{source_path.suffix}"
    )

    read_rows = 0
    written_rows = 0
    excluded_rows = 0
    changed_lines = 0
    answer_null_like = 0
    answer_non_string = 0
    generation_null_like = 0
    generation_non_string = 0

    with source_path.open("r", encoding="utf-8") as infile, sanitized_path.open("w", encoding="utf-8") as outfile:
        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            read_rows += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source_path} at line {line_number}: {exc}"
                ) from exc

            line_changed = False

            answer_value = record.get("answer")
            # Exclude rows with missing gold answers from correctness evaluation.
            if _is_null_like(answer_value):
                answer_null_like += 1
                excluded_rows += 1
                continue
            if not isinstance(answer_value, str):
                line_changed = True
                answer_non_string += 1
                record["answer"] = str(answer_value)

            generation_value = record.get("generation")
            if not isinstance(generation_value, str):
                line_changed = True
                if _is_null_like(generation_value):
                    generation_null_like += 1
                    record["generation"] = ""
                else:
                    generation_non_string += 1
                    record["generation"] = str(generation_value)

            if line_changed:
                changed_lines += 1

            outfile.write(json.dumps(record) + "\n")
            written_rows += 1

    if written_rows == 0:
        raise ValueError(
            "Natural Questions eval sanitizer excluded all rows; no valid records remain for evaluation."
        )

    if excluded_rows or changed_lines:
        logger.warning(
            "Sanitized Natural Questions eval input at {}: kept {}/{} rows, excluded {} rows with null-like answers; "
            "modified {} kept rows (answer non-string={}, generation null-like={}, generation non-string={})",
            sanitized_path,
            written_rows,
            read_rows,
            excluded_rows,
            changed_lines,
            answer_non_string,
            generation_null_like,
            generation_non_string,
        )
    else:
        logger.info(
            "Natural Questions eval input is already type-safe: {} ({} rows)",
            source_path,
            read_rows,
        )

    return str(sanitized_path)


def run_task_step(step, task, model, **kwargs):
    """Run a single task step by calling the task module directly (no subprocess).

    The task code runs in the calling thread, which is required for Jupyter notebook
    compatibility (avoids tqdm RAM explosion).

    Args:
        step: The step to run (generate, inference, eval)
        task: The task name
        model: The model name
        **kwargs: Additional task-specific arguments
    """
    logger.info(f"Running {step} step for {task} task with model {model}")

    # Temporarily set QA chunk size env var if specified
    qa_chunk_size = kwargs.get("qa_generation_chunk_size")
    _old_qa_chunk = os.environ.get("QA_GENERATION_CHUNK_SIZE")
    if qa_chunk_size is not None:
        os.environ["QA_GENERATION_CHUNK_SIZE"] = str(int(qa_chunk_size))

    logger.info("=" * 80)
    logger.info(f"Starting {step} step:")
    logger.info("=" * 80)

    try:
        if task == "precisewikiqa":
            from tasks.shortform.precise_wikiqa import run_step as _run
            _run(
                step=step,
                model=model,
                wiki_src=kwargs.get("wiki_src", "goodwiki"),
                mode=kwargs.get("mode", "dynamic"),
                N=kwargs.get("N", 1),
                qa_output_path=kwargs.get("qa_output_path", ""),
                q_generator=kwargs.get("q_generator"),
                max_workers_qgen=kwargs.get("max_workers_qgen", 1),
                generations_file_path=kwargs.get("generations_file_path", ""),
                eval_results_path=kwargs.get("eval_results_path", ""),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_inference_tokens=kwargs.get("max_inference_tokens", 256),
                logger_type=kwargs.get("logger_type", "zarr"),
                activations_path=kwargs.get("activations_path"),
                log_file=kwargs.get("log_file"),
                resume=kwargs.get("resume", True),
                resume_eval=kwargs.get("resume_eval", True),
            )

        elif task == "longwiki":
            from tasks.longwiki.longwiki_main import run_step as _run
            _run(
                step=step,
                model=model,
                exp_mode=kwargs.get("exp_mode", "longwiki"),
                N=kwargs.get("N", 5),
                db_path=kwargs.get("db_path", "data/wiki_data/.cache/enwiki-20230401.db"),
                q_generator=kwargs.get("q_generator"),
                claim_extractor=kwargs.get("claim_extractor", "meta-llama/Llama-3.1-405B-Instruct-FP8"),
                abstain_evaluator=kwargs.get("abstain_evaluator", "meta-llama/Llama-3.1-70B-Instruct"),
                verifier=kwargs.get("verifier", "meta-llama/Llama-3.1-405B-Instruct-FP8"),
                k=kwargs.get("k", 32),
                max_tokens=kwargs.get("max_tokens", 1024),
                max_workers=kwargs.get("max_workers", 64),
                max_workers_qgen=kwargs.get("max_workers_qgen", 1),
                inference_method=kwargs.get("inference_method", "vllm"),
                logger_type=kwargs.get("logger_type", "zarr"),
                activations_path=kwargs.get("activations_path"),
                log_file=kwargs.get("log_file"),
                resume=kwargs.get("resume", True),
                resume_eval=kwargs.get("resume_eval", True),
            )

        elif task == "mixedentities":
            from tasks.refusal_test.nonsense_mixed_entities import run_step as _run
            _run(
                step=step,
                model=model,
                exp=kwargs.get("exp", "nonsense_all"),
                N=kwargs.get("N", 2000),
                seed=kwargs.get("seed", 1),
                inference_method=kwargs.get("inference_method", "vllm"),
                logger_type=kwargs.get("logger_type", "zarr"),
                activations_path=kwargs.get("activations_path"),
                log_file=kwargs.get("log_file"),
                resume=kwargs.get("resume", True),
                resume_eval=kwargs.get("resume_eval", True),
            )

        elif task == "triviaqa":
            if step == "generate":
                logger.info("TriviaQA doesn't have a generate step - skipping")
                return None
            from tasks.triviaqa.triviaqa import run_step as _run
            _run(
                step=step,
                model=model,
                dataset_variant=kwargs.get("dataset_variant", "unfiltered"),
                split=kwargs.get("split", "dev"),
                N=kwargs.get("N", 1000),
                data_dir=kwargs.get("data_dir", ""),
                auto_download=kwargs.get("auto_download", True),
                generations_file_path=kwargs.get("generations_file_path", ""),
                eval_results_path=kwargs.get("eval_results_path", ""),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_inference_tokens=kwargs.get("max_inference_tokens", 256),
                logger_type=kwargs.get("logger_type", "zarr"),
                activations_path=kwargs.get("activations_path"),
                log_file=kwargs.get("log_file"),
                resume=kwargs.get("resume", True),
                resume_eval=kwargs.get("resume_eval", True),
            )

        elif task == "naturalquestions":
            if step == "generate":
                logger.info("Natural Questions doesn't have a generate step - skipping")
                return None
            from tasks.llmsknow.natural_questions import run_step as _run

            generations_file_path = kwargs.get("generations_file_path")
            if step == "eval":
                if not generations_file_path:
                    model_name = model.split("/")[-1]
                    output_base_dir = kwargs.get("output_dir") or "output"
                    generations_file_path = (
                        f"{output_base_dir}/natural_questions/{model_name}/generation.jsonl"
                    )
                generations_file_path = sanitize_naturalquestions_eval_input(generations_file_path)

            _run(
                step=step,
                model=model,
                data_dir=kwargs.get("data_dir", "external/LLMsKnow/data"),
                output_dir=kwargs.get("output_dir", "output"),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_tokens=kwargs.get("max_tokens", 64),
                temperature=kwargs.get("temperature", 0.0),
                N=kwargs.get("N"),
                generations_file_path=generations_file_path,
                eval_results_path=kwargs.get("eval_results_path"),
                log_file=kwargs.get("log_file"),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
            )

        elif task == "truthfulqa":
            if step == "generate":
                logger.info("TruthfulQA doesn't have a generate step — it's a fixed HuggingFace dataset")
                return None
            from tasks.shortform.truthfulqa import run_step as _run
            _run(
                step=step,
                model=model,
                output_dir=kwargs.get("output_dir", "output"),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_tokens=kwargs.get("max_tokens", 128),
                temperature=kwargs.get("temperature", 0.0),
                N=kwargs.get("N"),
                generations_file_path=kwargs.get("generations_file_path"),
                eval_results_path=kwargs.get("eval_results_path"),
                log_file=kwargs.get("log_file"),
                logger_type=kwargs.get("logger_type", "lmdb"),
                activations_path=kwargs.get("activations_path"),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
                resume=kwargs.get("resume", True),
            )

        elif task == "hotpotqa":
            if step == "generate":
                logger.info("HotpotQA doesn't have a generate step — loaded directly from HuggingFace")
                return None
            from tasks.llmsknow.hotpotqa import run_step as _run
            _run(
                step=step,
                model=model,
                output_dir=kwargs.get("output_dir", "output"),
                split=kwargs.get("split", "validation"),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_tokens=kwargs.get("max_inference_tokens") or kwargs.get("max_tokens", 128),
                temperature=kwargs.get("temperature", 0.0),
                N=kwargs.get("N"),
                generations_file_path=kwargs.get("generations_file_path"),
                eval_results_path=kwargs.get("eval_results_path"),
                log_file=kwargs.get("log_file"),
                logger_type=kwargs.get("logger_type", "lmdb"),
                activations_path=kwargs.get("activations_path"),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
                resume=kwargs.get("resume", True),
                llm_evaluator=kwargs.get("llm_evaluator"),
                batch_size=kwargs.get("batch_size"),
            )

        elif task == "movies":
            if step == "generate":
                logger.info("Movies QA doesn't have a generate step -- loaded directly from CSV")
                return None
            from tasks.llmsknow.movies import run_step as _run
            _run(
                step=step,
                model=model,
                data_dir=kwargs.get("data_dir", "external/LLMsKnow/data"),
                output_dir=kwargs.get("output_dir", "output"),
                split=kwargs.get("split", "test"),
                inference_method=kwargs.get("inference_method", "vllm"),
                max_tokens=kwargs.get("max_inference_tokens") or kwargs.get("max_tokens", 128),
                temperature=kwargs.get("temperature", 0.0),
                N=kwargs.get("N"),
                generations_file_path=kwargs.get("generations_file_path"),
                eval_results_path=kwargs.get("eval_results_path"),
                log_file=kwargs.get("log_file"),
                logger_type=kwargs.get("logger_type", "lmdb"),
                activations_path=kwargs.get("activations_path"),
                quick_debug_mode=kwargs.get("quick_debug_mode", False),
                resume=kwargs.get("resume", True),
                llm_evaluator=kwargs.get("llm_evaluator"),
                batch_size=kwargs.get("batch_size"),
            )

        else:
            raise ValueError(f"Unknown task: {task}")

    finally:
        # Restore QA chunk size env var
        if qa_chunk_size is not None:
            if _old_qa_chunk is None:
                os.environ.pop("QA_GENERATION_CHUNK_SIZE", None)
            else:
                os.environ["QA_GENERATION_CHUNK_SIZE"] = _old_qa_chunk

    logger.info("=" * 80)
    logger.success(f"Task {step} completed successfully")


def run_experiment(
    step, task, model,
    *,
    host="0.0.0.0",
    port=8000,
    server_startup_timeout=None,
    logger_type="zarr",
    activations_path=None,
    log_file=None,
    log_level=None,
    max_model_len=None,
    gpu_memory_utilization=None,
    N=1,
    wiki_src="goodwiki",
    mode="dynamic",
    inference_method="vllm",
    max_inference_tokens=256,
    generations_file_path=None,
    eval_results_path=None,
    q_generator=None,
    qa_output_path=None,
    quick_debug_mode=False,
    max_workers_qgen=1,
    qa_generation_chunk_size=None,
    db_path=None,
    claim_extractor=None,
    abstain_evaluator=None,
    verifier=None,
    k=32,
    max_tokens=1024,
    max_workers=64,
    exp="nonsense_all",
    seed=1,
    dataset_variant="unfiltered",
    split="dev",
    data_dir=None,
    auto_download=True,
    output_dir=None,
    temperature=0.0,
    resume=True,
    resume_eval=True,
    llm_evaluator=None,
    batch_size=None,
):
    """Run a HalluLens experiment with automatic server management.

    This is the main Python-callable entry point. The task code (inference/eval/generate)
    always runs in the calling thread — safe for Jupyter notebooks.
    Only the vLLM/llama.cpp server is started as a subprocess.

    Args:
        step: "generate", "inference", "eval", or "all"
        task: "precisewikiqa", "longwiki", "mixedentities", "triviaqa", or "naturalquestions"
        model: Model name or path (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        log_level: Console log verbosity (e.g. "DEBUG", "INFO", "WARNING").
                   Falls back to HALLULENS_LOG_LEVEL env var, then WARNING.
        All other arguments mirror the CLI flags.
    """
    # Apply log level if caller specified one (Python API entry point)
    if log_level:
        configure_logging(log_level, force=True)

    # Change to project root so task modules resolve relative paths correctly
    os.chdir(str(project_root))

    if not check_dependencies(task, step):
        raise RuntimeError(f"Missing dependencies for task={task}, step={step}")

    if requires_q_generator(step, task) and not q_generator:
        raise ValueError(
            "q_generator is required for generate/all runs on precisewikiqa and longwiki. "
            "Pass --q_generator in CLI or q_generator=... in run_experiment()."
        )

    log_file_path = log_file or "server.log"
    logger.info(f"Server log file: {log_file_path}")

    logger.info("=" * 80)
    logger.info("HalluLens Task Runner with Automatic Server Management")
    logger.info("Server will be started automatically when needed")
    logger.info("=" * 80)

    # Determine which model needs the server
    # When batch_size is set, inference runs directly via ModelAdapter — no server needed.
    server_model = None
    if batch_size and step in ["inference", "all"]:
        logger.info("batch_size is set — skipping server startup (using ModelAdapter directly)")
    elif step in ["generate", "all"]:
        server_model = q_generator or model
    elif step == "inference":
        server_model = model

    server_manager = None
    server_was_running = False

    if server_model and inference_method == "vllm":
        server_was_running = lm.check_server_health(f"http://{host}:{port}")

        if not server_was_running:
            model_lower = server_model.lower()
            is_gguf = (
                model_lower.endswith('.gguf') or
                '/gguf' in model_lower or
                'gguf/' in model_lower or
                '-gguf' in model_lower or
                'q6_k' in model_lower or
                'q4_k' in model_lower or
                'iq3_m' in model_lower or
                'iq4' in model_lower
            )

            if is_gguf:
                logger.info(f"GGUF model detected - using llama.cpp server instead of vLLM")
                logger.info(f"Starting llama.cpp server for model: {server_model}")

                env = os.environ.copy()
                if activations_path:
                    env["ACTIVATION_STORAGE_PATH"] = activations_path
                if logger_type:
                    env["ACTIVATION_LOGGER_TYPE"] = logger_type
                if log_file_path:
                    env["SERVER_LOG_FILE"] = log_file_path
                env["DEFAULT_MODEL"] = server_model

                if '/' in server_model:
                    gguf_dir = os.path.dirname(server_model)
                    env["GGUF_MODELS_DIR"] = gguf_dir
                    logger.info(f"GGUF models directory: {gguf_dir}")

                cmd = [sys.executable, "-m", "uvicorn", "activation_logging.server:app",
                       "--host", host, "--port", str(port)]

                logger.info(f"Server command: {' '.join(cmd)}")
                logger.info(f"Environment: DEFAULT_MODEL={server_model}")

                server_process = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                logger.info("Waiting for llama.cpp server to be ready...")
                max_wait = server_startup_timeout or 120
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    if lm.check_server_health(f"http://{host}:{port}"):
                        logger.success(f"Llama.cpp server started at http://{host}:{port}")

                        class _SimpleServerManager:
                            def __init__(self, process):
                                self.server_process = process
                            def stop_server(self):
                                if self.server_process:
                                    logger.info("Terminating llama.cpp server...")
                                    self.server_process.terminate()
                                    try:
                                        self.server_process.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        logger.warning("Server didn't terminate gracefully, killing...")
                                        self.server_process.kill()

                        server_manager = _SimpleServerManager(server_process)
                        break
                    time.sleep(3)
                else:
                    logger.error("Failed to start llama.cpp server within timeout")
                    server_process.terminate()
                    raise RuntimeError("Failed to start llama.cpp server within timeout")
            else:
                logger.info(f"Starting vLLM server for model: {server_model}")
                server_manager = lm.ServerManager(
                    model=server_model,
                    host=host,
                    port=port,
                    logger_type=logger_type,
                    activations_path=activations_path,
                    log_file_path=log_file_path,
                    startup_timeout=server_startup_timeout,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
                server_manager.start_server()
                lm.set_server_manager(server_manager)
                logger.success(f"Server started at http://{host}:{port}")
        else:
            logger.info(f"Server already running at http://{host}:{port}")
            logger.warning("Note: Using existing server (not managed by this script)")

    try:
        task_kwargs = dict(
            N=N,
            wiki_src=wiki_src,
            mode=mode,
            inference_method=inference_method,
            max_inference_tokens=max_inference_tokens,
            generations_file_path=generations_file_path,
            eval_results_path=eval_results_path,
            q_generator=q_generator,
            qa_output_path=qa_output_path,
            quick_debug_mode=quick_debug_mode,
            max_workers_qgen=max_workers_qgen,
            qa_generation_chunk_size=qa_generation_chunk_size,
            logger_type=logger_type,
            activations_path=activations_path,
            log_file=log_file_path,
            resume=resume,
            resume_eval=resume_eval,
            # LongWiki specific
            db_path=db_path,
            claim_extractor=claim_extractor,
            abstain_evaluator=abstain_evaluator,
            verifier=verifier,
            k=k,
            max_tokens=max_tokens,
            max_workers=max_workers,
            # Mixed entities specific
            exp=exp,
            seed=seed,
            # TriviaQA specific
            dataset_variant=dataset_variant,
            split=split,
            data_dir=data_dir,
            auto_download=auto_download,
            # Natural Questions specific
            output_dir=output_dir,
            temperature=temperature,
            # HotpotQA LLM eval specific
            llm_evaluator=llm_evaluator,
            # Batched inference (HotpotQA adapter path)
            batch_size=batch_size,
        )

        if step == "all":
            no_generate_tasks = {"triviaqa", "naturalquestions", "truthfulqa", "hotpotqa", "movies"}
            steps = ["inference", "eval"] if task in no_generate_tasks else ["generate", "inference", "eval"]
            for s in steps:
                logger.info(f"Running step {s}...")
                result = run_task_step(s, task, model, **task_kwargs)
                if result is None:
                    logger.info(f"Step {s} was skipped")
        else:
            result = run_task_step(step, task, model, **task_kwargs)
            if result is None:
                logger.info(f"Step {step} was skipped")

        logger.success("All steps completed successfully!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise
    finally:
        if server_manager:
            logger.info("Stopping server...")
            server_manager.stop_server()
            lm.set_server_manager(None)
            logger.success("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Run HalluLens tasks with automatic server management")

    # Required arguments
    parser.add_argument("--step", required=True, choices=["generate", "inference", "eval", "all"],
                       help="Which step to run (or 'all' for all steps)")
    parser.add_argument("--task", required=True, choices=["precisewikiqa", "longwiki", "mixedentities", "triviaqa", "naturalquestions", "truthfulqa", "hotpotqa", "movies"],
                       help="Which task to run")
    parser.add_argument("--model", required=True, help="Model to use")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--server-startup-timeout",
        type=int,
        default=None,
        help=(
            "Seconds to wait for the inference server to become healthy. "
            "If omitted, defaults to 600s (or env SERVER_STARTUP_TIMEOUT/VLLM_SERVER_STARTUP_TIMEOUT if set)."
        ),
    )
    parser.add_argument("--logger-type", default="lmdb", choices=["lmdb", "json", "zarr"],
                       help="Activation logger type")
    parser.add_argument("--activations-path", help="Path for storing activations")
    parser.add_argument("--log-file", help="Path for server behavior logs (if not specified and step is inference, will be placed in same directory as generations file)")

    # Task-specific arguments
    parser.add_argument("--N", type=int, default=1, help="Number of samples")
    parser.add_argument("--wiki_src", default="goodwiki", help="Wiki source for precisewikiqa")
    parser.add_argument("--mode", default="dynamic", help="Mode for precisewikiqa")
    parser.add_argument("--inference_method", default="vllm", help="Inference method")
    parser.add_argument("--max_inference_tokens", type=int, default=256, help="Maximum number of tokens to generate per inference")
    parser.add_argument("--generations_file_path", help="Path for generations file")
    parser.add_argument("--eval_results_path", help="Path for evaluation results (default: co-located with generations file)")
    parser.add_argument(
        "--q_generator",
        help="Question generator model (required for generate/all on precisewikiqa and longwiki)",
    )
    parser.add_argument("--qa_output_path", help="Custom QA output path")
    parser.add_argument("--quick_debug_mode", action="store_true", help="Quick debug mode (first 5 questions)")
    parser.add_argument("--max-workers-qgen", type=int, default=1, help="Maximum concurrent requests for question generation (default: 1)")
    parser.add_argument(
        "--qa-generation-chunk-size",
        type=int,
        default=None,
        help=(
            "Chunk size for QA generation batching (sets env QA_GENERATION_CHUNK_SIZE). "
            "Increase this to >= --max-workers-qgen to fully utilize concurrency. "
            "Default is controlled by the code (typically 5)."
        ),
    )

    # LongWiki specific
    parser.add_argument("--db_path", help="Database path for longwiki")
    parser.add_argument("--claim_extractor", help="Claim extractor model for longwiki")
    parser.add_argument("--abstain_evaluator", help="Abstain evaluator model for longwiki")
    parser.add_argument("--verifier", help="Verifier model for longwiki")
    parser.add_argument("--k", type=int, default=32, help="K parameter for longwiki")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for longwiki")
    parser.add_argument("--max_workers", type=int, default=64, help="Max workers for longwiki")

    # Mixed entities specific
    parser.add_argument("--exp", default="nonsense_all", help="Experiment name for mixed entities")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # TriviaQA specific
    parser.add_argument("--dataset_variant", default="unfiltered", help="TriviaQA dataset variant (filtered/unfiltered)")
    parser.add_argument("--split", default="dev", help="TriviaQA split (train/dev)")
    parser.add_argument("--data_dir", help="Directory containing TriviaQA data files")
    parser.add_argument("--auto_download", action="store_true", default=True, help="Automatically download TriviaQA data if not found")

    # Natural Questions specific
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for Natural Questions")
    parser.add_argument("--output_dir", help="Base output directory")

    # Resume control
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic resume from existing generations file (inference and evaluation)")
    parser.add_argument("--no-resume-eval", action="store_true", help="Disable automatic resume specifically for evaluation step")

    # Logging
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL",
                 "trace", "debug", "info", "success", "warning", "error", "critical"],
        help="Console log verbosity (default: WARNING). Also settable via HALLULENS_LOG_LEVEL env var.",
    )

    args = parser.parse_args()

    if requires_q_generator(args.step, args.task) and not args.q_generator:
        parser.error(
            "--q_generator is required when --step is 'generate' or 'all' for "
            "--task precisewikiqa or --task longwiki"
        )

    # Re-configure logging with the user-requested level (if provided)
    if args.log_level:
        configure_logging(args.log_level, force=True)

    try:
        run_experiment(
            step=args.step,
            task=args.task,
            model=args.model,
            host=args.host,
            port=args.port,
            server_startup_timeout=args.server_startup_timeout,
            logger_type=args.logger_type,
            activations_path=args.activations_path,
            log_file=args.log_file,
            N=args.N,
            wiki_src=args.wiki_src,
            mode=args.mode,
            inference_method=args.inference_method,
            max_inference_tokens=args.max_inference_tokens,
            generations_file_path=args.generations_file_path,
            eval_results_path=args.eval_results_path,
            q_generator=args.q_generator,
            qa_output_path=args.qa_output_path,
            quick_debug_mode=args.quick_debug_mode,
            max_workers_qgen=args.max_workers_qgen,
            qa_generation_chunk_size=args.qa_generation_chunk_size,
            db_path=args.db_path,
            claim_extractor=args.claim_extractor,
            abstain_evaluator=args.abstain_evaluator,
            verifier=args.verifier,
            k=args.k,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            exp=args.exp,
            seed=args.seed,
            dataset_variant=args.dataset_variant,
            split=args.split,
            data_dir=args.data_dir,
            auto_download=args.auto_download,
            output_dir=args.output_dir,
            temperature=args.temperature,
            resume=not args.no_resume,
            resume_eval=not args.no_resume_eval,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
