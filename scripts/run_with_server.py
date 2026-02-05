#!/usr/bin/env python3
"""
Unified script to run HalluLens tasks with automatic server management.

This script automatically starts the activation logging server and runs one of the three
generation steps, eliminating the need for manual server startup.

Usage:
    python scripts/run_with_server.py --step [generate|inference|eval] [task_options...]

Examples:
    # Generate prompts (PreciseWikiQA) - uses --model for question generation
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100

    # Generate with different model for questions vs inference
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --q_generator meta-llama/Llama-3.1-70B-Instruct --N 100

    # Run inference (PreciseWikiQA)
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct

    # Run evaluation (PreciseWikiQA)
    python scripts/run_with_server.py --step eval --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct

    # Run all steps in sequence (PreciseWikiQA)
    python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100

    # TriviaQA inference and evaluation
    python scripts/run_with_server.py --step all --task triviaqa --model meta-llama/Llama-3.1-8B-Instruct --N 1000

    # TriviaQA with custom dataset variant
    python scripts/run_with_server.py --step inference --task triviaqa --model meta-llama/Llama-3.1-8B-Instruct --dataset_variant unfiltered --split dev

    # Natural Questions inference and evaluation
    python scripts/run_with_server.py --step all --task naturalquestions --model meta-llama/Llama-3.1-8B-Instruct --N 1000

    # Natural Questions with custom settings
    python scripts/run_with_server.py --step inference --task naturalquestions --model meta-llama/Llama-3.1-8B-Instruct --max_tokens 64 --temperature 0.0

    # Custom server log file
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --log-file custom/server.log

    # Question generation with increased concurrency (8 parallel requests)
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100 --max-workers-qgen 8

    # LongWiki with concurrent question generation
    python scripts/run_with_server.py --step generate --task longwiki --model meta-llama/Llama-3.1-70B-Instruct --N 50 --max-workers-qgen 4

    # name for the q generator 
     Llama-3.3-70B-Instruct-Q6_K_L 
    """

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ServerManager from utils.lm
from utils import lm

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
    else:
        return task

def determine_generations_file_path(task_name, model, generations_file_path=None):
    """Determine the generations file path based on task and model."""
    if generations_file_path:
        return generations_file_path

    # Use the standard pattern: output/{task_name}/{model_name}/generation.jsonl
    model_name = model.split("/")[-1]
    return f"output/{task_name}/{model_name}/generation.jsonl"

# Use ServerManager from utils.lm
ServerManager = lm.ServerManager
get_server_manager = lm.get_server_manager
set_server_manager = lm.set_server_manager

def run_task_step(step, task, model, **kwargs):
    """Run the specified task step.

    Args:
        step: The step to run (generate, inference, eval)
        task: The task name
        model: The model name
        **kwargs: Additional task-specific arguments
    """
    logger.info(f"Running {step} step for {task} task with model {model}")

    # Use default environment
    env = os.environ.copy()

    # Build command based on task and step
    if task == "precisewikiqa":
        # If the user requested TensorRT-LLM for question generation, bypass the
        # Meta-authored task's internal qgen (which uses utils.lm/vLLM) and instead
        # produce the same QA JSONL output directly.
        if step == "generate" and kwargs.get("qgen_backend") == "tensorrt-llm":
            wiki_src = kwargs.get("wiki_src", "goodwiki")
            mode = kwargs.get("mode", "dynamic")
            model_name = model.split("/")[-1]
            qa_output_path = kwargs.get("qa_output_path")
            if not qa_output_path:
                qa_output_path = f"data/precise_qa/save/qa_{wiki_src}_{model_name}_{mode}.jsonl"

            cmd = [
                sys.executable,
                "-m",
                "question_generation.wikiqa_trtllm",
                "--task",
                "precise",
                "--wiki_input_path",
                str(project_root / "data" / "wiki_data" / "doc_goodwiki_h_score.jsonl"),
                "--output_path",
                qa_output_path,
                "--N",
                str(kwargs.get("N", 1)),
                "--q_model",
                (kwargs.get("q_generator") or "nvidia/Llama-3.3-70B-Instruct-FP8"),
                "--seed",
                str(kwargs.get("seed", 1)),
            ]
            # Keep chunk size / sampling defaults inside the module.
            if kwargs.get("max_workers_qgen"):
                # Currently implemented as batching, not threads; accept the flag
                # for forwards compatibility without breaking CLI.
                pass
        else:
            cmd = [sys.executable, "-m", "tasks.shortform.precise_wikiqa"]

        is_precise_task_entrypoint = "tasks.shortform.precise_wikiqa" in " ".join(cmd)

        # Add step-specific flags
        if is_precise_task_entrypoint:
            if step == "generate":
                cmd.append("--do_generate_prompt")
            elif step == "inference":
                cmd.append("--do_inference")
            elif step == "eval":
                cmd.append("--do_eval")

        # Add common parameters (only for the Meta task entrypoint)
        if is_precise_task_entrypoint:
            cmd.extend([
                "--model", model,
                "--wiki_src", kwargs.get("wiki_src", "goodwiki"),
                "--mode", kwargs.get("mode", "dynamic"),
                "--inference_method", kwargs.get("inference_method", "vllm"),
                "--max_inference_tokens", str(kwargs.get("max_inference_tokens", 256)),
                "--N", str(kwargs.get("N", 1))
            ])

        # Add optional parameters (only for the Meta task entrypoint)
        if is_precise_task_entrypoint:
            if kwargs.get("generations_file_path"):
                cmd.extend(["--generations_file_path", kwargs["generations_file_path"]])
            if kwargs.get("eval_results_path"):
                cmd.extend(["--eval_results_path", kwargs["eval_results_path"]])
            # Use --model for question generation if q_generator not explicitly specified
            if step == "generate":
                q_gen = kwargs.get("q_generator") or model
                cmd.extend(["--q_generator", q_gen])
            elif kwargs.get("q_generator"):
                cmd.extend(["--q_generator", kwargs["q_generator"]])
            if kwargs.get("qa_output_path"):
                cmd.extend(["--qa_output_path", kwargs["qa_output_path"]])
            if kwargs.get("quick_debug_mode"):
                cmd.append("--quick_debug_mode")
            if kwargs.get("max_workers_qgen"):
                cmd.extend(["--max_workers_qgen", str(kwargs["max_workers_qgen"])])

            # Add activation logging parameters
            if kwargs.get("logger_type"):
                cmd.extend(["--logger_type", kwargs["logger_type"]])
            if kwargs.get("activations_path"):
                cmd.extend(["--activations_path", kwargs["activations_path"]])
            if kwargs.get("log_file"):
                cmd.extend(["--log_file", kwargs["log_file"]])

            # Add resume control
            if not kwargs.get("resume", True):
                cmd.append("--no-resume")
            if not kwargs.get("resume_eval", True):
                cmd.append("--no-resume-eval")

    elif task == "longwiki":
        if step == "generate" and kwargs.get("qgen_backend") == "tensorrt-llm":
            model_name = model.split("/")[-1]
            qa_output_path = f"data/longwiki/save/longwiki_{model_name}.jsonl"
            if kwargs.get("qa_output_path"):
                qa_output_path = kwargs["qa_output_path"]

            cmd = [
                sys.executable,
                "-m",
                "question_generation.wikiqa_trtllm",
                "--task",
                "longform",
                "--wiki_input_path",
                str(project_root / "data" / "wiki_data" / "doc_goodwiki_h_score.jsonl"),
                "--output_path",
                qa_output_path,
                "--N",
                str(kwargs.get("N", 1)),
                "--q_model",
                (kwargs.get("q_generator") or "nvidia/Llama-3.3-70B-Instruct-FP8"),
                "--seed",
                str(kwargs.get("seed", 1)),
                "--min_ref_chars",
                "500",
                "--max_ref_chars",
                "750",
                "--low_level",
                "5",
                "--high_level",
                "10",
            ]
        else:
            cmd = [sys.executable, "-m", "tasks.longwiki.longwiki_main"]

        is_longwiki_task_entrypoint = "tasks.longwiki.longwiki_main" in " ".join(cmd)

        # Add step-specific flags
        if is_longwiki_task_entrypoint:
            if step == "generate":
                cmd.append("--do_generate_prompt")
            elif step == "inference":
                cmd.append("--do_inference")
            elif step == "eval":
                cmd.append("--do_eval")

        if is_longwiki_task_entrypoint:
            # Add common parameters
            cmd.extend([
                "--model", model,
                "--exp_mode", "longwiki",
                "--inference_method", kwargs.get("inference_method", "vllm"),
                "--N", str(kwargs.get("N", 5))
            ])

            # Add required parameters for longwiki
            db_path = kwargs.get("db_path", "data/wiki_data/.cache/enwiki-20230401.db")
            cmd.extend(["--db_path", db_path])

            if kwargs.get("q_generator"):
                cmd.extend(["--q_generator", kwargs["q_generator"]])
            if kwargs.get("claim_extractor"):
                cmd.extend(["--claim_extractor", kwargs["claim_extractor"]])
            if kwargs.get("abstain_evaluator"):
                cmd.extend(["--abstain_evaluator", kwargs["abstain_evaluator"]])
            if kwargs.get("verifier"):
                cmd.extend(["--verifier", kwargs["verifier"]])
            if kwargs.get("k"):
                cmd.extend(["--k", str(kwargs["k"])])
            if kwargs.get("max_tokens"):
                cmd.extend(["--max_tokens", str(kwargs["max_tokens"])])
            if kwargs.get("max_workers"):
                cmd.extend(["--max_workers", str(kwargs["max_workers"])])
            if kwargs.get("max_workers_qgen"):
                cmd.extend(["--max_workers_qgen", str(kwargs["max_workers_qgen"])])

            # Add activation logging parameters
            if kwargs.get("logger_type"):
                cmd.extend(["--logger_type", kwargs["logger_type"]])
            if kwargs.get("activations_path"):
                cmd.extend(["--activations_path", kwargs["activations_path"]])
            if kwargs.get("log_file"):
                cmd.extend(["--log_file", kwargs["log_file"]])

    elif task == "mixedentities":
        cmd = [sys.executable, "-m", "tasks.refusal_test.nonsense_mixed_entities"]

        # Add step-specific flags
        if step == "generate":
            cmd.append("--do_generate_prompt")
        elif step == "inference":
            cmd.append("--do_inference")
        elif step == "eval":
            cmd.append("--do_eval")

        # Add parameters
        cmd.extend([
            "--tested_model", model,
            "--exp", kwargs.get("exp", "nonsense_all"),
            "--N", str(kwargs.get("N", 2000)),
            "--seed", str(kwargs.get("seed", 1)),
            "--inference_method", kwargs.get("inference_method", "vllm")
        ])

        # Add activation logging parameters
        if kwargs.get("logger_type"):
            cmd.extend(["--logger_type", kwargs["logger_type"]])
        if kwargs.get("activations_path"):
            cmd.extend(["--activations_path", kwargs["activations_path"]])
        if kwargs.get("log_file"):
            cmd.extend(["--log_file", kwargs["log_file"]])

        # Add resume control
        if not kwargs.get("resume", True):
            cmd.append("--no-resume")
        if not kwargs.get("resume_eval", True):
            cmd.append("--no-resume-eval")

    elif task == "triviaqa":
        cmd = [sys.executable, "-m", "tasks.triviaqa.triviaqa"]

        # Add step-specific flags (TriviaQA only has inference and eval, no generate)
        if step == "generate":
            # TriviaQA doesn't have a generate step, skip
            logger.info("TriviaQA doesn't have a generate step - skipping")
            return None
        elif step == "inference":
            cmd.append("--do_inference")
        elif step == "eval":
            cmd.append("--do_eval")

        # Add common parameters
        cmd.extend([
            "--model", model,
            "--dataset_variant", kwargs.get("dataset_variant", "unfiltered"),
            "--split", kwargs.get("split", "dev"),
            "--inference_method", kwargs.get("inference_method", "vllm"),
            "--max_inference_tokens", str(kwargs.get("max_inference_tokens", 256)),
            "--N", str(kwargs.get("N", 1000))
        ])

        # Add optional parameters
        if kwargs.get("generations_file_path"):
            cmd.extend(["--generations_file_path", kwargs["generations_file_path"]])
        if kwargs.get("eval_results_path"):
            cmd.extend(["--eval_results_path", kwargs["eval_results_path"]])
        if kwargs.get("data_dir"):
            cmd.extend(["--data_dir", kwargs["data_dir"]])
        if kwargs.get("quick_debug_mode"):
            cmd.append("--quick_debug_mode")
        if not kwargs.get("auto_download", True):
            cmd.append("--no_auto_download")

        # Add activation logging parameters
        if kwargs.get("logger_type"):
            cmd.extend(["--logger_type", kwargs["logger_type"]])
        if kwargs.get("activations_path"):
            cmd.extend(["--activations_path", kwargs["activations_path"]])
        if kwargs.get("log_file"):
            cmd.extend(["--log_file", kwargs["log_file"]])

        # Add resume control
        if not kwargs.get("resume", True):
            cmd.append("--no-resume")
        if not kwargs.get("resume_eval", True):
            cmd.append("--no-resume-eval")

    elif task == "naturalquestions":
        cmd = [sys.executable, "-m", "tasks.llmsknow.natural_questions"]

        # Add step-specific flags (NQ only has inference and eval, no generate)
        if step == "generate":
            # Natural Questions doesn't have a generate step, skip
            logger.info("Natural Questions doesn't have a generate step - skipping")
            return None
        elif step == "inference":
            cmd.append("--do_inference")
        elif step == "eval":
            cmd.append("--do_eval")

        # Add common parameters
        cmd.extend([
            "--model", model,
            "--inference_method", kwargs.get("inference_method", "vllm"),
            "--max_tokens", str(kwargs.get("max_tokens", 64)),
            "--temperature", str(kwargs.get("temperature", 0.0)),
            "--N", str(kwargs.get("N")) if kwargs.get("N") is not None else "--N"
        ])

        # Remove --N flag if N is None (process all samples)
        if kwargs.get("N") is None:
            # Remove the last two items (--N and its value)
            cmd = cmd[:-2]

        # Add optional parameters
        if kwargs.get("data_dir"):
            cmd.extend(["--data_dir", kwargs["data_dir"]])
        if kwargs.get("output_dir"):
            cmd.extend(["--output_dir", kwargs["output_dir"]])
        if kwargs.get("generations_file_path"):
            cmd.extend(["--generations_file_path", kwargs["generations_file_path"]])
        if kwargs.get("eval_results_path"):
            cmd.extend(["--eval_results_path", kwargs["eval_results_path"]])
        if kwargs.get("log_file"):
            cmd.extend(["--log_file", kwargs["log_file"]])
        if kwargs.get("quick_debug_mode"):
            cmd.append("--quick_debug_mode")

    else:
        raise ValueError(f"Unknown task: {task}")

    logger.info(f"Task command: {' '.join(cmd)}")

    # Handle case where command is None (e.g., TriviaQA generate step)
    if cmd is None:
        logger.info(f"Step {step} skipped for task {task}")
        return None

    # Run the task with environment variables, streaming output in real-time
    logger.info("=" * 80)
    logger.info(f"Starting {step} step - output will be shown below:")
    logger.info("=" * 80)
    
    # Use Popen to stream output in real-time
    # Important: run tasks from the repository root so any relative paths
    # (e.g., data/precise_qa/save/...) resolve consistently.
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='', flush=True)
    
    # Wait for process to complete
    return_code = process.wait()
    
    logger.info("=" * 80)
    
    if return_code != 0:
        logger.error(f"Task failed with return code {return_code}")
        raise RuntimeError(f"Task execution failed with return code {return_code}")

    logger.success(f"Task {step} completed successfully")
    
    # Return a simple result object for compatibility
    class Result:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""
    
    return Result(return_code)

def main():
    parser = argparse.ArgumentParser(description="Run HalluLens tasks with automatic server management")
    
    # Required arguments
    parser.add_argument("--step", required=True, choices=["generate", "inference", "eval", "all"],
                       help="Which step to run (or 'all' for all steps)")
    parser.add_argument("--task", required=True, choices=["precisewikiqa", "longwiki", "mixedentities", "triviaqa", "naturalquestions"],
                       help="Which task to run")
    parser.add_argument("--model", required=True, help="Model to use")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--logger-type", default="lmdb", choices=["lmdb", "json"],
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
    parser.add_argument("--q_generator", help="Question generator model")
    parser.add_argument(
        "--qgen-backend",
        default=os.environ.get("HALLULENS_QGEN_BACKEND", "vllm"),
        choices=["vllm", "tensorrt-llm"],
        help="Backend used only for question generation (default: vllm).",
    )
    parser.add_argument("--qa_output_path", help="Custom QA output path")
    parser.add_argument("--quick_debug_mode", action="store_true", help="Quick debug mode (first 5 questions)")
    parser.add_argument("--max-workers-qgen", type=int, default=1, help="Maximum concurrent requests for question generation (default: 1)")

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

    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies(args.task, args.step):
        sys.exit(1)

    # Determine log file path for server behavior logs
    log_file_path = args.log_file
    if not log_file_path:
        # Default to server.log in current working directory
        log_file_path = "server.log"
    
    logger.info(f"Server log file: {log_file_path}")

    logger.info("=" * 80)
    logger.info("HalluLens Task Runner with Automatic Server Management")
    logger.info("Server will be started automatically when needed")
    logger.info("=" * 80)

    # Determine which model needs the server
    server_model = None
    if args.step == "generate":
        # For generate step, only start server when qgen uses vLLM.
        if args.qgen_backend == "vllm":
            server_model = args.q_generator or args.model
    elif args.step == "all":
        # For all steps, we start the server for inference (activation logging happens there).
        if args.inference_method == "vllm":
            server_model = args.model
    elif args.step in ["inference"]:
        # For inference step, use main model
        server_model = args.model

    # Start server if needed and not already running
    server_manager = None
    server_was_running = False

    if server_model and args.inference_method == "vllm":
        server_was_running = lm.check_server_health(f"http://{args.host}:{args.port}")

        if not server_was_running:
            # Check if it's a GGUF model (file or directory)
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
                logger.info(f"🔧 GGUF model detected - using llama.cpp server instead of vLLM")
                logger.info(f"🚀 Starting llama.cpp server for model: {server_model}")
                
                # Set environment variables for server configuration
                env = os.environ.copy()
                if args.activations_path:
                    env["ACTIVATION_STORAGE_PATH"] = args.activations_path
                if args.logger_type:
                    env["ACTIVATION_LOGGER_TYPE"] = args.logger_type
                if log_file_path:
                    env["SERVER_LOG_FILE"] = log_file_path
                
                # Set the default model to the GGUF model path
                env["DEFAULT_MODEL"] = server_model
                
                # Set GGUF models directory if model contains a directory path
                if '/' in server_model:
                    gguf_dir = os.path.dirname(server_model)
                    env["GGUF_MODELS_DIR"] = gguf_dir
                    logger.info(f"GGUF models directory: {gguf_dir}")
                
                # Build server command for llama.cpp
                cmd = [sys.executable, "-m", "uvicorn", "activation_logging.server:app",
                       "--host", args.host,
                       "--port", str(args.port)]
                    
                logger.info(f"Server command: {' '.join(cmd)}")
                logger.info(f"Environment: DEFAULT_MODEL={server_model}")
                
                # Start server process
                server_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for server to be ready
                logger.info("Waiting for llama.cpp server to be ready...")
                max_wait = 120  # seconds (GGUF models can take longer to load)
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    if lm.check_server_health(f"http://{args.host}:{args.port}"):
                        logger.success(f"✅ Llama.cpp server started at http://{args.host}:{args.port}")
                        # Create a simple server manager-like object to track the process
                        class SimpleServerManager:
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
                        
                        server_manager = SimpleServerManager(server_process)
                        break
                    time.sleep(3)
                else:
                    logger.error("Failed to start llama.cpp server within timeout")
                    logger.error("Check server logs for details")
                    server_process.terminate()
                    sys.exit(1)
            else:
                logger.info(f"🚀 Starting vLLM server for model: {server_model}")
                server_manager = lm.ServerManager(
                    model=server_model,
                    host=args.host,
                    port=args.port,
                    logger_type=args.logger_type,
                    activations_path=args.activations_path,
                    log_file_path=log_file_path
                )
                server_manager.start_server()
                lm.set_server_manager(server_manager)
                logger.success(f"✅ Server started at http://{args.host}:{args.port}")
        else:
            logger.info(f"✅ Server already running at http://{args.host}:{args.port}")
            logger.warning("⚠️  Note: Using existing server (not managed by this script)")

    try:

        # Prepare task kwargs
        task_kwargs = {
            "N": args.N,
            "wiki_src": args.wiki_src,
            "mode": args.mode,
            "inference_method": args.inference_method,
            "max_inference_tokens": args.max_inference_tokens,
            "generations_file_path": args.generations_file_path,
            "eval_results_path": args.eval_results_path,
            "q_generator": args.q_generator,
            "qgen_backend": args.qgen_backend,
            "qa_output_path": args.qa_output_path,
            "quick_debug_mode": args.quick_debug_mode,
            "max_workers_qgen": args.max_workers_qgen,
            # Activation logging parameters
            "logger_type": args.logger_type,
            "activations_path": args.activations_path,
            "log_file": log_file_path,
            # Resume control
            "resume": not args.no_resume,
            "resume_eval": not args.no_resume_eval,
            # LongWiki specific
            "db_path": args.db_path,
            "claim_extractor": args.claim_extractor,
            "abstain_evaluator": args.abstain_evaluator,
            "verifier": args.verifier,
            "k": args.k,
            "max_tokens": args.max_tokens,
            "max_workers": args.max_workers,
            # Mixed entities specific
            "exp": args.exp,
            "seed": args.seed,
            # TriviaQA specific
            "dataset_variant": args.dataset_variant,
            "split": args.split,
            "data_dir": args.data_dir,
            "auto_download": args.auto_download
        }

        # Run task step(s)
        if args.step == "all":
            # Run all steps in sequence
            if args.task == "triviaqa":
                # TriviaQA only has inference and eval steps
                steps = ["inference", "eval"]
            else:
                steps = ["generate", "inference", "eval"]

            for step in steps:
                logger.info(f"Running step {step}...")
                result = run_task_step(step, args.task, args.model, **task_kwargs)
                if result is None:
                    logger.info(f"Step {step} was skipped")
        else:
            # Run single step
            result = run_task_step(args.step, args.task, args.model, **task_kwargs)
            if result is None:
                logger.info(f"Step {args.step} was skipped")

        logger.success("All steps completed successfully!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Stop server if we started it
        if server_manager:
            logger.info("🛑 Stopping vLLM server...")
            server_manager.stop_server()
            lm.set_server_manager(None)
            logger.success("✅ Server stopped")

if __name__ == "__main__":
    main()
