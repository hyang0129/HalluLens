#!/usr/bin/env python3
"""
Unified script to run HalluLens tasks with automatic server management.

This script automatically starts the activation logging server and runs one of the three
generation steps, eliminating the need for manual server startup.

Usage:
    python scripts/run_with_server.py --step [generate|inference|eval] [task_options...]

Examples:
    # Generate prompts
    python scripts/run_with_server.py --step generate --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100

    # Run inference
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct

    # Run evaluation
    python scripts/run_with_server.py --step eval --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct

    # Run all steps in sequence
    python scripts/run_with_server.py --step all --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --N 100

    # Custom server log file
    python scripts/run_with_server.py --step inference --task precisewikiqa --model meta-llama/Llama-3.1-8B-Instruct --log-file custom/server.log
"""

import argparse
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    else:
        return task

def determine_generations_file_path(task_name, model, generations_file_path=None):
    """Determine the generations file path based on task and model."""
    if generations_file_path:
        return generations_file_path

    # Use the standard pattern: output/{task_name}/{model_name}/generation.jsonl
    model_name = model.split("/")[-1]
    return f"output/{task_name}/{model_name}/generation.jsonl"

class ServerManager:
    """Manages the activation logging server lifecycle."""

    def __init__(self, model, host="0.0.0.0", port=8000, logger_type="lmdb", activations_path=None, log_file_path=None):
        self.model = model
        self.host = host
        self.port = port
        self.logger_type = logger_type
        self.activations_path = activations_path or f"lmdb_data/{model.replace('/', '_')}_activations.lmdb"
        self.log_file_path = log_file_path
        self.server_process = None
        
    def start_server(self):
        """Start the activation logging server."""
        logger.info(f"Starting activation logging server for model: {self.model}")
        
        # Set environment variables for activation logging
        env = os.environ.copy()
        env["ACTIVATION_STORAGE_PATH"] = self.activations_path
        env["ACTIVATION_LOGGER_TYPE"] = self.logger_type
        env["ACTIVATION_TARGET_LAYERS"] = "all"
        env["ACTIVATION_SEQUENCE_MODE"] = "all"
        
        # Build server command
        cmd = [
            sys.executable, "-m", "activation_logging.vllm_serve",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--logger-type", self.logger_type,
            "--activations-path", self.activations_path
        ]

        # Add log file path if specified
        if self.log_file_path:
            cmd.extend(["--log-file", self.log_file_path])
            logger.info(f"Server behavior logs will be written to: {self.log_file_path}")
        
        logger.info(f"Server command: {' '.join(cmd)}")
        
        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        self._wait_for_server()
        
    def _wait_for_server(self, timeout=120):
        """Wait for server to be ready to accept requests."""
        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    logger.success("Server is ready!")
                    return
            except requests.exceptions.RequestException:
                pass
            
            # Check if server process is still running
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                logger.error(f"Server process died. STDOUT: {stdout}, STDERR: {stderr}")
                raise RuntimeError("Server process terminated unexpectedly")
            
            time.sleep(2)
        
        raise TimeoutError(f"Server did not become ready within {timeout} seconds")
    
    def stop_server(self):
        """Stop the activation logging server."""
        if self.server_process:
            logger.info("Stopping activation logging server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()
            logger.info("Server stopped")

def run_task_step(step, task, model, **kwargs):
    """Run the specified task step."""
    logger.info(f"Running {step} step for {task} task with model {model}")

    # Build command based on task and step
    if task == "precisewikiqa":
        cmd = [sys.executable, "-m", "tasks.shortform.precise_wikiqa"]

        # Add step-specific flags
        if step == "generate":
            cmd.append("--do_generate_prompt")
        elif step == "inference":
            cmd.append("--do_inference")
        elif step == "eval":
            cmd.append("--do_eval")

        # Add common parameters
        cmd.extend([
            "--model", model,
            "--wiki_src", kwargs.get("wiki_src", "goodwiki"),
            "--mode", kwargs.get("mode", "dynamic"),
            "--inference_method", kwargs.get("inference_method", "vllm"),
            "--max_inference_tokens", str(kwargs.get("max_inference_tokens", 256)),
            "--N", str(kwargs.get("N", 1))
        ])

        # Add optional parameters
        if kwargs.get("generations_file_path"):
            cmd.extend(["--generations_file_path", kwargs["generations_file_path"]])
        if kwargs.get("eval_results_path"):
            cmd.extend(["--eval_results_path", kwargs["eval_results_path"]])
        if kwargs.get("q_generator"):
            cmd.extend(["--q_generator", kwargs["q_generator"]])
        if kwargs.get("qa_output_path"):
            cmd.extend(["--qa_output_path", kwargs["qa_output_path"]])
        if kwargs.get("quick_debug_mode"):
            cmd.append("--quick_debug_mode")
            
    elif task == "longwiki":
        cmd = [sys.executable, "-m", "tasks.longwiki.longwiki_main"]

        # Add step-specific flags
        if step == "generate":
            cmd.append("--do_generate_prompt")
        elif step == "inference":
            cmd.append("--do_inference")
        elif step == "eval":
            cmd.append("--do_eval")

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
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    logger.info(f"Task command: {' '.join(cmd)}")
    
    # Run the task
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Task failed with return code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Task execution failed: {result.stderr}")
    
    logger.success(f"Task {step} completed successfully")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run HalluLens tasks with automatic server management")
    
    # Required arguments
    parser.add_argument("--step", required=True, choices=["generate", "inference", "eval", "all"],
                       help="Which step to run (or 'all' for all steps)")
    parser.add_argument("--task", required=True, choices=["precisewikiqa", "longwiki", "mixedentities"],
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
    parser.add_argument("--eval_results_path", help="Path for evaluation results")
    parser.add_argument("--q_generator", help="Question generator model")
    parser.add_argument("--qa_output_path", help="Custom QA output path")
    parser.add_argument("--quick_debug_mode", action="store_true", help="Quick debug mode (first 5 questions)")

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
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies(args.task, args.step):
        sys.exit(1)

    # Determine log file path for server behavior logs
    log_file_path = args.log_file
    if not log_file_path and (args.step == "inference" or args.step == "all"):
        # For inference steps, place server logs in same directory as generations file
        task_kwargs_for_name = {
            "wiki_src": args.wiki_src,
            "mode": args.mode,
            "exp": args.exp
        }
        task_name = get_task_name(args.task, **task_kwargs_for_name)
        generations_path = determine_generations_file_path(task_name, args.model, args.generations_file_path)

        # Create log file path in same directory as generations file
        generations_dir = Path(generations_path).parent
        generations_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = str(generations_dir / "server_behavior.log")

    # Create server manager
    server_manager = ServerManager(
        model=args.model,
        host=args.host,
        port=args.port,
        logger_type=args.logger_type,
        activations_path=args.activations_path,
        log_file_path=log_file_path
    )

    try:
        # Start server
        server_manager.start_server()

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
            "qa_output_path": args.qa_output_path,
            "quick_debug_mode": args.quick_debug_mode,
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
            "seed": args.seed
        }

        # Run task step(s)
        if args.step == "all":
            # Run all steps in sequence
            steps = ["generate", "inference", "eval"]
            for step in steps:
                logger.info(f"Running step {step}...")
                run_task_step(step, args.task, args.model, **task_kwargs)
        else:
            # Run single step
            run_task_step(args.step, args.task, args.model, **task_kwargs)

        logger.success("All steps completed successfully!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Always stop server
        server_manager.stop_server()

if __name__ == "__main__":
    main()
