# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from tqdm import tqdm
import hashlib
import json

from utils import lm

def process_with_incremental_save(all_prompts, inference_fn, generations_file_path, desc="Processing", resume_mode=False):
    """
    Process prompts sequentially and save each result immediately to JSONL file.

    Args:
        all_prompts: DataFrame with prompts and metadata
        inference_fn: Function to call for each prompt (takes prompt string, returns generation string)
        generations_file_path: Path to save generations
        desc: Description for progress bar
        resume_mode: If True, append to existing file; if False, overwrite

    Returns:
        DataFrame with generations added
    """
    prompts = all_prompts.prompt.to_list()
    generations = []

    # Determine file mode
    file_mode = 'a' if resume_mode else 'w'

    # Open file once and keep it open for all writes
    with open(generations_file_path, file_mode, encoding='utf-8') as f:
        # Process each prompt sequentially
        for idx, prompt in enumerate(tqdm(prompts, desc=desc)):
            # Generate response
            generation = inference_fn(prompt)
            generations.append(generation)

            # Create the full record with all metadata from original prompt
            record = all_prompts.iloc[idx].to_dict()
            record['generation'] = generation

            # Write to file immediately
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()  # Ensure it's written to disk immediately

    # Add generations to dataframe
    all_prompts = all_prompts.copy()
    all_prompts['generation'] = generations

    return all_prompts

def run_exp(
    task: str,
    model_path: str,
    all_prompts,
    generations_file_path=None,
    base_path="output",
    inference_method="vllm",
    max_workers=64,
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    return_gen = False,
    max_retries=3,
    base_delay=1.0,
    manage_server=True,
    server_host="0.0.0.0",
    server_port=8000,
    logger_type="lmdb",
    activations_path=None,
    log_file_path=None,
    resume=True
):
    """
    Run experiment with model inference.

    Args:
        task: Task name
        model_path: Model name/path
        all_prompts: DataFrame with prompts
        generations_file_path: Path to save generations
        base_path: Base output directory
        inference_method: Inference method (vllm, openai, custom)
        max_workers: DEPRECATED - kept for backward compatibility, inference is now single-threaded
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (default: 0.0 for greedy). Set > 0 for stochastic sampling.
        top_p: Top-p sampling parameter (default: 1.0)
        return_gen: Whether to return generations
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        manage_server: Whether to manage the server lifecycle (default: True)
        server_host: Server host (default: 0.0.0.0)
        server_port: Server port (default: 8000)
        logger_type: Activation logger type (lmdb or json)
        activations_path: Path for activation storage
        log_file_path: Path for server logs
        resume: Whether to resume from existing generations file (default: True)

    Note:
        Inference is now single-threaded with incremental saving. Each generation is saved
        to disk immediately after completion, ensuring no progress is lost on interruption.
    """
    # Start server if needed
    server_manager = None
    server_was_running = False

    if inference_method == "vllm" and manage_server:
        # Check if server is already running
        server_was_running = lm.check_server_health(f"http://{server_host}:{server_port}")

        if not server_was_running:
            print(f"Starting activation logging server for {model_path}...")
            server_manager = lm.ServerManager(
                model=model_path,
                host=server_host,
                port=server_port,
                logger_type=logger_type,
                activations_path=activations_path,
                log_file_path=log_file_path
            )
            server_manager.start_server()
            lm.set_server_manager(server_manager)
            print(f" Server started successfully at http://{server_host}:{server_port}")
        else:
            print(f"Server already running at http://{server_host}:{server_port}")
            print(f"  Note: Server restart will not be available (server not managed by this process)")

    try:
        # Determine generations file path first
        if not generations_file_path:
            base_path = Path(base_path)
            model_name = model_path.split("/")[-1]
            output_folder = base_path / task / model_name
            output_folder.mkdir(exist_ok=True, parents=True)
            generations_file_path = output_folder / "generation.jsonl"

        generations_file_path = str(generations_file_path)
        print('generations_file_path', generations_file_path)

        # Check for existing generations and resume if requested
        existing_generations = None
        original_prompt_count = len(all_prompts)
        already_completed_count = 0

        if resume and Path(generations_file_path).exists():
            import pandas as pd

            print(f" Found existing generations file: {generations_file_path}")
            try:
                # Load existing generations
                existing_generations = pd.read_json(generations_file_path, lines=True)
                print(f" Loaded {len(existing_generations)} existing generations")

                # Filter out prompts that already have generations
                # Match on the 'prompt' field to identify already-processed items
                existing_prompts = set(existing_generations['prompt'].tolist())

                # Create a mask for prompts that haven't been processed yet
                mask = ~all_prompts['prompt'].isin(existing_prompts)
                remaining_prompts = all_prompts[mask].copy()

                if len(remaining_prompts) == 0:
                    print(f" All {original_prompt_count} prompts already processed! Nothing to do.")
                    if return_gen:
                        return existing_generations
                    return None

                # Track how many were already completed for progress tracking
                already_completed_count = len(existing_generations)

                print(f" Resume statistics:")
                print(f"   - Total prompts: {original_prompt_count}")
                print(f"   - Already completed: {already_completed_count}")
                print(f"   - Remaining to process: {len(remaining_prompts)}")
                print(f"   - Progress: {already_completed_count/original_prompt_count*100:.1f}%")

                # Update all_prompts to only include remaining prompts
                all_prompts = remaining_prompts

            except Exception as e:
                print(f"  Warning: Could not load existing generations: {e}")
                print(f"   Starting from scratch...")
                existing_generations = None
                already_completed_count = 0

        # Initialize client logging for debugging
        if inference_method == "vllm":
            lm.setup_client_logging(generations_file_path)
            # Initialize progress tracking with total original count and already completed count
            lm.initialize_progress_tracking(original_prompt_count, already_completed=already_completed_count)
            print(f"Client logging initialized for {len(all_prompts)} remaining requests")
            print(f" Starting inference: {len(all_prompts)} remaining requests to process (total: {original_prompt_count}, completed: {already_completed_count})")

        # get the response from the model with incremental saving
        print(f" Processing {len(all_prompts)} inference requests with incremental saving...")
        print(f" Results will be saved to: {generations_file_path}")

        # Define inference function based on method
        if inference_method == 'openai':
            inference_fn = lambda p: lm.openai_generate(p, model=model_path, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        elif inference_method == "vllm":
            port = None
            inference_fn = lambda p: lm.call_vllm_api(p, model=model_path, temperature=temperature, top_p=top_p, max_tokens=max_tokens, port=port, max_retries=max_retries, base_delay=base_delay)
        elif inference_method == "custom":
            inference_fn = lambda p: lm.generate(p, model=model_path, temperature=temperature, top_p=top_p, max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay)
        else:
            raise NotImplementedError(f"No method {inference_method}")

        # Process with incremental saving (single-threaded, sequential)
        # If resuming, append to existing file; otherwise start fresh
        all_prompts = process_with_incremental_save(
            all_prompts=all_prompts,
            inference_fn=inference_fn,
            generations_file_path=generations_file_path,
            desc=f"{inference_method.upper()} inference",
            resume_mode=(existing_generations is not None)
        )

        print(f" All {len(all_prompts)} samples processed and saved to {generations_file_path}")

        # If resuming, we need to reload the full file to get complete dataset
        if existing_generations is not None:
            import pandas as pd
            print(f" Reloading complete dataset from {generations_file_path}...")
            all_prompts = pd.read_json(generations_file_path, lines=True)
            print(f" Total generations in file: {len(all_prompts)}")

        # Report final statistics if using vllm
        if inference_method == "vllm":
            skip_stats = lm.get_skip_statistics()
            progress_stats = lm.get_progress_stats()

            print(f"\n Experiment completed:")
            if existing_generations is not None:
                print(f"   - Previously completed: {len(existing_generations)}")
                print(f"   - Newly processed: {progress_stats['total_requests']}")
                print(f"   - Total in dataset: {original_prompt_count}")
            print(f"   - Total requests (this run): {progress_stats['total_requests']}")
            print(f"   - Successfully completed: {progress_stats['completed_requests']}")
            print(f"   - Failed requests: {progress_stats['failed_requests']}")
            if progress_stats['total_requests'] > 0:
                print(f"   - Success rate: {progress_stats['completed_requests']/progress_stats['total_requests']*100:.2f}%")

            if skip_stats["total_skipped"] > 0:
                print(f"   - Timeout skipped: {skip_stats['timeout_skipped']}")
                print(f"   - Error skipped: {skip_stats['error_skipped']}")
                print(f"   - Skip rate: {skip_stats['total_skipped']/progress_stats['total_requests']*100:.2f}%")
                print(f"   - Skipped samples list saved to: goodwiki_json/failed_requests/skipped_samples.json")
            else:
                print(f" All samples processed successfully!")

        if return_gen:
            return all_prompts

    finally:
        # Stop server if we started it
        if server_manager:
            print("Stopping activation logging server...")
            server_manager.stop_server()
            lm.set_server_manager(None)
            print(" Server stopped")


def export_generation_jsonl(zarr_path: str, qa_df, output_path: str):
    """Export generation results from a Zarr activation store to JSONL.

    Matches prompts in *qa_df* to entries stored in the Zarr index by
    prompt hash, then writes one JSON line per matched prompt containing
    the original DataFrame columns plus a ``generation`` field with the
    model response retrieved from the store.

    Args:
        zarr_path: Path to the ``.zarr`` activation store.
        qa_df: DataFrame with at least a ``prompt`` column.
        output_path: Destination path for the JSONL file.
    """
    from activation_logging.zarr_activations_logger import ZarrActivationsLogger

    reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)

    # Build a lookup: prompt_hash -> response
    hash_to_response = {}
    for _key, meta in reader._index.items():
        ph = meta.get("prompt_hash")
        if ph:
            hash_to_response[ph] = meta.get("response", "")
    reader.close()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for _, row in qa_df.iterrows():
            prompt = row["prompt"]
            ph = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            response = hash_to_response.get(ph)
            if response is None:
                continue
            record = row.to_dict()
            record["generation"] = response
            f.write(json.dumps(record, ensure_ascii=False) + "\n")