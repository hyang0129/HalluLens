# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from tqdm.contrib.concurrent import thread_map

from utils import lm

def run_exp(
    task: str,
    model_path: str,
    all_prompts,
    generations_file_path=None,
    base_path="output",
    inference_method="vllm",
    max_workers=64,
    max_tokens=512,
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
        max_workers: Number of parallel workers
        max_tokens: Maximum tokens to generate
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
            print(f"✅ Server started successfully at http://{server_host}:{server_port}")
        else:
            print(f"Server already running at http://{server_host}:{server_port}")
            print(f"⚠️  Note: Server restart will not be available (server not managed by this process)")

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

        if resume and Path(generations_file_path).exists():
            import pandas as pd
            import jsonlines

            print(f"📂 Found existing generations file: {generations_file_path}")
            try:
                # Load existing generations
                existing_generations = pd.read_json(generations_file_path, lines=True)
                print(f"✅ Loaded {len(existing_generations)} existing generations")

                # Filter out prompts that already have generations
                # Match on the 'prompt' field to identify already-processed items
                existing_prompts = set(existing_generations['prompt'].tolist())

                # Create a mask for prompts that haven't been processed yet
                mask = ~all_prompts['prompt'].isin(existing_prompts)
                remaining_prompts = all_prompts[mask].copy()

                if len(remaining_prompts) == 0:
                    print(f"✅ All {original_prompt_count} prompts already processed! Nothing to do.")
                    if return_gen:
                        return existing_generations
                    return None

                print(f"📊 Resume statistics:")
                print(f"   - Total prompts: {original_prompt_count}")
                print(f"   - Already completed: {len(existing_generations)}")
                print(f"   - Remaining to process: {len(remaining_prompts)}")
                print(f"   - Progress: {len(existing_generations)/original_prompt_count*100:.1f}%")

                # Update all_prompts to only include remaining prompts
                all_prompts = remaining_prompts

            except Exception as e:
                print(f"⚠️  Warning: Could not load existing generations: {e}")
                print(f"   Starting from scratch...")
                existing_generations = None

        # Initialize client logging for debugging
        if inference_method == "vllm":
            lm.setup_client_logging(generations_file_path)
            lm.initialize_progress_tracking(len(all_prompts))
            print(f"Client logging initialized for {len(all_prompts)} requests")
            print(f"📊 Starting inference: {len(all_prompts)} total requests to process")

        prompts =  all_prompts.prompt.to_list()

        # get the response from the model
        print(f"🚀 Processing {len(prompts)} inference requests...")
        if inference_method == 'openai':
            all_prompts["generation"] = thread_map(
                lambda p: lm.openai_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
                prompts,
                max_workers=max_workers,
                desc=f"OpenAI inference ({len(prompts)} requests)",
            )
        elif inference_method == "vllm":
            port = None
            all_prompts["generation"] = thread_map(
                lambda p: lm.call_vllm_api(p, model=model_path, temperature=0.0, top_p=1.0,  max_tokens=max_tokens, port=port, max_retries=max_retries, base_delay=base_delay),
                prompts,
                max_workers=max_workers,
                desc=f"vLLM inference ({len(prompts)} requests)",
            )
        elif inference_method == "custom":
            all_prompts["generation"] = thread_map(
                lambda p: lm.generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay),
                prompts,
                max_workers=max_workers,
                desc=f"Custom API inference ({len(prompts)} requests)",
            )
        else:
            raise NotImplementedError(f"No method {inference_method}")

        # Merge with existing generations if resuming
        if existing_generations is not None:
            import pandas as pd
            print(f"📝 Merging {len(all_prompts)} new generations with {len(existing_generations)} existing ones...")
            all_prompts = pd.concat([existing_generations, all_prompts], ignore_index=True)
            print(f"✅ Total generations: {len(all_prompts)}")

        # save the results
        all_prompts.to_json(generations_file_path, lines=True, orient="records")

        # Report final statistics if using vllm
        if inference_method == "vllm":
            skip_stats = lm.get_skip_statistics()
            progress_stats = lm.get_progress_stats()

            print(f"\n📊 Experiment completed:")
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
                print(f"✅ All samples processed successfully!")

        if return_gen:
            return all_prompts

    finally:
        # Stop server if we started it
        if server_manager:
            print("Stopping activation logging server...")
            server_manager.stop_server()
            lm.set_server_manager(None)
            print("✅ Server stopped")