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
    log_file_path=None
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
        # Initialize client logging for debugging
        if inference_method == "vllm":
            lm.setup_client_logging()
            print(f"Client logging initialized for {len(all_prompts)} requests")

        if not generations_file_path:
            base_path = Path(base_path)
            model_name = model_path.split("/")[-1]
            output_folder = base_path / task / model_name
            output_folder.mkdir(exist_ok=True, parents=True)
            generations_file_path = output_folder / "generation.jsonl"

        generations_file_path = str(generations_file_path)
        print('generations_file_path', generations_file_path)

        prompts =  all_prompts.prompt.to_list()

        # get the response from the model
        if inference_method == 'openai':
            all_prompts["generation"] = thread_map(
                lambda p: lm.openai_generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
                prompts,
                max_workers=max_workers,
                desc="Predict openai",
            )
        elif inference_method == "vllm":
            port = None
            all_prompts["generation"] = thread_map(
                lambda p: lm.call_vllm_api(p, model=model_path, temperature=0.0, top_p=1.0,  max_tokens=max_tokens, port=port, max_retries=max_retries, base_delay=base_delay),
                prompts,
                max_workers=max_workers,
                desc="Predict on vllm",
            )
        elif inference_method == "custom":
            all_prompts["generation"] = thread_map(
                lambda p: lm.generate(p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens, max_retries=max_retries, base_delay=base_delay),
                prompts,
                max_workers=max_workers,
                desc="Predict on custom API",
            )
        else:
            raise NotImplementedError(f"No method {inference_method}")

        # save the results
        all_prompts.to_json(generations_file_path, lines=True, orient="records")

        # Report skip statistics if using vllm
        if inference_method == "vllm":
            skip_stats = lm.get_skip_statistics()
            if skip_stats["total_skipped"] > 0:
                print(f"\n📊 Experiment completed with {skip_stats['total_skipped']} skipped samples:")
                print(f"   - Timeout skipped: {skip_stats['timeout_skipped']}")
                print(f"   - Error skipped: {skip_stats['error_skipped']}")
                print(f"   - Successfully processed: {len(all_prompts) - skip_stats['total_skipped']}")
                print(f"   - Skip rate: {skip_stats['total_skipped']/len(all_prompts)*100:.2f}%")
                print(f"   - Skipped samples list saved to: goodwiki_json/skipped_samples.json")
            else:
                print(f"✅ Experiment completed successfully with no skipped samples!")

        if return_gen:
            return all_prompts

    finally:
        # Stop server if we started it
        if server_manager:
            print("Stopping activation logging server...")
            server_manager.stop_server()
            lm.set_server_manager(None)
            print("✅ Server stopped")