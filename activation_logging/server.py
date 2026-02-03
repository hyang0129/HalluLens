"""
server.py
OpenAI API-compatible FastAPI server with activation logging.
Designed to be compatible with vLLM serve and the utils/lm.py interface.
"""
import os
import hashlib
import time
import uuid
import psutil
import threading
import gc
import glob
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import uvicorn
from activation_logging.activations_logger import ActivationsLogger, JsonActivationsLogger
from llama_cpp import Llama  # Import for llama.cpp Python bindings

# Configure logger to use the same log file as parent process if specified
if "SERVER_LOG_FILE" in os.environ:
    logger.remove()  # Remove default handler
    logger.add(
        os.environ["SERVER_LOG_FILE"],
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

# Default model if none specified in request (read from environment or use default)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
logger.info(f"Server initialization - DEFAULT_MODEL set to: {DEFAULT_MODEL}")
logger.info(f"Server initialization - Environment DEFAULT_MODEL: {os.environ.get('DEFAULT_MODEL', 'not set')}")
app = FastAPI()

# Default activation storage path (can be LMDB or JSON directory)
DEFAULT_ACTIVATIONS_PATH = os.environ.get("ACTIVATION_STORAGE_PATH",
                                        os.environ.get("ACTIVATION_LMDB_PATH", "lmdb_data/activations.lmdb"))
logger.info(f"Server initialization - DEFAULT_ACTIVATIONS_PATH: {DEFAULT_ACTIVATIONS_PATH}")

# Default logger type ('lmdb' or 'json')
DEFAULT_LOGGER_TYPE = os.environ.get("ACTIVATION_LOGGER_TYPE", "lmdb")
logger.info(f"Server initialization - DEFAULT_LOGGER_TYPE: {DEFAULT_LOGGER_TYPE}")

# Default LMDB path (for backward compatibility)
DEFAULT_LMDB_PATH = DEFAULT_ACTIVATIONS_PATH

# Default LMDB map size (16GB if not specified)
DEFAULT_MAP_SIZE = int(os.environ.get("ACTIVATION_LMDB_MAP_SIZE", 16 * (1 << 30)))

# Model cache to avoid reloading for each request
_model_cache = {}
_tokenizer_cache = {}
_llamacpp_model_cache = {}  # Cache for llama.cpp models

# Temperature overwrite variable, None means use request temperature
overwrite_temperature = None

# Activation storage path overwrite variable, None means use the default path from environment
overwrite_activations_path = None

# Logger type overwrite variable, None means use the default type from environment
overwrite_logger_type = None

# LMDB path overwrite variable, None means use the default path from environment (backward compatibility)
overwrite_lmdb_path = None

# Top-p overwrite variable, None means use request top_p
overwrite_top_p = None

# Default path for GGUF models
GGUF_MODELS_DIR = os.environ.get("GGUF_MODELS_DIR", "")


def _is_split_first_part_gguf(path: str) -> bool:
    """Return True if the file name looks like the first shard of a split GGUF."""
    base = os.path.basename(path)
    return ("-00001-of-" in base) or ("00001-of-" in base)


def _candidate_gguf_search_dirs() -> List[str]:
    """Return directories to search for GGUF models.

    Order matters: earlier directories are preferred.
    """
    dirs: List[str] = []

    if GGUF_MODELS_DIR:
        dirs.append(GGUF_MODELS_DIR)

    # Common repo-local conventions
    cwd = os.getcwd()
    dirs.extend(
        [
            cwd,
            os.path.join(cwd, "models"),
            os.path.join(cwd, "checkpoints"),
        ]
    )

    # Deduplicate while preserving order
    seen = set()
    unique_dirs: List[str] = []
    for d in dirs:
        d_norm = os.path.abspath(d)
        if d_norm not in seen:
            seen.add(d_norm)
            unique_dirs.append(d_norm)
    return unique_dirs


def _find_best_gguf_match(paths: List[str]) -> Optional[str]:
    """Pick the best GGUF file from a set of candidates.

    Prefers the first part of split models (e.g. *-00001-of-00002.gguf).
    """
    if not paths:
        return None

    # Prefer explicit first shard of split models
    split_first = [p for p in paths if _is_split_first_part_gguf(p)]
    if split_first:
        return sorted(split_first)[0]

    # Otherwise pick a stable deterministic first
    return sorted(paths)[0]


def resolve_gguf_model_path(model_path: str) -> str:
    """Resolve a user-provided GGUF model identifier to a concrete path.

    Accepts:
      - absolute path to a .gguf file
      - absolute path to a directory containing .gguf files
      - relative path to a .gguf file or directory
      - bare model name (e.g. "Llama-3.3-70B-Instruct-Q6_K_L")

    For split GGUF models, returns the first shard (llama.cpp auto-loads the rest).
    """
    logger.info(f"llama.cpp model path resolution - Input: {model_path}")
    logger.info(f"GGUF_MODELS_DIR from environment: {GGUF_MODELS_DIR}")
    logger.info(f"Is absolute path: {os.path.isabs(model_path)}")

    # Fast path: absolute file/dir
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    # Build candidate concrete paths first (cheap existence checks)
    candidates: List[str] = []
    search_dirs = _candidate_gguf_search_dirs()
    model_path_norm = model_path

    # If the user provided a relative path like models/foo, try it directly too.
    candidates.append(model_path_norm)

    if not model_path_norm.lower().endswith(".gguf"):
        candidates.append(model_path_norm + ".gguf")

    for base_dir in search_dirs:
        candidates.append(os.path.join(base_dir, model_path_norm))
        if not model_path_norm.lower().endswith(".gguf"):
            candidates.append(os.path.join(base_dir, model_path_norm + ".gguf"))

    # Try concrete candidates
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # If nothing exists yet, attempt a targeted glob search.
    # This supports split models when the user only provides the base name.
    glob_hits: List[str] = []
    if os.path.sep not in model_path_norm and "/" not in model_path_norm and "\\" not in model_path_norm:
        pattern = model_path_norm + "*.gguf"
        for base_dir in search_dirs:
            if not os.path.isdir(base_dir):
                continue
            # 1) Look in the directory itself
            glob_hits.extend(glob.glob(os.path.join(base_dir, pattern)))
            # 2) Look one level down (common layout: models/<name>/*.gguf)
            glob_hits.extend(glob.glob(os.path.join(base_dir, model_path_norm, "*.gguf")))

    best = _find_best_gguf_match(glob_hits)
    if best and os.path.exists(best):
        return best

    searched = []
    for d in search_dirs:
        if os.path.isdir(d):
            searched.append(d)
    hint = (
        "Could not resolve GGUF model path. "
        "If you downloaded via download_q6k_model.sh, pass --model 'models/Llama-3.3-70B-Instruct-Q6_K_L' "
        "or set GGUF_MODELS_DIR to the directory that contains your model folder/files."
    )
    raise FileNotFoundError(
        f"Model file/directory not found for: {model_path}. "
        f"Searched in: {searched}. {hint}"
    )

# Request tracking for debugging
active_requests = {}
request_lock = threading.Lock()

def log_system_resources(context: str = ""):
    """Log current system resource usage for debugging."""
    try:
        # Memory info
        memory = psutil.virtual_memory()

        # GPU info if available
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_info = f"GPU: {gpu_memory:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached"

        # Disk info for activation path
        disk_usage = ""
        if DEFAULT_ACTIVATIONS_PATH:
            try:
                disk = psutil.disk_usage(os.path.dirname(DEFAULT_ACTIVATIONS_PATH) or ".")
                disk_usage = f"Disk: {disk.free / 1024**3:.2f}GB free of {disk.total / 1024**3:.2f}GB"
            except:
                disk_usage = "Disk: info unavailable"

        # Active requests count
        with request_lock:
            active_count = len(active_requests)

        logger.info(f"[{context}] System Resources - "
                   f"RAM: {memory.percent}% used ({memory.available / 1024**3:.2f}GB free), "
                   f"{gpu_info}, {disk_usage}, Active requests: {active_count}")

    except Exception as e:
        logger.warning(f"Failed to log system resources: {e}")

def track_request_start(request_id: str, endpoint: str, model: str):
    """Track the start of a request for debugging."""
    with request_lock:
        active_requests[request_id] = {
            "start_time": time.time(),
            "endpoint": endpoint,
            "model": model,
            "status": "started"
        }
    logger.info(f"[{request_id}] Request started - Endpoint: {endpoint}, Model: {model}")
    log_system_resources(f"Request Start {request_id}")

def track_request_end(request_id: str, status: str = "completed"):
    """Track the end of a request for debugging."""
    with request_lock:
        if request_id in active_requests:
            duration = time.time() - active_requests[request_id]["start_time"]
            active_requests[request_id]["status"] = status
            active_requests[request_id]["duration"] = duration
            logger.info(f"[{request_id}] Request {status} - Duration: {duration:.2f}s")
            del active_requests[request_id]
        else:
            logger.warning(f"[{request_id}] Request end tracked but not found in active requests")
    log_system_resources(f"Request End {request_id}")

def log_long_running_requests():
    """Log any requests that have been running for more than 5 minutes."""
    current_time = time.time()
    with request_lock:
        for request_id, info in active_requests.items():
            duration = current_time - info["start_time"]
            if duration > 300:  # 5 minutes
                logger.warning(f"[{request_id}] Long-running request detected - "
                             f"Duration: {duration:.2f}s, Endpoint: {info['endpoint']}, "
                             f"Model: {info['model']}, Status: {info['status']}")

def get_model_and_tokenizer(model_name: str, auth_token: Optional[str] = None):
    """
    Get model and tokenizer, loading them if not already in cache.
    Uses GPU if available, otherwise falls back to CPU.
    
    Args:
        model_name: Name of the model to load (HuggingFace model ID)
        auth_token: HuggingFace authentication token for gated models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    cache_key = f"{model_name}_{auth_token if auth_token else 'public'}"
    
    if cache_key not in _model_cache:
        logger.info(f"Loading model: {model_name}")
        
        # Check if auth token is provided or in environment
        use_auth_token = auth_token
        if not use_auth_token and "HF_TOKEN" in os.environ:
            use_auth_token = os.environ["HF_TOKEN"]
            logger.info("Using HF_TOKEN from environment")
        
        tokenizer_kwargs = {
            "trust_remote_code": True
        }
        
        model_kwargs = {
            "trust_remote_code": True,
            "output_hidden_states": True
        }
        
        # Add auth token if available
        if use_auth_token:
            tokenizer_kwargs["use_auth_token"] = use_auth_token
            model_kwargs["use_auth_token"] = use_auth_token
            logger.info("Using authentication token for model access")
        
        # Use GPU if available, else CPU
        if torch.cuda.is_available():
            device_map = "cuda"
            torch_dtype = torch.float16
            logger.info("Using GPU for inference.")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            logger.info("Using CPU for inference.")
            
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = device_map
        
        # First load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        if not tokenizer.pad_token or tokenizer.pad_token == "":
            logger.info("Pad token not set or empty. Setting pad token to eos token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.info(f"Pad token is already set: {tokenizer.pad_token}")

        # Cache trim token IDs if TRIM_OUTPUT_AT is set
        trim_at = os.environ.get("TRIM_OUTPUT_AT")
        if trim_at is not None:
            logger.info(f"Caching token IDs for trim sequence: {repr(trim_at)}")
            trim_token_ids = []
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                if trim_at in token_str:
                    trim_token_ids.append(token_id)
            tokenizer.trim_token_ids = trim_token_ids
            logger.info(f"Found {len(trim_token_ids)} tokens containing trim sequence")

        # Then load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        logger.success(f"Model loaded: {model_name} on {device_map}")
        _model_cache[cache_key] = model
        _tokenizer_cache[cache_key] = tokenizer
    
    return _model_cache[cache_key], _tokenizer_cache[cache_key]


def get_llamacpp_model(model_path):
    """
    Get or load a llama.cpp model from cache.
    
    Args:
        model_path: Path to the GGUF model file or directory containing GGUF files
        
    Returns:
        Loaded llama.cpp model instance
    """
    full_model_path = resolve_gguf_model_path(model_path)
    logger.info(f"Resolved GGUF model path: {full_model_path}")
    
    # If path is a directory, search for .gguf files
    if os.path.isdir(full_model_path):
        logger.info(f"Path is a directory, searching for .gguf files...")
        gguf_files = []
        for root, dirs, files in os.walk(full_model_path):
            for file in files:
                if file.endswith('.gguf'):
                    gguf_files.append(os.path.join(root, file))
        
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in directory: {full_model_path}")
        
        logger.info(f"Found {len(gguf_files)} .gguf file(s)")
        
        # Check for split models (e.g., -00001-of-00002.gguf)
        split_files = [f for f in gguf_files if _is_split_first_part_gguf(f)]
        
        if len(gguf_files) == 1:
            full_model_path = gguf_files[0]
            logger.info(f"Using single .gguf file: {os.path.basename(full_model_path)}")
        elif split_files:
            full_model_path = split_files[0]
            logger.info(f"Split model detected, using first part: {os.path.basename(full_model_path)}")
        else:
            # Use the first file found
            full_model_path = gguf_files[0]
            logger.warning(f"Multiple .gguf files found, using: {os.path.basename(full_model_path)}")
    
    # Cache model by full path
    if full_model_path not in _llamacpp_model_cache:
        logger.info(f"Model not in cache, loading from disk: {full_model_path}")
        logger.info(f"Checking if path exists: {os.path.exists(full_model_path)}")
        
        # Check if model file exists
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found: {full_model_path}")
        
        try:
            # Initialize the model
            llm = Llama(
                model_path=full_model_path,
                n_gpu_layers=-1,  # Use all GPU layers available
                verbose=False,
                n_ctx=4096 * 4,   # 16K context
            )
            logger.success(f"Loaded llama.cpp model: {full_model_path}")
            _llamacpp_model_cache[full_model_path] = llm
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise
    
    return _llamacpp_model_cache[full_model_path]


def run_inference_llamacpp(prompt, max_tokens, temperature, top_p, model_path):
    """
    Run inference using llama.cpp Python bindings.
    
    Args:
        prompt: The input prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        model_path: Path to the GGUF model file
        
    Returns:
        Tuple of (response_text, model_outputs, input_length)
        - response_text: Generated text
        - model_outputs: None for llama.cpp models (activations not available)
        - input_length: Approximate input length (estimated)
    """
    logger.info(f"-" * 80)
    logger.info(f"run_inference_llamacpp() ENTRY")
    logger.info(f"  model_path: {model_path}")
    logger.info(f"  prompt length: {len(prompt)} chars")
    logger.info(f"  max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")
    logger.info(f"-" * 80)
    
    logger.info(f"Starting llama.cpp inference with model: {model_path}")
    logger.info(f"Inference parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    start_time = time.time()
    
    # Get model from cache or load it
    logger.info(f"Getting llama.cpp model from cache or loading...")
    llm = get_llamacpp_model(model_path)
    logger.info(f"✓ Model loaded/retrieved successfully")
    
    # Run inference
    logger.info(f"Running llama.cpp inference with prompt of length {len(prompt)} characters")
    logger.info(f"Starting generation...")
    inference_start = time.time()
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=False,  # Don't include prompt in output
    )
    
    inference_time = time.time() - inference_start
    logger.info(f"✓ llama.cpp inference completed in {inference_time:.2f} seconds")
    logger.info(f"  Output type: {type(output)}")
    logger.info(f"  Output keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
    
    # Extract the generated text
    if isinstance(output, dict):
        # Single response format
        response_text = output.get("choices", [{}])[0].get("text", "")
    elif isinstance(output, list) and len(output) > 0:
        # List format
        response_text = output[0].get("choices", [{}])[0].get("text", "")
    else:
        response_text = ""
        logger.warning("Unexpected output format from llama.cpp")
    
    # Approximate the input length (llama.cpp doesn't easily expose this)
    # This is a rough estimate based on characters
    input_length = len(prompt) // 4  # Very rough approximation
    
    # llama.cpp doesn't provide activations, so return None for model_outputs
    model_outputs = None
    
    logger.info(f"Generated {len(response_text)} characters of response")
    trim_pos = None # llama.cpp doesn't provide trim position and we want to stay consistent with the other models
    return response_text, model_outputs, input_length, trim_pos


def trim_response(tokenizer, gen_ids, response_text):
    """
    Trim the response text at a specific sequence if TRIM_OUTPUT_AT is set.
    Starts checking for trim tokens only after the first 10 tokens to ensure
    a reasonable amount of content.
    
    Args:
        tokenizer: The tokenizer used for decoding
        gen_ids: Generated token IDs
        response_text: Original response text
        
    Returns:
        Tuple of (trimmed_text, trim_position)
        - trimmed_text: The response text, possibly trimmed
        - trim_position: Position where response was trimmed (or None if no trimming)
    """
    trim_at = os.environ.get("TRIM_OUTPUT_AT")
    trim_pos = None
    
    if trim_at is not None:
        if not hasattr(tokenizer, 'trim_token_ids'):
            raise RuntimeError(f"Tokenizer does not have cached trim token IDs for sequence: {repr(trim_at)}")
        
        # Find first occurrence of any trim token in the generated sequence, starting after 10 tokens
        for i, token_id in enumerate(gen_ids[10:], start=10):
            if token_id.item() in tokenizer.trim_token_ids:
                trim_pos = i
                token_str = tokenizer.decode([token_id.item()], skip_special_tokens=False)
                logger.debug(f"Found trim token at position {i}: ID={token_id.item()}, str={repr(token_str)}")
                break
        
        if trim_pos is not None:
            original_length = len(response_text)
            # Decode only up to the trim position
            response_text = tokenizer.decode(gen_ids[:trim_pos], skip_special_tokens=True)
            logger.info(f"Trimmed output at token position {trim_pos} from {original_length} to {len(response_text)} characters")
    
    return response_text, trim_pos


def run_inference(prompt, max_tokens, temperature, top_p, model_name=DEFAULT_MODEL, auth_token=None):
    """
    Run inference using the model and tokenizer.

    Args:
        prompt: The input prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        model_name: Name of the model to use (HuggingFace model ID)
        auth_token: HuggingFace authentication token for gated models

    Returns:
        Tuple of (response_text, model_outputs, input_length, trim_position)
        - response_text: Generated text
        - model_outputs: Raw model outputs for the activation logger to process
        - input_length: Length of input tokens
        - trim_position: Position where response was trimmed (or None if no trimming)
    """
    logger.info(f"=" * 80)
    logger.info(f"run_inference() ENTRY")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  prompt length: {len(prompt)} chars")
    logger.info(f"  max_tokens: {max_tokens}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  top_p: {top_p}")
    logger.info(f"  auth_token: {'provided' if auth_token else 'not provided'}")
    logger.info(f"=" * 80)
    
    logger.info(f"Starting inference with model: {model_name}")
    logger.info(f"Inference parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    logger.info(f"Prompt length: {len(prompt)} characters")

    start_time = time.time()

    # Log system resources before inference
    log_system_resources("Pre-inference")

    # Check if this is a GGUF model path for llama.cpp
    # Match by file extension or directory/model name patterns
    model_lower = model_name.lower()
    logger.info(f"Checking model type - Original: {model_name}, Lowercase: {model_lower}")
    
    # Log each detection condition
    detection_checks = {
        "ends_with_gguf": model_lower.endswith('.gguf'),
        "contains_slash_gguf": '/gguf' in model_lower,
        "contains_gguf_slash": 'gguf/' in model_lower,
        "contains_dash_gguf": '-gguf' in model_lower,
        "contains_q6_k": 'q6_k' in model_lower,
        "contains_q4_k": 'q4_k' in model_lower,
        "contains_q5_k": 'q5_k' in model_lower,
        "contains_iq3_m": 'iq3_m' in model_lower,
        "contains_iq4": 'iq4' in model_lower
    }
    logger.info(f"GGUF detection checks: {detection_checks}")
    
    is_gguf = any(detection_checks.values())
    logger.info(f"GGUF detection result: {is_gguf}")
    
    if is_gguf:
        logger.info(f"✓ Detected GGUF model, using llama.cpp for inference")
        logger.info(f"Calling run_inference_llamacpp() with model_path={model_name}")
        try:
            logger.info(f"Entering run_inference_llamacpp()...")
            result = run_inference_llamacpp(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                model_path=model_name
            )
            logger.info(f"✓ GGUF inference completed successfully")
            logger.info(f"  Response length: {len(result[0]) if result and result[0] else 0} chars")
            return result
        except Exception as e:
            logger.error(f"✗ GGUF inference failed with exception: {type(e).__name__}")
            logger.error(f"  Error message: {e}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise

    # Otherwise use the standard Hugging Face model loading
    logger.info(f"Using Hugging Face transformers for inference")

    try:
        # Get model and tokenizer
        logger.info(f"Loading model and tokenizer for {model_name}")
        model_load_start = time.time()
        model, tokenizer = get_model_and_tokenizer(model_name, auth_token)
        model_load_time = time.time() - model_load_start
        logger.info(f"Model and tokenizer loaded in {model_load_time:.2f} seconds")

        # Ensure input_ids are on the same device as the model
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")

        # Tokenization
        tokenization_start = time.time()
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        tokenization_time = time.time() - tokenization_start

        logger.info(f"Input tokenized to {input_ids.shape[1]} tokens in {tokenization_time:.3f} seconds")

        # Log memory before generation
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory before generation: {gpu_memory_before:.2f}GB")

        # Start generation
        logger.info(f"Starting generation with {model_name}")
        generation_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=True if temperature > 0.0 else False,
                pad_token_id=tokenizer.pad_token_id
            )

        generation_time = time.time() - generation_start
        logger.info(f"Generation completed in {generation_time:.2f} seconds")

        # Log memory after generation
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after generation: {gpu_memory_after:.2f}GB")

        # Get generated tokens (excluding prompt)
        gen_ids = outputs.sequences[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Trim the response if needed
        trim_start = time.time()
        response_text, trim_pos = trim_response(tokenizer, gen_ids, response_text)
        trim_time = time.time() - trim_start

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Generated {len(gen_ids)} new tokens ({len(response_text)} characters)")
        logger.info(f"Timing breakdown - Model load: {model_load_time:.2f}s, "
                   f"Tokenization: {tokenization_time:.3f}s, Generation: {generation_time:.2f}s, "
                   f"Trimming: {trim_time:.3f}s")

        # Log system resources after inference
        log_system_resources("Post-inference")

        # Return the full outputs object for the activation logger to process
        return response_text, outputs, input_ids.shape[1], trim_pos

    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        # Log system resources on error
        log_system_resources("Error during inference")
        raise


# OpenAI API request/response models for compatibility
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    lmdb_path: Optional[str] = None  # Optional field for custom LMDB path
    auth_token: Optional[str] = None  # Optional field for HuggingFace auth token
    multi_sample: Optional[bool] = False  # Optional flag to enable multi-sample logging
    sample_group_id: Optional[str] = None  # Optional group ID to link samples
    sample_index: Optional[int] = None  # Optional index within the group
    request_id: Optional[str] = None  # Optional request ID for unique sample keys


class Choice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]


# Chat request/response models for OpenAI API compatibility
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    lmdb_path: Optional[str] = None  # Optional field for custom LMDB path
    auth_token: Optional[str] = None  # Optional field for HuggingFace auth token
    multi_sample: Optional[bool] = False  # Optional flag to enable multi-sample logging
    sample_group_id: Optional[str] = None  # Optional group ID to link samples
    sample_index: Optional[int] = None  # Optional index within the group
    request_id: Optional[str] = None  # Optional request ID for unique sample keys


class ChatCompletionChoice(BaseModel):
    message: Message
    index: int
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class SetOverwriteLMDBPathRequest(BaseModel):
    """Request model for setting the overwrite LMDB path."""
    lmdb_path: Optional[str] = None


class SetOverwriteLMDBPathResponse(BaseModel):
    """Response model for the set overwrite LMDB path endpoint."""
    success: bool
    previous_value: Optional[str]
    current_value: Optional[str]
    message: str


class SetOverwriteTemperatureRequest(BaseModel):
    """Request model for setting the overwrite temperature."""
    temperature: Optional[float] = None


class SetOverwriteTemperatureResponse(BaseModel):
    """Response model for the set overwrite temperature endpoint."""
    success: bool
    previous_value: Optional[float]
    current_value: Optional[float]
    message: str


class SetOverwriteTopPRequest(BaseModel):
    """Request model for setting the overwrite top_p."""
    top_p: Optional[float] = None


class SetOverwriteTopPResponse(BaseModel):
    """Response model for the set overwrite top_p endpoint."""
    success: bool
    previous_value: Optional[float]
    current_value: Optional[float]
    message: str


# Utility for hashing prompt
def prompt_hash(prompt: str) -> str:
    """
    Create a hash of the prompt to use as a unique key for LMDB.
    
    Args:
        prompt: The prompt text to hash
        
    Returns:
        SHA256 hash of the prompt
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_entry_key(prompt: str, request_id: Optional[str], multi_sample: bool) -> Tuple[str, str, Optional[str]]:
    """
    Build a unique entry key for logging.

    Args:
        prompt: Prompt string to hash
        request_id: Optional request ID for uniqueness
        multi_sample: Whether multi-sample logging is enabled

    Returns:
        Tuple of (entry_key, prompt_key, resolved_request_id)
    """
    prompt_key = prompt_hash(prompt)
    if not multi_sample:
        return prompt_key, prompt_key, None

    resolved_request_id = request_id or str(uuid.uuid4())[:8]
    entry_key = f"{prompt_key}_{resolved_request_id}"
    return entry_key, prompt_key, resolved_request_id


def apply_overwrites(request_params):
    """
    Apply global overwrites to request parameters.
    
    Args:
        request_params: Dictionary containing original request parameters
        
    Returns:
        Dictionary with overwritten parameters where applicable
    """
    # Create a copy to avoid modifying the original
    params = request_params.copy()
    
    # Set default values for None parameters
    if params.get('temperature') is None:
        params['temperature'] = 0.0
        logger.info(f"Using default temperature: 0.0 for None value in request")
        
    if params.get('top_p') is None:
        params['top_p'] = 1.0
        logger.info(f"Using default top_p: 1.0 for None value in request")
    
    # Apply temperature overwrite if set
    if overwrite_temperature is not None:
        original_temp = params.get('temperature')
        params['temperature'] = overwrite_temperature
        logger.info(f"Using overwrite temperature: {overwrite_temperature} instead of request temperature: {original_temp}")
    
    # Apply top_p overwrite if set
    if overwrite_top_p is not None:
        original_top_p = params.get('top_p')
        params['top_p'] = overwrite_top_p
        logger.info(f"Using overwrite top_p: {overwrite_top_p} instead of request top_p: {original_top_p}")
    
    # Apply activation storage path overwrite if set
    if overwrite_activations_path is not None:
        params['activations_path'] = overwrite_activations_path
        logger.info(f"Using overwrite activations path: {overwrite_activations_path}")
    elif 'activations_path' not in params or not params.get('activations_path'):
        # If no path specified and no overwrite, use the environment variable
        params['activations_path'] = DEFAULT_ACTIVATIONS_PATH
        logger.info(f"Using default activations path from environment: {params['activations_path']}")

    # Apply logger type overwrite if set
    if overwrite_logger_type is not None:
        params['logger_type'] = overwrite_logger_type
        logger.info(f"Using overwrite logger type: {overwrite_logger_type}")
    elif 'logger_type' not in params or not params.get('logger_type'):
        # If no logger type specified and no overwrite, use the environment variable
        params['logger_type'] = DEFAULT_LOGGER_TYPE
        logger.info(f"Using default logger type from environment: {params['logger_type']}")

    # Apply LMDB path overwrite if set (backward compatibility)
    if overwrite_lmdb_path is not None:
        params['lmdb_path'] = overwrite_lmdb_path
        params['activations_path'] = overwrite_lmdb_path  # Also set activations_path for consistency
        logger.info(f"Using overwrite LMDB path: {overwrite_lmdb_path}")
    elif 'lmdb_path' not in params or not params.get('lmdb_path'):
        # If no path specified and no overwrite, use the activations_path or environment variable
        params['lmdb_path'] = params.get('activations_path', DEFAULT_ACTIVATIONS_PATH)
        logger.info(f"Using activations path as LMDB path for backward compatibility: {params['lmdb_path']}")

    return params


def get_logger_for_request(request_params):
    """
    Create a new logger instance for each request to avoid LMDB assertion errors.

    Args:
        request_params: Dictionary containing request parameters (with overwrites applied)

    Returns:
        Tuple of (logger_to_use, custom_logger, used_custom_path)
        - logger_to_use: Logger to use for this request
        - custom_logger: Custom logger instance if created (to be closed after use)
        - used_custom_path: Boolean indicating if a custom path was used
    """
    activations_path = request_params.get('activations_path', request_params.get('lmdb_path'))
    logger_type = request_params.get('logger_type', DEFAULT_LOGGER_TYPE)

    # Get target layers and sequence mode from environment variables
    target_layers = os.environ.get("ACTIVATION_TARGET_LAYERS", "all")
    sequence_mode = os.environ.get("ACTIVATION_SEQUENCE_MODE", "all")

    logger.info(f"Creating activation logger - Type: {logger_type}, Path: {activations_path}, "
               f"Target layers: {target_layers}, Sequence mode: {sequence_mode}")

    try:
        # Always create a new logger instance to avoid assertion errors
        if activations_path and activations_path.strip():
            # Create and use a custom logger with the specified path and type
            logger.info(f"Using custom activations path: {activations_path}")

            if logger_type == "json":
                # Check if directory exists and log directory info
                if os.path.exists(activations_path):
                    file_count = len([f for f in os.listdir(activations_path) if f.endswith('.npy')])
                    dir_size = sum(os.path.getsize(os.path.join(activations_path, f))
                                 for f in os.listdir(activations_path)) / 1024**2  # MB
                    logger.info(f"Existing JSON activation directory: {file_count} files, {dir_size:.2f}MB")

                custom_logger = JsonActivationsLogger(
                    output_dir=activations_path,
                    target_layers=target_layers,
                    sequence_mode=sequence_mode,
                    read_only=False
                )
                logger.info(f"Created JsonActivationsLogger for {activations_path}")
            else:  # default to lmdb
                # Check if LMDB exists and log info
                if os.path.exists(activations_path):
                    file_size = os.path.getsize(activations_path) / 1024**2  # MB
                    logger.info(f"Existing LMDB file: {file_size:.2f}MB")

                custom_logger = ActivationsLogger(
                    lmdb_path=activations_path,
                    map_size=DEFAULT_MAP_SIZE,
                    target_layers=target_layers,
                    sequence_mode=sequence_mode
                )
                logger.info(f"Created ActivationsLogger for {activations_path}")

            return custom_logger, custom_logger, True
        else:
            # Create a new logger with the default path from environment
            default_path = DEFAULT_ACTIVATIONS_PATH
            logger.info(f"Using default activations path: {default_path}")

            if logger_type == "json":
                new_logger = JsonActivationsLogger(
                    output_dir=default_path,
                    target_layers=target_layers,
                    sequence_mode=sequence_mode,
                    read_only=False
                )
                logger.info(f"Created default JsonActivationsLogger")
            else:  # default to lmdb
                new_logger = ActivationsLogger(
                    lmdb_path=default_path,
                    map_size=DEFAULT_MAP_SIZE,
                    target_layers=target_layers,
                    sequence_mode=sequence_mode
                )
                logger.info(f"Created default ActivationsLogger")

            return new_logger, new_logger, False

    except Exception as e:
        logger.error(f"Failed to create activation logger: {e}")
        logger.error(f"Logger type: {logger_type}, Path: {activations_path}")
        raise


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the server is running.

    Returns:
        Dict with status and timestamp
    """
    logger.info("Health check endpoint called")
    return {
        "status": "ok",
        "timestamp": time.time()
    }


@app.get("/status")
async def status_check():
    """
    Detailed status check endpoint showing server state.

    Returns:
        Dict with detailed server status information
    """
    logger.info("Status check endpoint called")
    
    # Get active requests info
    with request_lock:
        active_count = len(active_requests)
        active_list = [
            {
                "request_id": req_id,
                "endpoint": info["endpoint"],
                "model": info["model"],
                "duration": time.time() - info["start_time"]
            }
            for req_id, info in active_requests.items()
        ]
    
    # Get resource info
    memory = psutil.virtual_memory()
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        }
    else:
        gpu_info = {"available": False}
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "default_model": DEFAULT_MODEL,
        "activations_path": DEFAULT_ACTIVATIONS_PATH,
        "logger_type": DEFAULT_LOGGER_TYPE,
        "active_requests": active_count,
        "active_request_details": active_list,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / 1024**3,
        "gpu": gpu_info,
        "model_cache_size": len(_model_cache),
        "tokenizer_cache_size": len(_tokenizer_cache),
        "llamacpp_cache_size": len(_llamacpp_model_cache)
    }


@app.post("/restart")
async def restart_server():
    """
    Restart the server by triggering a graceful shutdown and restart.
    This endpoint should be called when the server becomes unresponsive.

    Returns:
        Dict with restart status
    """
    logger.warning("=" * 80)
    logger.warning("SERVER RESTART REQUESTED")
    logger.warning("=" * 80)

    # Log current state
    with request_lock:
        if active_requests:
            logger.warning(f"Restarting with {len(active_requests)} active requests:")
            for request_id, info in active_requests.items():
                duration = time.time() - info["start_time"]
                logger.warning(f"  [{request_id}] {info['endpoint']} - {duration:.2f}s")
        else:
            logger.info("No active requests at restart")

    log_system_resources("Pre-Restart")

    # Clear GPU cache if available
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    # Clear model caches
    logger.info("Clearing model caches...")
    _model_cache.clear()
    _tokenizer_cache.clear()
    _llamacpp_model_cache.clear()
    logger.info("Model caches cleared")

    # Schedule restart in background
    import asyncio
    async def do_restart():
        await asyncio.sleep(1)  # Give time to send response
        logger.warning("Initiating server restart...")
        os._exit(0)  # Force exit, supervisor/systemd should restart

    asyncio.create_task(do_restart())

    return {
        "status": "restarting",
        "message": "Server restart initiated",
        "timestamp": time.time()
    }


@app.post("/set_overwrite_lmdb_path", response_model=SetOverwriteLMDBPathResponse)
async def set_overwrite_lmdb_path(request: SetOverwriteLMDBPathRequest):
    """
    Set the overwrite LMDB path for all completion requests.
    When set, this path will override any path in completion requests.
    Set to None to use the path specified in each request or the default path.
    
    Args:
        request: SetOverwriteLMDBPathRequest with the new LMDB path
        
    Returns:
        SetOverwriteLMDBPathResponse with status and path information
    """
    global overwrite_lmdb_path
    
    # Store the previous value
    previous_value = overwrite_lmdb_path
    
    # Update the global variable
    overwrite_lmdb_path = request.lmdb_path
    
    # If overwrite is None, we'll use the environment variable
    if overwrite_lmdb_path is None:
        logger.info(f"Overwrite LMDB path removed. Will use ACTIVATION_LMDB_PATH={os.environ.get('ACTIVATION_LMDB_PATH', DEFAULT_LMDB_PATH)}")
    else:
        logger.info(f"Overwrite LMDB path set to: {overwrite_lmdb_path}")
    
    return SetOverwriteLMDBPathResponse(
        success=True,
        previous_value=previous_value,
        current_value=overwrite_lmdb_path,
        message="Overwrite LMDB path successfully updated"
    )


@app.post("/set_overwrite_temperature", response_model=SetOverwriteTemperatureResponse)
async def set_overwrite_temperature(request: SetOverwriteTemperatureRequest):
    """
    Set the overwrite temperature for all completion requests.
    When set, this temperature will override any temperature in completion requests.
    Set to None to use the temperature specified in each request.
    
    Args:
        request: SetOverwriteTemperatureRequest with the new temperature value
        
    Returns:
        SetOverwriteTemperatureResponse with status and temperature information
    """
    global overwrite_temperature
    
    # Store the previous value
    previous_value = overwrite_temperature
    
    # Update the global variable
    overwrite_temperature = request.temperature
    
    logger.info(f"Overwrite temperature changed from {previous_value} to {overwrite_temperature}")
    
    return SetOverwriteTemperatureResponse(
        success=True,
        previous_value=previous_value,
        current_value=overwrite_temperature,
        message="Overwrite temperature successfully updated"
    )


@app.post("/set_overwrite_top_p", response_model=SetOverwriteTopPResponse)
async def set_overwrite_top_p(request: SetOverwriteTopPRequest):
    """
    Set the overwrite top_p for all completion requests.
    When set, this top_p will override any top_p in completion requests.
    Set to None to use the top_p specified in each request.
    
    Args:
        request: SetOverwriteTopPRequest with the new top_p value
        
    Returns:
        SetOverwriteTopPResponse with status and top_p information
    """
    global overwrite_top_p
    
    # Store the previous value
    previous_value = overwrite_top_p
    
    # Update the global variable
    overwrite_top_p = request.top_p
    
    logger.info(f"Overwrite top_p changed from {previous_value} to {overwrite_top_p}")
    
    return SetOverwriteTopPResponse(
        success=True,
        previous_value=previous_value,
        current_value=overwrite_top_p,
        message="Overwrite top_p successfully updated"
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    OpenAI API-compatible endpoint for text completions with activation logging.
    
    Args:
        request: CompletionRequest object containing prompt and generation parameters
        
    Returns:
        CompletionResponse with generated text
    """
    # Generate unique request ID for tracking
    req_id = str(uuid.uuid4())[:8]
    
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    logger.info(f"[{req_id}] ========== NEW COMPLETION REQUEST ==========")
    logger.info(f"[{req_id}] Received completion request for model: {model_name}")
    logger.info(f"[{req_id}] Request model field: {request.model}")
    logger.info(f"[{req_id}] DEFAULT_MODEL value: {DEFAULT_MODEL}")
    logger.info(f"[{req_id}] Prompt length: {len(request.prompt)} characters")
    logger.info(f"[{req_id}] Max tokens: {request.max_tokens}")
    
    # Track request start
    track_request_start(req_id, "completions", model_name)
    
    # Apply any overwrites to request parameters
    logger.info(f"[{req_id}] Applying parameter overwrites...")
    params = apply_overwrites({
        'temperature': request.temperature,
        'top_p': request.top_p,
        'lmdb_path': request.lmdb_path if hasattr(request, 'lmdb_path') else None
    })
    logger.info(f"[{req_id}] Final parameters after overwrites: {params}")
    
    # Use the extracted inference function
    logger.info(f"[{req_id}] Starting inference with run_inference()...")
    response_text, model_outputs, input_length, trim_pos = run_inference(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=params['temperature'],
        top_p=params['top_p'],
        model_name=model_name,
        auth_token=request.auth_token
    )
    
    # Log to LMDB only if not a GGUF model (which doesn't provide activations)
    multi_sample = bool(
        request.multi_sample
        or request.sample_index is not None
        or request.sample_group_id is not None
        or request.request_id is not None
    )
    entry_key, prompt_key, resolved_request_id = build_entry_key(
        request.prompt,
        request.request_id,
        multi_sample
    )
    sample_group_id = request.sample_group_id or prompt_key
    
    if not model_name.endswith('.gguf') and model_outputs is not None:
        # Get appropriate logger based on parameters with overwrites
        logger_to_use, _, _ = get_logger_for_request(params)
        
        try:
            # Pass the model outputs directly to the logger
            logger_to_use.log_entry(entry_key, {
                "prompt": request.prompt,
                "response": response_text,
                "model_outputs": model_outputs,  # Pass the full model outputs
                "input_length": input_length,    # Pass the input length for reference
                "model": model_name,
                "trim_position": trim_pos,       # Pass the trim position
                "prompt_hash": prompt_key,
                "multi_sample": multi_sample,
                "sample_group_id": sample_group_id,
                "sample_index": request.sample_index,
                "request_id": resolved_request_id,
            })
        finally:
            # Always close the logger to free up resources
            logger_to_use.close()
    else:
        logger.info(f"Skipping activation logging for GGUF model: {model_name}")
    
    # Build OpenAI-compatible response
    logger.info(f"[{req_id}] Building response - Generated text length: {len(response_text)} characters")
    response = CompletionResponse(
        id=entry_key,
        created=int(time.time()),
        model=model_name,
        choices=[Choice(text=response_text, index=0)]
    )
    
    # Track request completion
    track_request_end(req_id, "completed")
    logger.info(f"[{req_id}] ========== REQUEST COMPLETED ==========")
    
    return response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI API-compatible endpoint for chat completions with activation logging.

    Args:
        request: ChatCompletionRequest object containing messages and generation parameters

    Returns:
        ChatCompletionResponse with generated message
    """
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())[:8]

    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    logger.info(f"[{request_id}] Received chat completion request for model: {model_name}")

    # Track request start
    track_request_start(request_id, "chat_completions", model_name)

    try:
        # Convert chat messages to a single prompt - simplified for most models
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        logger.info(f"[{request_id}] Converted {len(request.messages)} messages to prompt of {len(prompt)} characters")

        # Apply any overwrites to request parameters
        params = apply_overwrites({
            'temperature': request.temperature,
            'top_p': request.top_p,
            'lmdb_path': request.lmdb_path if hasattr(request, 'lmdb_path') else None
        })
        logger.info(f"[{request_id}] Applied parameter overwrites: {params}")

        # Log activation logging setup
        multi_sample = bool(
            request.multi_sample
            or request.sample_index is not None
            or request.sample_group_id is not None
            or request.request_id is not None
        )
        entry_key, prompt_key, resolved_request_id = build_entry_key(
            prompt,
            request.request_id,
            multi_sample
        )
        sample_group_id = request.sample_group_id or prompt_key
        logger.info(f"[{request_id}] Generated entry key: {entry_key}")

        # Use the extracted inference function
        logger.info(f"[{request_id}] Starting inference")
        inference_start = time.time()

        response_text, model_outputs, input_length, trim_pos = run_inference(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=params['temperature'],
            top_p=params['top_p'],
            model_name=model_name,
            auth_token=request.auth_token
        )

        inference_time = time.time() - inference_start
        logger.info(f"[{request_id}] Inference completed in {inference_time:.2f} seconds")

        # Log activation logging
        if not model_name.endswith('.gguf') and model_outputs is not None:
            logger.info(f"[{request_id}] Starting activation logging")
            activation_start = time.time()

            # Get appropriate logger based on parameters with overwrites
            logger_to_use, _, _ = get_logger_for_request(params)

            try:
                # Pass the model outputs directly to the logger
                logger_to_use.log_entry(entry_key, {
                    "prompt": prompt,
                    "response": response_text,
                    "model_outputs": model_outputs,  # Pass the full model outputs
                    "input_length": input_length,    # Pass the input length for reference
                    "model": model_name,
                    "messages": [msg.model_dump() for msg in request.messages],
                    "trim_position": trim_pos,       # Pass the trim position
                    "prompt_hash": prompt_key,
                    "multi_sample": multi_sample,
                    "sample_group_id": sample_group_id,
                    "sample_index": request.sample_index,
                    "request_id": resolved_request_id,
                })

                activation_time = time.time() - activation_start
                logger.info(f"[{request_id}] Activation logging completed in {activation_time:.2f} seconds")

            finally:
                # Always close the logger to free up resources
                logger_to_use.close()
        else:
            logger.info(f"[{request_id}] Skipping activation logging for GGUF model: {model_name}")

        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=entry_key,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=Message(role="assistant", content=response_text),
                    index=0
                )
            ]
        )

        # Track successful completion
        track_request_end(request_id, "completed")
        logger.info(f"[{request_id}] Request completed successfully")
        logger.info(f"[{request_id}] Sending response to client - Response size: {len(response_text)} chars")

        return response

    except Exception as e:
        # Track failed completion
        track_request_end(request_id, "failed")
        logger.error(f"[{request_id}] Request failed with error: {e}")
        logger.error(f"[{request_id}] Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Add startup event to log system info and start monitoring
@app.on_event("startup")
async def startup_event():
    """Log system information on startup and start monitoring."""
    logger.info("=" * 80)
    logger.info("ACTIVATION LOGGING SERVER STARTUP")
    logger.info("=" * 80)

    # Log configuration
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Default activations path: {DEFAULT_ACTIVATIONS_PATH}")
    logger.info(f"Default logger type: {DEFAULT_LOGGER_TYPE}")
    logger.info(f"GGUF models directory: {GGUF_MODELS_DIR}")
    
    # Log all relevant environment variables
    logger.info("-" * 80)
    logger.info("ENVIRONMENT VARIABLES:")
    env_vars_to_log = [
        "DEFAULT_MODEL", "ACTIVATION_STORAGE_PATH", "ACTIVATION_LMDB_PATH",
        "ACTIVATION_LOGGER_TYPE", "GGUF_MODELS_DIR", "HF_TOKEN",
        "ACTIVATION_TARGET_LAYERS", "ACTIVATION_SEQUENCE_MODE",
        "TRIM_OUTPUT_AT", "SERVER_LOG_FILE"
    ]
    for var in env_vars_to_log:
        value = os.environ.get(var, "not set")
        # Mask token for security
        if var == "HF_TOKEN" and value != "not set":
            value = value[:10] + "..." if len(value) > 10 else "***"
        logger.info(f"  {var}: {value}")
    logger.info("-" * 80)

    # Log system resources
    log_system_resources("Startup")

    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.2f}GB")
    else:
        logger.info("CUDA not available, using CPU")

    # Start monitoring thread
    import threading
    def monitor_requests():
        while True:
            time.sleep(300)  # Check every 5 minutes
            log_long_running_requests()

    monitor_thread = threading.Thread(target=monitor_requests, daemon=True)
    monitor_thread.start()
    logger.info("Started request monitoring thread")

    logger.info("Server startup completed")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("=" * 80)
    logger.info("ACTIVATION LOGGING SERVER SHUTDOWN")
    logger.info("=" * 80)

    # Log any remaining active requests
    with request_lock:
        if active_requests:
            logger.warning(f"Shutting down with {len(active_requests)} active requests:")
            for request_id, info in active_requests.items():
                duration = time.time() - info["start_time"]
                logger.warning(f"  [{request_id}] {info['endpoint']} - {duration:.2f}s")
        else:
            logger.info("No active requests at shutdown")

    log_system_resources("Shutdown")
    logger.info("Server shutdown completed")
    logger.info("=" * 80)

if __name__ == "__main__":
    uvicorn.run("activation_logging.server:app", host="0.0.0.0", port=8000, reload=False)