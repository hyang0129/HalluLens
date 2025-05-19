"""
server.py
OpenAI API-compatible FastAPI server with activation logging.
Designed to be compatible with vLLM serve and the utils/lm.py interface.
"""
import os
import hashlib
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import uvicorn
from activation_logging.activations_logger import ActivationsLogger
from llama_cpp import Llama  # Import for llama.cpp Python bindings


# Default model if none specified in request
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
app = FastAPI()

# Default LMDB path
DEFAULT_LMDB_PATH = os.environ.get("ACTIVATION_LMDB_PATH", "lmdb_data/activations.lmdb")

# Model cache to avoid reloading for each request
_model_cache = {}
_tokenizer_cache = {}
_llamacpp_model_cache = {}  # Cache for llama.cpp models

# Temperature overwrite variable, None means use request temperature
overwrite_temperature = None

# LMDB path overwrite variable, None means use the default path from environment
overwrite_lmdb_path = None

# Top-p overwrite variable, None means use request top_p
overwrite_top_p = None

# Default path for GGUF models
GGUF_MODELS_DIR = os.environ.get("GGUF_MODELS_DIR", "")


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
        model_path: Path to the GGUF model file
        
    Returns:
        Loaded llama.cpp model instance
    """
    # Check if the model path is relative and if so, join with models directory
    if not os.path.isabs(model_path):
        full_model_path = os.path.join(GGUF_MODELS_DIR, model_path)
    else:
        full_model_path = model_path
    
    # Cache model by full path
    if full_model_path not in _llamacpp_model_cache:
        logger.info(f"Loading llama.cpp model: {full_model_path}")
        
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
    logger.info(f"Starting llama.cpp inference with model: {model_path}")
    logger.info(f"Inference parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    start_time = time.time()
    
    # Get model from cache or load it
    llm = get_llamacpp_model(model_path)
    
    # Run inference
    logger.info(f"Running llama.cpp inference with prompt of length {len(prompt)} characters")
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=False,  # Don't include prompt in output
    )
    
    inference_time = time.time() - start_time
    logger.info(f"llama.cpp inference completed in {inference_time:.2f} seconds")
    
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
        
        # Find first occurrence of any trim token in the generated sequence
        for i, token_id in enumerate(gen_ids):
            if token_id.item() in tokenizer.trim_token_ids:
                trim_pos = max(1, i)  # Ensure trim_pos is at least 1
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
    logger.info(f"Starting inference with model: {model_name}")
    logger.info(f"Inference parameters: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    start_time = time.time()
    
    # Check if this is a GGUF model path for llama.cpp
    if model_name.endswith('.gguf'):
        logger.info(f"Detected GGUF model, using llama.cpp for inference")
        return run_inference_llamacpp(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model_path=model_name
        )
    
    # Otherwise use the standard Hugging Face model loading
    logger.info(f"Using Hugging Face transformers for inference")
    
    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name, auth_token)
    
    # Ensure input_ids are on the same device as the model
    device = next(model.parameters()).device
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    logger.info(f"Input tokenized to {input_ids.shape[1]} tokens")
    
    # Start generation
    logger.info(f"Starting generation with {model_name}")
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
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    # Get generated tokens (excluding prompt)
    gen_ids = outputs.sequences[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Trim the response if needed
    response_text, trim_pos = trim_response(tokenizer, gen_ids, response_text)
    
    logger.info(f"Generated {len(gen_ids)} new tokens ({len(response_text)} characters)")
    
    # Return the full outputs object for the activation logger to process
    return response_text, outputs, input_ids.shape[1], trim_pos


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
    
    # Apply LMDB path overwrite if set
    if overwrite_lmdb_path is not None:
        params['lmdb_path'] = overwrite_lmdb_path
        logger.info(f"Using overwrite LMDB path: {overwrite_lmdb_path}")
    elif 'lmdb_path' not in params or not params.get('lmdb_path'):
        # If no path specified and no overwrite, use the environment variable
        params['lmdb_path'] = os.environ.get("ACTIVATION_LMDB_PATH", DEFAULT_LMDB_PATH)
        logger.info(f"Using default LMDB path from environment: {params['lmdb_path']}")
        
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
    lmdb_path = request_params.get('lmdb_path')
    
    # Always create a new logger instance to avoid LMDB assertion errors
    if lmdb_path and lmdb_path.strip():
        # Create and use a custom logger with the specified path
        custom_logger = ActivationsLogger(lmdb_path=lmdb_path)
        return custom_logger, custom_logger, True
    else:
        # Create a new logger with the default path from environment
        # This avoids the 'Assertion mp->mp_pgno != pgno failed in mdb_page_touch()' error
        default_path = os.environ.get("ACTIVATION_LMDB_PATH", DEFAULT_LMDB_PATH)
        new_logger = ActivationsLogger(lmdb_path=default_path)
        return new_logger, new_logger, False


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the server is running.
    
    Returns:
        Dict with status and timestamp
    """
    return {
        "status": "ok",
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
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    logger.info(f"Received completion request for model: {model_name}")
    
    # Apply any overwrites to request parameters
    params = apply_overwrites({
        'temperature': request.temperature,
        'top_p': request.top_p,
        'lmdb_path': request.lmdb_path if hasattr(request, 'lmdb_path') else None
    })
    
    # Use the extracted inference function
    response_text, model_outputs, input_length, trim_pos = run_inference(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=params['temperature'],
        top_p=params['top_p'],
        model_name=model_name,
        auth_token=request.auth_token
    )
    
    # Log to LMDB only if not a GGUF model (which doesn't provide activations)
    entry_key = prompt_hash(request.prompt)
    
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
            })
        finally:
            # Always close the logger to free up resources
            logger_to_use.close()
    else:
        logger.info(f"Skipping activation logging for GGUF model: {model_name}")
    
    # Build OpenAI-compatible response
    return CompletionResponse(
        id=entry_key,
        created=int(time.time()),
        model=model_name,
        choices=[Choice(text=response_text, index=0)]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI API-compatible endpoint for chat completions with activation logging.
    
    Args:
        request: ChatCompletionRequest object containing messages and generation parameters
        
    Returns:
        ChatCompletionResponse with generated message
    """
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    logger.info(f"Received chat completion request for model: {model_name}")
    
    # Convert chat messages to a single prompt - simplified for most models
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    # Apply any overwrites to request parameters
    params = apply_overwrites({
        'temperature': request.temperature,
        'top_p': request.top_p,
        'lmdb_path': request.lmdb_path if hasattr(request, 'lmdb_path') else None
    })
    
    # Use the extracted inference function
    response_text, model_outputs, input_length, trim_pos = run_inference(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=params['temperature'],
        top_p=params['top_p'],
        model_name=model_name,
        auth_token=request.auth_token
    )
    
    # Log to LMDB only if not a GGUF model (which doesn't provide activations)
    entry_key = prompt_hash(prompt)
    
    if not model_name.endswith('.gguf') and model_outputs is not None:
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
                "messages": [msg.dict() for msg in request.messages],
                "trim_position": trim_pos,       # Pass the trim position
            })
        finally:
            # Always close the logger to free up resources
            logger_to_use.close()
    else:
        logger.info(f"Skipping activation logging for GGUF model: {model_name}")
    
    # Build OpenAI-compatible response
    return ChatCompletionResponse(
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


if __name__ == "__main__":
    uvicorn.run("activation_logging.server:app", host="0.0.0.0", port=8000, reload=False) 