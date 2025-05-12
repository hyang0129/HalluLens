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


# Default model if none specified in request
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
app = FastAPI()
activation_logger = ActivationsLogger()

# Model cache to avoid reloading for each request
_model_cache = {}
_tokenizer_cache = {}

# Temperature overwrite variable, None means use request temperature
overwrite_temperature = None


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
        
        # Then load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        logger.success(f"Model loaded: {model_name} on {device_map}")
        _model_cache[cache_key] = model
        _tokenizer_cache[cache_key] = tokenizer
    
    return _model_cache[cache_key], _tokenizer_cache[cache_key]


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


class SetDefaultLMDBPathRequest(BaseModel):
    """Request model for changing the default LMDB path."""
    lmdb_path: str


class SetDefaultLMDBPathResponse(BaseModel):
    """Response model for the change default LMDB path endpoint."""
    success: bool
    previous_path: str
    current_path: str
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


@app.post("/set_default_lmdb_path", response_model=SetDefaultLMDBPathResponse)
async def set_default_lmdb_path(request: SetDefaultLMDBPathRequest):
    """
    Set the default LMDB path for activation logging.
    This will be used when no path is specified in completion requests.
    
    Args:
        request: SetDefaultLMDBPathRequest with the new LMDB path
        
    Returns:
        SetDefaultLMDBPathResponse with status and path information
    """
    global activation_logger
    
    # Store the previous path
    previous_path = activation_logger.env.path()
    
    # Close the current logger
    activation_logger.close()
    
    # Create a new logger with the new path
    try:
        new_path = request.lmdb_path
        
        # Create directory if it doesn't exist
        lmdb_dir = os.path.dirname(new_path)
        if lmdb_dir and not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir, exist_ok=True)
            logger.info(f"Created LMDB directory: {lmdb_dir}")
        
        # Create new logger
        activation_logger = ActivationsLogger(lmdb_path=new_path)
        
        logger.info(f"Default LMDB path changed from {previous_path} to {new_path}")
        
        return SetDefaultLMDBPathResponse(
            success=True,
            previous_path=previous_path,
            current_path=activation_logger.env.path(),
            message=f"Default LMDB path successfully changed to {new_path}"
        )
    except Exception as e:
        # If there's an error, try to recreate the original logger
        activation_logger = ActivationsLogger(lmdb_path=previous_path)
        logger.error(f"Error changing default LMDB path: {e}")
        
        return SetDefaultLMDBPathResponse(
            success=False,
            previous_path=previous_path,
            current_path=previous_path,
            message=f"Error changing default LMDB path: {str(e)}"
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


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    OpenAI API-compatible endpoint for text completions with activation logging.
    
    Args:
        request: CompletionRequest object containing prompt and generation parameters
        
    Returns:
        CompletionResponse with generated text
    """
    logger.info(f"Received completion request for model: {request.model if request.model else DEFAULT_MODEL}")
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    model, tokenizer = get_model_and_tokenizer(model_name, request.auth_token)
    
    # Use overwrite temperature if set, otherwise use request temperature
    temperature = overwrite_temperature if overwrite_temperature is not None else request.temperature
    if overwrite_temperature is not None:
        logger.info(f"Using overwrite temperature: {temperature} instead of request temperature: {request.temperature}")
    
    # Ensure input_ids are on the same device as the model
    device = next(model.parameters()).device
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=temperature,
            top_p=request.top_p,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    # Get generated tokens (excluding prompt)
    gen_ids = outputs.sequences[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Get last layer activations for generated tokens
    hidden_states = outputs.hidden_states[-1][0]  # (seq, hidden)
    activations = hidden_states[-len(gen_ids):].cpu().numpy()
    # Log to LMDB
    entry_key = prompt_hash(request.prompt)
    # Use custom LMDB path if provided and non-empty, else use default
    logger_to_use = activation_logger
    custom_logger = None
    if hasattr(request, 'lmdb_path') and request.lmdb_path and request.lmdb_path.strip():
        custom_logger = ActivationsLogger(lmdb_path=request.lmdb_path)
        logger_to_use = custom_logger
    logger_to_use.log_entry(entry_key, {
        "prompt": request.prompt,
        "response": response_text,
        "activations": activations,
        "model": model_name,
    })
    if custom_logger is not None:
        custom_logger.close()
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
    logger.info(f"Received chat completion request for model: {request.model if request.model else DEFAULT_MODEL}")
    # Convert chat messages to a single prompt - simplified for most models
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    model, tokenizer = get_model_and_tokenizer(model_name, request.auth_token)
    
    # Use overwrite temperature if set, otherwise use request temperature
    temperature = overwrite_temperature if overwrite_temperature is not None else request.temperature
    if overwrite_temperature is not None:
        logger.info(f"Using overwrite temperature: {temperature} instead of request temperature: {request.temperature}")
    
    # Ensure input_ids are on the same device as the model
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=temperature,
            top_p=request.top_p,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    # Get generated tokens (excluding prompt)
    gen_ids = outputs.sequences[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Get last layer activations for generated tokens
    hidden_states = outputs.hidden_states[-1][0]  # (seq, hidden)
    activations = hidden_states[-len(gen_ids):].cpu().numpy()
    
    # Log to LMDB
    entry_key = prompt_hash(prompt)
    
    # Use custom LMDB path if provided and non-empty, else use default
    logger_to_use = activation_logger
    custom_logger = None
    if hasattr(request, 'lmdb_path') and request.lmdb_path and request.lmdb_path.strip():
        custom_logger = ActivationsLogger(lmdb_path=request.lmdb_path)
        logger_to_use = custom_logger
    
    logger_to_use.log_entry(entry_key, {
        "prompt": prompt,
        "response": response_text,
        "activations": activations,
        "model": model_name,
        "messages": [msg.dict() for msg in request.messages]
    })
    
    if custom_logger is not None:
        custom_logger.close()
    
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


@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when shutting down the server."""
    activation_logger.close()


if __name__ == "__main__":
    uvicorn.run("activation_logging.server:app", host="0.0.0.0", port=8000, reload=False) 