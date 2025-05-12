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
    # Ensure input_ids are on the same device as the model
    device = next(model.parameters()).device
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
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
    
    # Ensure input_ids are on the same device as the model
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
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