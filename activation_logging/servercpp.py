"""
servercpp.py
OpenAI API-compatible FastAPI server for llama.cpp models.
Simplified version that only supports GGUF models via llama.cpp.
"""
import os
import time
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger
import uvicorn
from llama_cpp import Llama  # Import for llama.cpp Python bindings

# Default model path if none specified in request
DEFAULT_MODEL = "Llama-3.3-70B-Instruct-IQ3_M.gguf"
app = FastAPI()

# llama.cpp model cache to avoid reloading for each request
_llamacpp_model_cache = {}

# Default path for GGUF models
GGUF_MODELS_DIR = os.environ.get("GGUF_MODELS_DIR", "")


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
        Generated text
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
    
    logger.info(f"Generated {len(response_text)} characters of response")
    
    return response_text


# OpenAI API request/response models for compatibility
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None


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
    Create a hash of the prompt to use as a unique key.
    
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
    OpenAI API-compatible endpoint for text completions.
    
    Args:
        request: CompletionRequest object containing prompt and generation parameters
        
    Returns:
        CompletionResponse with generated text
    """
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    logger.info(f"Received completion request for model: {model_name}")
    
    # Use the inference function
    response_text = run_inference_llamacpp(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        model_path=model_name
    )
    
    # Generate a unique ID for this request
    entry_key = prompt_hash(request.prompt)
    
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
    OpenAI API-compatible endpoint for chat completions.
    
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
    
    # Run inference
    response_text = run_inference_llamacpp(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        model_path=model_name
    )
    
    # Generate a unique ID for this request
    entry_key = prompt_hash(prompt)
    
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
    uvicorn.run("activation_logging.servercpp:app", host="0.0.0.0", port=8000, reload=False) 