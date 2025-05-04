"""
server.py
OpenAI API-compatible FastAPI server for Llama 3.3 Instruct with activation logging.
"""
import os
import hashlib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hallu_llama_vllm.activations_logger import ActivationsLogger
from loguru import logger
import uvicorn


DEFAULT_MODEL =  "mistralai/Mistral-7B-Instruct-v0.2"
app = FastAPI()
activation_logger = ActivationsLogger()

# Model cache to avoid reloading for each request
_model_cache = {}
_tokenizer_cache = {}

def get_model_and_tokenizer(model_name: str):
    if model_name not in _model_cache:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Use GPU if available, else CPU
        if torch.cuda.is_available():
            device_map = "cuda"
            torch_dtype = torch.float16
            logger.info("Using GPU for inference.")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            logger.info("Using CPU for inference.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            output_hidden_states=True
        )
        logger.success(f"Model loaded: {model_name} on {device_map}")
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
    return _model_cache[model_name], _tokenizer_cache[model_name]

# OpenAI API request/response models
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

# Utility for hashing prompt
def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    logger.info(f"Received completion request for model: {request.model if request.model else DEFAULT_MODEL}")
    # Allow override of model via request, else use default
    model_name = request.model if request.model else DEFAULT_MODEL
    model, tokenizer = get_model_and_tokenizer(model_name)
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
    activation_logger.log_entry(entry_key, {
        "prompt": request.prompt,
        "response": response_text,
        "activations": activations,
        "model": model_name,
    })
    # Build OpenAI-compatible response
    import time
    return CompletionResponse(
        id=entry_key,
        created=int(time.time()),
        model=model_name,
        choices=[Choice(text=response_text, index=0)]
    )

@app.on_event("shutdown")
def shutdown_event():
    activation_logger.close()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
