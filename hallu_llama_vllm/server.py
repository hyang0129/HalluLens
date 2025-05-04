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
from activations_logger import ActivationsLogger
import uvicorn

MODEL_NAME = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")

app = FastAPI()
logger = ActivationsLogger()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", output_hidden_states=True)

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
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(model.device)
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
    logger.log_entry(entry_key, {
        "prompt": request.prompt,
        "response": response_text,
        "activations": activations,
    })
    # Build OpenAI-compatible response
    return CompletionResponse(
        id=entry_key,
        created=int(torch.time.time()),
        model=request.model,
        choices=[Choice(text=response_text, index=0)]
    )

@app.on_event("shutdown")
def shutdown_event():
    logger.close()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
