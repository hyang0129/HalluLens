"""
server.py
vLLM-based Llama 3.3 Instruct inference server with last-layer activation logging.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from vllm import LLM, SamplingParams
from activations_logger import ActivationsLogger
import hashlib

MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")

# ---- Request/Response Schemas ----
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    response: str
    activations_key: str

# ---- FastAPI Setup ----
app = FastAPI()
logger = ActivationsLogger()

# ---- Model Setup ----
llm = None
def get_llm():
    global llm
    if llm is None:
        llm = LLM(model=MODEL_PATH)
    return llm

# ---- Activation Hook ----
def get_last_layer_activations(model, input_ids: torch.Tensor, output_hidden_states: List[torch.Tensor]):
    # output_hidden_states: list of (batch, seq, hidden) for each layer
    # Take last layer's hidden states (per token)
    return output_hidden_states[-1].cpu().numpy()

# ---- Inference Endpoint ----
def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    llm = get_llm()
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    # vLLM API: Generate with hidden state return
    outputs = llm.generate(
        [request.prompt],
        sampling_params=sampling_params,
        return_hidden_states=True
    )
    if not outputs or len(outputs) == 0:
        raise HTTPException(status_code=500, detail="No output from model.")
    output = outputs[0]
    response = output.outputs[0].text
    # vLLM returns hidden_states as a list of tensors (layers, batch, seq, hidden)
    activations = get_last_layer_activations(llm.model, output.prompt_token_ids, output.hidden_states)
    # Use prompt hash as unique key
    entry_key = prompt_hash(request.prompt)
    logger.log_entry(entry_key, {
        "prompt": request.prompt,
        "response": response,
        "activations": activations,
    })
    return GenerateResponse(response=response, activations_key=entry_key)

# ---- Shutdown ----
@app.on_event("shutdown")
def shutdown_event():
    logger.close()
