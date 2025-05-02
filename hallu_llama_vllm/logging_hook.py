"""
logging_hook.py
Custom vLLM RequestOutputHook for logging activations, prompt, and response to LMDB.
"""
from vllm.engine.hooks import RequestOutputHook
from vllm.outputs import RequestOutput
from typing import Any, Dict
from activations_logger import ActivationsLogger
import hashlib
import os

class ActivationLoggingHook(RequestOutputHook):
    def __init__(self, lmdb_path: str = None):
        # Allow LMDB path to be set via environment variable for experiment parameterization
        env_path = os.environ.get("ACTIVATION_LMDB_PATH")
        if lmdb_path is None:
            lmdb_path = env_path if env_path is not None else "lmdb_data/activations.lmdb"
        self.logger = ActivationsLogger(lmdb_path)

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def post_request_output(self, request_id: str, request_dict: Dict[str, Any], output: RequestOutput) -> None:
        prompt = request_dict.get("prompt")
        if prompt is None:
            return
        # Get response text
        response = output.outputs[0].text if output.outputs and len(output.outputs) > 0 else ""
        # Get last layer activations if present
        activations = None
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            # output.hidden_states: list of (layers, batch, seq, hidden)
            activations = output.hidden_states[-1].cpu().numpy()
        entry_key = self._prompt_hash(prompt)
        self.logger.log_entry(entry_key, {
            "prompt": prompt,
            "response": response,
            "activations": activations,
        })

    def close(self):
        self.logger.close()
