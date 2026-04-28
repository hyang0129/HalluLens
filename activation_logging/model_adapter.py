"""
model_adapter.py
Abstracts model-specific inference and activation extraction behind a clean interface.

The server calls ModelAdapter.infer() or ModelAdapter.infer_batch() and receives
InferenceResult objects with pre-extracted, per-layer activations.  The server and
Zarr logger are model-agnostic — all HuggingFace / GGUF specifics live here.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from loguru import logger


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Clean output from a ModelAdapter. No model-specific objects."""

    prompt: str
    response_text: str
    input_length: int                                      # token count of prompt
    activations: Optional[List[Optional["torch.Tensor"]]]  # per-layer, shape (1, seq_len, hidden)
    logprobs: Optional[Dict[str, Any]] = None              # token_ids, token_logprobs, topk_*
    trim_position: Optional[int] = None


# ---------------------------------------------------------------------------
# ModelAdapter base class
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Abstracts model-specific inference and activation extraction.

    Subclasses implement:
    - infer(prompt, ...) → InferenceResult
    - infer_batch(prompts, ...) → List[InferenceResult]   (true batching or sequential fallback)
    - supports_activations() → bool
    """

    @abstractmethod
    def infer(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> InferenceResult:
        """Run single-prompt inference. Returns clean InferenceResult."""
        ...

    def infer_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[InferenceResult]:
        """Run batched inference. Default: sequential fallback.

        Subclasses override for true batching.
        """
        return [self.infer(p, max_tokens, temperature, top_p) for p in prompts]

    @abstractmethod
    def supports_activations(self) -> bool:
        """Whether this adapter can capture hidden-state activations."""
        ...


# ---------------------------------------------------------------------------
# HuggingFace Transformers adapter
# ---------------------------------------------------------------------------

class HFTransformersAdapter(ModelAdapter):
    """Adapter for HuggingFace Transformers models with activation capture.

    Wraps model.generate(output_hidden_states=True) and handles the HF-specific
    hidden-state format:
        outputs.hidden_states[0]  — prompt hidden states, tuple of L tensors (B, P, H)
        outputs.hidden_states[1:] — per-generation-step states, each tuple of L tensors (B, 1, H)

    For batched calls, left-padding is applied so all prompts share the same
    input length dimension.  Per-request activations are split out from dim=0
    and padding tokens are stripped.
    """

    def __init__(
        self,
        model_name: str,
        auth_token: Optional[str] = None,
        target_layers: str = "all",
        sequence_mode: str = "all",
        enable_logprobs: bool = True,
        logprobs_top_k: int = 5,
    ) -> None:
        if target_layers not in ("all", "first_half", "second_half"):
            raise ValueError(
                "target_layers must be one of: 'all', 'first_half', 'second_half'"
            )
        if sequence_mode not in ("all", "prompt", "response"):
            raise ValueError(
                "sequence_mode must be one of: 'all', 'prompt', 'response'"
            )
        self._model_name = model_name
        self._auth_token = auth_token
        self._target_layers = target_layers
        self._sequence_mode = sequence_mode
        self._enable_logprobs = enable_logprobs
        self._logprobs_top_k = logprobs_top_k

        self._model = None
        self._tokenizer = None
        self._target_layer_indices: Optional[Set[int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supports_activations(self) -> bool:
        return True

    def infer(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> InferenceResult:
        results = self.infer_batch([prompt], max_tokens, temperature, top_p)
        return results[0]

    def infer_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[InferenceResult]:
        """Batched inference with per-request activation extraction."""
        self._ensure_loaded()
        model = self._model
        tokenizer = self._tokenizer
        device = next(model.parameters()).device

        # Left-pad for causal LM batching
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # True per-prompt input lengths (before left-padding is added)
        input_lengths = [
            len(tokenizer.encode(p, add_special_tokens=True)) for p in prompts
        ]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            output_hidden_states=True,
            output_scores=self._enable_logprobs,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        if "qwen3" in self._model_name.lower():
            generate_kwargs["enable_thinking"] = False

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        # Split results per request
        results: List[InferenceResult] = []
        for i, prompt in enumerate(prompts):
            gen_start = inputs.input_ids.shape[1]
            gen_ids = outputs.sequences[i, gen_start:]
            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Trim response (mirrors existing server.py trim_response logic)
            response_text, trim_pos = self._trim_response(tokenizer, gen_ids, response_text)

            activations = self._extract_activations(
                outputs, i, input_lengths[i], trim_pos
            )

            logprobs = None
            if self._enable_logprobs and hasattr(outputs, "scores"):
                logprobs = self._extract_logprobs(outputs, i, gen_ids, trim_pos)

            results.append(
                InferenceResult(
                    prompt=prompt,
                    response_text=response_text,
                    input_length=input_lengths[i],
                    activations=activations,
                    logprobs=logprobs,
                    trim_position=trim_pos,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        # Re-use existing server.py helper to avoid duplicating loading logic
        from activation_logging.server import get_model_and_tokenizer
        self._model, self._tokenizer = get_model_and_tokenizer(
            self._model_name, self._auth_token
        )

    def _trim_response(self, tokenizer, gen_ids: "torch.Tensor", response_text: str):
        """Delegate to the existing server.py trim_response helper."""
        try:
            from activation_logging.server import trim_response
            return trim_response(tokenizer, gen_ids, response_text)
        except (ImportError, AttributeError):
            return response_text, None

    def _resolve_target_layers(self, num_layers: int) -> Set[int]:
        if self._target_layers == "first_half":
            return set(range(num_layers // 2))
        elif self._target_layers == "second_half":
            return set(range(num_layers // 2, num_layers))
        return set(range(num_layers))

    def _extract_activations(
        self,
        outputs,
        batch_idx: int,
        input_length: int,
        trim_pos: Optional[int],
    ) -> Optional[List[Optional["torch.Tensor"]]]:
        """Extract per-layer activations for one request from batched HF generate output.

        HF generate output format (output_hidden_states=True):
            outputs.hidden_states[0]     — prompt hidden states
                tuple of L tensors, each (B, prompt_padded_len, H)
            outputs.hidden_states[1:]    — per-generation-step hidden states
                each is a tuple of L tensors, each (B, 1, H)

        We extract batch_idx from dim=0, concatenate generation steps, and
        strip left-padding tokens from the prompt portion.
        """
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return None

        all_hidden_states = outputs.hidden_states
        prompt_hidden = all_hidden_states[0]   # tuple of L tensors
        gen_hiddens = list(all_hidden_states[1:])  # list of tuples

        if trim_pos is not None:
            gen_hiddens = gen_hiddens[:trim_pos]

        num_layers = len(prompt_hidden)
        if self._target_layer_indices is None:
            self._target_layer_indices = self._resolve_target_layers(num_layers)

        # For batched input with left-padding, skip padding tokens
        padded_prompt_len = prompt_hidden[0].shape[1]
        pad_offset = max(0, padded_prompt_len - input_length)
        prompt_len = min(input_length, padded_prompt_len)
        response_len = len(gen_hiddens)

        activations: List[Optional["torch.Tensor"]] = []
        for layer_idx in range(num_layers):
            if layer_idx not in self._target_layer_indices:
                activations.append(None)
                continue

            # Prompt activations: slice this request, skip padding
            prompt_act = prompt_hidden[layer_idx][
                batch_idx : batch_idx + 1, pad_offset : pad_offset + prompt_len, :
            ]

            # Response activations: concatenate per-step states for this request
            if response_len > 0:
                response_act = torch.cat(
                    [step[layer_idx][batch_idx : batch_idx + 1, :, :] for step in gen_hiddens],
                    dim=1,
                )
            else:
                response_act = prompt_act.new_zeros((1, 0, prompt_act.shape[-1]))

            if self._sequence_mode == "prompt":
                activations.append(prompt_act)
            elif self._sequence_mode == "response":
                activations.append(response_act)
            else:
                activations.append(torch.cat([prompt_act, response_act], dim=1))

        return activations

    def _extract_logprobs(
        self,
        outputs,
        batch_idx: int,
        gen_ids: "torch.Tensor",
        trim_pos: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Extract per-token logprobs for one request from batched HF output."""
        if not hasattr(outputs, "scores") or not outputs.scores:
            return None

        scores = list(outputs.scores)
        if trim_pos is not None:
            scores = scores[:trim_pos]

        num_tokens = len(scores)
        if num_tokens == 0:
            return None

        token_ids = gen_ids[:num_tokens].cpu().numpy().astype(np.int32)
        token_logprobs = np.zeros(num_tokens, dtype=np.float32)
        topk_ids = np.zeros((num_tokens, self._logprobs_top_k), dtype=np.int32)
        topk_logprobs_arr = np.zeros((num_tokens, self._logprobs_top_k), dtype=np.float32)

        for t, score in enumerate(scores):
            log_probs = torch.log_softmax(score[batch_idx], dim=-1)
            token_logprobs[t] = log_probs[token_ids[t]].item()
            topk_vals, topk_idx = torch.topk(log_probs, self._logprobs_top_k)
            topk_ids[t] = topk_idx.cpu().numpy()
            topk_logprobs_arr[t] = topk_vals.cpu().numpy()

        return {
            "response_token_ids": token_ids,
            "response_token_logprobs": token_logprobs,
            "response_topk_token_ids": topk_ids,
            "response_topk_logprobs": topk_logprobs_arr,
            "response_logprobs_top_k": self._logprobs_top_k,
        }


# ---------------------------------------------------------------------------
# GGUF / llama.cpp adapter
# ---------------------------------------------------------------------------

class GGUFAdapter(ModelAdapter):
    """Adapter for GGUF/llama.cpp models. No activation capture."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def supports_activations(self) -> bool:
        return False

    def infer(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> InferenceResult:
        """Delegates to existing llama.cpp inference path in server.py."""
        from activation_logging.server import run_inference
        response_text, _, input_length, trim_pos = run_inference(
            prompt, max_tokens, temperature, top_p, model_name=self._model_name
        )
        return InferenceResult(
            prompt=prompt,
            response_text=response_text,
            input_length=input_length,
            activations=None,
            trim_position=trim_pos,
        )
