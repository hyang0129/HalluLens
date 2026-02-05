"""TensorRT-LLM client wrapper used for question generation.

Notes
-----
- TensorRT-LLM is typically Linux-only and requires an NVIDIA GPU.
- This module is optional; importing it will not fail unless you instantiate
  :class:`TRTLLMClient`.

We use this only for *question generation*; the main activation-logging pipeline
continues to use the OpenAI API-compatible server (vLLM/llama.cpp) for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class TRTLLMGenerationConfig:
    """Sampling configuration for TensorRT-LLM generation."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None


class TRTLLMUnavailableError(RuntimeError):
    """Raised when TensorRT-LLM is not installed or not usable."""


class TRTLLMClient:
    """Small wrapper around `tensorrt_llm.LLM`.

    We cache the instantiated model because loading/initialization is expensive.
    """

    _singleton: Optional["TRTLLMClient"] = None

    def __init__(self, model: str):
        self.model_id = model
        try:
            # Imported lazily so environments without TRT-LLM can still use the repo.
            from tensorrt_llm import LLM  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TRTLLMUnavailableError(
                "TensorRT-LLM is not available. Install and run this component on a "
                "Linux NVIDIA environment with TensorRT-LLM set up."
            ) from exc

        self._LLM = LLM
        self._llm = self._LLM(model=self.model_id)

    @classmethod
    def get(cls, model: str) -> "TRTLLMClient":
        """Get (or create) a singleton client for a given model."""
        if cls._singleton is None or cls._singleton.model_id != model:
            cls._singleton = cls(model=model)
        return cls._singleton

    def generate_many(self, prompts: Iterable[str], config: TRTLLMGenerationConfig) -> List[str]:
        """Generate completions for a batch of prompts."""
        prompts_list = list(prompts)
        if not prompts_list:
            return []

        # SamplingParams import is also lazy.
        try:
            from tensorrt_llm import SamplingParams  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TRTLLMUnavailableError(
                "TensorRT-LLM SamplingParams import failed; check your TRT-LLM install."
            ) from exc

        sampling_kwargs = {
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
        }
        if config.max_tokens is not None:
            sampling_kwargs["max_tokens"] = int(config.max_tokens)

        sampling_params = SamplingParams(**sampling_kwargs)

        outputs = self._llm.generate(prompts_list, sampling_params)
        texts: List[str] = []
        for out in outputs:
            # TRT-LLM RequestOutput: out.outputs[0].text is the completion.
            texts.append(out.outputs[0].text)
        return texts

    def generate_one(self, prompt: str, config: TRTLLMGenerationConfig) -> str:
        """Generate a single completion."""
        return self.generate_many([prompt], config=config)[0]
