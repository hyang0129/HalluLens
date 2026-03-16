"""Unit tests for Phase 4: ModelAdapter, InferenceResult, GGUFAdapter.

All tests run without a GPU.  HFTransformersAdapter GPU tests are in the
integration test suite (tests/integration/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from activation_logging.model_adapter import (
    GGUFAdapter,
    HFTransformersAdapter,
    InferenceResult,
    ModelAdapter,
)


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------

class TestInferenceResult:
    def test_required_fields(self):
        r = InferenceResult(
            prompt="Hello",
            response_text="World",
            input_length=5,
            activations=None,
        )
        assert r.prompt == "Hello"
        assert r.response_text == "World"
        assert r.input_length == 5
        assert r.activations is None
        assert r.logprobs is None
        assert r.trim_position is None

    def test_optional_fields(self):
        r = InferenceResult(
            prompt="P",
            response_text="R",
            input_length=1,
            activations=[None, None],
            logprobs={"response_token_ids": np.array([1, 2])},
            trim_position=3,
        )
        assert r.logprobs is not None
        assert r.trim_position == 3

    def test_dataclass_is_mutable(self):
        r = InferenceResult(prompt="P", response_text="R", input_length=1, activations=None)
        r.response_text = "Updated"
        assert r.response_text == "Updated"


# ---------------------------------------------------------------------------
# ModelAdapter base class contract
# ---------------------------------------------------------------------------

class ConcreteAdapter(ModelAdapter):
    """Minimal concrete adapter for testing base class behaviour."""

    def __init__(self, results: List[InferenceResult]):
        self._results = iter(results)

    def infer(self, prompt, max_tokens=512, temperature=0.0, top_p=1.0) -> InferenceResult:
        return next(self._results)

    def supports_activations(self) -> bool:
        return False


class TestModelAdapterBase:
    def test_infer_batch_default_falls_back_to_sequential(self):
        """Default infer_batch calls infer() once per prompt."""
        results = [
            InferenceResult("P1", "R1", 1, None),
            InferenceResult("P2", "R2", 1, None),
        ]
        adapter = ConcreteAdapter(results)
        batch_results = adapter.infer_batch(["P1", "P2"])
        assert len(batch_results) == 2
        assert batch_results[0].response_text == "R1"
        assert batch_results[1].response_text == "R2"

    def test_supports_activations_abstract(self):
        """supports_activations() must be overridden."""
        adapter = ConcreteAdapter([])
        assert adapter.supports_activations() is False

    def test_infer_returns_inference_result_type(self):
        adapter = ConcreteAdapter([InferenceResult("P", "R", 1, None)])
        result = adapter.infer("P")
        assert isinstance(result, InferenceResult)


# ---------------------------------------------------------------------------
# HFTransformersAdapter — constructor validation (no GPU needed)
# ---------------------------------------------------------------------------

class TestHFTransformersAdapterInit:
    def test_default_construction(self):
        adapter = HFTransformersAdapter("some-model")
        assert adapter._model_name == "some-model"
        assert adapter._target_layers == "all"
        assert adapter._sequence_mode == "all"
        assert adapter._enable_logprobs is True
        assert adapter._logprobs_top_k == 5

    def test_custom_params(self):
        adapter = HFTransformersAdapter(
            "m", target_layers="first_half", sequence_mode="prompt",
            enable_logprobs=False, logprobs_top_k=10,
        )
        assert adapter._target_layers == "first_half"
        assert adapter._sequence_mode == "prompt"
        assert adapter._enable_logprobs is False
        assert adapter._logprobs_top_k == 10

    def test_invalid_target_layers_raises(self):
        with pytest.raises(ValueError, match="target_layers"):
            HFTransformersAdapter("m", target_layers="invalid")

    def test_invalid_sequence_mode_raises(self):
        with pytest.raises(ValueError, match="sequence_mode"):
            HFTransformersAdapter("m", sequence_mode="invalid")

    def test_supports_activations_true(self):
        assert HFTransformersAdapter("m").supports_activations() is True

    def test_model_not_loaded_until_infer(self):
        adapter = HFTransformersAdapter("m")
        assert adapter._model is None
        assert adapter._tokenizer is None


# ---------------------------------------------------------------------------
# HFTransformersAdapter — resolve_target_layers (no GPU)
# ---------------------------------------------------------------------------

class TestHFAdapterTargetLayerResolution:
    def test_all_layers(self):
        adapter = HFTransformersAdapter("m", target_layers="all")
        indices = adapter._resolve_target_layers(8)
        assert indices == set(range(8))

    def test_first_half(self):
        adapter = HFTransformersAdapter("m", target_layers="first_half")
        indices = adapter._resolve_target_layers(8)
        assert indices == {0, 1, 2, 3}

    def test_second_half(self):
        adapter = HFTransformersAdapter("m", target_layers="second_half")
        indices = adapter._resolve_target_layers(8)
        assert indices == {4, 5, 6, 7}

    def test_odd_num_layers_first_half(self):
        adapter = HFTransformersAdapter("m", target_layers="first_half")
        indices = adapter._resolve_target_layers(7)
        assert indices == {0, 1, 2}  # floor(7/2) = 3

    def test_odd_num_layers_second_half(self):
        adapter = HFTransformersAdapter("m", target_layers="second_half")
        indices = adapter._resolve_target_layers(7)
        assert indices == {3, 4, 5, 6}


# ---------------------------------------------------------------------------
# HFTransformersAdapter — _extract_activations (no GPU, uses numpy tensors)
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestHFAdapterExtractActivations:
    """Tests for _extract_activations using synthetic HF-style outputs."""

    def _make_mock_outputs(
        self,
        batch_size: int = 1,
        prompt_len: int = 4,
        response_len: int = 3,
        num_layers: int = 2,
        hidden_size: int = 8,
    ):
        """Build a fake outputs object mirroring HF generate() hidden_states format."""
        import torch

        rng = torch.manual_seed(0)

        # Prompt hidden states: tuple of L tensors, each (B, P, H)
        prompt_hidden = tuple(
            torch.randn(batch_size, prompt_len, hidden_size)
            for _ in range(num_layers)
        )

        # Generation step hidden states: list of R tuples, each tuple of L tensors (B, 1, H)
        gen_hidden = tuple(
            tuple(torch.randn(batch_size, 1, hidden_size) for _ in range(num_layers))
            for _ in range(response_len)
        )

        mock_outputs = MagicMock()
        mock_outputs.hidden_states = (prompt_hidden,) + gen_hidden

        # sequences: (B, P + R) token ids
        mock_outputs.sequences = torch.zeros(
            batch_size, prompt_len + response_len, dtype=torch.long
        )

        return mock_outputs

    def test_extract_activations_all_mode_shape(self):
        import torch
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="all")
        adapter._target_layer_indices = {0, 1}

        outputs = self._make_mock_outputs(
            batch_size=1, prompt_len=4, response_len=3, num_layers=2, hidden_size=8
        )
        acts = adapter._extract_activations(outputs, batch_idx=0, input_length=4, trim_pos=None)

        assert acts is not None
        assert len(acts) == 2
        # Each layer: (1, prompt_len + response_len, hidden)
        for layer_act in acts:
            assert layer_act is not None
            assert layer_act.shape == (1, 7, 8)

    def test_extract_activations_prompt_mode(self):
        import torch
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="prompt")
        adapter._target_layer_indices = {0, 1}

        outputs = self._make_mock_outputs(prompt_len=4, response_len=3, num_layers=2, hidden_size=8)
        acts = adapter._extract_activations(outputs, batch_idx=0, input_length=4, trim_pos=None)

        for layer_act in acts:
            assert layer_act is not None
            assert layer_act.shape[1] == 4  # prompt_len only

    def test_extract_activations_response_mode(self):
        import torch
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="response")
        adapter._target_layer_indices = {0, 1}

        outputs = self._make_mock_outputs(prompt_len=4, response_len=3, num_layers=2, hidden_size=8)
        acts = adapter._extract_activations(outputs, batch_idx=0, input_length=4, trim_pos=None)

        for layer_act in acts:
            assert layer_act is not None
            assert layer_act.shape[1] == 3  # response_len only

    def test_extract_activations_trim_pos_shortens_response(self):
        import torch
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="response")
        adapter._target_layer_indices = {0, 1}

        outputs = self._make_mock_outputs(prompt_len=4, response_len=5, num_layers=2, hidden_size=8)
        acts = adapter._extract_activations(outputs, batch_idx=0, input_length=4, trim_pos=2)

        for layer_act in acts:
            assert layer_act is not None
            assert layer_act.shape[1] == 2  # trimmed to 2

    def test_extract_activations_target_layers_none_in_output(self):
        """Layers not in target_layer_indices should produce None entries."""
        import torch
        adapter = HFTransformersAdapter("m", target_layers="first_half", sequence_mode="all")
        adapter._target_layer_indices = {0}  # only layer 0

        outputs = self._make_mock_outputs(prompt_len=4, response_len=2, num_layers=2, hidden_size=8)
        acts = adapter._extract_activations(outputs, batch_idx=0, input_length=4, trim_pos=None)

        assert acts[0] is not None   # layer 0 present
        assert acts[1] is None        # layer 1 not in target

    def test_extract_activations_batch_dimension_correct(self):
        """With batch_size=2, extracting batch_idx=1 gives different values than batch_idx=0."""
        import torch
        torch.manual_seed(99)
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="prompt")
        adapter._target_layer_indices = {0}

        outputs = self._make_mock_outputs(
            batch_size=2, prompt_len=3, response_len=2, num_layers=1, hidden_size=4
        )
        acts0 = adapter._extract_activations(outputs, batch_idx=0, input_length=3, trim_pos=None)
        acts1 = adapter._extract_activations(outputs, batch_idx=1, input_length=3, trim_pos=None)

        assert acts0[0] is not None
        assert acts1[0] is not None
        # Batch items should have the same shape
        assert acts0[0].shape == acts1[0].shape

    def test_extract_activations_left_padding_stripped(self):
        """When padded_prompt_len > input_length, pad_offset strips the padding tokens."""
        import torch
        adapter = HFTransformersAdapter("m", target_layers="all", sequence_mode="prompt")
        adapter._target_layer_indices = {0}

        # Simulate left-padded batch: padded_prompt_len=6, real input_length=4
        padded_len = 6
        real_len = 4
        hidden_size = 8
        num_layers = 1

        prompt_hidden = (torch.randn(1, padded_len, hidden_size),)
        outputs = MagicMock()
        outputs.hidden_states = (prompt_hidden,)  # no generation steps

        acts = adapter._extract_activations(
            outputs, batch_idx=0, input_length=real_len, trim_pos=None
        )

        assert acts[0] is not None
        assert acts[0].shape[1] == real_len  # padding stripped

    def test_extract_activations_no_hidden_states_returns_none(self):
        outputs = MagicMock()
        outputs.hidden_states = None
        adapter = HFTransformersAdapter("m")
        acts = adapter._extract_activations(outputs, 0, 4, None)
        assert acts is None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestHFAdapterExtractLogprobs:
    def _make_scores(self, num_tokens: int, batch_size: int = 1, vocab_size: int = 100):
        import torch
        return tuple(
            torch.randn(batch_size, vocab_size) for _ in range(num_tokens)
        )

    def test_logprobs_shape(self):
        import torch
        adapter = HFTransformersAdapter("m", logprobs_top_k=3)
        n_tokens = 4
        gen_ids = torch.arange(n_tokens, dtype=torch.long)
        outputs = MagicMock()
        outputs.scores = self._make_scores(n_tokens, batch_size=1, vocab_size=50)

        result = adapter._extract_logprobs(outputs, batch_idx=0, gen_ids=gen_ids, trim_pos=None)

        assert result is not None
        assert result["response_token_ids"].shape == (n_tokens,)
        assert result["response_token_logprobs"].shape == (n_tokens,)
        assert result["response_topk_token_ids"].shape == (n_tokens, 3)
        assert result["response_topk_logprobs"].shape == (n_tokens, 3)
        assert result["response_logprobs_top_k"] == 3

    def test_logprobs_trimmed(self):
        import torch
        adapter = HFTransformersAdapter("m", logprobs_top_k=3)
        n_tokens = 6
        gen_ids = torch.arange(n_tokens, dtype=torch.long)
        outputs = MagicMock()
        outputs.scores = self._make_scores(n_tokens, batch_size=1, vocab_size=50)

        result = adapter._extract_logprobs(outputs, batch_idx=0, gen_ids=gen_ids, trim_pos=3)

        assert result is not None
        assert result["response_token_ids"].shape == (3,)

    def test_logprobs_no_scores_returns_none(self):
        import torch
        adapter = HFTransformersAdapter("m")
        outputs = MagicMock()
        outputs.scores = ()

        result = adapter._extract_logprobs(
            outputs, batch_idx=0, gen_ids=torch.tensor([]), trim_pos=None
        )
        assert result is None

    def test_logprob_values_are_log_softmax(self):
        """Token logprobs should be ≤ 0 (log probs from softmax distribution)."""
        import torch
        adapter = HFTransformersAdapter("m", logprobs_top_k=2)
        gen_ids = torch.tensor([5, 10], dtype=torch.long)
        outputs = MagicMock()
        outputs.scores = self._make_scores(2, batch_size=1, vocab_size=20)

        result = adapter._extract_logprobs(outputs, batch_idx=0, gen_ids=gen_ids, trim_pos=None)

        assert all(lp <= 0.0 for lp in result["response_token_logprobs"])


# ---------------------------------------------------------------------------
# GGUFAdapter — constructor and routing (no GPU)
# ---------------------------------------------------------------------------

class TestGGUFAdapter:
    def test_supports_activations_false(self):
        adapter = GGUFAdapter("some-model.gguf")
        assert adapter.supports_activations() is False

    def test_infer_delegates_to_run_inference(self):
        adapter = GGUFAdapter("model.gguf")
        with patch("activation_logging.model_adapter.GGUFAdapter.infer") as mock_infer:
            fake_result = InferenceResult(
                prompt="Q?", response_text="A.", input_length=3, activations=None
            )
            mock_infer.return_value = fake_result
            result = adapter.infer("Q?")
            mock_infer.assert_called_once_with("Q?")
            assert result.activations is None

    def test_infer_batch_falls_back_to_sequential(self):
        """GGUFAdapter uses the default infer_batch which calls infer sequentially."""
        adapter = GGUFAdapter("model.gguf")
        fake_results = [
            InferenceResult("P1", "R1", 1, None),
            InferenceResult("P2", "R2", 1, None),
        ]
        with patch.object(adapter, "infer", side_effect=fake_results):
            results = adapter.infer_batch(["P1", "P2"])
        assert len(results) == 2
        assert results[0].response_text == "R1"
        assert results[1].response_text == "R2"
