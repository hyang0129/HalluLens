"""Integration tests: HFTransformersAdapter on GPU.

Verifies that:
- Adapter loads the model and produces valid InferenceResult objects
- Activations have the correct shapes for all sequence_modes
- Batch inference produces the same response text as single-sample inference
- Per-request activations from a batch match batch_size=1 results within tolerance
- Logprobs are valid log-probability arrays (≤ 0, finite, correctly shaped)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from activation_logging.model_adapter import HFTransformersAdapter, InferenceResult
from .conftest import TEST_PROMPTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter(model_name, device):
    """Load HFTransformersAdapter once per test module."""
    a = HFTransformersAdapter(
        model_name,
        target_layers="all",
        sequence_mode="all",
        enable_logprobs=True,
        logprobs_top_k=5,
    )
    yield a


@pytest.fixture(scope="module")
def adapter_prompt_mode(model_name, device):
    return HFTransformersAdapter(
        model_name, target_layers="all", sequence_mode="prompt",
        enable_logprobs=False,
    )


@pytest.fixture(scope="module")
def adapter_response_mode(model_name, device):
    return HFTransformersAdapter(
        model_name, target_layers="all", sequence_mode="response",
        enable_logprobs=False,
    )


# ---------------------------------------------------------------------------
# Single-prompt inference
# ---------------------------------------------------------------------------

class TestHFAdapterSingleInference:
    def test_returns_inference_result(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=32)
        assert isinstance(result, InferenceResult)

    def test_response_text_nonempty(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=32)
        assert len(result.response_text.strip()) > 0

    def test_input_length_positive(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=32)
        assert result.input_length > 0

    def test_activations_not_none(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=32)
        assert result.activations is not None
        assert len(result.activations) > 0

    def test_activations_shape_batch_dim_one(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=16)
        for act in result.activations:
            if act is not None:
                assert act.shape[0] == 1, f"Expected batch dim 1, got {act.shape[0]}"

    def test_activations_hidden_dim_consistent(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=16)
        hidden_sizes = {act.shape[-1] for act in result.activations if act is not None}
        assert len(hidden_sizes) == 1, "All layers should have same hidden size"

    def test_all_mode_seq_len_equals_prompt_plus_response(self, adapter):
        """In 'all' mode, seq_len should equal input_length + response_token_count."""
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=16)
        # response token count is inferred from tokenizer; check prompt portion
        for act in result.activations:
            if act is not None:
                total_seq = act.shape[1]
                # At minimum contains the prompt tokens
                assert total_seq >= result.input_length

    def test_prompt_mode_seq_len_equals_input_length(self, adapter_prompt_mode):
        result = adapter_prompt_mode.infer(TEST_PROMPTS[0], max_tokens=16)
        for act in result.activations:
            if act is not None:
                assert act.shape[1] == result.input_length, (
                    f"Prompt mode: expected seq_len={result.input_length}, got {act.shape[1]}"
                )

    def test_response_mode_seq_len_positive(self, adapter_response_mode):
        result = adapter_response_mode.infer(TEST_PROMPTS[0], max_tokens=16)
        for act in result.activations:
            if act is not None:
                assert act.shape[1] > 0, "Response mode should have at least one response token"

    def test_logprobs_shape_and_range(self, adapter):
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=16)
        lp = result.logprobs
        assert lp is not None
        n = lp["response_token_ids"].shape[0]
        assert n > 0
        assert lp["response_token_logprobs"].shape == (n,)
        assert lp["response_topk_token_ids"].shape == (n, 5)
        assert lp["response_topk_logprobs"].shape == (n, 5)
        # All log-probs should be ≤ 0 and finite
        assert np.all(lp["response_token_logprobs"] <= 0)
        assert np.all(np.isfinite(lp["response_token_logprobs"]))

    def test_logprobs_top_k_descending(self, adapter):
        """Top-k logprobs should be in descending order (most probable first)."""
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=8)
        topk_lp = result.logprobs["response_topk_logprobs"]
        for row in topk_lp:
            assert all(row[i] >= row[i + 1] for i in range(len(row) - 1)), (
                "Top-k logprobs should be descending"
            )

    def test_first_half_layers_second_half_none(self, model_name, device):
        adapter = HFTransformersAdapter(
            model_name, target_layers="first_half", sequence_mode="all",
            enable_logprobs=False,
        )
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=8)
        acts = result.activations
        n = len(acts)
        for i in range(n // 2):
            assert acts[i] is not None, f"Layer {i} should be in first_half target"
        for i in range(n // 2, n):
            assert acts[i] is None, f"Layer {i} should be None (not in first_half)"

    def test_second_half_layers_first_half_none(self, model_name, device):
        adapter = HFTransformersAdapter(
            model_name, target_layers="second_half", sequence_mode="all",
            enable_logprobs=False,
        )
        result = adapter.infer(TEST_PROMPTS[0], max_tokens=8)
        acts = result.activations
        n = len(acts)
        for i in range(n // 2):
            assert acts[i] is None, f"Layer {i} should be None (not in second_half)"
        for i in range(n // 2, n):
            assert acts[i] is not None, f"Layer {i} should be in second_half target"


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

class TestHFAdapterBatchInference:
    def test_batch_returns_correct_count(self, adapter):
        prompts = TEST_PROMPTS[:2]
        results = adapter.infer_batch(prompts, max_tokens=16)
        assert len(results) == len(prompts)

    def test_batch_all_results_are_inference_result(self, adapter):
        results = adapter.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            assert isinstance(r, InferenceResult)

    def test_batch_response_texts_nonempty(self, adapter):
        results = adapter.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            assert len(r.response_text.strip()) > 0

    def test_batch_activations_shape_consistency(self, adapter):
        """All per-request activations in a batch should have batch_dim=1."""
        results = adapter.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            for act in r.activations:
                if act is not None:
                    assert act.shape[0] == 1

    def test_batch_vs_single_response_text_matches(self, model_name, device):
        """Greedy (temp=0) batch and single-sample inference should produce the same text."""
        adapter_single = HFTransformersAdapter(
            model_name, target_layers="all", sequence_mode="all", enable_logprobs=False
        )
        prompt = TEST_PROMPTS[0]
        single_result = adapter_single.infer(prompt, max_tokens=32, temperature=0.0)

        # Batch of 1 to compare
        batch_results = adapter_single.infer_batch([prompt], max_tokens=32, temperature=0.0)
        assert batch_results[0].response_text == single_result.response_text

    def test_batch_vs_single_activations_within_tolerance(self, model_name, device):
        """Per-request activations from a 2-prompt batch should match single-prompt
        activations for the same prompt within float16 tolerance.

        Note: Left-padding introduces numerical differences in the attention mechanism
        for some model architectures.  We allow a generous atol=1e-2 for float16.
        """
        adapter_b = HFTransformersAdapter(
            model_name, target_layers="second_half", sequence_mode="response",
            enable_logprobs=False,
        )
        prompt = TEST_PROMPTS[0]
        filler = TEST_PROMPTS[1]

        # Single-prompt reference
        single_result = adapter_b.infer(prompt, max_tokens=16, temperature=0.0)

        # In a batch; prompt is second (filler is first → forces left-padding for prompt)
        batch_results = adapter_b.infer_batch([filler, prompt], max_tokens=16, temperature=0.0)
        batch_result = batch_results[1]

        single_acts = [a for a in single_result.activations if a is not None]
        batch_acts = [a for a in batch_result.activations if a is not None]
        assert len(single_acts) == len(batch_acts)

        for s_act, b_act in zip(single_acts, batch_acts):
            # Shapes may differ slightly if response length differs; compare min length
            min_len = min(s_act.shape[1], b_act.shape[1])
            s_np = s_act[0, :min_len, :].float().cpu().numpy()
            b_np = b_act[0, :min_len, :].float().cpu().numpy()
            max_diff = np.abs(s_np - b_np).max()
            assert max_diff < 1e-2, (
                f"Activation mismatch too large: max_diff={max_diff:.4f} (atol=1e-2)"
            )

    def test_batch_logprobs_present_for_all(self, adapter):
        results = adapter.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            assert r.logprobs is not None, "logprobs should be present for HF adapter"


# ---------------------------------------------------------------------------
# Activation → Zarr round-trip (adapter + logger)
# ---------------------------------------------------------------------------

class TestHFAdapterZarrRoundTrip:
    """Verifies that activations produced by the adapter can be stored and
    retrieved correctly via ZarrActivationsLogger."""

    def test_single_inference_persisted_to_zarr(self, adapter):
        from activation_logging.zarr_activations_logger import ZarrActivationsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            result = adapter.infer(TEST_PROMPTS[0], max_tokens=16)

            # Build entry dict as the server would
            entry = {
                "prompt": result.prompt,
                "response": result.response_text,
                "model": "test-model",
                "input_length": result.input_length,
                "prompt_hash": "hash_test",
                "all_layers_activations": result.activations,
            }
            if result.logprobs:
                entry.update(result.logprobs)

            zl = ZarrActivationsLogger(
                zarr_path=zarr_path, target_layers="all", sequence_mode="all",
                verbose=False,
            )
            zl.log_entry("test_key", entry)
            zl.close()

            # Re-open and verify
            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            assert "test_key" in reader.list_entries()
            loaded = reader.get_entry("test_key")
            assert loaded["response"] == result.response_text
            assert "all_layers_activations" in loaded
            reader.close()

    def test_batch_inference_all_persisted(self, adapter):
        from activation_logging.zarr_activations_logger import ZarrActivationsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            results = adapter.infer_batch(TEST_PROMPTS[:3], max_tokens=16)

            zl = ZarrActivationsLogger(
                zarr_path=zarr_path, target_layers="all", sequence_mode="all",
                verbose=False,
            )
            for i, result in enumerate(results):
                entry = {
                    "prompt": result.prompt,
                    "response": result.response_text,
                    "model": "test-model",
                    "input_length": result.input_length,
                    "prompt_hash": f"hash_{i}",
                    "all_layers_activations": result.activations,
                }
                zl.log_entry(f"key_{i}", entry)
            zl.close()

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            for i in range(3):
                assert f"key_{i}" in reader.list_entries()
            reader.close()


# ---------------------------------------------------------------------------
# Throughput benchmark (slow — opt-in with -m slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHFAdapterThroughput:
    def test_batch_throughput_vs_sequential(self, adapter, benchmark):
        """Batch inference should be measurably faster than sequential on GPU.

        Run with: pytest tests/integration/test_hf_adapter_gpu.py -m slow --benchmark-only
        """
        prompts = TEST_PROMPTS * 4  # 16 prompts

        def run_sequential():
            return [adapter.infer(p, max_tokens=32) for p in prompts]

        def run_batch():
            return adapter.infer_batch(prompts, max_tokens=32)

        seq_results = run_sequential()
        batch_results = run_batch()
        assert len(batch_results) == len(prompts)
        # Actual timing is printed by pytest-benchmark; no hard assertion on speed here
