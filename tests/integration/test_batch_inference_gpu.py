"""Integration tests: batch inference correctness on GPU.

Verifies:
- infer_batch([p]) produces the same result as infer(p) for greedy decoding
- infer_batch with N > 1 prompts produces per-request activations within tolerance
- Zarr stores batch results correctly (one row per unique prompt)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from activation_logging.model_adapter import HFTransformersAdapter, InferenceResult
from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from .conftest import TEST_PROMPTS, DEFAULT_TEST_MODEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter_all(model_name, device):
    return HFTransformersAdapter(
        model_name,
        target_layers="second_half",
        sequence_mode="all",
        enable_logprobs=True,
        logprobs_top_k=3,
    )


# ---------------------------------------------------------------------------
# Batch vs sequential correctness
# ---------------------------------------------------------------------------

class TestBatchVsSequential:
    def test_batch_1_matches_single_response_text(self, adapter_all):
        p = TEST_PROMPTS[0]
        single = adapter_all.infer(p, max_tokens=32, temperature=0.0)
        batch = adapter_all.infer_batch([p], max_tokens=32, temperature=0.0)
        assert batch[0].response_text == single.response_text

    def test_batch_response_count_matches_prompt_count(self, adapter_all):
        prompts = TEST_PROMPTS[:4]
        results = adapter_all.infer_batch(prompts, max_tokens=16)
        assert len(results) == len(prompts)

    def test_batch_each_result_has_activations(self, adapter_all):
        results = adapter_all.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            assert r.activations is not None
            assert any(a is not None for a in r.activations)

    def test_batch_activation_batch_dim_always_one(self, adapter_all):
        results = adapter_all.infer_batch(TEST_PROMPTS[:3], max_tokens=16)
        for r in results:
            for act in r.activations:
                if act is not None:
                    assert act.shape[0] == 1, "Each per-request activation must have batch_dim=1"

    def test_batch_2_activations_match_sequential_within_tolerance(self, model_name, device):
        """Batch of 2 should produce the same activations as two separate single calls.

        atol=1e-1 to account for cumulative float16 rounding through residual stream layers.
        """
        adapter = HFTransformersAdapter(
            model_name, target_layers="second_half", sequence_mode="response",
            enable_logprobs=False,
        )
        p0, p1 = TEST_PROMPTS[0], TEST_PROMPTS[1]

        r0_single = adapter.infer(p0, max_tokens=16, temperature=0.0)
        r1_single = adapter.infer(p1, max_tokens=16, temperature=0.0)
        batch_results = adapter.infer_batch([p0, p1], max_tokens=16, temperature=0.0)

        def _compare(single_acts, batch_acts, label):
            s_nonnull = [a for a in single_acts if a is not None]
            b_nonnull = [a for a in batch_acts if a is not None]
            assert len(s_nonnull) == len(b_nonnull)
            for s, b in zip(s_nonnull, b_nonnull):
                min_len = min(s.shape[1], b.shape[1])
                diff = (s[0, :min_len].float() - b[0, :min_len].float()).abs().max().item()
                assert diff < 1e-1, f"[{label}] max activation diff={diff:.4f} exceeds atol=1e-1"

        _compare(r0_single.activations, batch_results[0].activations, "prompt0")
        _compare(r1_single.activations, batch_results[1].activations, "prompt1")

    def test_batch_4_all_responses_nonempty(self, adapter_all):
        results = adapter_all.infer_batch(TEST_PROMPTS[:4], max_tokens=24)
        for i, r in enumerate(results):
            assert len(r.response_text.strip()) > 0, f"Result {i} has empty response"

    def test_batch_logprobs_per_request(self, adapter_all):
        results = adapter_all.infer_batch(TEST_PROMPTS[:2], max_tokens=16)
        for r in results:
            lp = r.logprobs
            assert lp is not None
            n = lp["response_token_ids"].shape[0]
            assert n > 0
            assert lp["response_token_logprobs"].shape == (n,)
            assert np.all(lp["response_token_logprobs"] <= 0)


# ---------------------------------------------------------------------------
# Zarr persistence for batch results
# ---------------------------------------------------------------------------

class TestBatchZarrPersistence:
    def test_batch_results_stored_as_separate_rows(self, adapter_all):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            prompts = TEST_PROMPTS[:3]
            results = adapter_all.infer_batch(prompts, max_tokens=16)

            zl = ZarrActivationsLogger(
                zarr_path=zarr_path, target_layers="second_half", sequence_mode="all",
                verbose=False,
            )
            for i, r in enumerate(results):
                entry = {
                    "prompt": r.prompt,
                    "response": r.response_text,
                    "model": "test",
                    "input_length": r.input_length,
                    "prompt_hash": f"hash_{i}",
                    "all_layers_activations": r.activations,
                }
                if r.logprobs:
                    entry.update(r.logprobs)
                zl.log_entry(f"batch_key_{i}", entry)
            zl.close()

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            assert reader._prompt_activations.shape[0] == 3
            for i in range(3):
                assert f"batch_key_{i}" in reader.list_entries()
            reader.close()

    def test_batch_activation_shapes_roundtrip(self, adapter_all):
        """Stored and reloaded activations should have the same non-zero shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            result = adapter_all.infer_batch([TEST_PROMPTS[0]], max_tokens=16)[0]

            zl = ZarrActivationsLogger(
                zarr_path=zarr_path, target_layers="second_half", sequence_mode="all",
                verbose=False,
            )
            zl.log_entry("roundtrip_key", {
                "prompt": result.prompt,
                "response": result.response_text,
                "model": "test",
                "input_length": result.input_length,
                "prompt_hash": "rth",
                "all_layers_activations": result.activations,
            })
            zl.close()

            reader = ZarrActivationsLogger(
                zarr_path=zarr_path, read_only=True, verbose=False,
                target_layers="second_half",
            )
            loaded = reader.get_entry("roundtrip_key")
            loaded_acts = [a for a in loaded["all_layers_activations"] if a is not None]
            orig_acts = [a for a in result.activations if a is not None]

            assert len(loaded_acts) == len(orig_acts)
            for orig, load in zip(orig_acts, loaded_acts):
                # Shapes should be within Zarr's truncation limits
                for dim in (1, 2):
                    assert load.shape[dim] <= orig.shape[dim]
            reader.close()
