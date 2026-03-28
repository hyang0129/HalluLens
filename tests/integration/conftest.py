"""Shared fixtures and marks for GPU integration tests."""
from __future__ import annotations

import os
import pytest
import torch

# Model to use for all integration tests.  Override via environment variable.
DEFAULT_TEST_MODEL = os.environ.get(
    "TEST_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
)

# Small prompts for fast inference
TEST_PROMPTS = [
    "What is the capital of France?",
    "Name the largest planet in the solar system.",
    "Who wrote Hamlet?",
    "What is 2 + 2?",
]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (throughput benchmarks)")
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")


@pytest.fixture(scope="session")
def model_name() -> str:
    return DEFAULT_TEST_MODEL


@pytest.fixture(scope="session")
def device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available — skipping GPU integration tests")
    return "cuda"
