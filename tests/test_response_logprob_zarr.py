"""Tests for response token top-k logprob persistence in Zarr activation logging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from activation_logging.activation_parser import ActivationDataset
from activation_logging.zarr_activations_logger import ZarrActivationsLogger


def _make_layers(num_layers: int, seq_len: int, hidden_size: int) -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    layers = []
    for _ in range(num_layers):
        layers.append(rng.standard_normal((1, seq_len, hidden_size), dtype=np.float32).astype(np.float16))
    return layers


def test_response_logprob_roundtrip_and_dataset_loading() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = str(Path(tmpdir) / "activations.zarr")

        logger = ZarrActivationsLogger(
            zarr_path=zarr_path,
            target_layers="all",
            sequence_mode="all",
            prompt_max_tokens=4,
            response_max_tokens=6,
            prompt_chunk_tokens=4,
            response_chunk_tokens=6,
            response_logprobs_top_k=3,
            dtype="float16",
            read_only=False,
            verbose=False,
        )

        key = "sample_key_1"
        response_len = 5
        entry = {
            "prompt": "Question?",
            "response": "Answer.",
            "model": "dummy-model",
            "input_length": 4,
            "prompt_hash": key,
            "all_layers_activations": _make_layers(num_layers=2, seq_len=9, hidden_size=8),
            "response_token_ids": np.array([101, 102, 103, 104, 105], dtype=np.int32),
            "response_token_logprobs": np.array([-0.10, -0.20, -0.30, -0.40, -0.50], dtype=np.float32),
            "response_topk_token_ids": np.array(
                [
                    [101, 201, 301],
                    [102, 202, 302],
                    [103, 203, 303],
                    [104, 204, 304],
                    [105, 205, 305],
                ],
                dtype=np.int32,
            ),
            "response_topk_logprobs": np.array(
                [
                    [-0.10, -0.40, -0.70],
                    [-0.20, -0.50, -0.80],
                    [-0.30, -0.60, -0.90],
                    [-0.40, -0.70, -1.00],
                    [-0.50, -0.80, -1.10],
                ],
                dtype=np.float32,
            ),
        }

        logger.log_entry(key, entry)
        logger.close()

        reader = ZarrActivationsLogger(
            zarr_path=zarr_path,
            read_only=True,
            target_layers="all",
            sequence_mode="all",
            verbose=False,
        )

        meta = reader.get_entry_by_key(key, metadata_only=True)
        assert meta is not None
        assert meta["has_response_logprobs"] is True
        assert int(meta["response_logprobs_len"]) == response_len
        assert int(meta["response_logprobs_top_k"]) == 3

        payload = reader.get_response_logprobs(key)
        assert payload is not None
        assert tuple(payload["response_token_ids"].shape) == (response_len,)
        assert tuple(payload["response_topk_token_ids"].shape) == (response_len, 3)

        df = pd.DataFrame(
            {
                "prompt_hash": [key],
                "halu": [0],
                "split": ["train"],
            }
        )
        dataset = ActivationDataset(
            df=df,
            activations_path=zarr_path,
            split="train",
            relevant_layers=[0, 1],
            pad_length=6,
            min_target_layers=2,
            num_views=2,
            include_response_logprobs=True,
            response_logprobs_top_k=3,
            verbose=False,
        )

        sample = dataset[0]
        assert tuple(sample["response_token_ids"].shape) == (6,)
        assert tuple(sample["response_topk_token_ids"].shape) == (6, 3)
        assert int(sample["response_logprob_len"].item()) == response_len
        assert int(sample["response_logprobs_top_k"].item()) == 3
        assert int(sample["response_logprob_mask"].sum().item()) == response_len
