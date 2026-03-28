"""Unit tests for Phase 1: ZarrActivationsLogger — log_metadata, overwrite, get_entry fix.

All tests run without a GPU.  Activations are synthesized as numpy arrays.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from activation_logging.activations_logger import ActivationsLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(num_layers: int, seq_len: int, hidden_size: int) -> list:
    """Return a list of float32 numpy arrays shaped (1, seq_len, hidden_size)."""
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((1, seq_len, hidden_size)).astype(np.float32)
        for _ in range(num_layers)
    ]


def _make_entry(prompt_len: int = 4, response_len: int = 3,
                num_layers: int = 2, hidden_size: int = 8,
                prompt: str = "Hello?", response: str = "World.") -> dict:
    """Build a minimal entry dict with all_layers_activations."""
    total_len = prompt_len + response_len
    return {
        "prompt": prompt,
        "response": response,
        "model": "dummy",
        "input_length": prompt_len,
        "prompt_hash": "abc123",
        "all_layers_activations": _make_layers(num_layers, total_len, hidden_size),
    }


def _open_zarr(tmpdir: str, read_only: bool = False) -> ZarrActivationsLogger:
    zarr_path = str(Path(tmpdir) / "activations.zarr")
    return ZarrActivationsLogger(
        zarr_path=zarr_path,
        target_layers="all",
        sequence_mode="all",
        prompt_max_tokens=8,
        response_max_tokens=8,
        prompt_chunk_tokens=8,
        response_chunk_tokens=8,
        response_logprobs_top_k=3,
        dtype="float16",
        read_only=read_only,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Phase 1a — log_metadata()
# ---------------------------------------------------------------------------

class TestLogMetadata:
    def test_metadata_entry_appears_in_list_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("key1", {"prompt": "Q?", "response": "A.", "prompt_hash": "key1"})
            assert "key1" in zl.list_entries()
            zl.close()

    def test_metadata_entry_persisted_to_index_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("key2", {"prompt": "Q?", "response": "A.", "prompt_hash": "key2"})
            zl.close()

            index_path = Path(tmpdir) / "activations.zarr" / "meta" / "index.jsonl"
            lines = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
            keys = [l["key"] for l in lines]
            assert "key2" in keys

    def test_metadata_entry_has_activations_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("key3", {"prompt": "Q?"})
            meta = zl.get_entry("key3")
            assert meta.get("has_activations") is False
            assert meta.get("sample_index") is None
            zl.close()

    def test_metadata_entry_sample_index_is_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("key4", {"prompt": "Q?"})
            meta = zl._index["key4"]
            assert meta["sample_index"] is None
            zl.close()

    def test_metadata_strips_activation_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            entry = {
                "prompt": "Q?",
                "all_layers_activations": _make_layers(2, 4, 8),
                "model_outputs": "should_be_stripped",
                "response_token_ids": np.array([1, 2]),
            }
            zl.log_metadata("key5", entry)
            meta = zl._index["key5"]
            assert "all_layers_activations" not in meta
            assert "model_outputs" not in meta
            assert "response_token_ids" not in meta
            zl.close()

    def test_metadata_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")

            zl = ZarrActivationsLogger(zarr_path=zarr_path, verbose=False)
            zl.log_metadata("key6", {"prompt": "Q?", "response": "A."})
            zl.close()

            zl2 = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            assert "key6" in zl2.list_entries()
            zl2.close()

    def test_log_metadata_raises_in_read_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            # Create store first
            zl = ZarrActivationsLogger(zarr_path=zarr_path, verbose=False)
            zl.close()
            # Open read-only and verify exception
            zl_ro = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            with pytest.raises(ValueError, match="read-only"):
                zl_ro.log_metadata("x", {})
            zl_ro.close()


# ---------------------------------------------------------------------------
# Phase 1b — log_entry() overwrite support
# ---------------------------------------------------------------------------

class TestLogEntryOverwrite:
    def test_overwrite_same_key_no_orphan_rows(self):
        """Writing the same key twice should not create two Zarr rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            entry1 = _make_entry(prompt_len=4, response_len=3, response="First.")
            entry2 = _make_entry(prompt_len=4, response_len=3, response="Second.")

            zl.log_entry("dup", entry1)
            zl.log_entry("dup", entry2)

            # Only one row in Zarr arrays
            assert zl._prompt_activations.shape[0] == 1

            # Index still has one entry
            assert len(zl.list_entries()) == 1

            # Entry reflects the overwritten data
            meta = zl.get_entry("dup", metadata_only=True)
            assert meta["response"] == "Second."
            zl.close()

    def test_overwrite_updates_index_file(self):
        """After two writes of the same key, the index file has two lines but
        both point to the same row (idempotent on read because dict is keyed by 'key')."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_entry("dup2", _make_entry(response="First."))
            zl.log_entry("dup2", _make_entry(response="Second."))
            zl.close()

            # After reopen, only one logical entry
            zl2 = ZarrActivationsLogger(
                zarr_path=str(Path(tmpdir) / "activations.zarr"),
                read_only=True, verbose=False,
            )
            assert len(zl2.list_entries()) == 1
            meta = zl2.get_entry("dup2", metadata_only=True)
            assert meta["response"] == "Second."
            zl2.close()

    def test_metadata_upgrade_to_full_entry(self):
        """Writing a metadata-only entry then a full entry for the same key
        should upgrade the entry and create exactly one Zarr row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)

            # Step 1: metadata-only entry (no activations)
            zl.log_metadata("upgrade_key", {"prompt": "Q?", "response": "A."})
            assert zl._prompt_activations is None  # arrays not created yet

            # Step 2: full entry
            zl.log_entry("upgrade_key", _make_entry(prompt="Q?", response="A2."))

            # One Zarr row created
            assert zl._prompt_activations.shape[0] == 1

            meta = zl.get_entry("upgrade_key", metadata_only=True)
            # sample_index now set
            assert meta.get("sample_index") == 0
            zl.close()

    def test_multiple_distinct_keys_no_interference(self):
        """Overwriting one key does not affect rows for different keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_entry("a", _make_entry(response="A"))
            zl.log_entry("b", _make_entry(response="B"))
            zl.log_entry("a", _make_entry(response="A2"))  # overwrite a

            assert zl._prompt_activations.shape[0] == 2

            meta_a = zl.get_entry("a", metadata_only=True)
            meta_b = zl.get_entry("b", metadata_only=True)
            assert meta_a["response"] == "A2"
            assert meta_b["response"] == "B"
            zl.close()


# ---------------------------------------------------------------------------
# Phase 1c — get_entry() for metadata-only entries
# ---------------------------------------------------------------------------

class TestGetEntryMetadataOnly:
    def test_get_entry_no_flag_returns_meta_for_metadata_only_entry(self):
        """get_entry(key) (no flags) should not raise for a metadata-only entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("m1", {"prompt": "Q?", "response": "A."})
            meta = zl.get_entry("m1")  # should not raise
            assert meta["prompt"] == "Q?"
            zl.close()

    def test_get_entry_metadata_only_true_returns_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("m2", {"prompt": "Q2?"})
            meta = zl.get_entry("m2", metadata_only=True)
            assert "prompt" in meta
            zl.close()

    def test_get_entry_metadata_only_false_returns_meta_not_raises(self):
        """metadata_only=False on a metadata-only entry should still return metadata (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_metadata("m3", {"prompt": "Q3?", "response": "R3."})
            meta = zl.get_entry("m3", metadata_only=False)
            assert meta is not None
            assert meta.get("has_activations") is False
            # No activation tensor in result (idx is None → returned early)
            assert "all_layers_activations" not in meta
            zl.close()

    def test_get_entry_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            with pytest.raises(KeyError):
                zl.get_entry("nonexistent")
            zl.close()

    def test_full_entry_get_entry_still_works(self):
        """Regression: normal full entries should still return activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            zl.log_entry("full", _make_entry())
            result = zl.get_entry("full")
            assert "all_layers_activations" in result
            assert len(result["all_layers_activations"]) == 2  # num_layers
            zl.close()


# ---------------------------------------------------------------------------
# ActivationsLogger wrapper
# ---------------------------------------------------------------------------

class TestActivationsLoggerWrapper:
    def test_log_metadata_delegates_to_zarr_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            al = ActivationsLogger(lmdb_path=zarr_path, verbose=False)
            al.log_metadata("wk1", {"prompt": "Q?", "response": "A."})
            assert "wk1" in al.list_entries()
            al.close()

    def test_log_metadata_raises_for_lmdb_backend(self):
        """ActivationsLogger without Zarr backend should raise NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = str(Path(tmpdir) / "activations.lmdb")
            al = ActivationsLogger(lmdb_path=lmdb_path, verbose=False)
            with pytest.raises(NotImplementedError):
                al.log_metadata("x", {})
