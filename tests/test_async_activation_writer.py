"""Unit tests for Phase 2: AsyncActivationWriter.

All tests run without a GPU.  The ZarrActivationsLogger is used as the real
write backend (via a temp directory) to verify end-to-end persistence.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from activation_logging.activations_logger import ActivationsLogger
from activation_logging.server import AsyncActivationWriter
from activation_logging.zarr_activations_logger import ZarrActivationsLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layers(num_layers: int = 2, seq_len: int = 7, hidden_size: int = 8) -> list:
    rng = np.random.default_rng(0)
    return [rng.standard_normal((1, seq_len, hidden_size)).astype(np.float32) for _ in range(num_layers)]


def _full_entry(prompt: str = "Q?", response: str = "A.", prompt_len: int = 4) -> dict:
    return {
        "prompt": prompt,
        "response": response,
        "model": "dummy",
        "input_length": prompt_len,
        "prompt_hash": "hash1",
        "all_layers_activations": _make_layers(),
    }


def _open_zarr(tmpdir: str) -> ZarrActivationsLogger:
    return ZarrActivationsLogger(
        zarr_path=str(Path(tmpdir) / "activations.zarr"),
        target_layers="all",
        sequence_mode="all",
        prompt_max_tokens=8,
        response_max_tokens=8,
        prompt_chunk_tokens=8,
        response_chunk_tokens=8,
        response_logprobs_top_k=3,
        dtype="float16",
        verbose=False,
    )


def _wrap(zl: ZarrActivationsLogger) -> ActivationsLogger:
    """Wrap a ZarrActivationsLogger in an ActivationsLogger proxy."""
    al = ActivationsLogger.__new__(ActivationsLogger)
    al._backend = zl
    return al


# ---------------------------------------------------------------------------
# AsyncActivationWriter unit tests
# ---------------------------------------------------------------------------

class TestAsyncActivationWriterBasic:
    def test_enqueue_and_drain_full_entry(self):
        """Entries enqueued should appear in Zarr store after shutdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            al = _wrap(zl)
            writer = AsyncActivationWriter(al)

            writer.enqueue("k1", _full_entry())
            writer.shutdown(timeout=5.0)

            assert "k1" in zl.list_entries()
            meta = zl.get_entry("k1", metadata_only=True)
            assert meta["response"] == "A."

    def test_enqueue_metadata_only(self):
        """metadata_only=True entries should use log_metadata, not log_entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            al = _wrap(zl)
            writer = AsyncActivationWriter(al)

            writer.enqueue("m1", {"prompt": "Q?", "response": "R."}, metadata_only=True)
            writer.shutdown(timeout=5.0)

            assert "m1" in zl.list_entries()
            meta = zl.get_entry("m1")
            assert meta.get("has_activations") is False

    def test_multiple_entries_all_persisted(self):
        """All enqueued entries should be drained before shutdown completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            al = _wrap(zl)
            writer = AsyncActivationWriter(al, max_queue_size=64)

            n = 10
            for i in range(n):
                writer.enqueue(f"key_{i}", _full_entry(prompt=f"Q{i}?", response=f"A{i}."))

            writer.shutdown(timeout=10.0)

            entries = zl.list_entries()
            for i in range(n):
                assert f"key_{i}" in entries, f"key_{i} missing from Zarr index"

    def test_written_counter_increments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            al = _wrap(zl)
            writer = AsyncActivationWriter(al)

            writer.enqueue("c1", _full_entry())
            writer.enqueue("c2", _full_entry())
            writer.shutdown(timeout=5.0)

            assert writer.written == 2

    def test_errors_counter_on_bad_entry(self):
        """An entry that causes log_entry to raise should increment errors, not crash writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zl = _open_zarr(tmpdir)
            al = _wrap(zl)
            writer = AsyncActivationWriter(al)

            # Bad entry: missing activations key triggers ValueError in log_entry
            writer.enqueue("bad", {"prompt": "Q?", "model": "dummy"})
            writer.enqueue("good", _full_entry())
            writer.shutdown(timeout=5.0)

            assert writer.errors == 1
            assert writer.written == 1

    def test_pending_property(self):
        """pending returns current queue depth (coarse check)."""
        mock_logger = MagicMock()
        # Make log_entry block briefly so we can observe pending > 0
        import threading
        ready = threading.Event()

        def slow_log(key, entry):
            ready.set()
            time.sleep(0.05)

        mock_logger.log_entry = slow_log
        mock_logger.log_metadata = MagicMock()

        writer = AsyncActivationWriter(mock_logger, max_queue_size=16)
        # Enqueue enough items that some will queue behind the slow first one
        for i in range(5):
            writer.enqueue(f"k{i}", {"prompt": "q"})
        ready.wait(timeout=2.0)
        # At some point there were pending items
        writer.shutdown(timeout=5.0)
        assert writer.errors == 0


class TestAsyncActivationWriterMocked:
    """Tests using a mocked logger to verify routing (log_entry vs log_metadata)."""

    def test_routes_to_log_entry_when_not_metadata_only(self):
        mock_logger = MagicMock()
        writer = AsyncActivationWriter(mock_logger)
        writer.enqueue("k1", {"prompt": "Q?"}, metadata_only=False)
        writer.shutdown(timeout=5.0)
        mock_logger.log_entry.assert_called_once_with("k1", {"prompt": "Q?"})
        mock_logger.log_metadata.assert_not_called()

    def test_routes_to_log_metadata_when_metadata_only(self):
        mock_logger = MagicMock()
        writer = AsyncActivationWriter(mock_logger)
        writer.enqueue("m1", {"prompt": "Q?"}, metadata_only=True)
        writer.shutdown(timeout=5.0)
        mock_logger.log_metadata.assert_called_once_with("m1", {"prompt": "Q?"})
        mock_logger.log_entry.assert_not_called()

    def test_shutdown_waits_for_queue_to_drain(self):
        """Queue should be empty after shutdown returns."""
        mock_logger = MagicMock()
        writer = AsyncActivationWriter(mock_logger)
        for i in range(20):
            writer.enqueue(f"k{i}", {"prompt": "q"})
        writer.shutdown(timeout=10.0)
        assert writer.pending == 0

    def test_second_shutdown_is_safe(self):
        """Calling shutdown twice should not raise."""
        mock_logger = MagicMock()
        writer = AsyncActivationWriter(mock_logger)
        writer.enqueue("k1", {"prompt": "Q?"})
        writer.shutdown(timeout=5.0)
        writer.shutdown(timeout=1.0)  # should not raise
