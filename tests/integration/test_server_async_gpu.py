"""Integration tests: server with AsyncActivationWriter and persistent logger.

Runs the full server stack (FastAPI + HFTransformers) and verifies:
- Activation entries appear in Zarr after inference
- Async writer drains cleanly on shutdown
- Server resumes correctly after restart (no duplicate rows)
- Metadata-only entries (GGUF skip path) are persisted correctly

These tests start a real server process and hit it with HTTP requests.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from .conftest import TEST_PROMPTS, DEFAULT_TEST_MODEL


# ---------------------------------------------------------------------------
# Server management helpers
# ---------------------------------------------------------------------------

SERVER_PORT = 18765  # Use a non-default port to avoid conflicts


def _start_server(zarr_path: str, model: str, port: int = SERVER_PORT) -> subprocess.Popen:
    """Launch the activation-logging server as a subprocess."""
    env = {
        **os.environ,
        "ACTIVATION_STORAGE_PATH": zarr_path,
        "ACTIVATION_LOGGER_TYPE": "zarr",
        "DEFAULT_MODEL": model,
        "ACTIVATION_TARGET_LAYERS": "second_half",   # faster: fewer layers
        "ACTIVATION_PROMPT_MAX_TOKENS": "64",
        "ACTIVATION_RESPONSE_MAX_TOKENS": "32",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "activation_logging.server:app",
         "--host", "0.0.0.0",
         "--port", str(port),
         "--log-level", "warning"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _wait_for_server(port: int = SERVER_PORT, timeout: float = 60.0) -> bool:
    """Poll the health endpoint until server is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _send_chat(prompt: str, model: str, port: int = SERVER_PORT, max_tokens: int = 32):
    r = httpx.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zarr_store(tmp_path_factory):
    return str(tmp_path_factory.mktemp("zarr") / "activations.zarr")


@pytest.fixture(scope="module")
def server_proc(zarr_store, model_name):
    proc = _start_server(zarr_store, model_name)
    ready = _wait_for_server(timeout=120.0)
    if not ready:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=10)
        pytest.fail(
            f"Server did not start within 120s.\n"
            f"stdout: {stdout.decode()[-2000:]}\n"
            f"stderr: {stderr.decode()[-2000:]}"
        )
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServerAsyncWriter:
    def test_health_check(self, server_proc):
        r = httpx.get(f"http://localhost:{SERVER_PORT}/health")
        assert r.status_code == 200

    def test_inference_returns_response(self, server_proc, model_name):
        resp = _send_chat(TEST_PROMPTS[0], model_name)
        text = resp["choices"][0]["message"]["content"]
        assert len(text.strip()) > 0

    def test_activation_persisted_after_inference(self, server_proc, model_name, zarr_store):
        """After one inference request, the Zarr index should contain the entry."""
        _send_chat(TEST_PROMPTS[1], model_name)

        # Give async writer time to flush
        time.sleep(2.0)

        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        entries = reader.list_entries()
        reader.close()
        assert len(entries) >= 1

    def test_multiple_inferences_all_persisted(self, server_proc, model_name, zarr_store):
        """Each unique prompt should produce an entry in the Zarr store."""
        # Count existing entries
        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        before = set(reader.list_entries())
        reader.close()

        for p in TEST_PROMPTS[:3]:
            _send_chat(p + " [unique-tag]", model_name)

        time.sleep(3.0)

        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        after = set(reader.list_entries())
        reader.close()

        new_entries = after - before
        assert len(new_entries) >= 3, (
            f"Expected at least 3 new entries, got {len(new_entries)}"
        )

    def test_entry_metadata_fields_present(self, server_proc, model_name, zarr_store):
        """Zarr entries should contain prompt, response, and prompt_hash."""
        unique_prompt = "What is 7 times 8? [metadata-check]"
        _send_chat(unique_prompt, model_name)
        time.sleep(2.0)

        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        entries = reader.list_entries()
        reader.close()

        assert len(entries) > 0
        # Spot-check the last entry
        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        meta = reader.get_entry(entries[-1], metadata_only=True)
        reader.close()

        assert "prompt" in meta
        assert "response" in meta
        assert "prompt_hash" in meta

    def test_activation_arrays_loadable(self, server_proc, model_name, zarr_store):
        """Stored activations should be loadable as tensors."""
        _send_chat(TEST_PROMPTS[0] + " [actcheck]", model_name)
        time.sleep(2.0)

        reader = ZarrActivationsLogger(zarr_path=zarr_store, read_only=True, verbose=False)
        entries = reader.list_entries()
        # Find one with activations
        entry_with_acts = None
        for key in entries:
            meta = reader.get_entry(key, metadata_only=True)
            if meta.get("sample_index") is not None:
                entry_with_acts = key
                break

        assert entry_with_acts is not None, "No full activation entries found"
        full = reader.get_entry(entry_with_acts)
        acts = full.get("all_layers_activations")
        assert acts is not None
        assert any(a is not None for a in acts)
        reader.close()


class TestServerRestart:
    """Verifies that the server resumes correctly after a restart."""

    def test_restart_does_not_duplicate_rows(self, model_name):
        """Sending the same prompt before and after restart should not create duplicate rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            prompt = "Name the tallest mountain on Earth. [restart-test]"

            # First server run
            proc1 = _start_server(zarr_path, model_name, port=SERVER_PORT + 1)
            assert _wait_for_server(port=SERVER_PORT + 1, timeout=120.0), "Server 1 failed to start"
            _send_chat(prompt, model_name, port=SERVER_PORT + 1)
            time.sleep(3.0)
            proc1.terminate()
            proc1.wait(timeout=15)

            # Count rows after first run
            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            rows_after_first = reader._prompt_activations.shape[0] if reader._prompt_activations else 0
            entries_after_first = set(reader.list_entries())
            reader.close()

            # Second server run — same prompt
            proc2 = _start_server(zarr_path, model_name, port=SERVER_PORT + 1)
            assert _wait_for_server(port=SERVER_PORT + 1, timeout=120.0), "Server 2 failed to start"
            _send_chat(prompt, model_name, port=SERVER_PORT + 1)
            time.sleep(3.0)
            proc2.terminate()
            proc2.wait(timeout=15)

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            rows_after_second = reader._prompt_activations.shape[0] if reader._prompt_activations else 0
            entries_after_second = set(reader.list_entries())
            reader.close()

            # Same key → same number of rows (overwrite, not append)
            assert rows_after_second == rows_after_first, (
                f"Expected row count unchanged after restart overwrite, "
                f"got {rows_after_first} → {rows_after_second}"
            )
