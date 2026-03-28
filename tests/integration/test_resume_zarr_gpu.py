"""Integration tests: resume-from-Zarr workflow on GPU.

Verifies:
- Client correctly identifies already-completed prompts from Zarr index
- Interrupted inference resumes without re-running completed prompts
- generate.jsonl can be exported from a Zarr store
- AsyncActivationWriter survives a simulated crash (SIGTERM mid-run)
  and resumes from correct state on restart.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from activation_logging.zarr_activations_logger import ZarrActivationsLogger
from .conftest import TEST_PROMPTS, DEFAULT_TEST_MODEL
from .test_server_async_gpu import (
    SERVER_PORT,
    _start_server,
    _wait_for_server,
    _send_chat,
)


RESUME_PORT = SERVER_PORT + 10


def _prompt_hash(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Resume from Zarr index
# ---------------------------------------------------------------------------

class TestResumeFromZarr:
    def test_completed_prompts_not_rerun(self, model_name):
        """If N prompts are already in Zarr, run_exp should skip them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            prompts = TEST_PROMPTS[:4]

            # Pre-populate the Zarr store with the first 2 prompts
            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0), "Server failed to start"

            for p in prompts[:2]:
                _send_chat(p, model_name, port=RESUME_PORT)

            time.sleep(3.0)
            proc.terminate()
            proc.wait(timeout=15)

            # Verify 2 entries exist
            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            completed_hashes = {
                meta.get("prompt_hash")
                for meta in reader._index.values()
                if meta.get("prompt_hash")
            }
            reader.close()
            assert len(completed_hashes) >= 2

            # Simulate run_exp resume logic: filter out already-done prompts
            import pandas as pd
            all_df = pd.DataFrame({"prompt": prompts})
            all_df["_ph"] = all_df["prompt"].apply(_prompt_hash)
            remaining = all_df[~all_df["_ph"].isin(completed_hashes)]
            already_done = len(all_df) - len(remaining)

            assert already_done == 2, f"Expected 2 skipped, got {already_done}"
            assert len(remaining) == 2, f"Expected 2 remaining, got {len(remaining)}"

    def test_full_run_then_resume_completes_cleanly(self, model_name):
        """Full run + identical resume should produce the same number of entries (no duplicates)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            prompts = TEST_PROMPTS[:2]

            # Full run
            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)
            for p in prompts:
                _send_chat(p, model_name, port=RESUME_PORT)
            time.sleep(3.0)
            proc.terminate()
            proc.wait(timeout=15)

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            rows_after_full_run = reader._prompt_activations.shape[0] if reader._prompt_activations else 0
            reader.close()

            # "Resume" run: same prompts → should overwrite, not append
            proc2 = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)
            for p in prompts:
                _send_chat(p, model_name, port=RESUME_PORT)
            time.sleep(3.0)
            proc2.terminate()
            proc2.wait(timeout=15)

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            rows_after_resume = reader._prompt_activations.shape[0] if reader._prompt_activations else 0
            reader.close()

            assert rows_after_resume == rows_after_full_run, (
                f"Resume should not add rows: {rows_after_full_run} → {rows_after_resume}"
            )


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    def test_partial_run_data_persisted_after_sigterm(self, model_name):
        """Even after SIGTERM mid-run, already-written entries should survive in Zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")

            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)

            # Send some requests
            for p in TEST_PROMPTS[:2]:
                _send_chat(p, model_name, port=RESUME_PORT)

            time.sleep(2.0)

            # SIGTERM — give async writer 30s to drain (server handles graceful shutdown)
            proc.terminate()
            try:
                proc.wait(timeout=35)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            entries = reader.list_entries()
            reader.close()

            # At least the entries that were sent before SIGTERM should be present
            assert len(entries) >= 1, (
                "Expected at least 1 entry after graceful shutdown with async writer drain"
            )

    def test_zarr_index_readable_after_kill(self, model_name):
        """Hard SIGKILL (no drain) should leave a valid (possibly partial) Zarr index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")

            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)

            _send_chat(TEST_PROMPTS[0], model_name, port=RESUME_PORT)
            time.sleep(1.0)

            # SIGKILL — no graceful shutdown
            proc.kill()
            proc.wait()

            # Zarr store should still be readable
            try:
                reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
                _ = reader.list_entries()
                reader.close()
            except Exception as e:
                pytest.fail(f"Zarr store unreadable after SIGKILL: {e}")


# ---------------------------------------------------------------------------
# Export generation.jsonl from Zarr
# ---------------------------------------------------------------------------

class TestExportGenerationJsonl:
    def test_export_produces_jsonl_with_correct_fields(self, model_name):
        """export_generation_jsonl should create a JSONL file with prompt/generation fields."""
        import pandas as pd
        from utils.exp import export_generation_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            output_path = str(Path(tmpdir) / "generation.jsonl")
            prompts = TEST_PROMPTS[:2]

            # Run inference
            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)
            for p in prompts:
                _send_chat(p, model_name, port=RESUME_PORT)
            time.sleep(3.0)
            proc.terminate()
            proc.wait(timeout=15)

            # Export
            qa_df = pd.DataFrame({"prompt": prompts, "answer": ["Paris", "Jupiter"]})
            export_generation_jsonl(zarr_path, qa_df, output_path)

            assert Path(output_path).exists()
            with open(output_path) as f:
                records = [json.loads(line) for line in f if line.strip()]

            assert len(records) >= 1
            for rec in records:
                assert "prompt" in rec or "generation" in rec, (
                    f"Record missing expected fields: {rec.keys()}"
                )

    def test_export_generation_matches_zarr_responses(self, model_name):
        """Exported generation field should match the response stored in Zarr."""
        import pandas as pd
        from utils.exp import export_generation_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = str(Path(tmpdir) / "activations.zarr")
            output_path = str(Path(tmpdir) / "generation.jsonl")
            prompt = TEST_PROMPTS[0]

            proc = _start_server(zarr_path, model_name, port=RESUME_PORT)
            assert _wait_for_server(RESUME_PORT, timeout=120.0)
            _send_chat(prompt, model_name, port=RESUME_PORT)
            time.sleep(3.0)
            proc.terminate()
            proc.wait(timeout=15)

            # Get the stored response from Zarr
            reader = ZarrActivationsLogger(zarr_path=zarr_path, read_only=True, verbose=False)
            ph = _prompt_hash(prompt)
            zarr_response = None
            for key, meta in reader._index.items():
                if meta.get("prompt_hash") == ph:
                    zarr_response = meta.get("response")
                    break
            reader.close()

            assert zarr_response is not None, "Prompt not found in Zarr index"

            # Export and compare
            qa_df = pd.DataFrame({"prompt": [prompt], "answer": ["expected"]})
            export_generation_jsonl(zarr_path, qa_df, output_path)

            with open(output_path) as f:
                records = [json.loads(line) for line in f if line.strip()]

            assert len(records) == 1
            assert records[0].get("generation") == zarr_response
