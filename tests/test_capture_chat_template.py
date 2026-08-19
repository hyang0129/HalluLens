"""
tests/test_capture_chat_template.py — CPU-only unit tests for the opt-in
--chat-template support added to scripts/capture_inference.py and
scripts/dispatch/generate_manifest.py.

No GPU, no model downloads, no torch import — stub tokenizer objects stand
in for the real HF tokenizer.

Test inventory:
  1. apply_chat_template_to_prompt / build_prompt — template wrapping,
     enable_thinking gating, raw-prompt passthrough when disabled.
  2. response_len_from_ids — EOS scanning over a set of eos ids.
  3. normalize_eos_ids — int / list / None generation_config shapes.
  4. generate_manifest --chat-template — cell JSON field wiring.
  5. check_capture_mode_compat — chat_template/legacy out_dir mixing guard.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 1. Chat-template prompt building
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Minimal stand-in for a HF tokenizer's apply_chat_template.

    Records the kwargs it was called with so tests can assert on them, and
    renders a template string that trivially contains the raw user content
    (so tests can assert the raw prompt survives templating).
    """

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        user_content = messages[0]["content"]
        return f"<bos><|user|>{user_content}<|assistant|>"


def test_apply_chat_template_wraps_raw_prompt_as_user_message():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    raw = "Answer the question concisely.\n\nQ: What is 2+2?\nA:"
    templated = apply_chat_template_to_prompt(raw, tok, "meta-llama/Llama-3.1-8B-Instruct")

    assert len(tok.calls) == 1
    assert tok.calls[0]["messages"] == [{"role": "user", "content": raw}]
    assert raw in templated, "raw prompt content must survive templating"


def test_apply_chat_template_add_generation_prompt_true():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    apply_chat_template_to_prompt("hi", tok, "meta-llama/Llama-3.1-8B-Instruct")

    assert tok.calls[0]["kwargs"]["add_generation_prompt"] is True
    assert tok.calls[0]["kwargs"]["tokenize"] is False


def test_apply_chat_template_qwen3_gets_enable_thinking_false():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    apply_chat_template_to_prompt("hi", tok, "Qwen/Qwen3-8B")

    assert tok.calls[0]["kwargs"].get("enable_thinking") is False


def test_apply_chat_template_qwen3_case_insensitive():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    apply_chat_template_to_prompt("hi", tok, "SomeOrg/QWEN3-1.5b-custom")

    assert tok.calls[0]["kwargs"].get("enable_thinking") is False


def test_apply_chat_template_smollm3_gets_enable_thinking_false():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    apply_chat_template_to_prompt("hi", tok, "HuggingFaceTB/SmolLM3-3B")

    assert tok.calls[0]["kwargs"].get("enable_thinking") is False


def test_apply_chat_template_llama_no_enable_thinking():
    from scripts.capture_inference import apply_chat_template_to_prompt

    tok = _StubTokenizer()
    apply_chat_template_to_prompt("hi", tok, "meta-llama/Llama-3.1-8B-Instruct")

    assert "enable_thinking" not in tok.calls[0]["kwargs"]


def _make_fake_task_module() -> types.ModuleType:
    m = types.ModuleType("tasks.llmsknow._fake")

    def format_prompt(question: str) -> str:
        return f"Answer the question concisely.\n\nQ: {question}\nA:"

    m.format_prompt = format_prompt
    return m


def test_build_prompt_flag_off_returns_raw_prompt_unchanged():
    from scripts.capture_inference import build_prompt

    task_module = _make_fake_task_module()
    sample = {"question": "What is the capital of France?"}

    prompt = build_prompt(sample, task_module, chat_template=False)

    assert prompt == "Answer the question concisely.\n\nQ: What is the capital of France?\nA:"


def test_build_prompt_flag_on_applies_template():
    from scripts.capture_inference import build_prompt

    task_module = _make_fake_task_module()
    sample = {"question": "What is the capital of France?"}
    tok = _StubTokenizer()

    prompt = build_prompt(
        sample, task_module, tokenizer=tok, model_name="Qwen/Qwen3-8B", chat_template=True,
    )

    assert "What is the capital of France?" in prompt
    assert prompt.startswith("<bos><|user|>")
    assert tok.calls[0]["kwargs"].get("enable_thinking") is False


def test_build_prompt_flag_on_requires_tokenizer_and_model_name():
    from scripts.capture_inference import build_prompt

    task_module = _make_fake_task_module()
    sample = {"question": "x"}

    with pytest.raises(AssertionError):
        build_prompt(sample, task_module, chat_template=True)


# ---------------------------------------------------------------------------
# 2. response_len_from_ids
# ---------------------------------------------------------------------------

def test_response_len_no_eos_returns_full_length():
    from scripts.capture_inference import response_len_from_ids

    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert response_len_from_ids(ids, {999}) == 5


def test_response_len_eos_mid_sequence_returns_index_plus_one():
    from scripts.capture_inference import response_len_from_ids

    ids = np.array([1, 2, 999, 4, 5], dtype=np.int64)
    assert response_len_from_ids(ids, {999}) == 3


def test_response_len_multiple_eos_ids_all_detected():
    from scripts.capture_inference import response_len_from_ids

    # Llama-3.1-Instruct style: 128001 / 128008 / 128009 are all valid stops.
    eos_ids = {128001, 128008, 128009}

    ids_a = np.array([10, 20, 128008, 30], dtype=np.int64)
    assert response_len_from_ids(ids_a, eos_ids) == 3

    ids_b = np.array([10, 20, 128001, 30], dtype=np.int64)
    assert response_len_from_ids(ids_b, eos_ids) == 3

    ids_c = np.array([10, 20, 128009, 30], dtype=np.int64)
    assert response_len_from_ids(ids_c, eos_ids) == 3


def test_response_len_first_eos_wins():
    from scripts.capture_inference import response_len_from_ids

    eos_ids = {128001, 128009}
    ids = np.array([128009, 1, 128001, 2], dtype=np.int64)
    assert response_len_from_ids(ids, eos_ids) == 1


def test_response_len_qwen3_eos_ids():
    from scripts.capture_inference import response_len_from_ids

    eos_ids = {151643, 151645}
    ids = np.array([5, 6, 151645, 7], dtype=np.int64)
    assert response_len_from_ids(ids, eos_ids) == 3


# ---------------------------------------------------------------------------
# 3. normalize_eos_ids
# ---------------------------------------------------------------------------

def test_normalize_eos_ids_int():
    from scripts.capture_inference import normalize_eos_ids

    assert normalize_eos_ids(128009, None) == {128009}


def test_normalize_eos_ids_list():
    from scripts.capture_inference import normalize_eos_ids

    assert normalize_eos_ids([128001, 128008, 128009], None) == {128001, 128008, 128009}


def test_normalize_eos_ids_none_falls_back_to_tokenizer_eos():
    from scripts.capture_inference import normalize_eos_ids

    assert normalize_eos_ids(None, 128001) == {128001}


def test_normalize_eos_ids_none_and_no_tokenizer_eos_is_empty():
    from scripts.capture_inference import normalize_eos_ids

    assert normalize_eos_ids(None, None) == set()


def test_normalize_eos_ids_merges_generation_config_and_tokenizer():
    from scripts.capture_inference import normalize_eos_ids

    # tokenizer.eos_token_id is always folded in, even if not already in the list.
    assert normalize_eos_ids([151643], 151645) == {151643, 151645}


# ---------------------------------------------------------------------------
# 4. generate_manifest --chat-template wiring
# ---------------------------------------------------------------------------

def test_generate_manifest_chat_template_true_sets_cell_field(tmp_path):
    from scripts.dispatch.generate_manifest import generate_manifest

    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr_chat"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["sciq"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"],
        n_samples=None,
        chat_template=True,
    )
    assert n == 1

    pending = list((dispatch_root / "pending").glob("*.json"))
    assert len(pending) == 1
    cell = json.loads(pending[0].read_text())
    assert cell["chat_template"] is True


def test_generate_manifest_chat_template_false_by_default(tmp_path):
    from scripts.dispatch.generate_manifest import generate_manifest

    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr"

    n = generate_manifest(
        dispatch_root=dispatch_root,
        out_base_dir=out_base,
        tasks=["sciq"],
        models=["meta-llama/Llama-3.1-8B-Instruct"],
        splits=["test"],
        n_samples=None,
    )
    assert n == 1

    pending = list((dispatch_root / "pending").glob("*.json"))
    cell = json.loads(pending[0].read_text())
    # Why the assertion is `not cell["chat_template"]` rather than absence:
    # generate_manifest always writes the key so worker.sh's `d.get('chat_template')`
    # default-false behavior is exercised on an explicit `false` here, and on
    # missing keys in cells written by older manifest versions.
    assert not cell["chat_template"]


def test_generate_manifest_chat_template_cli_flag(tmp_path):
    """--chat-template on the CLI must flow through to the emitted cell JSON."""
    import subprocess

    dispatch_root = tmp_path / "_dispatch"
    out_base = tmp_path / "icr_chat"
    repo_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/dispatch/generate_manifest.py"),
            "--dispatch-root", str(dispatch_root),
            "--out-base-dir", str(out_base),
            "--tasks", "sciq",
            "--models", "meta-llama/Llama-3.1-8B-Instruct",
            "--splits", "test",
            "--chat-template",
        ],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr

    pending = list((dispatch_root / "pending").glob("*.json"))
    assert len(pending) == 1
    cell = json.loads(pending[0].read_text())
    assert cell["chat_template"] is True


# ---------------------------------------------------------------------------
# 5. check_capture_mode_compat — chat_template/legacy out_dir mixing guard
# ---------------------------------------------------------------------------

def test_check_capture_mode_compat_both_false_matches():
    from scripts.capture_inference import check_capture_mode_compat

    assert check_capture_mode_compat({"chat_template": False}, False) is True


def test_check_capture_mode_compat_both_true_matches():
    from scripts.capture_inference import check_capture_mode_compat

    assert check_capture_mode_compat({"chat_template": True}, True) is True


def test_check_capture_mode_compat_legacy_missing_key_treated_as_false():
    from scripts.capture_inference import check_capture_mode_compat

    # config.json written before this flag existed has no "chat_template" key.
    legacy_config = {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "r_max": 64}
    assert check_capture_mode_compat(legacy_config, False) is True
    assert check_capture_mode_compat(legacy_config, True) is False


def test_check_capture_mode_compat_mismatch_existing_true_requested_false():
    from scripts.capture_inference import check_capture_mode_compat

    assert check_capture_mode_compat({"chat_template": True}, False) is False


def test_check_capture_mode_compat_mismatch_existing_false_requested_true():
    from scripts.capture_inference import check_capture_mode_compat

    assert check_capture_mode_compat({"chat_template": False}, True) is False


def _make_fake_hotpotqa_module_for_capture():
    """Fake tasks.llmsknow.hotpotqa exposing load_hotpotqa for the guard test.

    Avoids importing the real task module (heavy deps) and avoids any network
    or dataset I/O — the guard must reject the run before the dataset loader
    result even matters.
    """
    m = types.ModuleType("tasks.llmsknow.hotpotqa")

    def load_hotpotqa(split, n_samples=None):
        rows = [{"question": "Q1?", "answer": "A1"}, {"question": "Q2?", "answer": "A2"}]
        return rows[:n_samples] if n_samples is not None else rows

    def is_correct(generation, answer):
        return answer.lower() in generation.lower()

    def format_prompt(question):
        return f"Q: {question}\nA:"

    m.load_hotpotqa = load_hotpotqa
    m.is_correct = is_correct
    m.format_prompt = format_prompt
    return m


def test_run_capture_mismatch_exits_nonzero_without_loading_model(tmp_path):
    """main(--step capture --chat-template) must bail before load_model_eager()
    when out_dir/config.json already exists with a different chat_template mode.

    Regression target: resume keys off sha256(prompt), and a templated prompt
    hashes differently from its raw equivalent, so append mode alone would
    silently write mixed-convention rows into the same memmap instead of
    erroring. This exercises the full main() -> _run_capture() path (torch is
    importable in this environment, but the guard must trip before any model
    load, dataset network I/O beyond the stubbed loader, or writer/file I/O).
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Existing capture was legacy (no chat_template key at all).
    (out_dir / "config.json").write_text(json.dumps({"model_name": "fake-model", "r_max": 64}))

    fake_task_mod = _make_fake_hotpotqa_module_for_capture()

    test_args = [
        "capture_inference.py",
        "--task", "hotpotqa",
        "--model", "fake-model",
        "--out-dir", str(out_dir),
        "--step", "capture",
        "--chat-template",
    ]

    import importlib
    import sys as _sys
    import scripts.capture_inference as ci_mod

    def _fail_if_called(model_name):
        raise AssertionError(
            "load_model_eager must not be called when the chat_template guard trips"
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_sys, "argv", test_args)
        mp.setitem(_sys.modules, "tasks.llmsknow.hotpotqa", fake_task_mod)

        importlib.reload(ci_mod)
        # Reload rebinds module globals, so the patch must be (re-)applied after.
        mp.setattr(ci_mod, "load_model_eager", _fail_if_called)
        ret = ci_mod.main()

    assert ret == 1, "main() must return 1 on chat_template mode mismatch"

    # Guard must trip before the writer opens — config.json content untouched,
    # and no generation.jsonl / meta.jsonl were ever created.
    assert not (out_dir / "generation.jsonl").exists()
    assert not (out_dir / "meta.jsonl").exists()
    existing_config = json.loads((out_dir / "config.json").read_text())
    assert existing_config == {"model_name": "fake-model", "r_max": 64}
