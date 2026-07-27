#!/usr/bin/env python3
"""judge_core.py — shared LLM-judge primitives (Sonnet via `claude -p`).

Used by both:
  - judge.py                 (the 1k-sample audit, issue #143)
  - backfill_judge_labels.py (full-capture relabel into judge_labels.jsonl, #145)

The judge grades whether a model's answer is correct given the reference
answer(s), reference-grounded ("answer matching"). Verdict: CORRECT | INCORRECT.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

PROMPT_VERSION = "v1"

INSTRUCTION = """You are grading whether a model's ANSWER to a QUESTION is correct, given the REFERENCE answer(s).

Rules:
- Judge SEMANTIC correctness, not wording. A paraphrase, synonym, or equivalent value/number is CORRECT.
- If the answer contradicts the reference, omits the answer, or adds a wrong claim that changes the conclusion, it is INCORRECT.
- If the answer rambles or self-corrects, judge its final/overall committed answer.
- For MULTIPLE CHOICE, the reference is the correct option. CORRECT iff the answer selects or states that option (by letter OR by its text), even with extra text. INCORRECT if it commits to a different option.
- Extra correct detail is fine; a different factual answer is INCORRECT.

Output ONLY a JSON array, one object per item, in the SAME ORDER as given:
[{"id": "<id>", "verdict": "CORRECT" or "INCORRECT", "reason": "<=12 words"}]
No markdown, no code fences, no prose before or after the array.

Items to grade:
"""


def build_item(rec: dict) -> dict:
    """Map a sample record to the compact judge item. Accepts either the audit
    sample schema (gold dict) or a raw generation.jsonl record (answer/possible_answers)."""
    gold = rec.get("gold")
    if gold is None:
        gold = {}
        if rec.get("answer") is not None:
            gold["answer"] = rec["answer"]
        for k in ("possible_answers", "answers", "aliases"):
            if rec.get(k):
                gold["possible_answers"] = rec[k]
                break
    ref = {}
    if gold.get("answer") is not None:
        ref["answer"] = gold["answer"]
    if gold.get("possible_answers"):
        ref["acceptable_answers"] = gold["possible_answers"]
    item = {
        "id": rec["id"] if "id" in rec else rec.get("audit_id"),
        "question": rec.get("question") or rec.get("prompt", ""),
        "reference": ref,
        "model_answer": (rec.get("generation") or "").strip(),
    }
    if rec.get("choices"):
        item["multiple_choice_options"] = rec["choices"]
    return item


def extract_json_array(text: str):
    """Pull the first top-level JSON array out of a model response."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "[":
            depth += 1
        elif t[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def call_claude(prompt: str, model: str, timeout: int, max_retries: int = 5) -> str:
    """Call `claude -p` as a stateless, no-logging function, retrying transient
    failures (rate-limit / overload / timeout) with exponential backoff before
    giving up. Raises RuntimeError only after exhausting retries.

    Grading fans this out tens of thousands of times. By default each call writes
    a session transcript to ~/.claude/projects/<slug>/<uuid>.jsonl plus a
    prompt-history entry; at ~50k calls that flood is enough to crash the VSCode
    extension that watches the directory. `--no-session-persistence` (print mode
    only) skips the transcript and CLAUDE_CODE_SKIP_PROMPT_HISTORY=1 skips the
    history append, so each invocation leaves no on-disk trace."""
    env = {**os.environ, "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1"}
    last = "unknown error"
    for attempt in range(max_retries):
        try:
            r = subprocess.run(
                ["claude", "-p", "--no-session-persistence", "--model", model],
                input=prompt, capture_output=True, text=True, timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            last = "timeout"
            time.sleep(min(45, 3 * (2 ** attempt)))
            continue
        if r.returncode == 0:
            return r.stdout
        last = (r.stderr or r.stdout or "")[:200]
        # rc != 0 is typically a transient rate-limit / overload → backoff + retry
        time.sleep(min(45, 3 * (2 ** attempt)))
    raise RuntimeError(f"claude failed after {max_retries} retries: {last}")


def judge_batch(batch: list[dict], model: str, timeout: int, *, id_key: str) -> dict[str, dict]:
    """Judge a batch of sample records. Returns {id: {verdict, reason}} keyed by rec[id_key].

    A batch that won't parse or is missing ids is retried at batch size 1 before
    giving up (verdict=UNKNOWN)."""
    items = [build_item(r) for r in batch]
    prompt = INSTRUCTION + json.dumps(items, ensure_ascii=False)
    ids = {str(r[id_key]) for r in batch}
    try:
        arr = extract_json_array(call_claude(prompt, model, timeout))
    except (subprocess.TimeoutExpired, RuntimeError):
        arr = None

    verdicts: dict[str, dict] = {}
    if arr:
        for o in arr:
            oid = str(o.get("id", ""))
            v = str(o.get("verdict", "")).upper()
            if oid in ids and v in ("CORRECT", "INCORRECT"):
                verdicts[oid] = {"verdict": v, "reason": str(o.get("reason", ""))[:160]}

    missing = [r for r in batch if str(r[id_key]) not in verdicts]
    if missing and len(batch) > 1:
        for r in missing:
            verdicts.update(judge_batch([r], model, timeout, id_key=id_key))
    elif missing:  # single item still failed
        for r in missing:
            verdicts[str(r[id_key])] = {"verdict": "UNKNOWN", "reason": "judge parse/timeout failure"}
    return verdicts
