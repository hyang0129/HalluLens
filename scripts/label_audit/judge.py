#!/usr/bin/env python3
"""judge.py — re-label audit samples with an LLM judge (Sonnet via `claude -p`).

Reads the sample files from export_sample.py, sends batches to `claude -p`, and
records a CORRECT/INCORRECT verdict per item to compare against the current
substring-match label. Runs LOCALLY (where the `claude` CLI is authenticated).

- Batched (default 20 items/call) to amortize CLI/model start-up latency.
- Concurrent (default 8 workers).
- Resumable: audit_ids already present in the judged output are skipped.
- Robust parse: strips fences, extracts the JSON array; a batch that won't parse
  or is missing ids is retried at batch size 1 before giving up (verdict=UNKNOWN).

Usage (local):
  python scripts/label_audit/judge.py \
      --samples-dir output/label_audit/samples \
      --out-dir output/label_audit/judged \
      --model sonnet --batch-size 20 --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    g = rec.get("gold") or {}
    ref = {}
    if g.get("answer") is not None:
        ref["answer"] = g["answer"]
    if g.get("possible_answers"):
        ref["acceptable_answers"] = g["possible_answers"]
    item = {
        "id": rec["audit_id"],
        "question": rec.get("question", ""),
        "reference": ref,
        "model_answer": rec.get("generation", ""),
    }
    if rec.get("choices"):
        item["multiple_choice_options"] = rec["choices"]
    return item


def _extract_json_array(text: str):
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


def call_claude(prompt: str, model: str, timeout: int) -> str:
    r = subprocess.run(
        ["claude", "-p", "--model", model],
        input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude rc={r.returncode}: {r.stderr[:200]}")
    return r.stdout


def judge_batch(batch: list[dict], model: str, timeout: int) -> dict[str, dict]:
    """Return {audit_id: {verdict, reason}} for a batch; UNKNOWN on unrecoverable failure."""
    items = [build_item(r) for r in batch]
    prompt = INSTRUCTION + json.dumps(items, ensure_ascii=False)
    ids = {r["audit_id"] for r in batch}
    try:
        out = call_claude(prompt, model, timeout)
        arr = _extract_json_array(out)
    except (subprocess.TimeoutExpired, RuntimeError):
        arr = None

    verdicts: dict[str, dict] = {}
    if arr:
        for o in arr:
            oid = str(o.get("id", ""))
            v = str(o.get("verdict", "")).upper()
            if oid in ids and v in ("CORRECT", "INCORRECT"):
                verdicts[oid] = {"verdict": v, "reason": str(o.get("reason", ""))[:160]}

    missing = [r for r in batch if r["audit_id"] not in verdicts]
    # Retry the missing ones one-at-a-time (only if the batch was >1, to avoid loops)
    if missing and len(batch) > 1:
        for r in missing:
            verdicts.update(judge_batch([r], model, timeout))
    elif missing:  # single item still failed
        for r in missing:
            verdicts[r["audit_id"]] = {"verdict": "UNKNOWN", "reason": "judge parse/timeout failure"}
    return verdicts


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["audit_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def process_file(sample_path: Path, out_dir: Path, model: str,
                 batch_size: int, workers: int, timeout: int) -> str:
    recs = [json.loads(l) for l in sample_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = out_dir / sample_path.name.replace(".jsonl", ".judged.jsonl")
    done = load_done(out_path)
    todo = [r for r in recs if r["audit_id"] not in done]
    if not todo:
        return f"OK    {sample_path.name}: all {len(recs)} already judged"

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    write_lock = threading.Lock()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(judge_batch, b, model, timeout): b for b in batches}
            for fut in as_completed(futs):
                batch = futs[fut]
                verdicts = fut.result()
                by_id = {r["audit_id"]: r for r in batch}
                with write_lock:
                    for aid, v in verdicts.items():
                        rec = by_id[aid]
                        w.write(json.dumps({
                            "audit_id": aid,
                            "dataset": rec["dataset"],
                            "model": rec["model"],
                            "substring_label": rec["substring_label"],
                            "judge_verdict": v["verdict"],
                            "judge_correct": v["verdict"] == "CORRECT",
                            "judge_reason": v["reason"],
                            "judge_model": model,
                        }, ensure_ascii=False) + "\n")
                        n_done += 1
                    w.flush()
    return f"OK    {sample_path.name}: judged {n_done}/{len(todo)} new ({len(recs)} total)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="output/label_audit/samples")
    ap.add_argument("--out-dir", default="output/label_audit/judged")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--glob", default="*.jsonl", help="which sample files to judge")
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(samples_dir.glob(args.glob))
    if not files:
        print(f"no sample files in {samples_dir}/{args.glob}", file=sys.stderr)
        return 1
    for f in files:
        print(process_file(f, out_dir, args.model, args.batch_size,
                           args.workers, args.timeout), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
