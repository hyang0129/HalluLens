#!/usr/bin/env python3
"""judge.py — re-label audit samples with an LLM judge (Sonnet via `claude -p`).

Reads the sample files from export_sample.py, sends batches to `claude -p`, and
records a CORRECT/INCORRECT verdict per item to compare against the current
substring-match label. Runs LOCALLY (where the `claude` CLI is authenticated).
Shared judging primitives live in judge_core.py (also used by backfill_judge_labels.py).

- Batched (default 20 items/call), concurrent (default 8 workers).
- Resumable: audit_ids already in the judged output are skipped.

Usage (local):
  python scripts/label_audit/judge.py \
      --samples-dir output/label_audit/samples \
      --out-dir output/label_audit/judged \
      --model sonnet --batch-size 20 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from judge_core import judge_batch  # same dir on sys.path when run as a script


def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
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
            futs = {ex.submit(judge_batch, b, model, timeout, id_key="audit_id"): b for b in batches}
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
