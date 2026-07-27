#!/usr/bin/env python3
"""backfill_judge_labels.py — write LLM-judge labels into icr_capture dirs (#145).

For each capture dir, reads generation.jsonl, judges every sample (Sonnet via
`claude -p`), and writes the sidecar consumed by activation_research/labels.py:

  <dir>/judge_labels.jsonl     one JSON/line keyed by sample_index:
      {"sample_index": 0, "judge_verdict": "CORRECT|INCORRECT|UNKNOWN",
       "hallucinated": false}   # hallucinated == (verdict == "INCORRECT")
  <dir>/judge_labels_meta.json  provenance (judge_model, prompt_version, ...).

Resumable: sample_indexes already in judge_labels.jsonl are skipped. Full train
captures are large (up to ~90k); run incrementally / per capture as needed.

Usage (local, `claude` authenticated):
  python scripts/label_audit/backfill_judge_labels.py \
      --capture-dirs shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct ... \
      --model sonnet --batch-size 20 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from judge_core import PROMPT_VERSION, judge_batch  # same dir on sys.path as a script


def _load_generation(cap: Path) -> list[dict]:
    recs: list[dict] = []
    with (cap / "generation.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _compact_sidecar(sidecar: Path) -> set[int]:
    """Rewrite the sidecar keeping only the last CORRECT/INCORRECT verdict per
    sample_index (drops UNKNOWN rows + duplicates). Returns the finalized
    sample_indexes, so a continue-run re-judges UNKNOWN and missing samples."""
    if not sidecar.exists():
        return set()
    good: dict[int, dict] = {}
    with sidecar.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                si = int(o["sample_index"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if str(o.get("judge_verdict", "")).upper() == "UNKNOWN":
                continue
            good[si] = o  # last write wins
    tmp = sidecar.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as w:
        for si in sorted(good):
            w.write(json.dumps(good[si]) + "\n")
    tmp.replace(sidecar)
    return set(good)


def _judge_record(rec: dict) -> dict:
    """Normalize a generation.jsonl row to a judging record keyed by sample_index.

    Note: some datasets (e.g. hotpotqa) carry their own `id`; we key on
    sample_index to align with meta.jsonl, so set the judge `id` explicitly.
    """
    return {
        "id": str(rec["sample_index"]),
        "sample_index": int(rec["sample_index"]),
        "question": rec.get("question") or rec.get("prompt", ""),
        "answer": rec.get("answer"),
        "possible_answers": rec.get("possible_answers"),
        "choices": rec.get("choices"),
        "generation": rec.get("generation", ""),
    }


def backfill_one(cap: Path, model: str, batch_size: int, workers: int, timeout: int) -> str:
    if not (cap / "generation.jsonl").exists():
        return f"SKIP  {cap.name}: no generation.jsonl"
    recs = [_judge_record(r) for r in _load_generation(cap)]
    sidecar = cap / "judge_labels.jsonl"
    done = _compact_sidecar(sidecar)  # drops prior UNKNOWN/dupes; retries them
    todo = [r for r in recs if r["sample_index"] not in done]
    if not todo:
        return f"OK    {cap.name}: all {len(recs)} already judged"

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    lock = threading.Lock()
    n_new = n_unknown = 0
    with sidecar.open("a", encoding="utf-8") as w:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(judge_batch, b, model, timeout, id_key="id"): b for b in batches}
            for fut in as_completed(futs):
                by_id = {r["id"]: r for r in futs[fut]}
                with lock:
                    for rid, v in fut.result().items():
                        verdict = v["verdict"]
                        n_unknown += verdict == "UNKNOWN"
                        w.write(json.dumps({
                            "sample_index": by_id[rid]["sample_index"],
                            "judge_verdict": verdict,
                            "hallucinated": verdict == "INCORRECT",
                        }) + "\n")
                        n_new += 1
                    w.flush()

    (cap / "judge_labels_meta.json").write_text(json.dumps({
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "n_judged": len(done) + n_new,
        "n_unknown_this_run": n_unknown,
        "source": "generation.jsonl",
    }, indent=2))
    return f"OK    {cap.name}: +{n_new} new ({n_unknown} UNKNOWN), total {len(done)+n_new}/{len(recs)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dirs", nargs="+", required=True,
                    help="icr_capture dirs to backfill (each needs generation.jsonl)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    for d in args.capture_dirs:
        print(backfill_one(Path(d), args.model, args.batch_size, args.workers, args.timeout), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
