#!/usr/bin/env python3
"""export_sample.py — sample N test records per (dataset, model) for the LLM-judge label audit.

Runs where the test captures live (Empire AI). Reads each test capture's
generation.jsonl and emits a small sample file with exactly the fields the judge
needs — question, gold answer(s), MC choices, the model generation, and the
current substring-match label — so the (local) judge step never needs the
multi-TB activation memmaps.

Deterministic: a seeded permutation of line indices, first N taken. Re-running
with the same seed reproduces the sample.

Usage (on the cluster):
  python scripts/label_audit/export_sample.py \
      --capture-root shared/icr_capture \
      --out-dir output/label_audit/samples \
      --n 1000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATASETS = ["hotpotqa", "mmlu", "popqa", "natural_questions",
            "sciq", "searchqa", "triviaqa", "simpleqa"]
# HF id -> capture-dir slug (last path component)
MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _gold(rec: dict) -> dict:
    """Extract every gold-answer form present, so the judge sees full context."""
    g: dict = {}
    if rec.get("answer") is not None:
        g["answer"] = rec["answer"]
    # popqa / triviaqa carry a list of acceptable answers
    for k in ("possible_answers", "answers", "aliases"):
        if rec.get(k):
            g["possible_answers"] = rec[k]
            break
    if rec.get("answer_letter") is not None:
        g["answer_letter"] = rec["answer_letter"]
    return g


def export_one(capture_root: Path, dataset: str, hf_model: str, slug: str,
               n: int, seed: int, out_dir: Path) -> str:
    cap = capture_root / f"{dataset}_test_{slug}"
    gen = cap / "generation.jsonl"
    if not gen.exists():
        return f"SKIP  {dataset}/{slug}: no generation.jsonl at {gen}"

    recs = _load_jsonl(gen)
    total = len(recs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    take = perm[: min(n, total)]

    out_path = out_dir / f"{dataset}__{slug}.jsonl"
    with out_path.open("w", encoding="utf-8") as w:
        for gi in take:
            r = recs[int(gi)]
            hallu = bool(r.get("hallucinated"))
            item = {
                "audit_id": f"{dataset}__{slug}__{int(gi)}",
                "dataset": dataset,
                "model": hf_model,
                "gen_index": int(gi),
                "question": r.get("question") or r.get("prompt", ""),
                "gold": _gold(r),
                "choices": r.get("choices"),          # MMLU only; else None
                "generation": (r.get("generation") or "").strip(),
                # current label: substring match. hallucinated=True => NOT correct.
                "substring_label": "hallucinated" if hallu else "correct",
            }
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
    return f"OK    {dataset}/{slug}: sampled {len(take)}/{total} -> {out_path.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", default="shared/icr_capture")
    ap.add_argument("--out-dir", default="output/label_audit/samples")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    capture_root = Path(args.capture_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dataset in datasets:
        for hf_model, slug in MODELS.items():
            print(export_one(capture_root, dataset, hf_model, slug,
                             args.n, args.seed, out_dir), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
