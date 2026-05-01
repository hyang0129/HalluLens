#!/usr/bin/env python3
"""Extract generation.jsonl from a zarr activation store's meta/index.jsonl.

The zarr store contains prompt+response in meta/index.jsonl but lacks the
task-level metadata (id, question, answer, type, level) that eval needs.
This script reloads the original task data, matches on prompt text, and
writes a generation.jsonl identical in structure to what inference produces.

Usage:
    python scripts/extract_generation_from_zarr.py \\
        --task hotpotqa --split validation \\
        --zarr shared/hotpotqa_llama_3_1_8b_instruct/activations.zarr \\
        --out output/hotpotqa/Llama-3.1-8B-Instruct/generation.jsonl

    # Dry-run parity check against existing generation.jsonl:
    python scripts/extract_generation_from_zarr.py \\
        --task hotpotqa --split validation \\
        --zarr shared/hotpotqa_qwen3_8b/activations.zarr \\
        --out /tmp/gen_check.jsonl \\
        --verify output/hotpotqa/Qwen3-8B/generation.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_zarr_index(zarr_path: Path) -> dict:
    """Return {prompt_text: response_text} from meta/index.jsonl."""
    index_path = zarr_path / "meta" / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"No meta/index.jsonl found at {zarr_path}")
    mapping = {}
    with open(index_path) as f:
        for line in f:
            entry = json.loads(line)
            mapping[entry["prompt"]] = entry["response"]
    return mapping


def load_task_prompts(task: str, split: str, split_seed: int = 42):
    """Load task dataset rows with id/question/answer/etc and built prompts.

    Inlines data loading to avoid importing task modules that pull in openai.
    """
    if task == "hotpotqa":
        from datasets import load_dataset
        print(f"Loading HotpotQA (distractor / {split}) from HuggingFace...")
        dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)[split]
        rows = []
        for item in dataset:
            question = item["question"]
            prompt = f"Answer the question concisely.\n\nQ: {question}\nA:"
            rows.append({
                "id":       item["id"],
                "question": question,
                "answer":   item["answer"],
                "type":     item.get("type", ""),
                "level":    item.get("level", ""),
                "prompt":   prompt,
            })
        return rows

    elif task == "naturalquestions":
        import pandas as pd
        data_path = ROOT / "external/LLMsKnow/data/nq_wc_dataset.csv"
        print(f"Loading NQ from {data_path}...")
        df = pd.read_csv(data_path)
        df = df.rename(columns={"Question": "question", "Answer": "answer"})
        # Load all rows — the zarr index itself determines the split.
        # Do not re-apply a sklearn split here; the inference may have used a
        # different seed, and the zarr store is the authoritative split record.
        rows = []
        for _, item in df.iterrows():
            question = item["question"]
            prompt = f"Answer the question concisely.\n\nQuestion: {question}\n\nAnswer:"
            rows.append({
                "question": question,
                "answer":   item["answer"],
                "prompt":   prompt,
            })
        return rows

    else:
        raise ValueError(f"Unsupported task: {task}. Add it to load_task_prompts().")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   required=True, help="Task name: hotpotqa, naturalquestions")
    parser.add_argument("--split",  required=True, help="Dataset split: validation, train, test")
    parser.add_argument("--zarr",   required=True, help="Path to activations.zarr directory")
    parser.add_argument("--out",    required=True, help="Output generation.jsonl path")
    parser.add_argument("--split-seed", type=int, default=42, help="Split seed for NQ (default 42)")
    parser.add_argument("--verify", default=None,
                        help="Path to existing generation.jsonl to verify parity against")
    args = parser.parse_args()

    zarr_path = ROOT / args.zarr
    out_path  = ROOT / args.out

    print(f"Loading zarr index from: {zarr_path}")
    prompt_to_response = load_zarr_index(zarr_path)
    print(f"  {len(prompt_to_response):,} entries in zarr index")

    print(f"Loading task data: task={args.task} split={args.split}")
    rows = load_task_prompts(args.task, args.split, args.split_seed)
    print(f"  {len(rows):,} rows from task dataset")

    # Match and write
    matched = missing = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            response = prompt_to_response.get(row["prompt"])
            if response is None:
                missing += 1
                continue
            record = {**row, "generation": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            matched += 1

    print(f"\nWrote {matched:,} records to {out_path}")
    if missing:
        print(f"WARNING: {missing:,} rows had no matching prompt in the zarr index")

    # Parity check
    if args.verify:
        verify_path = Path(args.verify)
        if not verify_path.is_absolute():
            verify_path = ROOT / args.verify
        print(f"\n--- Parity check vs {verify_path} ---")

        ref = {}
        with open(verify_path) as f:
            for line in f:
                rec = json.loads(line)
                ref[rec["prompt"]] = rec["generation"]

        out_records = {}
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                out_records[rec["prompt"]] = rec["generation"]

        n_ref = len(ref)
        n_out = len(out_records)
        common = set(ref) & set(out_records)
        matches = sum(ref[p] == out_records[p] for p in common)
        mismatches = len(common) - matches

        print(f"  Reference rows : {n_ref:,}")
        print(f"  Extracted rows : {n_out:,}")
        print(f"  Common prompts : {len(common):,}")
        print(f"  Generation match : {matches:,} / {len(common):,}")
        if mismatches:
            print(f"  MISMATCHES: {mismatches}")
            # Show first mismatch
            for p in common:
                if ref[p] != out_records[p]:
                    print(f"\n  First mismatch prompt: {p[:80]!r}")
                    print(f"  ref : {ref[p][:120]!r}")
                    print(f"  out : {out_records[p][:120]!r}")
                    break
        else:
            print("  ALL MATCH ✓")


if __name__ == "__main__":
    main()
