# Label-quality audit (issue #143)

Re-evaluate the substring-match correctness labels with an LLM judge (Sonnet via
`claude -p`) on a 1k sample of each dataset's test set, and quantify the
disagreement. Both `clr` and `act_vit` are scored against these labels, so label
noise is a shared confounder for the clr-vs-act_vit comparison (epic #136).
Generalizes the HotpotQA-only #76 LLM-judge check to all 8 datasets.

## Pipeline

```
export_sample.py   (cluster) → output/label_audit/samples/{dataset}__{model}.jsonl
        │  sample 1k/(dataset,model) from the test capture generation.jsonl
        ▼
judge.py           (local)   → output/label_audit/judged/{dataset}__{model}.judged.jsonl
        │  batch → `claude -p --model sonnet` → CORRECT/INCORRECT + reason
        ▼
report.py          (local)   → output/label_audit/report.md, report.csv, disagreements_*.md
```

`export_sample.py` runs where the captures live (Empire AI); it emits only the
small fields the judge needs (question, gold, choices, generation, current
label), never the activation memmaps. `judge.py` runs locally where `claude` is
authenticated.

## Run

```bash
# 1. on the cluster (data lives there):
python scripts/label_audit/export_sample.py --n 1000 --seed 0
#    then copy output/label_audit/samples/ to a machine with the claude CLI.

# 2. locally (claude CLI authenticated):
python scripts/label_audit/judge.py --model sonnet --batch-size 20 --workers 8
python scripts/label_audit/report.py
```

For a full-capture backfill, inspect coverage and sidecar integrity without
calling Claude:

```bash
python scripts/label_audit/backfill_status.py \
  output/label_audit/backfill_staging/shared/icr_capture
# Add --json for automation or --strict to require complete, valid sidecars.
```

`judge.py` is resumable — re-running skips audit_ids already judged, so an
interrupted pass just continues.

## Scope
- 8 datasets: hotpotqa, mmlu, popqa, natural_questions, sciq, searchqa, triviaqa, simpleqa (test split)
- 2 models: Llama-3.1-8B-Instruct, Qwen3-8B
- 1000 samples per (dataset, model), deterministic (seed 0) → 16k judgments

## Reading the report
- `disagree_rate` — fraction where judge and substring-match disagree on correctness.
- `false_hallucinated` — substring marked a **correct** answer as a hallucination (the main suspected failure, esp. MMLU).
- `false_correct` — substring marked a **wrong** answer as correct (lucky substring hit).

## Caveats
- The Sonnet judge is a strong proxy, not ground truth. Every judge `reason` is
  retained for spot-checking; `disagreements_*.md` lists concrete examples.
- Judge self-consistency: re-run `judge.py` on a subset into a second
  `--out-dir` to measure agreement of the judge with itself.
