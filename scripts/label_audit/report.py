#!/usr/bin/env python3
"""report.py — compare LLM-judge verdicts against substring-match labels.

Joins export_sample.py samples with judge.py verdicts and reports, per dataset
(and per model), the disagreement rate and its two directions:
  - false_hallucinated: substring says WRONG (hallucinated) but judge says CORRECT
                        (good answers mislabeled as hallucinations)
  - false_correct:      substring says RIGHT but judge says INCORRECT (lucky hits)

Outputs report.md, report.csv, and disagreements_<dataset>.md (examples).

Usage (local):
  python scripts/label_audit/report.py \
      --samples-dir output/label_audit/samples \
      --judged-dir output/label_audit/judged \
      --out-dir output/label_audit
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="output/label_audit/samples")
    ap.add_argument("--judged-dir", default="output/label_audit/judged")
    ap.add_argument("--out-dir", default="output/label_audit")
    ap.add_argument("--examples", type=int, default=12, help="disagreement examples per dataset per direction")
    args = ap.parse_args()

    samples_dir, judged_dir, out_dir = Path(args.samples_dir), Path(args.judged_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[str, dict] = {}
    for f in samples_dir.glob("*.jsonl"):
        for r in _load_jsonl(f):
            samples[r["audit_id"]] = r

    # group verdicts by (dataset, model)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in judged_dir.glob("*.judged.jsonl"):
        for r in _load_jsonl(f):
            cells[(r["dataset"], r["model"])].append(r)

    def summarize(rows: list[dict]) -> dict:
        n = both_c = both_w = false_hallu = false_corr = unknown = 0
        for r in rows:
            if r["judge_verdict"] == "UNKNOWN":
                unknown += 1
                continue
            n += 1
            sub_correct = r["substring_label"] == "correct"
            jc = bool(r["judge_correct"])
            if sub_correct and jc:
                both_c += 1
            elif (not sub_correct) and (not jc):
                both_w += 1
            elif (not sub_correct) and jc:
                false_hallu += 1
            else:
                false_corr += 1
        disagree = false_hallu + false_corr
        return {
            "n": n, "unknown": unknown,
            "substring_correct_rate": round((both_c + false_corr) / n, 4) if n else 0.0,
            "judge_correct_rate": round((both_c + false_hallu) / n, 4) if n else 0.0,
            "disagree_rate": round(disagree / n, 4) if n else 0.0,
            "false_hallucinated": false_hallu,
            "false_hallucinated_rate": round(false_hallu / n, 4) if n else 0.0,
            "false_correct": false_corr,
            "false_correct_rate": round(false_corr / n, 4) if n else 0.0,
        }

    # per (dataset, model) and per dataset (pooled)
    per_cell = {k: summarize(v) for k, v in cells.items()}
    per_dataset: dict[str, dict] = {}
    ds_rows: dict[str, list[dict]] = defaultdict(list)
    for (ds, _m), rows in cells.items():
        ds_rows[ds].extend(rows)
    for ds, rows in ds_rows.items():
        per_dataset[ds] = summarize(rows)

    # ---- report.csv ----
    csv_path = out_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["dataset", "model", "n", "unknown", "substring_correct_rate",
                "judge_correct_rate", "disagree_rate", "false_hallucinated",
                "false_hallucinated_rate", "false_correct", "false_correct_rate"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for (ds, m), s in sorted(per_cell.items()):
            w.writerow({"dataset": ds, "model": m.split("/")[-1], **s})
        for ds, s in sorted(per_dataset.items()):
            w.writerow({"dataset": ds, "model": "BOTH", **s})

    # ---- report.md ----
    lines = ["# Label-quality audit: LLM-judge (Sonnet) vs substring-match\n"]
    lines.append("Disagreement = judge and substring-match disagree on correctness. "
                 "`false_hallucinated` = substring marked a correct answer as a hallucination; "
                 "`false_correct` = substring marked a wrong answer as correct.\n")
    lines.append("## Per dataset (both models pooled), sorted by disagreement\n")
    lines.append("| dataset | n | substring_acc | judge_acc | **disagree** | false_hallu | false_correct |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for ds, s in sorted(per_dataset.items(), key=lambda kv: -kv[1]["disagree_rate"]):
        lines.append(f"| {ds} | {s['n']} | {s['substring_correct_rate']:.3f} | {s['judge_correct_rate']:.3f} "
                     f"| **{s['disagree_rate']:.3f}** | {s['false_hallucinated_rate']:.3f} | {s['false_correct_rate']:.3f} |")
    lines.append("\n## Per (dataset, model)\n")
    lines.append("| dataset | model | n | disagree | false_hallu | false_correct | unknown |")
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    for (ds, m), s in sorted(per_cell.items()):
        lines.append(f"| {ds} | {m.split('/')[-1]} | {s['n']} | {s['disagree_rate']:.3f} "
                     f"| {s['false_hallucinated_rate']:.3f} | {s['false_correct_rate']:.3f} | {s['unknown']} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- disagreement examples ----
    for ds, rows in ds_rows.items():
        fh_ex = ["# Disagreements: %s\n" % ds]
        for direction, want in (("false_hallucinated (substring=hallucinated, judge=CORRECT)", "fh"),
                                ("false_correct (substring=correct, judge=INCORRECT)", "fc")):
            fh_ex.append(f"\n## {direction}\n")
            shown = 0
            for r in rows:
                if r["judge_verdict"] == "UNKNOWN":
                    continue
                sub_correct = r["substring_label"] == "correct"
                jc = bool(r["judge_correct"])
                is_fh = (not sub_correct) and jc
                is_fc = sub_correct and (not jc)
                if (want == "fh" and not is_fh) or (want == "fc" and not is_fc):
                    continue
                s = samples.get(r["audit_id"], {})
                gold = s.get("gold", {})
                fh_ex.append(f"- **Q:** {str(s.get('question',''))[:300]}")
                fh_ex.append(f"  **gold:** {json.dumps(gold, ensure_ascii=False)[:200]}")
                fh_ex.append(f"  **model:** {str(s.get('generation',''))[:300]}")
                fh_ex.append(f"  **judge:** {r['judge_verdict']} — {r['judge_reason']}\n")
                shown += 1
                if shown >= args.examples:
                    break
        (out_dir / f"disagreements_{ds}.md").write_text("\n".join(fh_ex) + "\n", encoding="utf-8")

    # console summary
    print("dataset             n   disagree  false_hallu  false_correct")
    for ds, s in sorted(per_dataset.items(), key=lambda kv: -kv[1]["disagree_rate"]):
        print(f"{ds:18s} {s['n']:5d}   {s['disagree_rate']:.3f}      "
              f"{s['false_hallucinated_rate']:.3f}        {s['false_correct_rate']:.3f}")
    print(f"\nwrote {out_dir}/report.md, report.csv, disagreements_*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
