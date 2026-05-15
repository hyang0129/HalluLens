"""Audit P(true) output completeness and correctness.

Usage:
    # Check expected vs actual row counts for all 12 cells
    python scripts/audit_p_true.py --check-counts

    # Spot-check 20 random rows per (dataset, model), save to reports/
    python scripts/audit_p_true.py --spot-check 20

    # Both
    python scripts/audit_p_true.py --check-counts --spot-check 20
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.p_true.paths import DATASETS, MODELS, ptrue_scores_path
from tasks.sampling_baselines.paths import eval_results_json, generation_jsonl, model_name


# Expected test-split row counts per dataset (informational; actual count from
# generation.jsonl is the authoritative source since SearchQA naming is inverted).
_EXPECTED_COUNTS = {
    "hotpotqa": 7405,
    "nq": 4155,
    "popqa": 2854,
    "sciq": 1000,
    "searchqa": 151140,  # ~151K; varies slightly by model
    "mmlu": 10225,
}


def count_generation_rows(dataset: str, model_id: str) -> int:
    p = generation_jsonl(dataset, model_id, "test")
    if not p.exists():
        return -1
    with open(p) as f:
        return sum(1 for _ in f)


def count_ptrue_rows(dataset: str, model_id: str) -> int:
    p = ptrue_scores_path(dataset, model_id, "test")
    if not p.exists():
        return -1
    with open(p) as f:
        return sum(1 for _ in f)


def check_counts(models, datasets) -> bool:
    print("=== P(true) count audit ===\n")
    header = f"{'Dataset':<14} {'Model':<30} {'Gen rows':>10} {'ptrue rows':>12} {'Status':>8}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for mid in models:
        for ds in datasets:
            gen_n = count_generation_rows(ds, mid)
            pt_n = count_ptrue_rows(ds, mid)
            if gen_n < 0:
                status = "NO GEN"
                all_ok = False
            elif pt_n < 0:
                status = "MISSING"
                all_ok = False
            elif pt_n < gen_n:
                status = f"PARTIAL"
                all_ok = False
            else:
                status = "OK"
            print(
                f"{ds:<14} {model_name(mid):<30} {gen_n:>10,} {pt_n:>12,} {status:>8}"
            )
    print()
    return all_ok


def spot_check(models, datasets, n: int, save_dir: Path) -> None:
    print(f"=== P(true) spot-check ({n} rows per cell) ===\n")
    save_dir.mkdir(parents=True, exist_ok=True)

    for mid in models:
        # Load labels once per model
        label_cache = {}

        for ds in datasets:
            pt_path = ptrue_scores_path(ds, mid, "test")
            if not pt_path.exists():
                print(f"SKIP {ds}/{model_name(mid)}: ptrue.jsonl not found")
                continue

            eval_path = eval_results_json(ds, mid, "test")
            if ds not in label_cache and eval_path.exists():
                with open(eval_path) as f:
                    label_cache[ds] = json.load(f).get("halu_test_res", [])

            gen_path = generation_jsonl(ds, mid, "test")

            # Load ptrue.jsonl
            rows = []
            with open(pt_path) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass

            if not rows:
                print(f"SKIP {ds}/{model_name(mid)}: ptrue.jsonl is empty")
                continue

            sample = random.sample(rows, min(n, len(rows)))
            sample.sort(key=lambda r: r["row_idx"])

            # Load generation rows for context
            gen_rows = {}
            if gen_path.exists():
                with open(gen_path) as f:
                    for i, line in enumerate(f):
                        try:
                            gen_rows[i] = json.loads(line)
                        except Exception:
                            pass

            output_lines = [
                f"=== Spot-check: {ds} / {model_name(mid)} ({len(sample)} rows) ===\n"
            ]

            labels = label_cache.get(ds, [])
            for rec in sample:
                row_idx = rec["row_idx"]
                gen_row = gen_rows.get(row_idx, {})
                question = gen_row.get("question", gen_row.get("prompt", "")[:80])
                answer = gen_row.get("generation", "")[:120]
                halu = labels[row_idx] if row_idx < len(labels) else rec.get("halu_label")
                line = (
                    f"row={row_idx:6d} | halu={halu} | "
                    f"p_true={rec['p_true']:.4f} | p_true_rev={rec['p_true_reversed']:.4f}\n"
                    f"  Q: {question[:100]}\n"
                    f"  A: {answer[:100]}\n"
                )
                output_lines.append(line)

            block = "\n".join(output_lines)
            print(block)

            out_file = save_dir / f"spotcheck_{ds}_{model_name(mid)}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(block)

    print(f"Spot-check reports saved → {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audit P(true) output.")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--check-counts", action="store_true")
    parser.add_argument("--spot-check", type=int, metavar="N",
                        help="Print N random rows per (dataset, model).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="reports/p_true_spotchecks")
    args = parser.parse_args()

    if not args.check_counts and args.spot_check is None:
        parser.error("Specify --check-counts and/or --spot-check N")

    random.seed(args.seed)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    ok = True
    if args.check_counts:
        ok = check_counts(models, datasets)

    if args.spot_check is not None:
        spot_check(models, datasets, args.spot_check, Path(args.save_dir))

    if args.check_counts and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
