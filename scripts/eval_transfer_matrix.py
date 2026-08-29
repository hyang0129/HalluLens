"""CLI: evaluate cross-dataset transfer matrix for hallucination detectors.

Zero new training; zero new inference. CPU-only.

Usage:
  # Smoke test (1 cell, diagonal):
  python scripts/eval_transfer_matrix.py \
    --source-datasets hotpotqa --target-datasets hotpotqa \
    --methods linear_probe --model-slugs llama --seeds 0

  # Full run:
  python scripts/eval_transfer_matrix.py --model-slugs llama --resume
  python scripts/eval_transfer_matrix.py --model-slugs qwen3 --resume
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_research.transfer_eval import discover_runs, evaluate_transfer_cell

DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
METHODS = ["contrastive_logprob_recon", "linear_probe", "saplma"]
MODEL_SLUGS = ["llama", "qwen3"]


def parse_layer_range(spec: str) -> list:
    """Parse '14-29' → [14..29] or '22,26' → [22, 26]."""
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",")]


def _dataset_cfg_name(dataset: str, model_slug: str) -> str:
    return dataset if model_slug == "llama" else f"{dataset}_qwen3"


def _slug_from_experiment(experiment_name: str) -> str:
    if "_qwen3" in experiment_name:
        return "qwen3"
    return "llama"


def _resolve_probe_layer(run_dir: str, config: dict) -> int:
    """Determine probe layer: eval_metrics > config.json > default 26."""
    eval_metrics_path = os.path.join(run_dir, "eval_metrics.json")
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path) as f:
            em = json.load(f)
        if "selected_layer" in em:
            return int(em["selected_layer"])
    method_cfg = config.get("method_cfg", {})
    if "probe_layer" in method_cfg:
        return int(method_cfg["probe_layer"])
    return 26


def aggregate_results(output_dir: str) -> None:
    """Read all per-cell JSON files; write transfer_matrix*.csv."""
    records = []
    for slug_dir in Path(output_dir).iterdir():
        if not slug_dir.is_dir():
            continue
        for jf in slug_dir.glob("*.json"):
            try:
                with open(jf) as f:
                    records.append(json.load(f))
            except Exception:
                pass

    if not records:
        print("[aggregate] No cell JSON files found.")
        return

    df = pd.DataFrame(records)

    cols = [
        "source_dataset", "target_dataset", "method", "model_slug", "seed",
        "auroc", "status", "mahalanobis_auroc", "knn_auroc",
        "n_src_train", "n_tgt_test",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    flat_path = os.path.join(output_dir, "transfer_matrix.csv")
    df.to_csv(flat_path, index=False)
    print(f"[aggregate] Wrote {flat_path} ({len(df)} rows)")

    group_cols = ["source_dataset", "target_dataset", "method", "model_slug"]
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    ok["auroc"] = pd.to_numeric(ok["auroc"], errors="coerce")
    agg = ok.groupby(group_cols)["auroc"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = group_cols + ["auroc_mean", "auroc_std", "n_seeds"]
    agg["auroc_ci95_lo"] = agg["auroc_mean"] - 1.96 * agg["auroc_std"] / np.sqrt(agg["n_seeds"])
    agg["auroc_ci95_hi"] = agg["auroc_mean"] + 1.96 * agg["auroc_std"] / np.sqrt(agg["n_seeds"])

    mean_path = os.path.join(output_dir, "transfer_matrix_mean.csv")
    agg.to_csv(mean_path, index=False)
    print(f"[aggregate] Wrote {mean_path} ({len(agg)} rows)")

    ci_path = os.path.join(output_dir, "transfer_matrix_ci.csv")
    agg.to_csv(ci_path, index=False)
    print(f"[aggregate] Wrote {ci_path} ({len(agg)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-dataset transfer matrix for hallucination detectors"
    )
    parser.add_argument("--runs-dir", default="runs",
                        help="Root of existing training runs (default: runs)")
    parser.add_argument("--configs-dir", default="configs",
                        help="Root of configs/ directory (default: configs)")
    parser.add_argument("--output-dir", default="runs/transfer_matrix",
                        help="Output directory (default: runs/transfer_matrix)")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="Methods to evaluate")
    parser.add_argument("--model-slugs", nargs="+", default=MODEL_SLUGS,
                        choices=["llama", "qwen3"],
                        help="Model families to include")
    parser.add_argument("--source-datasets", nargs="+", default=DATASETS,
                        help="Source dataset names")
    parser.add_argument("--target-datasets", nargs="+", default=DATASETS,
                        help="Target dataset names")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to include (default: all discovered)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--relevant-layers", default="14-29",
                        help="Layer range for contrastive model (e.g. '14-29')")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells where output JSON already exists")
    args = parser.parse_args()

    relevant_layers = parse_layer_range(args.relevant_layers)

    # Load target dataset configs
    target_cfgs = {}
    for dataset in args.target_datasets:
        for slug in args.model_slugs:
            cfg_name = _dataset_cfg_name(dataset, slug)
            cfg_path = os.path.join(args.configs_dir, "datasets", f"{cfg_name}.json")
            if not os.path.exists(cfg_path):
                print(f"[warn] Dataset config not found: {cfg_path} — skipping {dataset}/{slug}")
                continue
            with open(cfg_path) as f:
                target_cfgs[(dataset, slug)] = json.load(f)

    # Discover source runs across all requested methods
    all_runs = []
    for method in args.methods:
        found = discover_runs(args.runs_dir, method)
        all_runs += found
    print(f"[info] Discovered {len(all_runs)} runs before filtering")

    all_runs = [r for r in all_runs if r["dataset"] in args.source_datasets]
    all_runs = [r for r in all_runs if _slug_from_experiment(r["experiment_name"]) in args.model_slugs]
    if args.seeds is not None:
        all_runs = [r for r in all_runs if r["seed"] in args.seeds]
    print(f"[info] {len(all_runs)} runs after filtering (sources={args.source_datasets}, slugs={args.model_slugs})")

    total = 0
    skipped = 0
    errors = 0

    for run in all_runs:
        model_slug = _slug_from_experiment(run["experiment_name"])

        src_cfg_name = _dataset_cfg_name(run["dataset"], model_slug)
        src_cfg_path = os.path.join(args.configs_dir, "datasets", f"{src_cfg_name}.json")
        if not os.path.exists(src_cfg_path):
            print(f"[warn] Source config not found: {src_cfg_path} — skipping run")
            continue
        with open(src_cfg_path) as f:
            src_dataset_cfg = json.load(f)

        probe_layer = _resolve_probe_layer(run["run_dir"], run["config"])

        for target_dataset in args.target_datasets:
            key = (target_dataset, model_slug)
            if key not in target_cfgs:
                continue

            cell_id = f"{run['dataset']}__{target_dataset}__{run['method']}__{run['seed']}"
            output_path = os.path.join(args.output_dir, model_slug, f"{cell_id}.json")

            if args.resume and os.path.exists(output_path):
                skipped += 1
                continue

            total += 1
            try:
                result = evaluate_transfer_cell(
                    source_run_dir=run["run_dir"],
                    source_dataset_cfg=src_dataset_cfg,
                    target_test_cfg=target_cfgs[key],
                    method=run["method"],
                    relevant_layers=relevant_layers,
                    probe_layer=probe_layer,
                    device=args.device,
                )
            except Exception as exc:
                result = {"status": f"error: {exc}"}
                errors += 1

            result.update({
                "source_dataset": run["dataset"],
                "target_dataset": target_dataset,
                "method": run["method"],
                "seed": run["seed"],
                "model_slug": model_slug,
                "experiment_name": run["experiment_name"],
            })

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            auroc = result.get("auroc", "ERR")
            status = result.get("status", "?")
            if isinstance(auroc, float) and not np.isnan(auroc):
                print(f"[{cell_id}] status={status} auroc={auroc:.4f}")
            else:
                print(f"[{cell_id}] status={status} auroc={auroc}")

    print(f"\n[done] Evaluated {total} cells, skipped {skipped} (resume), errors {errors}")
    aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
