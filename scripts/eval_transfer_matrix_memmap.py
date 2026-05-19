"""CLI: evaluate cross-dataset transfer matrix on memmap checkpoints (issue #89).

Port of scripts/eval_transfer_matrix.py (feat/issue-62-transfer-matrix) to the
icr_capture memmap backend produced by issue #79.  The protocol — load source
checkpoint → forward target test → AUROC — is identical; only the parser and
run-dir layout differ.

Zero new training, zero new inference.  CPU-only (each cell is one forward pass
over the target test split; see spec §Compute).

Usage:
  # Smoketest — one diagonal cell and one off-diagonal cell, llama, seed 0:
  python scripts/eval_transfer_matrix_memmap.py \\
      --source-datasets hotpotqa --target-datasets hotpotqa mmlu \\
      --methods contrastive_logprob_recon saplma llmsknow_probe \\
      --model-slugs llama --seeds 0

  # Full run (both models, resume-safe):
  python scripts/eval_transfer_matrix_memmap.py --model-slugs llama --resume
  python scripts/eval_transfer_matrix_memmap.py --model-slugs qwen3 --resume

  # Single-cell mode (used by worker_89.sh / cell worker dispatch):
  python scripts/eval_transfer_matrix_memmap.py --cell-json <path/to/cell.json>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from activation_research.transfer_eval_memmap import (
    _resolve_probe_layer,
    discover_runs,
    evaluate_transfer_cell,
)

DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
METHODS = ["contrastive_logprob_recon", "saplma", "llmsknow_probe"]
MODEL_SLUGS = ["llama", "qwen3"]


def parse_layer_range(spec: str) -> list:
    """Parse '14-29' → [14..29] or '22,26' → [22, 26]."""
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",")]


def _dataset_cfg_name(dataset: str, model_slug: str) -> str:
    """Return the configs/datasets/<name>.json stem for a (dataset, slug) pair."""
    if model_slug == "llama":
        return f"{dataset}_memmap"
    return f"{dataset}_qwen3_memmap"


def _slug_from_experiment(experiment_name: str) -> str:
    """Infer model slug from the experiment directory name."""
    if "_qwen3_memmap" in experiment_name:
        return "qwen3"
    return "llama"


def aggregate_results(output_dir: str) -> None:
    """Read all per-cell JSON files and write the three aggregate CSVs.

    Output:
      transfer_matrix.csv       — long-form: one row per cell
      transfer_matrix_mean.csv  — mean AUROC per (source, target, method, model_slug)
      transfer_matrix_ci.csv    — same plus 95% CI columns (mean ± 1.96·std/√n)
    """
    records = []
    for slug_dir in Path(output_dir).iterdir():
        if not slug_dir.is_dir():
            continue
        for jf in sorted(slug_dir.glob("*.json")):
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
        "auroc", "mahalanobis_auroc", "knn_auroc",
        "n_src_train", "n_tgt_test", "status",
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

    # Summary scalars: off-diagonal mean ± CI, worst-pair, diagonal mean.
    summary = {}
    ok["is_diagonal"] = ok["source_dataset"] == ok["target_dataset"]
    for (slug, meth), grp in ok.groupby(["model_slug", "method"]):
        off = grp[~grp["is_diagonal"]]["auroc"].dropna()
        diag = grp[grp["is_diagonal"]]["auroc"].dropna()
        key = f"{slug}__{meth}"
        summary[key] = {
            "off_diag_mean": float(off.mean()) if len(off) else float("nan"),
            "off_diag_std": float(off.std()) if len(off) else float("nan"),
            "off_diag_n": int(len(off)),
            "off_diag_ci95_lo": (
                float(off.mean() - 1.96 * off.std() / np.sqrt(len(off)))
                if len(off) > 1 else float("nan")
            ),
            "off_diag_ci95_hi": (
                float(off.mean() + 1.96 * off.std() / np.sqrt(len(off)))
                if len(off) > 1 else float("nan")
            ),
            "worst_pair_auroc": float(off.min()) if len(off) else float("nan"),
            "diag_mean": float(diag.mean()) if len(diag) else float("nan"),
        }

    summary_path = os.path.join(output_dir, "transfer_matrix_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] Wrote {summary_path}")


def run_cell_json(cell_json_path: str, output_dir: str | None = None) -> int:
    """Evaluate a single transfer matrix cell from a dispatch cell JSON.

    Cell JSON shape (produced by generate_manifest_89.py):
      {
        "cell_id":            str   — unique cell identifier
        "source_dataset":     str   — bare dataset name, e.g. "hotpotqa"
        "target_dataset":     str   — bare dataset name, e.g. "mmlu"
        "method":             str
        "model_slug":         str   — "llama" or "qwen3"
        "seed":               int
        "source_run_dir":     str   — project-relative path to seed_* run dir
        "source_dataset_cfg": str   — project-relative path to configs/datasets/*.json
        "target_dataset_cfg": str   — project-relative path to configs/datasets/*.json
        "output_check":       str   — project-relative path; written on success
        "relevant_layers":    list[int]
        "probe_layer":        int
      }

    Returns 0 on success (ok or single_class), 1 on error.
    """
    project_root = Path(__file__).parent.parent

    with open(cell_json_path) as f:
        cell = json.load(f)

    cell_id = cell["cell_id"]
    output_check = project_root / cell["output_check"]

    if output_check.exists():
        print(f"[{cell_id}] output exists — skipping")
        return 0

    src_cfg_path = project_root / cell["source_dataset_cfg"]
    tgt_cfg_path = project_root / cell["target_dataset_cfg"]
    if not src_cfg_path.exists():
        print(f"[{cell_id}] source dataset config not found: {src_cfg_path}", file=sys.stderr)
        return 1
    if not tgt_cfg_path.exists():
        print(f"[{cell_id}] target dataset config not found: {tgt_cfg_path}", file=sys.stderr)
        return 1

    with open(src_cfg_path) as f:
        src_dataset_cfg = json.load(f)
    with open(tgt_cfg_path) as f:
        tgt_dataset_cfg = json.load(f)

    source_run_dir = str(project_root / cell["source_run_dir"])

    try:
        result = evaluate_transfer_cell(
            source_run_dir=source_run_dir,
            source_dataset_cfg=src_dataset_cfg,
            target_dataset_cfg=tgt_dataset_cfg,
            method=cell["method"],
            relevant_layers=cell["relevant_layers"],
            probe_layer=cell["probe_layer"],
            device="cpu",
            training_seed=cell["seed"],
        )
    except Exception as exc:
        print(f"[{cell_id}] error: {exc}", file=sys.stderr)
        return 1

    result.update({
        "source_dataset": cell["source_dataset"],
        "target_dataset": cell["target_dataset"],
        "method": cell["method"],
        "seed": cell["seed"],
        "model_slug": cell["model_slug"],
        "cell_id": cell_id,
    })

    out_path = output_check if output_dir is None else (
        Path(output_dir) / cell["model_slug"] / f"{cell_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    auroc = result.get("auroc")
    status = result.get("status", "?")
    if isinstance(auroc, float) and not np.isnan(auroc):
        print(f"[{cell_id}] status={status} auroc={auroc:.4f}")
    else:
        print(f"[{cell_id}] status={status} auroc={auroc}")

    return 0 if result.get("status") in ("ok", "single_class") else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-dataset transfer matrix on memmap checkpoints (issue #89)"
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Root of existing training runs (default: runs)",
    )
    parser.add_argument(
        "--configs-dir", default="configs",
        help="Root of configs/ directory (default: configs)",
    )
    parser.add_argument(
        "--output-dir", default="runs/transfer_matrix_memmap",
        help="Output directory (default: runs/transfer_matrix_memmap)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=METHODS,
        choices=METHODS,
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--model-slugs", nargs="+", default=MODEL_SLUGS,
        choices=["llama", "qwen3"],
        help="Model families to include",
    )
    parser.add_argument(
        "--source-datasets", nargs="+", default=DATASETS,
        help="Source dataset names (bare, e.g. hotpotqa)",
    )
    parser.add_argument(
        "--target-datasets", nargs="+", default=DATASETS,
        help="Target dataset names (bare, e.g. hotpotqa)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds to include (default: all discovered)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--relevant-layers", default="14-29",
        help="Layer range for contrastive model, e.g. '14-29' or '22,26'",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells where the per-cell JSON already exists",
    )
    parser.add_argument(
        "--cell-json", default=None, metavar="FILE",
        help="Single-cell mode: evaluate one cell from a dispatch cell JSON "
             "(produced by scripts/dispatch/generate_manifest_89.py). "
             "All other args are ignored when this is set.",
    )
    args = parser.parse_args()

    if args.cell_json is not None:
        raise SystemExit(run_cell_json(args.cell_json, output_dir=args.output_dir))

    relevant_layers = parse_layer_range(args.relevant_layers)

    # Load target dataset configs up front so we can look them up cheaply per cell.
    target_cfgs: dict = {}
    for dataset in args.target_datasets:
        for slug in args.model_slugs:
            cfg_name = _dataset_cfg_name(dataset, slug)
            cfg_path = os.path.join(args.configs_dir, "datasets", f"{cfg_name}.json")
            if not os.path.exists(cfg_path):
                print(f"[warn] Dataset config not found: {cfg_path} — skipping {dataset}/{slug}")
                continue
            with open(cfg_path) as f:
                target_cfgs[(dataset, slug)] = json.load(f)

    # Discover completed runs across all requested methods.
    all_runs: list = []
    for method in args.methods:
        found = discover_runs(args.runs_dir, method)
        all_runs.extend(found)
    print(f"[info] Discovered {len(all_runs)} runs before filtering")

    # Filter: strip <dataset>_memmap suffix to compare against bare dataset names.
    def _bare_dataset(dataset_field: str) -> str:
        # dataset field is e.g. "hotpotqa_memmap" or "hotpotqa_qwen3_memmap".
        # Strip known suffixes to get the bare name for matching.
        for suffix in ("_qwen3_memmap", "_memmap"):
            if dataset_field.endswith(suffix):
                return dataset_field[: -len(suffix)]
        return dataset_field

    all_runs = [r for r in all_runs if _bare_dataset(r["dataset"]) in args.source_datasets]
    all_runs = [
        r for r in all_runs
        if _slug_from_experiment(r["experiment_name"]) in args.model_slugs
    ]
    if args.seeds is not None:
        all_runs = [r for r in all_runs if r["seed"] in args.seeds]
    print(
        f"[info] {len(all_runs)} runs after filtering "
        f"(sources={args.source_datasets}, slugs={args.model_slugs})"
    )

    total = 0
    skipped = 0
    errors = 0

    for run in all_runs:
        model_slug = _slug_from_experiment(run["experiment_name"])
        bare_src = _bare_dataset(run["dataset"])

        # Load source dataset config (for input_dim, icr_capture.train_dir, outlier_class).
        src_cfg_name = _dataset_cfg_name(bare_src, model_slug)
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

            cell_id = (
                f"{bare_src}__{target_dataset}"
                f"__{run['method']}"
                f"__{run['seed']}"
            )
            output_path = os.path.join(args.output_dir, model_slug, f"{cell_id}.json")

            if args.resume and os.path.exists(output_path):
                skipped += 1
                continue

            total += 1
            try:
                result = evaluate_transfer_cell(
                    source_run_dir=run["run_dir"],
                    source_dataset_cfg=src_dataset_cfg,
                    target_dataset_cfg=target_cfgs[key],
                    method=run["method"],
                    relevant_layers=relevant_layers,
                    probe_layer=probe_layer,
                    device=args.device,
                    training_seed=run["seed"],
                )
            except Exception as exc:
                result = {"status": f"error: {exc}"}
                errors += 1

            result.update({
                "source_dataset": bare_src,
                "target_dataset": target_dataset,
                "method": run["method"],
                "seed": run["seed"],
                "model_slug": model_slug,
                "experiment_name": run["experiment_name"],
            })

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            auroc = result.get("auroc")
            status = result.get("status", "?")
            if isinstance(auroc, float) and not np.isnan(auroc):
                print(f"[{cell_id}] status={status} auroc={auroc:.4f}")
            else:
                print(f"[{cell_id}] status={status} auroc={auroc}")

    print(
        f"\n[done] Evaluated {total} cells, "
        f"skipped {skipped} (--resume), "
        f"errors {errors}"
    )
    aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
