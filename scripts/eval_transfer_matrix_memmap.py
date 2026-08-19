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

  # chatv1 suite (issue #151/#156 chat-template re-capture; see
  # activation_research/transfer_eval_chatv1.py). Full 5x5 matrix x 3 method
  # families x both models, resume-safe, writes to output/transfer_matrix_chatv1/:
  python scripts/eval_transfer_matrix_memmap.py --suite chatv1 --resume

  # chatv1 smoketest — one method, one model, two datasets:
  python scripts/eval_transfer_matrix_memmap.py --suite chatv1 \\
      --methods tokenwise_contrastive_first_anchored \\
      --model-slugs llama --source-datasets hotpotqa --target-datasets hotpotqa nq
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

from activation_research.transfer_eval_chatv1 import (
    CHATV1_DATASETS,
    CHATV1_METHODS,
    build_source_scorer_chatv1,
    discover_runs_chatv1,
    score_on_target_chatv1,
)
from activation_research.transfer_eval_memmap import (
    _resolve_probe_layer,
    build_source_scorer,
    discover_runs,
    evaluate_transfer_cell,
    score_on_target,
)

DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
METHODS = ["contrastive_logprob_recon", "saplma", "llmsknow_probe", "act_vit"]
MODEL_SLUGS = ["llama", "qwen3"]

# --- chatv1 suite (issue #151/#156 chat-template re-capture) --------------
CHATV1_METHOD_LIST = sorted(CHATV1_METHODS)


def _chatv1_dataset_cfg_name(dataset: str, model_slug: str) -> str:
    """Return the configs/datasets/<name>.json stem for a chatv1 (dataset, slug) pair."""
    if model_slug == "llama":
        return f"{dataset}_chat_memmap"
    return f"{dataset}_qwen3_chat_memmap"


def run_chatv1_suite(
    runs_dir: str,
    configs_dir: str,
    output_dir: str,
    source_datasets: list,
    target_datasets: list,
    model_slugs: list,
    methods: list,
    resume: bool,
    device: str = "cpu",
) -> None:
    """Evaluate the full chatv1 transfer matrix: datasets x methods x models.

    The diagonal (source == target) is INTENTIONALLY included as a sanity
    check — it should reproduce the in-domain t0_cosine_knn_auroc / auroc
    number (module docstring of activation_research/transfer_eval_chatv1.py
    explains why the transfer-time scoring path matches the in-domain path).

    Transfer is within-model only: llama (32 captured layers) and qwen3 (36)
    checkpoints are architecture-incompatible, so a run is only scored
    against target dataset configs for its own model_slug.

    Writes one JSON per cell to
      {output_dir}/{model_slug}/{method}/{source}__{target}.json
    plus aggregate transfer_matrix_chatv1.csv / .json in {output_dir}.
    --resume skips any cell whose JSON already exists. Missing checkpoints
    (run still training) are SKIPPED with a logged warning, not an error —
    rerun later to fill them in.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load target dataset configs up front (dataset, model_slug) -> cfg dict.
    target_cfgs: dict = {}
    for dataset in target_datasets:
        for slug in model_slugs:
            cfg_name = _chatv1_dataset_cfg_name(dataset, slug)
            cfg_path = os.path.join(configs_dir, "datasets", f"{cfg_name}.json")
            if not os.path.exists(cfg_path):
                print(f"[warn] chatv1 dataset config not found: {cfg_path} — skipping {dataset}/{slug}")
                continue
            with open(cfg_path) as f:
                target_cfgs[(dataset, slug)] = json.load(f)

    all_runs = discover_runs_chatv1(runs_dir)
    all_runs = [r for r in all_runs if r["dataset"] in source_datasets]
    all_runs = [r for r in all_runs if r["model_slug"] in model_slugs]
    all_runs = [r for r in all_runs if r["method"] in methods]
    print(f"[info] chatv1: discovered {len(all_runs)} matching runs "
          f"(sources={source_datasets}, slugs={model_slugs}, methods={methods})")

    total = 0
    skipped_resume = 0
    skipped_not_ready = 0
    errors = 0
    records: list = []

    for run in all_runs:
        model_slug = run["model_slug"]
        source_dataset = run["dataset"]
        method = run["method"]
        seed = run["seed"]

        if not run["ready"]:
            print(
                f"[warn] {run['experiment_name']}/{method}/seed_{seed}: "
                "no artifacts/final_weights.pt yet (still training?) — skipping"
            )
            skipped_not_ready += 1
            continue

        src_cfg_name = _chatv1_dataset_cfg_name(source_dataset, model_slug)
        src_cfg_path = os.path.join(configs_dir, "datasets", f"{src_cfg_name}.json")
        if not os.path.exists(src_cfg_path):
            print(f"[warn] source config not found: {src_cfg_path} — skipping run")
            continue
        with open(src_cfg_path) as f:
            src_dataset_cfg = json.load(f)

        scorer = None
        for target_dataset in target_datasets:
            key = (target_dataset, model_slug)
            if key not in target_cfgs:
                continue

            cell_dir = os.path.join(output_dir, model_slug, method)
            cell_path = os.path.join(cell_dir, f"{source_dataset}__{target_dataset}.json")

            if resume and os.path.exists(cell_path):
                with open(cell_path) as f:
                    records.append(json.load(f))
                skipped_resume += 1
                continue

            if scorer is None:
                scorer = build_source_scorer_chatv1(
                    method=method,
                    source_run_dir=run["run_dir"],
                    source_dataset_cfg=src_dataset_cfg,
                    training_seed=seed,
                    device=device,
                )
                if scorer.get("status") in ("missing_artifact", "missing_checkpoint"):
                    print(
                        f"[warn] {run['experiment_name']}/{method}/seed_{seed}: "
                        f"{scorer['status']} — skipping all targets for this run"
                    )

            total += 1
            try:
                result = score_on_target_chatv1(scorer, target_cfgs[key])
            except Exception as exc:
                result = {"status": f"error: {exc}", "auroc": None, "n_test": None, "n_src_train": None}
                errors += 1
                print(f"[{source_dataset}->{target_dataset}/{method}/{model_slug}] error: {exc}", file=sys.stderr)

            record = {
                "model": model_slug,
                "method": method,
                "source": source_dataset,
                "target": target_dataset,
                "seed": seed,
                "auroc": result.get("auroc"),
                "n_test": result.get("n_test"),
                "n_src_train": result.get("n_src_train"),
                "status": result.get("status"),
            }
            records.append(record)

            os.makedirs(cell_dir, exist_ok=True)
            with open(cell_path, "w") as f:
                json.dump(record, f, indent=2)

            auroc = record["auroc"]
            auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) and auroc == auroc else str(auroc)
            print(
                f"[{source_dataset}->{target_dataset}/{method}/{model_slug}] "
                f"status={record['status']} auroc={auroc_str}"
            )

    print(
        f"\n[done] chatv1: evaluated {total} cells, "
        f"resumed {skipped_resume}, not-ready {skipped_not_ready}, errors {errors}"
    )

    if records:
        df = pd.DataFrame(records)
        cols = ["model", "method", "source", "target", "seed", "auroc", "n_test", "n_src_train", "status"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
        csv_path = os.path.join(output_dir, "transfer_matrix_chatv1.csv")
        json_path = os.path.join(output_dir, "transfer_matrix_chatv1.json")
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[aggregate] Wrote {csv_path} ({len(df)} rows)")
        print(f"[aggregate] Wrote {json_path} ({len(records)} rows)")
    else:
        print("[aggregate] No chatv1 cells evaluated — nothing to aggregate.")


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


def run_batch_cell_json(cell_json_path: str, output_dir: str | None = None) -> int:
    """Run a batch cell: all (method × target_dataset) for one (source, model, seed).

    Batch cell JSON shape (produced by generate_manifest_89.py):
      {
        "cell_id":             str   — "hotpotqa__llama__0"
        "task_type":           "transfer_eval"
        "source_dataset":      str
        "model_slug":          str
        "seed":                int
        "source_run_dirs":     {method: project-relative run_dir, ...}
        "source_dataset_cfg":  str   — project-relative path
        "target_datasets":     list[str]
        "target_dataset_cfgs": {dataset: project-relative path, ...}
        "probe_layers":        {method: int, ...}
        "relevant_layers":     list[int]   — informational only; methods read from config
        "output_check":        str   — project-relative sentinel path (.done file)
      }

    Output JSONs are written as:
      <output_dir>/<model_slug>/<source>__<target>__<method>__<seed>.json

    Sentinel is written only when all scorers build and all targets score successfully.
    Individual (method, target) pairs whose output JSON already exists are skipped.
    Returns 0 on full success, 1 if any scorer or scoring step failed.
    """
    project_root = Path(__file__).parent.parent

    with open(cell_json_path) as f:
        cell = json.load(f)

    cell_id = cell["cell_id"]
    sentinel_path = project_root / cell["output_check"]

    if sentinel_path.exists():
        print(f"[{cell_id}] sentinel exists — skipping")
        return 0

    source_dataset = cell["source_dataset"]
    model_slug = cell["model_slug"]
    seed = cell["seed"]
    source_run_dirs: dict = cell["source_run_dirs"]
    target_datasets: list = cell["target_datasets"]
    target_dataset_cfgs: dict = cell["target_dataset_cfgs"]
    probe_layers: dict = cell.get("probe_layers", {})

    src_cfg_path = project_root / cell["source_dataset_cfg"]
    if not src_cfg_path.exists():
        print(f"[{cell_id}] source dataset config not found: {src_cfg_path}", file=sys.stderr)
        return 1
    with open(src_cfg_path) as f:
        src_dataset_cfg = json.load(f)

    if output_dir is not None:
        base_out = Path(output_dir) / model_slug
    else:
        base_out = project_root / "runs" / "transfer_matrix_memmap" / model_slug
    base_out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for method, rel_run_dir in source_run_dirs.items():
        source_run_dir = str(project_root / rel_run_dir)
        probe_layer = probe_layers.get(method, 22)

        try:
            scorer = build_source_scorer(
                method=method,
                source_run_dir=source_run_dir,
                source_dataset_cfg=src_dataset_cfg,
                probe_layer=probe_layer,
                training_seed=seed,
            )
        except Exception as exc:
            print(f"[{cell_id}/{method}] build_source_scorer error: {exc}", file=sys.stderr)
            n_fail += 1
            continue

        if scorer.get("status") in ("missing_artifact", "missing_checkpoint"):
            print(f"[{cell_id}/{method}] {scorer['status']} — skipping method")
            n_fail += 1
            continue

        for tgt_ds in target_datasets:
            out_stem = f"{source_dataset}__{tgt_ds}__{method}__{seed}"
            out_path = base_out / f"{out_stem}.json"

            if out_path.exists():
                n_skip += 1
                continue

            tgt_cfg_path = project_root / target_dataset_cfgs[tgt_ds]
            if not tgt_cfg_path.exists():
                print(f"[{cell_id}] target config not found: {tgt_cfg_path}", file=sys.stderr)
                n_fail += 1
                continue
            with open(tgt_cfg_path) as f:
                tgt_dataset_cfg = json.load(f)

            try:
                result = score_on_target(scorer, tgt_dataset_cfg)
            except Exception as exc:
                print(f"[{cell_id}] {method}->{tgt_ds} error: {exc}", file=sys.stderr)
                n_fail += 1
                continue

            result.update({
                "source_dataset": source_dataset,
                "target_dataset": tgt_ds,
                "method": method,
                "seed": seed,
                "model_slug": model_slug,
                "cell_id": f"{source_dataset}__{tgt_ds}__{method}__{model_slug}__{seed}",
            })

            sample_scores = result.pop("_scores", None)
            sample_labels = result.pop("_labels", None)
            if sample_scores is not None and sample_labels is not None:
                preds_path = out_path.with_suffix(".predictions.csv")
                with open(preds_path, "w") as pf:
                    pf.write("example_id,score_halu,label_halu\n")
                    for i, (s, l) in enumerate(zip(sample_scores, sample_labels)):
                        pf.write(f"{i},{float(s):.10f},{int(l)}\n")

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            auroc = result.get("auroc")
            status = result.get("status", "?")
            auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) and auroc == auroc else str(auroc)
            print(f"[{cell_id}] {method}->{tgt_ds}: status={status} auroc={auroc_str}")
            n_ok += 1

    print(f"[{cell_id}] done: {n_ok} written, {n_skip} skipped, {n_fail} failed")

    if n_fail > 0:
        return 1

    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text("")
    return 0


def _run_single_cell_json(cell_json_path: str, output_dir: str | None = None) -> int:
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


def run_cell_json(cell_json_path: str, output_dir: str | None = None) -> int:
    """Dispatch to batch or single-cell runner based on cell format.

    Batch cells have "source_run_dirs" (dict of method→run_dir).
    Single cells have "source_run_dir" (a single string path).
    """
    with open(cell_json_path) as f:
        cell = json.load(f)
    if "source_run_dirs" in cell:
        return run_batch_cell_json(cell_json_path, output_dir=output_dir)
    return _run_single_cell_json(cell_json_path, output_dir=output_dir)


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
        "--output-dir", default=None,
        help="Output directory (default: runs/transfer_matrix_memmap, or "
             "output/transfer_matrix_chatv1 when --suite chatv1)",
    )
    parser.add_argument(
        "--suite", default="issue89", choices=["issue89", "chatv1"],
        help="'issue89' (default) evaluates the legacy contrastive/saplma/"
             "llmsknow_probe/act_vit matrix over baseline_comparison_*_memmap "
             "runs. 'chatv1' evaluates the tokenwise_contrastive_first_anchored"
             "/tokenwise_arch_v1_input_norm_only/token_zero_mlp_probe matrix "
             "(and their _qwen3 variants) over chatv1_* / chatv1_qwen3_* runs.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to evaluate (default: all methods for the selected --suite)",
    )
    parser.add_argument(
        "--model-slugs", nargs="+", default=MODEL_SLUGS,
        choices=["llama", "qwen3"],
        help="Model families to include",
    )
    parser.add_argument(
        "--source-datasets", nargs="+", default=None,
        help="Source dataset names (bare, e.g. hotpotqa); "
             "default: all datasets for the selected --suite",
    )
    parser.add_argument(
        "--target-datasets", nargs="+", default=None,
        help="Target dataset names (bare, e.g. hotpotqa); "
             "default: all datasets for the selected --suite",
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

    if args.suite == "chatv1":
        output_dir = args.output_dir or "output/transfer_matrix_chatv1"
        methods = args.methods or CHATV1_METHOD_LIST
        unknown = sorted(set(methods) - CHATV1_METHODS)
        if unknown:
            parser.error(
                f"--suite chatv1 does not support methods {unknown}; "
                f"choose from {CHATV1_METHOD_LIST}"
            )
        source_datasets = args.source_datasets or CHATV1_DATASETS
        target_datasets = args.target_datasets or CHATV1_DATASETS
        run_chatv1_suite(
            runs_dir=args.runs_dir,
            configs_dir=args.configs_dir,
            output_dir=output_dir,
            source_datasets=source_datasets,
            target_datasets=target_datasets,
            model_slugs=args.model_slugs,
            methods=methods,
            resume=args.resume,
            device=args.device,
        )
        return

    output_dir = args.output_dir or "runs/transfer_matrix_memmap"
    methods = args.methods or METHODS
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        parser.error(
            f"--suite issue89 does not support methods {unknown}; choose from {METHODS}"
        )
    source_datasets = args.source_datasets or DATASETS
    target_datasets = args.target_datasets or DATASETS

    relevant_layers = parse_layer_range(args.relevant_layers)

    # Load target dataset configs up front so we can look them up cheaply per cell.
    target_cfgs: dict = {}
    for dataset in target_datasets:
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
    for method in methods:
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

    all_runs = [r for r in all_runs if _bare_dataset(r["dataset"]) in source_datasets]
    all_runs = [
        r for r in all_runs
        if _slug_from_experiment(r["experiment_name"]) in args.model_slugs
    ]
    if args.seeds is not None:
        all_runs = [r for r in all_runs if r["seed"] in args.seeds]
    print(
        f"[info] {len(all_runs)} runs after filtering "
        f"(sources={source_datasets}, slugs={args.model_slugs})"
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

        for target_dataset in target_datasets:
            key = (target_dataset, model_slug)
            if key not in target_cfgs:
                continue

            cell_id = (
                f"{bare_src}__{target_dataset}"
                f"__{run['method']}"
                f"__{run['seed']}"
            )
            output_path = os.path.join(output_dir, model_slug, f"{cell_id}.json")

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

            # Per-sample (score_halu, label_halu) live under _scores/_labels in the
            # cell dict; pull them out into a sidecar CSV so downstream stats (e.g.
            # AUPR, calibration) can be recomputed without re-running the cell.
            sample_scores = result.pop("_scores", None)
            sample_labels = result.pop("_labels", None)
            if sample_scores is not None and sample_labels is not None:
                preds_path = output_path[:-len(".json")] + ".predictions.csv"
                with open(preds_path, "w") as pf:
                    pf.write("example_id,score_halu,label_halu\n")
                    for i, (s, l) in enumerate(zip(sample_scores, sample_labels)):
                        pf.write(f"{i},{float(s):.10f},{int(l)}\n")

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
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
