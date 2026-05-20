"""
generate_manifest_89.py — populate pending/ with one JSON cell per
(source_run × target_dataset) for the issue #89 transfer matrix sweep.

Scans the runs directory for completed memmap training checkpoints via
discover_runs(), then emits one cell per
(source_dataset × target_dataset × method × model_slug × seed).

Usage:
    python scripts/dispatch/generate_manifest_89.py \\
        --dispatch-root shared/issue_89_dispatch \\
        [--runs-dir runs] \\
        [--configs-dir configs] \\
        [--output-dir runs/transfer_matrix_memmap] \\
        [--methods contrastive_logprob_recon,saplma,llmsknow_probe] \\
        [--model-slugs llama,qwen3] \\
        [--source-datasets hotpotqa,mmlu,...] \\
        [--target-datasets hotpotqa,mmlu,...] \\
        [--seeds 0,1,2,3,4] \\
        [--relevant-layers 14-29] \\
        [--skip-existing]

Cell shape (consumed by scripts/dispatch/worker_89.sh):

    {
      "cell_id":            "hotpotqa__hotpotqa__contrastive_logprob_recon__llama__0",
      "source_dataset":     "hotpotqa",
      "target_dataset":     "hotpotqa",
      "method":             "contrastive_logprob_recon",
      "model_slug":         "llama",
      "seed":               0,
      "source_run_dir":     "runs/baseline_comparison_hotpotqa_memmap/hotpotqa_memmap/contrastive_logprob_recon/seed_0",
      "source_dataset_cfg": "configs/datasets/hotpotqa_memmap.json",
      "target_dataset_cfg": "configs/datasets/hotpotqa_memmap.json",
      "output_check":       "runs/transfer_matrix_memmap/llama/hotpotqa__hotpotqa__contrastive_logprob_recon__0.json",
      "relevant_layers":    [14, 15, ..., 29],
      "probe_layer":        22
    }

cell_id ordering is source → target → method → seed, so the diagonal
(in-dist sanity check) cells all land early in the sorted claim order.

Skip rules:
  - Cells already in pending/claimed/done/failed are never re-queued.
  - --skip-existing additionally skips cells whose output_check exists at
    manifest time (default OFF — emit all cells and let the worker decide).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_research.transfer_eval_memmap import (  # noqa: E402
    _resolve_probe_layer,
    discover_runs,
)
from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

_DEFAULT_METHODS = ["contrastive_logprob_recon", "saplma", "llmsknow_probe"]
_DEFAULT_DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
_DEFAULT_SLUGS = ["llama", "qwen3"]
_DEFAULT_LAYERS = list(range(14, 30))  # 14-29 inclusive


def _parse_layer_range(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",")]


def _slug_from_experiment(experiment_name: str) -> str:
    if "_qwen3_memmap" in experiment_name:
        return "qwen3"
    return "llama"


def _bare_dataset(dataset_field: str) -> str:
    for suffix in ("_qwen3_memmap", "_memmap"):
        if dataset_field.endswith(suffix):
            return dataset_field[: -len(suffix)]
    return dataset_field


def _dataset_cfg_name(dataset: str, model_slug: str) -> str:
    if model_slug == "llama":
        return f"{dataset}_memmap"
    return f"{dataset}_qwen3_memmap"


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    fname = cell_id + ".json"
    for sub in ("pending", "done", "failed"):
        if (dispatch_root / sub / fname).exists():
            return True
    claimed = dispatch_root / "claimed"
    if claimed.exists():
        for wd in claimed.iterdir():
            if wd.is_dir() and (wd / fname).exists():
                return True
    return False


def generate_manifest(
    dispatch_root: Path,
    runs_dir: Path,
    configs_dir: Path,
    output_dir: Path,
    methods: list[str],
    model_slugs: list[str],
    source_datasets: list[str],
    target_datasets: list[str],
    seed_filter: list[int] | None,
    relevant_layers: list[int],
    skip_existing: bool,
) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0

    # Discover all completed source runs.
    all_runs: list[dict] = []
    for method in methods:
        found = discover_runs(str(runs_dir), method)
        all_runs.extend(found)

    # Apply filters.
    all_runs = [r for r in all_runs if _bare_dataset(r["dataset"]) in source_datasets]
    all_runs = [r for r in all_runs if _slug_from_experiment(r["experiment_name"]) in model_slugs]
    if seed_filter is not None:
        all_runs = [r for r in all_runs if r["seed"] in seed_filter]

    print(f"Discovered {len(all_runs)} source runs after filtering.")

    if not all_runs:
        print("No completed runs found — nothing to queue.", file=sys.stderr)
        return 0

    # Pre-load target dataset configs for quick lookup.
    target_cfgs: dict[tuple[str, str], Path] = {}
    for tgt_ds in target_datasets:
        for slug in model_slugs:
            cfg_name = _dataset_cfg_name(tgt_ds, slug)
            cfg_path = configs_dir / "datasets" / f"{cfg_name}.json"
            if cfg_path.exists():
                target_cfgs[(tgt_ds, slug)] = cfg_path
            else:
                print(f"  [warn] Target config not found: {cfg_path} — skipping {tgt_ds}/{slug}")

    for run in all_runs:
        model_slug = _slug_from_experiment(run["experiment_name"])
        bare_src = _bare_dataset(run["dataset"])

        src_cfg_name = _dataset_cfg_name(bare_src, model_slug)
        src_cfg_path = configs_dir / "datasets" / f"{src_cfg_name}.json"
        if not src_cfg_path.exists():
            print(f"  [warn] Source config not found: {src_cfg_path} — skipping")
            continue

        probe_layer = _resolve_probe_layer(run["run_dir"], run["config"])
        # Make run_dir project-relative for portability.
        try:
            rel_run_dir = str(Path(run["run_dir"]).relative_to(_PROJECT_ROOT))
        except ValueError:
            rel_run_dir = run["run_dir"]

        for tgt_ds in target_datasets:
            key = (tgt_ds, model_slug)
            if key not in target_cfgs:
                continue

            tgt_cfg_path = target_cfgs[key]
            cell_id = f"{bare_src}__{tgt_ds}__{run['method']}__{model_slug}__{run['seed']}"

            if _dispatch_has_cell(dispatch_root, cell_id):
                continue

            output_check = str(
                (output_dir / model_slug / f"{bare_src}__{tgt_ds}__{run['method']}__{run['seed']}.json")
                .relative_to(_PROJECT_ROOT)
            )

            if skip_existing and (_PROJECT_ROOT / output_check).exists():
                continue

            cell = {
                "cell_id": cell_id,
                "task_type": "transfer_eval",
                "source_dataset": bare_src,
                "target_dataset": tgt_ds,
                "method": run["method"],
                "model_slug": model_slug,
                "seed": int(run["seed"]),
                "source_run_dir": rel_run_dir,
                "source_dataset_cfg": str(src_cfg_path.relative_to(_PROJECT_ROOT)),
                "target_dataset_cfg": str(tgt_cfg_path.relative_to(_PROJECT_ROOT)),
                "output_check": output_check,
                "relevant_layers": relevant_layers,
                "probe_layer": probe_layer,
            }
            cell_path = dispatch_root / "pending" / f"{cell_id}.json"
            cell_path.write_text(json.dumps(cell, indent=2), encoding="utf-8")
            written += 1
            print(f"  queued: {cell_id}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate dispatch pending/ queue for issue #89 transfer matrix sweep."
    )
    parser.add_argument("--dispatch-root", required=True,
                        help="Path to dispatch queue root (e.g. shared/issue_89_dispatch).")
    parser.add_argument("--runs-dir", default="runs",
                        help="Root of training runs (default: runs).")
    parser.add_argument("--configs-dir", default="configs",
                        help="Root of configs/ dir (default: configs).")
    parser.add_argument("--output-dir", default="runs/transfer_matrix_memmap",
                        help="Output dir for cell result JSONs (default: runs/transfer_matrix_memmap).")
    parser.add_argument("--methods", default=",".join(_DEFAULT_METHODS),
                        help=f"Comma-separated methods (default: {','.join(_DEFAULT_METHODS)}).")
    parser.add_argument("--model-slugs", default=",".join(_DEFAULT_SLUGS),
                        help=f"Comma-separated model slugs (default: {','.join(_DEFAULT_SLUGS)}).")
    parser.add_argument("--source-datasets", default=",".join(_DEFAULT_DATASETS),
                        help=f"Comma-separated source datasets (default: all 6).")
    parser.add_argument("--target-datasets", default=",".join(_DEFAULT_DATASETS),
                        help=f"Comma-separated target datasets (default: all 6).")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed filter (default: all discovered).")
    parser.add_argument("--relevant-layers", default="14-29",
                        help="Layer range spec for contrastive model (default: 14-29).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cells whose output_check exists. Default OFF.")
    args = parser.parse_args()

    total = generate_manifest(
        dispatch_root=_PROJECT_ROOT / args.dispatch_root,
        runs_dir=_PROJECT_ROOT / args.runs_dir,
        configs_dir=_PROJECT_ROOT / args.configs_dir,
        output_dir=_PROJECT_ROOT / args.output_dir,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        model_slugs=[s.strip() for s in args.model_slugs.split(",") if s.strip()],
        source_datasets=[d.strip() for d in args.source_datasets.split(",") if d.strip()],
        target_datasets=[d.strip() for d in args.target_datasets.split(",") if d.strip()],
        seed_filter=(
            [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            if args.seeds else None
        ),
        relevant_layers=_parse_layer_range(args.relevant_layers),
        skip_existing=args.skip_existing,
    )
    print(f"Done — {total} cells queued in {args.dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
