"""
generate_manifest_89.py — populate pending/ with one JSON cell per
(source_dataset × model_slug × seed) for the issue #89 transfer matrix sweep.

Each batch cell covers all requested methods and all target datasets, so the
worker loads each source checkpoint once and scores all 6 targets in one pass
instead of 6 separate cells.  Cell count: 6 sources × 2 slugs × 5 seeds = 60
instead of the old 6×6×4×2×5 = 1,440.

Usage:
    python scripts/dispatch/generate_manifest_89.py \\
        --dispatch-root shared/issue_79_dispatch \\
        [--runs-dir runs] \\
        [--configs-dir configs] \\
        [--output-dir runs/transfer_matrix_memmap] \\
        [--methods contrastive_logprob_recon,saplma,llmsknow_probe,act_vit] \\
        [--model-slugs llama,qwen3] \\
        [--source-datasets hotpotqa,mmlu,...] \\
        [--target-datasets hotpotqa,mmlu,...] \\
        [--seeds 0,1,2,3,4] \\
        [--relevant-layers 14-29] \\
        [--skip-existing]

Batch cell shape (consumed by worker_79.sh via eval_transfer_matrix_memmap.py):

    {
      "cell_id":             "hotpotqa__llama__0",
      "task_type":           "transfer_eval",
      "source_dataset":      "hotpotqa",
      "model_slug":          "llama",
      "seed":                0,
      "source_run_dirs":     {"contrastive_logprob_recon": "runs/.../seed_0", ...},
      "source_dataset_cfg":  "configs/datasets/hotpotqa_memmap.json",
      "target_datasets":     ["hotpotqa", "mmlu", ...],
      "target_dataset_cfgs": {"hotpotqa": "configs/...", ...},
      "probe_layers":        {"saplma": 18, ...},
      "relevant_layers":     [14, ..., 29],
      "output_check":        "runs/transfer_matrix_memmap/llama/hotpotqa__0.done"
    }

Skip rules:
  - Cells already in pending/claimed/done/failed are never re-queued.
  - --skip-existing additionally skips cells whose output_check sentinel exists.
  - Only methods with a completed source run are included in source_run_dirs;
    cells with no discovered runs are silently skipped.
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

    # Discover all completed source runs across requested methods.
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

    # Group runs by (source_dataset, model_slug, seed) — one batch cell per group.
    from collections import defaultdict
    groups: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for run in all_runs:
        model_slug = _slug_from_experiment(run["experiment_name"])
        bare_src = _bare_dataset(run["dataset"])
        key = (bare_src, model_slug, int(run["seed"]))
        groups[key][run["method"]] = run

    for (bare_src, model_slug, seed), method_runs in sorted(groups.items()):
        cell_id = f"{bare_src}__{model_slug}__{seed}"

        if _dispatch_has_cell(dispatch_root, cell_id):
            continue

        src_cfg_name = _dataset_cfg_name(bare_src, model_slug)
        src_cfg_path = configs_dir / "datasets" / f"{src_cfg_name}.json"
        if not src_cfg_path.exists():
            print(f"  [warn] Source config not found: {src_cfg_path} — skipping {cell_id}")
            continue

        # Build per-method source_run_dirs and probe_layers.
        source_run_dirs: dict[str, str] = {}
        probe_layers: dict[str, int] = {}
        for method, run in method_runs.items():
            try:
                rel_run_dir = str(Path(run["run_dir"]).relative_to(_PROJECT_ROOT))
            except ValueError:
                rel_run_dir = run["run_dir"]
            source_run_dirs[method] = rel_run_dir
            probe_layers[method] = _resolve_probe_layer(run["run_dir"], run["config"])

        # Build target dataset config map for this model slug.
        tgt_ds_list: list[str] = []
        tgt_cfg_map: dict[str, str] = {}
        for tgt_ds in target_datasets:
            cfg_name = _dataset_cfg_name(tgt_ds, model_slug)
            cfg_path = configs_dir / "datasets" / f"{cfg_name}.json"
            if cfg_path.exists():
                tgt_ds_list.append(tgt_ds)
                tgt_cfg_map[tgt_ds] = str(cfg_path.relative_to(_PROJECT_ROOT))
            else:
                print(f"  [warn] Target config not found: {cfg_path} — skipping {tgt_ds}/{model_slug}")

        if not tgt_ds_list:
            print(f"  [warn] No target configs found for {cell_id} — skipping")
            continue

        output_check = str(
            (output_dir / model_slug / f"{bare_src}__{seed}.done")
            .relative_to(_PROJECT_ROOT)
        )

        if skip_existing and (_PROJECT_ROOT / output_check).exists():
            continue

        cell = {
            "cell_id": cell_id,
            "task_type": "transfer_eval",
            "source_dataset": bare_src,
            "model_slug": model_slug,
            "seed": seed,
            "source_run_dirs": source_run_dirs,
            "source_dataset_cfg": str(src_cfg_path.relative_to(_PROJECT_ROOT)),
            "target_datasets": tgt_ds_list,
            "target_dataset_cfgs": tgt_cfg_map,
            "probe_layers": probe_layers,
            "relevant_layers": relevant_layers,
            "output_check": output_check,
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
