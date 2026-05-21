"""
build_actvit_transfer_cells.py — backfill act_vit-only transfer-eval cells for
all (source_dataset × model_slug × seed) combinations that are missing from the
transfer matrix.

The regular issue-89 cells (e.g. hotpotqa__llama__0) were generated without
act_vit in source_run_dirs, and those cell_ids are already in done/.  This
script generates act_vit-only cells with a distinct cell_id prefix
("actvit_<source>__<slug>__<seed>") and a matching output sentinel
("runs/transfer_matrix_memmap/<slug>/<source>__<seed>__actvit.done"), so they
can be queued alongside the existing cells without collision.

Usage (from repo root):
    python scripts/dispatch/build_actvit_transfer_cells.py \\
        --dispatch-root shared/issue_79_dispatch \\
        [--runs-dir runs] \\
        [--configs-dir configs] \\
        [--output-dir runs/transfer_matrix_memmap] \\
        [--model-slugs llama,qwen3] \\
        [--source-datasets hotpotqa,mmlu,nq,popqa,sciq,searchqa] \\
        [--seeds 0,1,2,3,4] \\
        [--skip-existing]
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

_DEFAULT_DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
_DEFAULT_SLUGS = ["llama", "qwen3"]
_DEFAULT_LAYERS = list(range(14, 30))


def _slug_from_experiment(experiment_name: str) -> str:
    return "qwen3" if "_qwen3_memmap" in experiment_name else "llama"


def _bare_dataset(dataset_field: str) -> str:
    for suffix in ("_qwen3_memmap", "_memmap"):
        if dataset_field.endswith(suffix):
            return dataset_field[: -len(suffix)]
    return dataset_field


def _dataset_cfg_name(dataset: str, model_slug: str) -> str:
    return f"{dataset}_qwen3_memmap" if model_slug == "qwen3" else f"{dataset}_memmap"


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


def build(
    dispatch_root: Path,
    runs_dir: Path,
    configs_dir: Path,
    output_dir: Path,
    model_slugs: list[str],
    source_datasets: list[str],
    target_datasets: list[str],
    seed_filter: list[int] | None,
    relevant_layers: list[int],
    skip_existing: bool,
    project_root: Path = _PROJECT_ROOT,
) -> int:
    init_dispatch_dirs(dispatch_root)

    runs = discover_runs(str(runs_dir), "act_vit")
    runs = [r for r in runs if _bare_dataset(r["dataset"]) in source_datasets]
    runs = [r for r in runs if _slug_from_experiment(r["experiment_name"]) in model_slugs]
    if seed_filter is not None:
        runs = [r for r in runs if r["seed"] in seed_filter]

    print(f"Discovered {len(runs)} completed act_vit source runs.")

    written = 0
    for run in sorted(runs, key=lambda r: (r["dataset"], r["experiment_name"], r["seed"])):
        model_slug = _slug_from_experiment(run["experiment_name"])
        bare_src = _bare_dataset(run["dataset"])
        seed = int(run["seed"])

        cell_id = f"actvit_{bare_src}__{model_slug}__{seed}"

        sentinel = output_dir / model_slug / f"{bare_src}__{seed}__actvit.done"
        try:
            sentinel_rel = str(sentinel.relative_to(project_root))
        except ValueError:
            sentinel_rel = str(sentinel)

        if _dispatch_has_cell(dispatch_root, cell_id):
            print(f"  skip (already queued): {cell_id}")
            continue

        if skip_existing and sentinel.exists():
            print(f"  skip (output exists): {cell_id}")
            continue

        src_cfg_name = _dataset_cfg_name(bare_src, model_slug)
        src_cfg_path = configs_dir / "datasets" / f"{src_cfg_name}.json"
        if not src_cfg_path.exists():
            print(f"  [warn] Source config missing: {src_cfg_path} — skipping {cell_id}")
            continue

        try:
            src_cfg_rel = str(src_cfg_path.relative_to(project_root))
            run_dir_rel = str(Path(run["run_dir"]).relative_to(project_root))
        except ValueError:
            src_cfg_rel = str(src_cfg_path)
            run_dir_rel = run["run_dir"]

        probe_layer = _resolve_probe_layer(run["run_dir"], run.get("config", {}))

        tgt_ds_list: list[str] = []
        tgt_cfg_map: dict[str, str] = {}
        for tgt in target_datasets:
            cfg_name = _dataset_cfg_name(tgt, model_slug)
            cfg_path = configs_dir / "datasets" / f"{cfg_name}.json"
            if cfg_path.exists():
                tgt_ds_list.append(tgt)
                try:
                    tgt_cfg_map[tgt] = str(cfg_path.relative_to(project_root))
                except ValueError:
                    tgt_cfg_map[tgt] = str(cfg_path)

        if not tgt_ds_list:
            print(f"  [warn] No target configs for {cell_id} — skipping")
            continue

        cell = {
            "cell_id": cell_id,
            "task_type": "transfer_eval",
            "source_dataset": bare_src,
            "model_slug": model_slug,
            "seed": seed,
            "source_run_dirs": {"act_vit": run_dir_rel},
            "source_dataset_cfg": src_cfg_rel,
            "target_datasets": tgt_ds_list,
            "target_dataset_cfgs": tgt_cfg_map,
            "probe_layers": {"act_vit": probe_layer},
            "relevant_layers": relevant_layers,
            "output_check": sentinel_rel,
        }
        (dispatch_root / "pending" / f"{cell_id}.json").write_text(
            json.dumps(cell, indent=2), encoding="utf-8"
        )
        written += 1
        print(f"  queued: {cell_id}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Queue act_vit-only transfer-eval backfill cells."
    )
    parser.add_argument("--dispatch-root", default="shared/issue_79_dispatch")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--output-dir", default="runs/transfer_matrix_memmap")
    parser.add_argument("--model-slugs", default=",".join(_DEFAULT_SLUGS))
    parser.add_argument("--source-datasets", default=",".join(_DEFAULT_DATASETS))
    parser.add_argument("--target-datasets", default=",".join(_DEFAULT_DATASETS))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--relevant-layers", default="14-29")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cells whose output sentinel already exists.")
    args = parser.parse_args()

    def _parse_layers(spec: str) -> list[int]:
        if "-" in spec and "," not in spec:
            a, b = spec.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in spec.split(",")]

    total = build(
        dispatch_root=_PROJECT_ROOT / args.dispatch_root,
        runs_dir=_PROJECT_ROOT / args.runs_dir,
        configs_dir=_PROJECT_ROOT / args.configs_dir,
        output_dir=_PROJECT_ROOT / args.output_dir,
        model_slugs=[s.strip() for s in args.model_slugs.split(",") if s.strip()],
        source_datasets=[d.strip() for d in args.source_datasets.split(",") if d.strip()],
        target_datasets=[d.strip() for d in args.target_datasets.split(",") if d.strip()],
        seed_filter=(
            [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
            if args.seeds else None
        ),
        relevant_layers=_parse_layers(args.relevant_layers),
        skip_existing=args.skip_existing,
        project_root=_PROJECT_ROOT,
    )
    print(f"Done — {total} cells queued in {args.dispatch_root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
