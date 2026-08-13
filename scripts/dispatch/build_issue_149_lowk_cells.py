"""Build the symmetric low-k HalluLens/ACT-ViT queue for issue #149.

The queue covers the five in-scope Llama memmap benchmark datasets (MMLU is
explicitly excluded). Each cell is exactly one dataset x method x training
seed. Numeric cell priorities put both HotpotQA methods first under the
filesystem queue's lexicographic claiming rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

_DATASETS = (
    ("00", "hotpotqa_memmap"),
    ("20", "nq_memmap"),
    ("30", "popqa_memmap"),
    ("40", "sciq_memmap"),
    ("50", "searchqa_memmap"),
)
_METHODS = (
    "act_vit_prefix_multik_lowk",
    "contrastive_logprob_recon_prefix_mixed_lowk",
)
_SEEDS = (0, 1, 2, 3, 4)


def _load_cell(cell_path: Path) -> dict:
    try:
        return json.loads(cell_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _is_bundled_cell(cell: dict) -> bool:
    seed = cell.get("seed")
    return isinstance(seed, str) and "," in seed


def _active_bundled_pairs(dispatch_root: Path) -> set[tuple[str, str]]:
    """Return dataset/method pairs still owned by legacy bundled workers."""
    active: set[tuple[str, str]] = set()
    for cell_path in (dispatch_root / "claimed").glob("*/*.json"):
        cell = _load_cell(cell_path)
        if _is_bundled_cell(cell) and cell.get("dataset") and cell.get("method"):
            active.add((str(cell["dataset"]), str(cell["method"])))
    return active


def remove_nonrunning_bundled_cells(dispatch_root: Path) -> list[Path]:
    """Delete legacy seed-bundle records that are not actively claimed.

    Run artifacts are deliberately outside the dispatch root and are never
    touched. Claimed bundles are also preserved so live workers can finish.
    """
    removed: list[Path] = []
    for subdir in ("pending", "done", "failed", "cancelled"):
        for cell_path in (dispatch_root / subdir).glob("*.json"):
            if not _is_bundled_cell(_load_cell(cell_path)):
                continue
            err_path = cell_path.with_name(cell_path.name + ".err")
            if err_path.exists():
                err_path.unlink()
            cell_path.unlink()
            removed.append(cell_path)
    return removed


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    filename = f"{cell_id}.json"
    for subdir in ("pending", "done", "failed", "cancelled"):
        if (dispatch_root / subdir / filename).exists():
            return True
    claimed = dispatch_root / "claimed"
    return claimed.exists() and any(
        worker.is_dir() and (worker / filename).exists()
        for worker in claimed.iterdir()
    )


def _experiment_name(dataset_name: str) -> str:
    return f"prefix149_lowk_{dataset_name.removesuffix('_memmap')}"


def _run_dir(dataset_name: str, method_name: str, seed: int) -> Path:
    return (
        Path("runs")
        / _experiment_name(dataset_name)
        / dataset_name
        / method_name
        / f"seed_{seed}"
    )


def _terminal_output(run_dir: Path, method_name: str) -> Path:
    # Contrastive writes predictions after eval_metrics, making predictions the
    # stronger single-file completion sentinel understood by the existing
    # worker. ACT-ViT completion is represented by eval_metrics.
    if method_name == "contrastive_logprob_recon_prefix_mixed_lowk":
        return run_dir / "predictions.csv"
    return run_dir / "eval_metrics.json"


def _seed_is_complete(project_root: Path, run_dir: Path, method_name: str) -> bool:
    absolute = project_root / run_dir
    if (absolute / "run_error.json").exists():
        return False
    if not (absolute / "eval_metrics.json").exists():
        return False
    if method_name == "contrastive_logprob_recon_prefix_mixed_lowk":
        return (absolute / "predictions.csv").exists()
    return True


def build(dispatch_root: Path) -> int:
    init_dispatch_dirs(dispatch_root)
    active_bundled = _active_bundled_pairs(dispatch_root)
    written = 0
    for priority, dataset_name in _DATASETS:
        experiment_name = _experiment_name(dataset_name)
        experiment_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / experiment_rel).exists():
            raise FileNotFoundError(experiment_rel)
        for method_name in _METHODS:
            if (dataset_name, method_name) in active_bundled:
                print(
                    "  skip (legacy bundle actively claimed): "
                    f"{dataset_name} / {method_name}"
                )
                continue
            for seed in _SEEDS:
                cell_id = (
                    f"{priority}_issue149_lowk__{dataset_name}__"
                    f"{method_name}__seed_{seed}"
                )
                if _dispatch_has_cell(dispatch_root, cell_id):
                    print(f"  skip (already queued): {cell_id}")
                    continue
                run_dir = _run_dir(dataset_name, method_name, seed)
                if _seed_is_complete(_PROJECT_ROOT, run_dir, method_name):
                    print(f"  skip (seed output complete): {cell_id}")
                    continue
                cell = {
                    "cell_id": cell_id,
                    "kind": "experiment",
                    "experiment_config": experiment_rel,
                    "dataset": dataset_name,
                    "method": method_name,
                    "seed": seed,
                    "seeded": True,
                    "output_check": str(
                        _terminal_output(run_dir, method_name)
                    ),
                }
                (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                    json.dumps(cell, indent=2) + "\n"
                )
                written += 1
                print(f"  queued: {cell_id}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root", default="shared/issue_149_lowk_dispatch"
    )
    parser.add_argument(
        "--replace-bundled",
        action="store_true",
        help=(
            "delete non-claimed legacy seed-bundle records before building "
            "missing per-seed cells; run artifacts are preserved"
        ),
    )
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root
    if args.replace_bundled:
        removed = remove_nonrunning_bundled_cells(dispatch_root)
        for cell_path in removed:
            print(f"  removed legacy bundle: {cell_path}")
        print(f"removed {len(removed)} non-running legacy bundles")
    count = build(dispatch_root)
    print(f"\nqueued {count} cells under {dispatch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
