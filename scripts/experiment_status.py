#!/usr/bin/env python3
"""CLI tool for checking experiment run status.

Usage:
    # From experiment config (enumerates expected runs)
    python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_hotpotqa.json

    # From a runs directory directly
    python scripts/experiment_status.py --runs-dir runs/baseline_comparison_hotpotqa

    # Machine-readable JSON output
    python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_hotpotqa.json --json

    # Verbose: list every run
    python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_hotpotqa.json --verbose

Exit codes:
    0 - All runs complete
    1 - Some runs pending or running
    2 - Any runs failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

# Allow running from repo root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiment_utils import (
    RunSpec,
    RunStatus,
    classify_run,
    count_summary,
    enumerate_runs,
    load_experiment_config,
)

# Status display symbols
_STATUS_SYMBOL = {
    RunStatus.COMPLETE: "\u2713",  # checkmark
    RunStatus.FAILED: "\u2717",    # x mark
    RunStatus.RUNNING: "~",
    RunStatus.PENDING: "\u00b7",   # middle dot
}


def _discover_runs_from_dir(runs_dir: str) -> List[RunSpec]:
    """Discover runs by walking a runs directory tree.

    Expected structure: {runs_dir}/{dataset}/{method}/[seed_{n}/]
    """
    runs: List[RunSpec] = []
    if not os.path.isdir(runs_dir):
        return runs

    for dataset_name in sorted(os.listdir(runs_dir)):
        dataset_path = os.path.join(runs_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for method_name in sorted(os.listdir(dataset_path)):
            method_path = os.path.join(dataset_path, method_name)
            if not os.path.isdir(method_path):
                continue

            # Check if this method dir has seed subdirectories
            sub_entries = os.listdir(method_path)
            seed_dirs = [e for e in sub_entries if e.startswith("seed_")]

            if seed_dirs:
                for seed_dir in sorted(seed_dirs):
                    seed_val = int(seed_dir.split("_", 1)[1])
                    run_dir = os.path.join(method_path, seed_dir)
                    runs.append(
                        RunSpec(
                            dataset_name=dataset_name,
                            method_name=method_name,
                            seed=seed_val,
                            run_dir=run_dir,
                            is_learned=True,
                        )
                    )
            else:
                # Non-learned method (no seed dirs) or single run
                runs.append(
                    RunSpec(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        seed=None,
                        run_dir=method_path,
                        is_learned=False,
                    )
                )

    return runs


def _get_error_message(run_dir: str) -> Optional[str]:
    """Extract error message from run_error.json if it exists."""
    error_path = os.path.join(run_dir, "run_error.json")
    if not os.path.exists(error_path):
        return None
    try:
        with open(error_path) as f:
            err = json.load(f)
        # Support both flat and nested formats
        msg = err.get("error", err.get("message", str(err)))
        if isinstance(msg, dict):
            msg = msg.get("message", str(msg))
        # Truncate long messages
        if len(str(msg)) > 120:
            msg = str(msg)[:117] + "..."
        return str(msg)
    except Exception:
        return "(could not read error file)"


def _print_status_matrix(
    classified_runs: List[tuple[RunSpec, RunStatus]],
    experiment_cfg: Optional[dict],
) -> None:
    """Print the status matrix to stdout."""
    # Gather unique datasets and methods in order
    datasets_seen: list[str] = []
    methods_seen: list[str] = []
    for run_spec, _ in classified_runs:
        if run_spec.dataset_name not in datasets_seen:
            datasets_seen.append(run_spec.dataset_name)
        if run_spec.method_name not in methods_seen:
            methods_seen.append(run_spec.method_name)

    # Group runs: (method, dataset) -> list of (RunSpec, RunStatus)
    grid: dict[tuple[str, str], list[tuple[RunSpec, RunStatus]]] = defaultdict(list)
    for run_spec, status in classified_runs:
        grid[(run_spec.method_name, run_spec.dataset_name)].append((run_spec, status))

    # Header info
    if experiment_cfg:
        exp_name = experiment_cfg.get("experiment_name", "unknown")
        seeds = experiment_cfg.get("training_seeds", [42])
        print(f"Experiment: {exp_name}")
        print(f"Datasets: {', '.join(datasets_seen)}")
        print(f"Methods: {', '.join(methods_seen)}")
        print(f"Seeds: {seeds}")

        # Count expected runs
        n_learned = sum(
            1
            for m in methods_seen
            if experiment_cfg.get("method_configs", {}).get(m, {}).get("training") is not None
        )
        n_non_learned = len(methods_seen) - n_learned
        total = (n_learned * len(seeds) + n_non_learned) * len(datasets_seen)
        print(
            f"Expected runs: {total}  "
            f"({n_learned} learned x {len(seeds)} seeds + {n_non_learned} non-learned)"
            f" x {len(datasets_seen)} dataset(s)"
        )
    print()

    # Calculate column widths
    method_col_width = max(len(m) for m in methods_seen) if methods_seen else 10
    dataset_col_width = max(len(d) for d in datasets_seen) if datasets_seen else 10

    # Header row
    header = " " * (method_col_width + 4)
    for ds in datasets_seen:
        header += ds.ljust(dataset_col_width + 4)
    print(header)

    # Method rows
    for method in methods_seen:
        row = method.ljust(method_col_width + 4)
        for dataset in datasets_seen:
            cell_runs = grid.get((method, dataset), [])
            if not cell_runs:
                cell = "-"
                detail = ""
            else:
                symbols = "".join(_STATUS_SYMBOL[s] for _, s in cell_runs)
                total_in_cell = len(cell_runs)
                complete_in_cell = sum(1 for _, s in cell_runs if s == RunStatus.COMPLETE)
                failed_in_cell = sum(1 for _, s in cell_runs if s == RunStatus.FAILED)

                parts = [f"{complete_in_cell}/{total_in_cell} complete"]
                if failed_in_cell:
                    parts.append(f"{failed_in_cell} failed")
                detail = f"  ({', '.join(parts)})"
                cell = symbols + detail

            row += cell.ljust(dataset_col_width + 4)
        print(row)


def _print_verbose(classified_runs: List[tuple[RunSpec, RunStatus]]) -> None:
    """Print every run with its status."""
    print("\nAll runs:")
    for run_spec, status in classified_runs:
        seed_str = f"seed={run_spec.seed}" if run_spec.seed is not None else "no-seed"
        label = f"  {run_spec.dataset_name}/{run_spec.method_name}/{seed_str}"
        status_str = status.value.upper()
        line = f"{label:<60} {status_str}"
        if status == RunStatus.FAILED:
            msg = _get_error_message(run_spec.run_dir)
            if msg:
                line += f"  -- {msg}"
        print(line)


def _print_failed(classified_runs: List[tuple[RunSpec, RunStatus]]) -> None:
    """Print failed runs with error messages."""
    failed = [(r, s) for r, s in classified_runs if s == RunStatus.FAILED]
    if not failed:
        return

    print("\nFailed runs:")
    for run_spec, _ in failed:
        seed_str = f"seed_{run_spec.seed}" if run_spec.seed is not None else ""
        path_label = f"  {run_spec.method_name}"
        if seed_str:
            path_label += f"/{seed_str}"
        msg = _get_error_message(run_spec.run_dir)
        error_detail = f": {msg}" if msg else ""
        print(f"{path_label}{error_detail} (run_error.json)")


def _build_json_output(
    classified_runs: List[tuple[RunSpec, RunStatus]],
    summary: dict,
    experiment_cfg: Optional[dict],
) -> dict:
    """Build a JSON-serializable dict of the full status."""
    runs_list = []
    for run_spec, status in classified_runs:
        entry = {
            "dataset": run_spec.dataset_name,
            "method": run_spec.method_name,
            "seed": run_spec.seed,
            "run_dir": run_spec.run_dir,
            "is_learned": run_spec.is_learned,
            "status": status.value,
        }
        if status == RunStatus.FAILED:
            entry["error"] = _get_error_message(run_spec.run_dir)
        runs_list.append(entry)

    output = {
        "summary": summary,
        "runs": runs_list,
    }
    if experiment_cfg:
        output["experiment_name"] = experiment_cfg.get("experiment_name", "unknown")
        output["datasets"] = experiment_cfg.get("datasets", [])
        output["methods"] = experiment_cfg.get("methods", [])
        output["training_seeds"] = experiment_cfg.get("training_seeds", [42])

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check experiment run status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        type=str,
        help="Path to experiment config JSON",
    )
    group.add_argument(
        "--runs-dir",
        type=str,
        help="Path to a runs directory to scan directly",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from experiment config",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List every run with its status",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_cfg: Optional[dict] = None

    if args.experiment:
        experiment_cfg = load_experiment_config(
            args.experiment, project_root=str(project_root)
        )
        output_base = args.output_dir or experiment_cfg.get("output_dir", "runs")
        run_specs = enumerate_runs(
            experiment_cfg,
            output_base=output_base,
            project_root=str(project_root),
        )
    else:
        # --runs-dir mode: discover from directory
        run_specs = _discover_runs_from_dir(args.runs_dir)

    # Classify each run
    classified_runs = [(spec, classify_run(spec.run_dir)) for spec in run_specs]

    # Summary
    summary = count_summary(classified_runs)

    if args.json_output:
        output = _build_json_output(classified_runs, summary, experiment_cfg)
        print(json.dumps(output, indent=2))
    else:
        _print_status_matrix(classified_runs, experiment_cfg)
        print()
        print(
            f"Overall: {summary['complete']}/{summary['total']} complete, "
            f"{summary['failed']} failed, "
            f"{summary['running']} running, "
            f"{summary['pending']} pending"
        )
        _print_failed(classified_runs)

        if args.verbose:
            _print_verbose(classified_runs)

    # Exit code
    if summary["failed"] > 0:
        sys.exit(2)
    elif summary["complete"] < summary["total"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
