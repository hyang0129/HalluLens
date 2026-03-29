"""Shared helpers for experiment status tracking and run enumeration.

Used by experiment_status.py and (potentially) run_experiment.py to enumerate
expected runs from an experiment config and classify their completion status.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class RunStatus(Enum):
    COMPLETE = "complete"
    FAILED = "failed"
    RUNNING = "running"
    PENDING = "pending"


@dataclass
class RunSpec:
    dataset_name: str
    method_name: str
    seed: Optional[int]
    run_dir: str
    is_learned: bool


def load_experiment_config(
    path: str, project_root: Optional[str] = None
) -> dict:
    """Load experiment JSON, resolving method configs from configs/methods/{name}.json.

    Returns a dict with at least:
      - experiment_name, datasets (list), methods (list), training_seeds (list)
      - method_configs: dict mapping method name -> loaded method config
      - output_dir: str
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent)

    with open(path) as f:
        experiment_cfg = json.load(f)

    # Normalise dataset(s) to a list
    dataset_value = experiment_cfg.get("dataset") or experiment_cfg.get("datasets")
    if dataset_value is None:
        raise ValueError("Experiment config must have 'dataset' or 'datasets' key")
    if isinstance(dataset_value, str):
        experiment_cfg["datasets"] = [dataset_value]
    else:
        experiment_cfg["datasets"] = list(dataset_value)

    # Ensure methods list
    if "methods" not in experiment_cfg:
        raise ValueError("Experiment config must have 'methods' key")

    # Ensure training_seeds
    if "training_seeds" not in experiment_cfg:
        experiment_cfg["training_seeds"] = [42]

    # Ensure output_dir
    if "output_dir" not in experiment_cfg:
        experiment_cfg["output_dir"] = "runs"

    # Load method configs
    method_configs = {}
    for method_name in experiment_cfg["methods"]:
        method_cfg_path = os.path.join(
            project_root, "configs", "methods", f"{method_name}.json"
        )
        if os.path.exists(method_cfg_path):
            with open(method_cfg_path) as f:
                method_configs[method_name] = json.load(f)
        else:
            # If method config doesn't exist, assume non-learned
            method_configs[method_name] = {"name": method_name, "training": None}

    experiment_cfg["method_configs"] = method_configs
    return experiment_cfg


def enumerate_runs(
    experiment_cfg: dict,
    output_base: Optional[str] = None,
    project_root: Optional[str] = None,
) -> List[RunSpec]:
    """Produce the full list of expected runs from an experiment config.

    Replicates the directory path construction from run_experiment.py:
      - Learned methods (those with "training" in their config): one run per seed
      - Non-learned methods: one run with seed=None
      - Path: {output_base}/{exp_name}/{dataset}/{method}/seed_{seed}
              or {output_base}/{exp_name}/{dataset}/{method}
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent)

    if output_base is None:
        output_base = experiment_cfg.get("output_dir", "runs")

    exp_name = experiment_cfg.get("experiment_name", "default")
    datasets = experiment_cfg.get("datasets", [])
    methods = experiment_cfg.get("methods", [])
    training_seeds = experiment_cfg.get("training_seeds", [42])
    method_configs = experiment_cfg.get("method_configs", {})

    runs: List[RunSpec] = []

    for dataset_name in datasets:
        for method_name in methods:
            method_cfg = method_configs.get(method_name, {})
            is_learned = method_cfg.get("training") is not None
            seeds = training_seeds if is_learned else [None]

            for seed in seeds:
                if seed is not None:
                    run_dir = os.path.join(
                        output_base, exp_name, dataset_name, method_name, f"seed_{seed}"
                    )
                else:
                    run_dir = os.path.join(
                        output_base, exp_name, dataset_name, method_name
                    )

                runs.append(
                    RunSpec(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        seed=seed,
                        run_dir=run_dir,
                        is_learned=is_learned,
                    )
                )

    return runs


def classify_run(run_dir: str) -> RunStatus:
    """Classify the status of a single run based on its directory contents.

    - Has eval_metrics.json -> COMPLETE
    - Has run_error.json -> FAILED
    - Has config.json or other files but no eval_metrics/run_error -> RUNNING
    - Directory doesn't exist or is empty -> PENDING
    """
    if not os.path.isdir(run_dir):
        return RunStatus.PENDING

    contents = os.listdir(run_dir)
    if not contents:
        return RunStatus.PENDING

    if "eval_metrics.json" in contents:
        return RunStatus.COMPLETE

    if "run_error.json" in contents:
        return RunStatus.FAILED

    # Has some files (e.g. config.json) but not finished
    return RunStatus.RUNNING


def count_summary(runs: List[tuple[RunSpec, RunStatus]]) -> dict:
    """Aggregate status counts from a list of (RunSpec, RunStatus) pairs."""
    counts = {status: 0 for status in RunStatus}
    for _, status in runs:
        counts[status] += 1
    return {
        "total": len(runs),
        "complete": counts[RunStatus.COMPLETE],
        "failed": counts[RunStatus.FAILED],
        "running": counts[RunStatus.RUNNING],
        "pending": counts[RunStatus.PENDING],
    }
