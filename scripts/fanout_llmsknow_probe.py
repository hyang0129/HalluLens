"""Fan out llmsknow_probe re-runs across Jupyter GPU nodes (issue #57 / PR #58).

We split work by **(dataset, model) cache unit** — each unit is one
``baseline_comparison_*.json`` experiment config. All 5 seeds for a unit
run in the same ``run_experiment.py`` invocation so the preloaded
activation memmap is shared across seeds (the dev-subset materialization
is the heavy I/O step; splitting seeds across nodes would force each
node to re-warm ~60 GB of cache per seed).

Default plan: 12 cache units (6 datasets × 2 models) round-robin'd
across 3 nodes → 4 units per node, each running 5 seeds serially.

Usage::

    # Dry-run to see the plan
    python scripts/fanout_llmsknow_probe.py --dry-run \\
        --nodes alphagpu01-8888,alphagpu03-8887,alphagpu04-8884

    # Dispatch all three shards
    python scripts/fanout_llmsknow_probe.py \\
        --nodes alphagpu01-8888,alphagpu03-8887,alphagpu04-8884

    # Only re-run Qwen3 results
    python scripts/fanout_llmsknow_probe.py \\
        --nodes alphagpu01-8888,alphagpu03-8887,alphagpu04-8884 \\
        --models Qwen/Qwen3-8B

Each shard writes its assigned units to
``reports/llmsknow_probe_runs/fanout_<stamp>/units_<node>.tsv`` and is
dispatched via ``gpu_dispatch.py run --jupyter --node <node>``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Default 6 datasets × 2 models matches the shipped baseline_comparison_* configs.
DEFAULT_DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]
DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"]

# Map HF model id -> experiment-config suffix used by configs/experiments/baseline_comparison_<ds><suffix>.json
MODEL_SUFFIX = {
    "meta-llama/Llama-3.1-8B-Instruct": "",       # baseline_comparison_<ds>.json
    "Qwen/Qwen3-8B": "_qwen3",                    # baseline_comparison_<ds>_qwen3.json
}


def build_units(datasets, models, project_root: Path):
    """Return list of (label, experiment_config_path) tuples, validating each
    config exists on disk."""
    units = []
    missing = []
    for ds in datasets:
        for model in models:
            suffix = MODEL_SUFFIX.get(model)
            if suffix is None:
                raise ValueError(
                    f"Unknown model {model!r}: extend MODEL_SUFFIX in "
                    f"scripts/fanout_llmsknow_probe.py to map it to an experiment-config suffix."
                )
            cfg_path = project_root / "configs" / "experiments" / f"baseline_comparison_{ds}{suffix}.json"
            label = f"{ds}/{model.split('/')[-1]}"
            if not cfg_path.exists():
                missing.append((label, cfg_path))
                continue
            units.append((label, str(cfg_path.relative_to(project_root))))
    if missing:
        for lbl, p in missing:
            print(f"  WARNING: missing config for {lbl}: {p}", file=sys.stderr)
    return units


def assign_round_robin(units, nodes):
    """Round-robin units across nodes. Returns {node: [(label, cfg_path), ...]}."""
    bins = {n: [] for n in nodes}
    for i, u in enumerate(units):
        bins[nodes[i % len(nodes)]].append(u)
    return bins


def write_units_file(path: Path, units) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# experiment_config_path  (one per line; comments start with #)\n")
        for label, cfg in units:
            f.write(f"{cfg}\t# {label}\n")


def dispatch_shard(node: str, units_file: Path, seeds: str, dry_run: bool) -> str:
    cmd = [
        "python", "scripts/gpu_dispatch.py", "run",
        "--jupyter", "--node", node,
        "--",
        "env", f"UNITS_FILE={units_file}", f"SEEDS={seeds}",
        "bash", "scripts/run_llmsknow_probe_shard.sh",
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return ""
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(f"  FAILED ({node}): {out.stderr}", file=sys.stderr)
        return ""
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("Job ID:"):
            return line.split(":", 1)[1].strip()
    print(f"  WARNING ({node}): dispatched but no job id found", file=sys.stderr)
    return ""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help=f"Comma-separated datasets (default: {','.join(DEFAULT_DATASETS)})")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help=f"Comma-separated HF model ids (default: {','.join(DEFAULT_MODELS)})")
    parser.add_argument("--nodes", required=True,
                        help="Comma-separated jupyter node names (e.g. alphagpu01-8888,alphagpu03-8887,alphagpu04-8884)")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated training seeds (default: 0,1,2,3,4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without dispatching")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    nodes = [s.strip() for s in args.nodes.split(",") if s.strip()]

    units = build_units(datasets, models, project_root)
    if not units:
        print("No units to dispatch (all configs missing?).", file=sys.stderr)
        sys.exit(1)
    bins = assign_round_robin(units, nodes)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fanout_dir = project_root / "reports" / "llmsknow_probe_runs" / f"fanout_{stamp}"

    print(f"Units: {len(units)} | Nodes: {len(nodes)} | Seeds: {args.seeds} | Stamp: {stamp}")
    print(f"Cache-unit granularity: (dataset, model) — all seeds share preloaded activation cache.\n")
    for n, nunits in bins.items():
        print(f"  {n}: {len(nunits)} units")
        for label, _ in nunits:
            print(f"    - {label}")
    print()

    job_ids = []
    for node, nunits in bins.items():
        if not nunits:
            continue
        units_file = fanout_dir / f"units_{node}.tsv"
        write_units_file(units_file, nunits)
        jid = dispatch_shard(node, units_file, args.seeds, dry_run=args.dry_run)
        if jid:
            job_ids.append((jid, node, units_file, len(nunits)))

    if args.dry_run:
        print(f"\nDRY-RUN. Would have written shard files in {fanout_dir} and dispatched {len(bins)} shard jobs.")
        return

    if job_ids:
        manifest = fanout_dir / "dispatch.txt"
        with open(manifest, "w") as f:
            f.write(f"# fanout dispatch {stamp}\n")
            for jid, node, cf, n in job_ids:
                f.write(f"{jid}\t{node}\t{n}\t{cf}\n")
        print(f"\nDispatched {len(job_ids)}/{len(bins)} shards.")
        for jid, node, _, n in job_ids:
            print(f"  {jid}  {node}  ({n} units)")
        print(f"\nManifest: {manifest}")
        print(f"\nMonitor:")
        print(f"  python scripts/gpu_dispatch.py jobs")
        print(f"  tail -f reports/llmsknow_probe_runs/shard_*.log")


if __name__ == "__main__":
    main()
