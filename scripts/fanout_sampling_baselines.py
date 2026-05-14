"""Fan out the 20 sampling-baseline cells across available jupyter GPU slots.

Each node gets ONE dispatched job that runs its assigned cells sequentially
(via scripts/run_sampling_baselines_shard.sh). Cells are written to a per-node
TSV file under reports/sampling_baselines_runs/fanout_<stamp>/. Resumable: a
shard that fails mid-way picks up the next cell on the next dispatch.

Usage:
    python scripts/fanout_sampling_baselines.py --dry-run
    python scripts/fanout_sampling_baselines.py --nodes alphagpu04-8884
    python scripts/fanout_sampling_baselines.py \
        --nodes alphagpu04-8884,alphagpu03-8887 \
        --models meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa"]
DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"]
DEFAULT_SPLITS = ["test", "train"]


def build_cells(datasets, splits, models):
    cells = []
    for ds in datasets:
        for split in splits:
            for model in models:
                cells.append((ds, split, model))
    return cells


def assign_round_robin(cells, nodes):
    """Round-robin cells across nodes. Returns {node: [cell, ...]}."""
    bins = {n: [] for n in nodes}
    for i, c in enumerate(cells):
        bins[nodes[i % len(nodes)]].append(c)
    return bins


def write_shard_file(path: Path, cells) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# dataset\tsplit\tmodel\n")
        for ds, split, model in cells:
            f.write(f"{ds}\t{split}\t{model}\n")


def dispatch_shard(node: str, cells_file: Path, dry_run: bool) -> str:
    cmd = [
        "python", "scripts/gpu_dispatch.py", "run",
        "--jupyter", "--node", node,
        "--",
        "env", f"CELLS_FILE={cells_file}",
        "bash", "scripts/run_sampling_baselines_shard.sh",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--nodes", required=True,
                        help="Comma-separated jupyter node names (e.g. alphagpu04-8884,...).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    nodes = [s.strip() for s in args.nodes.split(",") if s.strip()]

    cells = build_cells(datasets, splits, models)
    bins = assign_round_robin(cells, nodes)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fanout_dir = Path("reports/sampling_baselines_runs") / f"fanout_{stamp}"

    print(f"Cells: {len(cells)} | Nodes: {len(nodes)} | Stamp: {stamp}")
    for n, ncells in bins.items():
        print(f"  {n}: {len(ncells)} cells")
        for ds, split, model in ncells:
            print(f"    - {ds}/{split}/{model.split('/')[-1]}")
    print()

    # Write per-node shard files
    job_ids = []
    for node, ncells in bins.items():
        cells_file = fanout_dir / f"cells_{node}.tsv"
        write_shard_file(cells_file, ncells)
        jid = dispatch_shard(node, cells_file, dry_run=args.dry_run)
        if jid:
            job_ids.append((jid, node, cells_file, len(ncells)))

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
            print(f"  {jid}  {node}  ({n} cells)")
        print(f"\nManifest: {manifest}")
        print(f"\nMonitor:")
        print(f"  python scripts/gpu_dispatch.py jobs")
        print(f"  tail -f reports/sampling_baselines_runs/shard_*.log")


if __name__ == "__main__":
    main()
