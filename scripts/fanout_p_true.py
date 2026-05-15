"""Fan out P(true) scoring across available Jupyter GPU nodes.

12 cells: 6 datasets × 2 models, test split only (P(true) is deterministic;
no seed axis). Cells are grouped by model within each shard so the model is
loaded once per model per node.

Usage:
    python scripts/fanout_p_true.py --dry-run
    python scripts/fanout_p_true.py --nodes alphagpu04-8884
    python scripts/fanout_p_true.py \\
        --nodes alphagpu04-8884,alphagpu03-8887,alphagpu01-8888
    python scripts/fanout_p_true.py \\
        --nodes alphagpu04-8884 \\
        --models meta-llama/Llama-3.1-8B-Instruct

Dispatch note: searchqa has ~151K rows (85% of total forward passes).
For balanced fanout across 3 nodes, assign searchqa to its own node and
split the remaining 5 datasets across the other nodes.  The default
round-robin assignment is fine for 1–2 nodes.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from tasks.p_true.paths import DATASETS, MODELS

DEFAULT_SPLIT = "test"


def build_cells(datasets, models):
    return [(ds, DEFAULT_SPLIT, model) for ds in datasets for model in models]


def assign_round_robin(cells, nodes):
    bins = {n: [] for n in nodes}
    for i, c in enumerate(cells):
        bins[nodes[i % len(nodes)]].append(c)
    return bins


def write_shard_file(path: Path, cells) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# dataset\tmodel\n")
        for ds, _split, model in cells:
            f.write(f"{ds}\t{model}\n")


def dispatch_shard(node: str, cells_file: Path, dry_run: bool) -> str:
    cmd = [
        "python", "scripts/gpu_dispatch.py", "run",
        "--jupyter", "--node", node,
        "--",
        "env", f"CELLS_FILE={cells_file}",
        "bash", "scripts/run_p_true_shard.sh",
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
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--nodes",
        required=True,
        help="Comma-separated Jupyter node names (e.g. alphagpu04-8884,...).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]
    nodes = [s.strip() for s in args.nodes.split(",") if s.strip()]

    cells = build_cells(datasets, models)
    bins = assign_round_robin(cells, nodes)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fanout_dir = Path("reports/p_true_runs") / f"fanout_{stamp}"

    print(f"Cells: {len(cells)} | Nodes: {len(nodes)} | Stamp: {stamp}")
    for n, ncells in bins.items():
        print(f"  {n}: {len(ncells)} cells")
        for ds, split, model in ncells:
            print(f"    - {ds}/{split}/{model.split('/')[-1]}")
    print()

    job_ids = []
    for node, ncells in bins.items():
        cells_file = fanout_dir / f"cells_{node}.tsv"
        write_shard_file(cells_file, ncells)
        jid = dispatch_shard(node, cells_file, dry_run=args.dry_run)
        if jid:
            job_ids.append((jid, node, cells_file, len(ncells)))

    if args.dry_run:
        print(
            f"\nDRY-RUN. Would have written shard files in {fanout_dir} "
            f"and dispatched {len(bins)} shard jobs."
        )
        return

    if job_ids:
        manifest = fanout_dir / "dispatch.txt"
        with open(manifest, "w") as f:
            f.write(f"# P(true) fanout dispatch {stamp}\n")
            for jid, node, cf, n in job_ids:
                f.write(f"{jid}\t{node}\t{n}\t{cf}\n")
        print(f"\nDispatched {len(job_ids)}/{len(bins)} shards.")
        for jid, node, _, n in job_ids:
            print(f"  {jid}  {node}  ({n} cells)")
        print(f"\nManifest: {manifest}")
        print("\nMonitor:")
        print("  python scripts/gpu_dispatch.py jobs")
        print("  python scripts/audit_p_true.py --check-counts")


if __name__ == "__main__":
    main()
