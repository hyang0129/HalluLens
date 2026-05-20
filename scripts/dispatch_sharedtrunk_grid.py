"""Dispatch the shared-trunk grid (issue #102) across two GPU nodes.

Splits by model so each node only processes one model's cached activations:
  Node A (--node-llama):  sharedtrunk_grid_{sciq,nq}_llama_memmap
  Node B (--node-qwen3):  sharedtrunk_grid_{sciq,nq}_qwen3_memmap

Configs include C0 (unlabeled SimCLR) and D2a/D2b (projection-head variants).
D1 is intentionally excluded.

Usage (from the cluster login node, after sync-jupyter)::

    python scripts/dispatch_sharedtrunk_grid.py \\
        --node-llama alphagpu10-8887 \\
        --node-qwen3 alphagpu03-8887

Dry-run (print commands, don't dispatch)::

    python scripts/dispatch_sharedtrunk_grid.py \\
        --node-llama alphagpu10-8887 \\
        --node-qwen3 alphagpu03-8887 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/mnt/home/hyang1/.local/share/mamba/envs/py312/bin/python"

EXPERIMENTS = {
    "llama": [
        "configs/experiments/sharedtrunk_grid_sciq_llama_memmap.json",
        "configs/experiments/sharedtrunk_grid_nq_llama_memmap.json",
    ],
    "qwen3": [
        "configs/experiments/sharedtrunk_grid_sciq_qwen3_memmap.json",
        "configs/experiments/sharedtrunk_grid_nq_qwen3_memmap.json",
    ],
}


def build_command(experiment_paths: list[str]) -> str:
    """Build a chained bash command that runs each experiment sequentially."""
    parts = []
    for exp in experiment_paths:
        parts.append(
            f"{shlex.quote(PYTHON)} scripts/run_experiment.py"
            f" --experiment {shlex.quote(exp)}"
        )
    return " && ".join(parts)


def dispatch(node: str, command: str, description: str, dry_run: bool) -> None:
    cmd = [
        PYTHON, "scripts/gpu_dispatch.py", "run",
        "--node", node,
        "--desc", description,
        command,
    ]
    print(f"\n>>> {description}")
    print(f"    node: {node}")
    print(f"    cmd:  {command}")
    if dry_run:
        print("    [dry-run — not dispatched]")
        return
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: dispatch to {node} failed (rc={result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node-llama", required=True, help="Node for Llama experiments (e.g. alphagpu10-8887)")
    p.add_argument("--node-qwen3", required=True, help="Node for Qwen3 experiments (e.g. alphagpu03-8887)")
    p.add_argument("--dry-run", action="store_true", help="Print commands without dispatching")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dispatch(
        node=args.node_llama,
        command=build_command(EXPERIMENTS["llama"]),
        description="sharedtrunk-grid Llama: sciq + nq (C0, D2a, D2b)",
        dry_run=args.dry_run,
    )
    dispatch(
        node=args.node_qwen3,
        command=build_command(EXPERIMENTS["qwen3"]),
        description="sharedtrunk-grid Qwen3: sciq + nq (C0, D2a, D2b)",
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nBoth jobs dispatched. Monitor with:")
        print(f"  python scripts/gpu_dispatch.py jobs --all")
        print(f"  ssh empire-ai 'tail -f ~/LLM_research/HalluLens/shared/logs/<job_id>.log'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
