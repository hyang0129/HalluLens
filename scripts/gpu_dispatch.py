#!/usr/bin/env python3
"""Multi-node GPU job dispatch and tracking.

A single-file CLI tool for dispatching jobs to GPU nodes, checking node
health, and tracking running jobs. Uses only Python stdlib.

Usage:
    python scripts/gpu_dispatch.py status
    python scripts/gpu_dispatch.py run [--node HOST] [--desc DESC] [--min-vram GB] COMMAND...
    python scripts/gpu_dispatch.py jobs [--all]
    python scripts/gpu_dispatch.py kill JOB_ID
"""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    hostname: str
    python: str
    project_root: str
    max_concurrent_jobs: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass
class NodeStatus:
    hostname: str
    reachable: bool
    gpu_name: Optional[str] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    our_procs: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class JobRecord:
    job_id: str
    hostname: str
    command: str
    pid: Optional[int]
    started_at: str
    status: str  # "running", "completed", "failed", "killed", "unknown"
    log_file: str
    description: str = ""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    """Return default path to configs/nodes.json relative to this script."""
    return Path(__file__).resolve().parent.parent / "configs" / "nodes.json"


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the node registry from JSON."""
    path = Path(config_path) if config_path else _default_config_path()
    with open(path) as f:
        return json.load(f)


def load_nodes(config: dict) -> List[NodeConfig]:
    """Parse nodes from config, applying defaults where fields are missing."""
    defaults = config.get("defaults", {})
    nodes = []
    for entry in config.get("nodes", []):
        nodes.append(NodeConfig(
            hostname=entry["hostname"],
            python=entry.get("python", defaults.get("python", "python3")),
            project_root=entry.get("project_root", defaults.get("project_root", ".")),
            max_concurrent_jobs=entry.get("max_concurrent_jobs", 1),
            tags=entry.get("tags", []),
        ))
    return nodes


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------

def ssh_run(hostname: str, cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a command on a remote host via SSH with BatchMode enabled."""
    return subprocess.run(
        [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            hostname,
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Node health
# ---------------------------------------------------------------------------

def check_node_health(node: NodeConfig, ssh_timeout: int = 10) -> NodeStatus:
    """SSH to a node, query nvidia-smi and running processes."""
    status = NodeStatus(hostname=node.hostname, reachable=False)

    # nvidia-smi query
    try:
        result = ssh_run(
            node.hostname,
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null; echo '---PROCS---'; "
            "ps aux | grep -E 'python|bash' | grep hyang1 | grep -v grep | grep -v sshd",
            timeout=ssh_timeout + 5,
        )
    except subprocess.TimeoutExpired:
        status.error = "SSH timeout"
        return status
    except Exception as exc:
        status.error = str(exc)
        return status

    if result.returncode != 0 and "---PROCS---" not in result.stdout:
        status.error = result.stderr.strip() or f"exit code {result.returncode}"
        return status

    status.reachable = True
    parts = result.stdout.split("---PROCS---")

    # Parse GPU info (take first GPU line)
    gpu_lines = [l.strip() for l in parts[0].strip().splitlines() if l.strip()]
    if gpu_lines:
        fields = [f.strip() for f in gpu_lines[0].split(",")]
        if len(fields) >= 4:
            status.gpu_name = fields[0]
            try:
                status.gpu_util_pct = float(fields[1])
            except ValueError:
                pass
            try:
                status.gpu_mem_used_mb = float(fields[2])
            except ValueError:
                pass
            try:
                status.gpu_mem_total_mb = float(fields[3])
            except ValueError:
                pass

    # Parse our processes
    if len(parts) > 1:
        proc_lines = [l.strip() for l in parts[1].strip().splitlines() if l.strip()]
        status.our_procs = proc_lines

    return status


# ---------------------------------------------------------------------------
# Job manifest (file-locked JSON)
# ---------------------------------------------------------------------------

def _manifest_path(config: dict) -> Path:
    """Resolve the manifest path from config, relative to project root."""
    defaults = config.get("defaults", {})
    project_root = Path(defaults.get("project_root", "."))
    manifest_rel = defaults.get("job_manifest", "shared/gpu_jobs.json")
    return project_root / manifest_rel


def load_manifest(manifest_path: Path) -> List[JobRecord]:
    """Load the job manifest, returning an empty list if it doesn't exist."""
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return [JobRecord(**rec) for rec in data]


def save_manifest(manifest_path: Path, jobs: List[JobRecord]) -> None:
    """Save the job manifest with exclusive file locking.

    Acquires lock before truncating to prevent races with concurrent writers.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in r+/create mode so we can lock before truncating
    try:
        f = open(manifest_path, "r+")
    except FileNotFoundError:
        f = open(manifest_path, "w")
    with f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        f.truncate()
        json.dump([asdict(j) for j in jobs], f, indent=2, default=str)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def update_manifest(manifest_path: Path, fn) -> List[JobRecord]:
    """Atomically read-modify-write the manifest under an exclusive lock.

    fn receives the current job list and must return the updated list.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(manifest_path, "r+")
    except FileNotFoundError:
        f = open(manifest_path, "w+")
    with f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        content = f.read()
        try:
            data = json.loads(content) if content.strip() else []
        except json.JSONDecodeError:
            data = []
        jobs = [JobRecord(**rec) for rec in data]
        jobs = fn(jobs)
        f.seek(0)
        f.truncate()
        json.dump([asdict(j) for j in jobs], f, indent=2, default=str)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return jobs


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

def select_best_node(
    nodes: List[NodeConfig],
    manifest: List[JobRecord],
    statuses: dict,  # hostname -> NodeStatus
    min_vram_gb: Optional[float] = None,
) -> Optional[NodeConfig]:
    """Pick the reachable node with fewest running jobs and most free VRAM."""
    # Count running jobs per node
    running_counts: dict = {}
    for job in manifest:
        if job.status == "running":
            running_counts[job.hostname] = running_counts.get(job.hostname, 0) + 1

    candidates = []
    for node in nodes:
        ns = statuses.get(node.hostname)
        if ns is None or not ns.reachable:
            continue
        running = running_counts.get(node.hostname, 0)
        if running >= node.max_concurrent_jobs:
            continue
        free_vram_mb = 0.0
        if ns.gpu_mem_total_mb is not None and ns.gpu_mem_used_mb is not None:
            free_vram_mb = ns.gpu_mem_total_mb - ns.gpu_mem_used_mb
        free_vram_gb = free_vram_mb / 1024.0
        if min_vram_gb is not None and free_vram_gb < min_vram_gb:
            continue
        candidates.append((node, running, free_vram_mb))

    if not candidates:
        return None

    # Sort: fewest running jobs first, then most free VRAM
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------

def dispatch_job(
    node: NodeConfig,
    command: str,
    description: str,
    config: dict,
) -> JobRecord:
    """Dispatch a command to a node via SSH nohup bash, record in manifest."""
    job_id = uuid.uuid4().hex[:12]
    log_file = f"shared/logs/{job_id}.log"

    # Ensure log directory exists
    ssh_run(node.hostname, f"mkdir -p {shlex.quote(node.project_root + '/shared/logs')}", timeout=10)

    # Build the dispatch command (quote paths and user command to prevent injection)
    abs_log = f"{node.project_root}/{log_file}"
    quoted_cmd = shlex.quote(command)
    quoted_root = shlex.quote(node.project_root)
    dispatch_cmd = (
        f"cd {quoted_root} && "
        f"nohup bash -c {quoted_cmd} > {shlex.quote(abs_log)} 2>&1 & echo $!"
    )

    result = ssh_run(node.hostname, dispatch_cmd, timeout=15)

    pid = None
    pid_str = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        pid = int(pid_str)
    except ValueError:
        pass

    job = JobRecord(
        job_id=job_id,
        hostname=node.hostname,
        command=command,
        pid=pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running" if pid else "unknown",
        log_file=log_file,
        description=description,
    )

    # Append to manifest atomically
    manifest_path = _manifest_path(config)
    update_manifest(manifest_path, lambda jobs: jobs + [job])

    return job


# ---------------------------------------------------------------------------
# Refresh job statuses
# ---------------------------------------------------------------------------

def refresh_job_statuses(
    manifest_path: Path,
    nodes: List[NodeConfig],
    ssh_timeout: int = 10,
) -> List[JobRecord]:
    """Check PIDs on nodes for running jobs, update statuses."""
    jobs = load_manifest(manifest_path)
    node_map = {n.hostname: n for n in nodes}

    # Group running jobs by hostname
    running_by_host: dict = {}
    for i, job in enumerate(jobs):
        if job.status == "running" and job.pid is not None:
            running_by_host.setdefault(job.hostname, []).append((i, job))

    changed = False
    for hostname, job_list in running_by_host.items():
        if hostname not in node_map:
            continue
        pids = [str(j.pid) for _, j in job_list]
        # Check which PIDs are still alive
        try:
            result = ssh_run(
                hostname,
                f"ps -p {','.join(pids)} -o pid= 2>/dev/null || true",
                timeout=ssh_timeout + 5,
            )
            alive_pids = set()
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                try:
                    alive_pids.add(int(line))
                except ValueError:
                    pass
            for idx, job in job_list:
                if job.pid not in alive_pids:
                    jobs[idx].status = "finished"
                    changed = True
        except (subprocess.TimeoutExpired, Exception):
            # Can't reach node -- leave status as-is
            pass

    if changed:
        save_manifest(manifest_path, jobs)

    return jobs


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Handle the 'status' subcommand."""
    config = load_config(args.config)
    nodes = load_nodes(config)
    ssh_timeout = config.get("defaults", {}).get("ssh_timeout", 10)
    manifest_path = _manifest_path(config)
    manifest = load_manifest(manifest_path)

    # Count running jobs per node
    running_counts: dict = {}
    for job in manifest:
        if job.status == "running":
            running_counts[job.hostname] = running_counts.get(job.hostname, 0) + 1

    # Check all nodes in parallel
    statuses: dict = {}
    if not nodes:
        print("No nodes configured.")
        return
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        futures = {
            pool.submit(check_node_health, node, ssh_timeout): node
            for node in nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            statuses[node.hostname] = future.result()

    # Print table
    header = f"{'NODE':<15} {'GPU':<18} {'VRAM USED/TOTAL':<18} {'GPU UTIL':<10} {'OUR JOBS':<10} {'STATUS':<12}"
    print(header)
    print("-" * len(header))

    for node in nodes:
        ns = statuses.get(node.hostname)
        if ns is None or not ns.reachable:
            err = ns.error if ns else "unknown"
            print(f"{node.hostname:<15} {'-':<18} {'-':<18} {'-':<10} {'-':<10} {'unreachable':<12}")
            continue

        gpu_name = ns.gpu_name or "-"
        if len(gpu_name) > 16:
            gpu_name = gpu_name[:16] + ".."

        if ns.gpu_mem_used_mb is not None and ns.gpu_mem_total_mb is not None:
            vram = f"{ns.gpu_mem_used_mb / 1024:.1f}/{ns.gpu_mem_total_mb / 1024:.1f} GB"
        else:
            vram = "-"

        util = f"{ns.gpu_util_pct:.0f}%" if ns.gpu_util_pct is not None else "-"

        running = running_counts.get(node.hostname, 0)
        node_status = "busy" if running >= node.max_concurrent_jobs else "available"

        print(f"{node.hostname:<15} {gpu_name:<18} {vram:<18} {util:<10} {running:<10} {node_status:<12}")


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    config = load_config(args.config)
    nodes = load_nodes(config)
    ssh_timeout = config.get("defaults", {}).get("ssh_timeout", 10)
    manifest_path = _manifest_path(config)
    manifest = load_manifest(manifest_path)
    command = " ".join(args.cmd)
    description = args.desc or ""

    # Health check nodes
    if args.node:
        target_nodes = [n for n in nodes if n.hostname == args.node]
        if not target_nodes:
            print(f"Error: node '{args.node}' not found in config", file=sys.stderr)
            sys.exit(1)
    else:
        target_nodes = nodes

    statuses: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, len(target_nodes))) as pool:
        futures = {
            pool.submit(check_node_health, node, ssh_timeout): node
            for node in target_nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            statuses[node.hostname] = future.result()

    if args.node:
        # Validate specified node is reachable
        ns = statuses.get(args.node)
        if ns is None or not ns.reachable:
            err = ns.error if ns else "unknown"
            print(f"Error: node '{args.node}' is unreachable ({err})", file=sys.stderr)
            sys.exit(1)
        selected = target_nodes[0]
        # Check min-vram
        if args.min_vram and ns.gpu_mem_total_mb and ns.gpu_mem_used_mb:
            free_gb = (ns.gpu_mem_total_mb - ns.gpu_mem_used_mb) / 1024.0
            if free_gb < args.min_vram:
                print(
                    f"Error: node '{args.node}' has {free_gb:.1f} GB free VRAM, "
                    f"need {args.min_vram:.1f} GB",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        selected = select_best_node(nodes, manifest, statuses, min_vram_gb=args.min_vram)
        if selected is None:
            print("Error: no suitable node available", file=sys.stderr)
            sys.exit(1)

    job = dispatch_job(selected, command, description, config)

    print(f"Job dispatched:")
    print(f"  Job ID:   {job.job_id}")
    print(f"  Node:     {job.hostname}")
    print(f"  PID:      {job.pid}")
    print(f"  Log file: {job.log_file}")
    print(f"  Command:  {job.command}")


def cmd_jobs(args: argparse.Namespace) -> None:
    """Handle the 'jobs' subcommand."""
    config = load_config(args.config)
    nodes = load_nodes(config)
    ssh_timeout = config.get("defaults", {}).get("ssh_timeout", 10)
    manifest_path = _manifest_path(config)

    jobs = refresh_job_statuses(manifest_path, nodes, ssh_timeout)

    if not args.all:
        jobs = [j for j in jobs if j.status == "running"]

    if not jobs:
        print("No jobs found.")
        return

    # Print table
    header = f"{'JOB_ID':<14} {'NODE':<15} {'STATUS':<12} {'PID':<8} {'STARTED':<22} {'COMMAND'}"
    print(header)
    print("-" * len(header))

    for job in jobs:
        started = job.started_at[:19] if job.started_at else "-"
        cmd_display = job.command
        if len(cmd_display) > 60:
            cmd_display = cmd_display[:57] + "..."
        print(
            f"{job.job_id:<14} {job.hostname:<15} {job.status:<12} "
            f"{str(job.pid or '-'):<8} {started:<22} {cmd_display}"
        )


def cmd_kill(args: argparse.Namespace) -> None:
    """Handle the 'kill' subcommand."""
    config = load_config(args.config)
    manifest_path = _manifest_path(config)
    jobs = load_manifest(manifest_path)

    target = None
    target_idx = None
    for i, job in enumerate(jobs):
        if job.job_id == args.job_id:
            target = job
            target_idx = i
            break

    if target is None:
        print(f"Error: job '{args.job_id}' not found in manifest", file=sys.stderr)
        sys.exit(1)

    if target.status != "running":
        print(f"Warning: job '{args.job_id}' status is '{target.status}', not 'running'")

    if target.pid:
        try:
            result = ssh_run(target.hostname, f"kill {target.pid}", timeout=10)
            if result.returncode == 0:
                print(f"Sent kill signal to PID {target.pid} on {target.hostname}")
            else:
                print(
                    f"Warning: kill command returned code {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            print(f"Warning: SSH timeout trying to kill PID {target.pid} on {target.hostname}")
    else:
        print(f"Warning: no PID recorded for job '{args.job_id}'")

    jobs[target_idx].status = "killed"
    save_manifest(manifest_path, jobs)
    print(f"Job '{args.job_id}' marked as killed")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-node GPU job dispatch and tracking"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to nodes.json config (default: configs/nodes.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Check GPU node health")

    # run
    run_parser = subparsers.add_parser("run", help="Dispatch a job to a GPU node")
    run_parser.add_argument("--node", help="Target specific node")
    run_parser.add_argument("--desc", help="Job description")
    run_parser.add_argument(
        "--min-vram", type=float, help="Minimum free VRAM in GB"
    )
    run_parser.add_argument("cmd", nargs="+", help="Command to run")

    # jobs
    jobs_parser = subparsers.add_parser("jobs", help="List tracked jobs")
    jobs_parser.add_argument(
        "--all", action="store_true", help="Show all jobs including completed/killed"
    )

    # kill
    kill_parser = subparsers.add_parser("kill", help="Kill a running job")
    kill_parser.add_argument("job_id", help="Job ID to kill")

    args = parser.parse_args()

    handlers = {
        "status": cmd_status,
        "run": cmd_run,
        "jobs": cmd_jobs,
        "kill": cmd_kill,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
