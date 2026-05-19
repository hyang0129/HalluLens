#!/usr/bin/env python3
"""Multi-node GPU job dispatch and tracking.

A single-file CLI tool for dispatching jobs to GPU nodes, checking node
health, and tracking running jobs. Uses only Python stdlib (plus utils/
for Jupyter dispatch).

Usage:
    python scripts/gpu_dispatch.py status
    python scripts/gpu_dispatch.py run [--node HOST] [--desc DESC] [--min-vram GB] COMMAND...
    python scripts/gpu_dispatch.py jobs [--all]
    python scripts/gpu_dispatch.py kill JOB_ID

Transport (Jupyter is the default):
    All node interactions — health probes, dispatch, status, kill — go
    through each node's already-allocated Jupyter kernel by default.
    The login node spends ~60s per SSH channel on PAM/profile setup and
    leaks orphan processes against the user.slice TasksMax=512 cap, so
    we avoid it entirely for dispatch work.

    Pass --ssh on `status` or `run` to force the legacy SSH path. Useful
    only for debugging or for nodes that genuinely have no jupyter_url.
    `kill` routes per-job based on each job's recorded dispatch_method.

    Auto-select (no --node on `run`) probes all jupyter_url-configured
    nodes in parallel via Jupyter, then picks the best by free VRAM and
    current job count. Nodes without jupyter_url are skipped from the
    Jupyter pool — add their endpoint via `sync-jupyter` or pass --ssh.

    Job tracking remains transport-agnostic. Jupyter-dispatched jobs are
    marked with dispatch_method=jupyter in the manifest; `jobs` and
    `kill` route per-job from that field. Probe failures on `jobs` mark
    the job status as "unknown" rather than blocking (5s SSH timeout).
"""

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path so utils.jupyter_exec is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    name: str  # logical identity; can differ from hostname (e.g. two Jupyter ports on one host)
    hostname: str
    python: str
    project_root: str
    max_concurrent_jobs: int = 1
    tags: List[str] = field(default_factory=list)
    jupyter_url: Optional[str] = None
    jupyter_password: str = "123"
    source: Optional[str] = None  # "squeue" → entry is managed by sync-jupyter and will be regenerated


@dataclass
class NodeStatus:
    name: str
    hostname: str
    reachable: bool
    gpu_name: Optional[str] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    our_procs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    via_jupyter: bool = False


@dataclass
class JobRecord:
    job_id: str
    node_name: str  # logical node identity (matches NodeConfig.name)
    hostname: str
    command: str
    pid: Optional[int]
    started_at: str
    status: str  # "running", "finished", "killed", "unknown"
    log_file: str
    description: str = ""
    dispatch_method: str = "ssh"  # "ssh" or "jupyter"


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
        hostname = entry["hostname"]
        nodes.append(NodeConfig(
            name=entry.get("name", hostname),
            hostname=hostname,
            python=entry.get("python", defaults.get("python", "python3")),
            project_root=entry.get("project_root", defaults.get("project_root", ".")),
            max_concurrent_jobs=entry.get("max_concurrent_jobs", 1),
            tags=entry.get("tags", []),
            jupyter_url=entry.get("jupyter_url", defaults.get("jupyter_url")),
            jupyter_password=entry.get("jupyter_password", defaults.get("jupyter_password", "123")),
            source=entry.get("source"),
        ))
    names = [n.name for n in nodes]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate node names in config: {dupes}")
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
    status = NodeStatus(name=node.name, hostname=node.hostname, reachable=False)

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
    _parse_health_output(status, result.stdout)
    return status


def check_node_health_jupyter(node: NodeConfig) -> NodeStatus:
    """Query a node's GPU health via Jupyter API (fallback when SSH is unavailable)."""
    status = NodeStatus(name=node.name, hostname=node.hostname, reachable=False)
    if not node.jupyter_url:
        status.error = "no jupyter_url configured"
        return status
    try:
        from utils.jupyter_exec import JupyterExecutor
        code = (
            "import subprocess\n"
            "smi = subprocess.run(\n"
            "    ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',\n"
            "     '--format=csv,noheader,nounits'],\n"
            "    capture_output=True, text=True)\n"
            "ps = subprocess.run(\n"
            "    'ps aux | grep -E \"python|bash\" | grep hyang1 | grep -v grep | grep -v sshd',\n"
            "    shell=True, capture_output=True, text=True)\n"
            "print('---PROCS---'.join([smi.stdout.strip(), ps.stdout.strip()]))\n"
        )
        with JupyterExecutor(base_url=node.jupyter_url, password=node.jupyter_password) as jup:
            result = jup.run(code)
        if result.status != "ok":
            status.error = f"jupyter kernel error: {result.stdout[:200]}"
            return status
        status.reachable = True
        status.via_jupyter = True
        _parse_health_output(status, result.stdout)
    except Exception as exc:
        status.error = str(exc)
    return status


def _parse_health_output(status: NodeStatus, raw: str) -> None:
    """Parse combined nvidia-smi + ps output into a NodeStatus (mutates in place)."""
    parts = raw.split("---PROCS---")

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

    if len(parts) > 1:
        status.our_procs = [l.strip() for l in parts[1].strip().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Job manifest (file-locked JSON)
# ---------------------------------------------------------------------------

def _manifest_path(config: dict) -> Path:
    """Resolve the manifest path from config, relative to project root."""
    defaults = config.get("defaults", {})
    project_root = Path(defaults.get("project_root", "."))
    manifest_rel = defaults.get("job_manifest", "shared/gpu_jobs.json")
    return project_root / manifest_rel


def _record_from_dict(rec: dict) -> JobRecord:
    """Deserialise a JobRecord, back-filling fields absent from older manifests."""
    rec = {"dispatch_method": "ssh", **rec}
    rec.setdefault("node_name", rec["hostname"])
    return JobRecord(**rec)


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
    return [_record_from_dict(rec) for rec in data]


def save_manifest(manifest_path: Path, jobs: List[JobRecord]) -> None:
    """Save the job manifest with exclusive file locking.

    Acquires lock before truncating to prevent races with concurrent writers.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
        jobs = [_record_from_dict(rec) for rec in data]
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
    statuses: dict,  # name -> NodeStatus
    min_vram_gb: Optional[float] = None,
) -> Optional[NodeConfig]:
    """Pick the reachable node with fewest running jobs and most free VRAM."""
    running_counts: dict = {}
    for job in manifest:
        if job.status == "running":
            running_counts[job.node_name] = running_counts.get(job.node_name, 0) + 1

    candidates = []
    for node in nodes:
        ns = statuses.get(node.name)
        if ns is None or not ns.reachable:
            continue
        running = running_counts.get(node.name, 0)
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

    ssh_run(node.hostname, f"mkdir -p {shlex.quote(node.project_root + '/shared/logs')}", timeout=10)

    abs_log = f"{node.project_root}/{log_file}"
    inner_cmd = shlex.quote(f"cd {node.project_root} && {command}")
    dispatch_cmd = (
        f"setsid nohup bash -c {inner_cmd} > {shlex.quote(abs_log)} 2>&1 & echo $!"
    )

    result = ssh_run(node.hostname, dispatch_cmd, timeout=60)

    pid = None
    pid_str = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        pid = int(pid_str)
    except ValueError:
        pass

    job = JobRecord(
        job_id=job_id,
        node_name=node.name,
        hostname=node.hostname,
        command=command,
        pid=pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running" if pid else "unknown",
        log_file=log_file,
        description=description,
        dispatch_method="ssh",
    )

    manifest_path = _manifest_path(config)
    update_manifest(manifest_path, lambda jobs: jobs + [job])

    return job


def dispatch_job_jupyter(
    node: NodeConfig,
    command: str,
    description: str,
    config: dict,
) -> JobRecord:
    """Dispatch a command to a node via Jupyter API, record in manifest.

    Requires the node to have jupyter_url configured. This is an explicit
    opt-in path — it is never called automatically when SSH fails.
    """
    if not node.jupyter_url:
        raise ValueError(
            f"Node '{node.hostname}' has no jupyter_url configured. "
            "Add jupyter_url to its entry in configs/nodes.json to use --jupyter."
        )

    from utils.jupyter_exec import JupyterExecutor

    job_id = uuid.uuid4().hex[:12]
    log_file = f"shared/logs/{job_id}.log"
    abs_log = f"{node.project_root}/{log_file}"
    log_dir = str(Path(abs_log).parent)
    inner_cmd = f"cd {node.project_root} && {command}"

    code = (
        f"import subprocess, pathlib\n"
        f"pathlib.Path({repr(log_dir)}).mkdir(parents=True, exist_ok=True)\n"
        f"with open({repr(abs_log)}, 'w') as _lf:\n"
        f"    _p = subprocess.Popen(\n"
        f"        ['bash', '-c', {repr(inner_cmd)}],\n"
        f"        stdout=_lf, stderr=_lf, start_new_session=True)\n"
        f"print(_p.pid)\n"
    )

    with JupyterExecutor(base_url=node.jupyter_url, password=node.jupyter_password) as jup:
        result = jup.run(code)

    if result.status != "ok":
        raise RuntimeError(
            f"Jupyter dispatch to '{node.hostname}' failed:\n{result.stdout}"
        )

    pid = None
    try:
        pid = int(result.stdout.strip())
    except ValueError:
        pass

    job = JobRecord(
        job_id=job_id,
        node_name=node.name,
        hostname=node.hostname,
        command=command,
        pid=pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running" if pid else "unknown",
        log_file=log_file,
        description=description,
        dispatch_method="jupyter",
    )

    manifest_path = _manifest_path(config)
    update_manifest(manifest_path, lambda jobs: jobs + [job])

    return job


# ---------------------------------------------------------------------------
# squeue-driven Jupyter node discovery
# ---------------------------------------------------------------------------

# Match SLURM job names like "jupyter_empire_8889" — the trailing digits are the
# Jupyter port. Anchored at end so "jupyter_foo_8889_extra" is not matched.
JUPYTER_JOB_NAME_RE = re.compile(r"jupyter_\w+_(\d+)$")


def discover_jupyter_allocations(squeue_timeout: int = 15) -> List[dict]:
    """Run `squeue --me` and return one dict per RUNNING jupyter_*_<port> allocation.

    Each dict has: job_id, hostname, port, time_left. Multi-node allocations
    (NODELIST containing '[') are skipped — we don't know which host serves the
    Jupyter endpoint in that case.
    """
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N|%L"],
            capture_output=True,
            text=True,
            timeout=squeue_timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "squeue not found in PATH — sync-jupyter must run on a SLURM submit host."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"squeue timed out after {squeue_timeout}s") from exc

    if result.returncode != 0:
        raise RuntimeError(f"squeue failed (exit {result.returncode}): {result.stderr.strip()}")

    allocations = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        job_id, name, state, nodelist, time_left = parts[:5]
        if state != "RUNNING":
            continue
        m = JUPYTER_JOB_NAME_RE.match(name)
        if not m:
            continue
        if "[" in nodelist or "," in nodelist:
            # Multi-node allocation — ambiguous which host hosts the Jupyter
            # server. Skip rather than guess.
            continue
        allocations.append({
            "job_id": job_id,
            "hostname": nodelist,
            "port": int(m.group(1)),
            "time_left": time_left,
        })
    return allocations


def _build_squeue_node_entry(alloc: dict, defaults: dict) -> dict:
    """Compose a nodes.json entry for one squeue allocation."""
    return {
        "name": f"{alloc['hostname']}-{alloc['port']}",
        "hostname": alloc["hostname"],
        "python": defaults.get("python", "python3"),
        "project_root": defaults.get("project_root", "."),
        "max_concurrent_jobs": 1,
        "tags": ["training", "inference", "jupyter-only"],
        "jupyter_url": f"http://{alloc['hostname']}:{alloc['port']}",
        "jupyter_password": defaults.get("jupyter_password", "123"),
        "source": "squeue",
        "slurm_job_id": alloc["job_id"],
        "slurm_time_left": alloc["time_left"],
    }


def sync_jupyter_nodes(config_path: Path, dry_run: bool = False) -> dict:
    """Reconcile squeue-derived Jupyter entries in nodes.json with live SLURM state.

    Removes every entry with source="squeue" and re-adds one per RUNNING
    `jupyter_*_<port>` allocation reported by `squeue --me`. SSH-only and
    hybrid (SSH+Jupyter) entries are left untouched.

    Returns a summary dict: {added, removed, kept, allocations, written}.
    """
    with open(config_path) as f:
        raw = json.load(f)

    defaults = raw.get("defaults", {})
    existing = raw.get("nodes", [])
    managed_old = [n for n in existing if n.get("source") == "squeue"]
    unmanaged = [n for n in existing if n.get("source") != "squeue"]
    old_names = {n.get("name") or n.get("hostname") for n in managed_old}

    allocations = discover_jupyter_allocations()
    new_entries = [_build_squeue_node_entry(a, defaults) for a in allocations]
    new_names = {e["name"] for e in new_entries}

    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)
    kept = sorted(new_names & old_names)

    raw["nodes"] = unmanaged + new_entries

    written = False
    if not dry_run:
        with open(config_path, "w") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")
        written = True

    return {
        "added": added,
        "removed": removed,
        "kept": kept,
        "allocations": allocations,
        "written": written,
    }


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
    name_map = {n.name: n for n in nodes}
    hostname_seen: set = set()
    for n in nodes:
        hostname_seen.add(n.hostname)

    # Separate running jobs by dispatch method
    # SSH jobs are deduped by hostname (ps result is shared across all logical
    # nodes on the same host). Jupyter jobs are keyed by node_name because
    # each logical node has its own Jupyter URL.
    ssh_by_host: dict = {}
    jupyter_by_name: dict = {}
    for i, job in enumerate(jobs):
        if job.status != "running" or job.pid is None:
            continue
        if job.dispatch_method == "jupyter":
            jupyter_by_name.setdefault(job.node_name, []).append((i, job))
        else:
            ssh_by_host.setdefault(job.hostname, []).append((i, job))

    changed = False

    # SSH jobs: check via SSH (one probe per hostname, not per logical node).
    # Tight timeout (5s) — the login node can be slow/wedged, and `jobs` must
    # never hang. On any failure, mark each affected job's status as
    # "unknown" so the caller sees that the probe didn't conclude.
    for hostname, job_list in ssh_by_host.items():
        if hostname not in hostname_seen:
            continue
        pids = [str(j.pid) for _, j in job_list]
        try:
            result = ssh_run(
                hostname,
                f"ps -p {','.join(pids)} -o pid= 2>/dev/null || true",
                timeout=5,
            )
            alive_pids = set()
            for line in result.stdout.strip().splitlines():
                try:
                    alive_pids.add(int(line.strip()))
                except ValueError:
                    pass
            for idx, job in job_list:
                if job.pid not in alive_pids:
                    jobs[idx].status = "finished"
                    changed = True
        except (subprocess.TimeoutExpired, Exception):
            for idx, _ in job_list:
                if jobs[idx].status == "running":
                    jobs[idx].status = "unknown"
                    changed = True

    # Jupyter jobs: check via Jupyter API (one probe per logical node).
    # Same failure semantics — mark "unknown" if the probe fails.
    for node_name, job_list in jupyter_by_name.items():
        node = name_map.get(node_name)
        if node is None or not node.jupyter_url:
            for idx, _ in job_list:
                if jobs[idx].status == "running":
                    jobs[idx].status = "unknown"
                    changed = True
            continue
        pids = [str(j.pid) for _, j in job_list]
        try:
            from utils.jupyter_exec import JupyterExecutor
            code = (
                f"import subprocess\n"
                f"r = subprocess.run(['ps', '-p', {repr(','.join(pids))}, '-o', 'pid='],\n"
                f"    capture_output=True, text=True)\n"
                f"print(r.stdout)\n"
            )
            with JupyterExecutor(base_url=node.jupyter_url, password=node.jupyter_password) as jup:
                result = jup.run(code)
            alive_pids = set()
            for line in result.stdout.strip().splitlines():
                try:
                    alive_pids.add(int(line.strip()))
                except ValueError:
                    pass
            for idx, job in job_list:
                if job.pid not in alive_pids:
                    jobs[idx].status = "finished"
                    changed = True
        except Exception:
            for idx, _ in job_list:
                if jobs[idx].status == "running":
                    jobs[idx].status = "unknown"
                    changed = True

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

    running_counts: dict = {}
    for job in manifest:
        if job.status == "running":
            running_counts[job.node_name] = running_counts.get(job.node_name, 0) + 1

    # Default transport is Jupyter (the login node is unreliable for SSH).
    # --ssh forces SSH-only probes for backward compatibility / debugging.
    # When Jupyter is the default, a node without jupyter_url is probed via
    # SSH as a last resort; failures are reported as "unreachable" with no
    # silent SSH fallback for nodes that DO have jupyter_url.
    use_ssh = getattr(args, "ssh", False)
    statuses: dict = {}

    if use_ssh:
        with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
            futures = {
                pool.submit(check_node_health, node, ssh_timeout): node
                for node in nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                statuses[node.name] = future.result()
    else:
        jupyter_capable = [n for n in nodes if n.jupyter_url]
        ssh_only = [n for n in nodes if not n.jupyter_url]

        if jupyter_capable:
            with ThreadPoolExecutor(max_workers=max(1, len(jupyter_capable))) as pool:
                futures = {
                    pool.submit(check_node_health_jupyter, node): node
                    for node in jupyter_capable
                }
                for future in as_completed(futures):
                    node = futures[future]
                    statuses[node.name] = future.result()

        # Nodes lacking jupyter_url fall back to SSH on a per-node basis.
        if ssh_only:
            with ThreadPoolExecutor(max_workers=max(1, len(ssh_only))) as pool:
                futures = {
                    pool.submit(check_node_health, node, ssh_timeout): node
                    for node in ssh_only
                }
                for future in as_completed(futures):
                    node = futures[future]
                    statuses[node.name] = future.result()

    header = f"{'NODE':<20} {'GPU':<18} {'VRAM USED/TOTAL':<18} {'GPU UTIL':<10} {'OUR JOBS':<10} {'STATUS':<12}"
    print(header)
    print("-" * len(header))

    for node in nodes:
        ns = statuses.get(node.name)
        if ns is None or not ns.reachable:
            print(f"{node.name:<20} {'-':<18} {'-':<18} {'-':<10} {'-':<10} {'unreachable':<12}")
            continue

        gpu_name = ns.gpu_name or "-"
        if len(gpu_name) > 16:
            gpu_name = gpu_name[:16] + ".."

        if ns.gpu_mem_used_mb is not None and ns.gpu_mem_total_mb is not None:
            vram = f"{ns.gpu_mem_used_mb / 1024:.1f}/{ns.gpu_mem_total_mb / 1024:.1f} GB"
        else:
            vram = "-"

        util = f"{ns.gpu_util_pct:.0f}%" if ns.gpu_util_pct is not None else "-"
        running = running_counts.get(node.name, 0)
        node_status = "busy" if running >= node.max_concurrent_jobs else "available"
        if ns.via_jupyter:
            node_status += " (jupyter)"

        print(f"{node.name:<20} {gpu_name:<18} {vram:<18} {util:<10} {running:<10} {node_status:<12}")


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
    config = load_config(args.config)
    nodes = load_nodes(config)
    ssh_timeout = config.get("defaults", {}).get("ssh_timeout", 10)
    manifest_path = _manifest_path(config)
    manifest = refresh_job_statuses(manifest_path, nodes, ssh_timeout)
    command = " ".join(args.cmd)
    description = args.desc or ""

    if args.node:
        target_nodes = [n for n in nodes if n.name == args.node]
        if not target_nodes:
            print(f"Error: node '{args.node}' not found in config", file=sys.stderr)
            sys.exit(1)
    else:
        target_nodes = nodes

    # Transport selection: Jupyter is the default (the cluster's login node
    # is unreliable for SSH; Jupyter goes directly to the GPU node's
    # already-allocated kernel). Pass --ssh to opt back to SSH. The old
    # --jupyter flag is accepted but is now a no-op (Jupyter is default).
    use_ssh = getattr(args, "ssh", False)

    if not use_ssh:
        # Jupyter dispatch (default). Probe candidate nodes via Jupyter in
        # parallel, then either honor --node or auto-select.
        jupyter_candidates = [n for n in target_nodes if n.jupyter_url]
        if not jupyter_candidates:
            print(
                "Error: no candidate nodes have jupyter_url configured. "
                "Either add jupyter_url entries to configs/nodes.json "
                "(run `gpu_dispatch.py sync-jupyter`) or pass --ssh to "
                "dispatch via SSH.",
                file=sys.stderr,
            )
            sys.exit(1)

        statuses: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, len(jupyter_candidates))) as pool:
            futures = {
                pool.submit(check_node_health_jupyter, node): node
                for node in jupyter_candidates
            }
            for future in as_completed(futures):
                node = futures[future]
                statuses[node.name] = future.result()

        if args.node:
            ns = statuses.get(args.node)
            if ns is None or not ns.reachable:
                err = ns.error if ns else "unknown"
                print(
                    f"Error: node '{args.node}' is unreachable via Jupyter ({err}). "
                    f"Pass --ssh to try the SSH path.",
                    file=sys.stderr,
                )
                sys.exit(1)
            selected = jupyter_candidates[0]
        else:
            selected = select_best_node(
                jupyter_candidates, manifest, statuses, min_vram_gb=args.min_vram
            )
            if selected is None:
                print(
                    "Error: no suitable Jupyter-reachable node available",
                    file=sys.stderr,
                )
                sys.exit(1)
            ns = statuses[selected.name]

        running = sum(1 for j in manifest if j.status == "running" and j.node_name == selected.name)
        if running >= selected.max_concurrent_jobs and not args.force_concurrent:
            print(
                f"Error: node '{selected.name}' is at max_concurrent_jobs "
                f"({selected.max_concurrent_jobs}). Use 'jobs' to check running jobs, "
                "or pass --force-concurrent to dispatch a co-tenant (e.g. a CPU-only side job).",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.min_vram and ns.gpu_mem_total_mb and ns.gpu_mem_used_mb:
            free_gb = (ns.gpu_mem_total_mb - ns.gpu_mem_used_mb) / 1024.0
            if free_gb < args.min_vram:
                print(
                    f"Error: node '{selected.name}' has {free_gb:.1f} GB free VRAM, "
                    f"need {args.min_vram:.1f} GB",
                    file=sys.stderr,
                )
                sys.exit(1)

        job = dispatch_job_jupyter(selected, command, description, config)

    else:
        # SSH dispatch (--ssh opt-out from Jupyter default).
        statuses: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, len(target_nodes))) as pool:
            futures = {
                pool.submit(check_node_health, node, ssh_timeout): node
                for node in target_nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                statuses[node.name] = future.result()

        if args.node:
            ns = statuses.get(args.node)
            if ns is None or not ns.reachable:
                err = ns.error if ns else "unknown"
                print(f"Error: node '{args.node}' is unreachable ({err})", file=sys.stderr)
                if target_nodes[0].jupyter_url:
                    print(
                        f"Hint: this node has jupyter_url configured. "
                        f"Drop --ssh to dispatch via Jupyter instead.",
                        file=sys.stderr,
                    )
                sys.exit(1)
            selected = target_nodes[0]
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
    print(f"  Node:     {job.node_name}")
    print(f"  Host:     {job.hostname}")
    print(f"  PID:      {job.pid}")
    print(f"  Via:      {job.dispatch_method}")
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

    header = f"{'JOB_ID':<14} {'NODE':<20} {'STATUS':<12} {'VIA':<8} {'PID':<8} {'STARTED':<22} {'COMMAND'}"
    print(header)
    print("-" * len(header))

    for job in jobs:
        started = job.started_at[:19] if job.started_at else "-"
        cmd_display = job.command
        if len(cmd_display) > 60:
            cmd_display = cmd_display[:57] + "..."
        print(
            f"{job.job_id:<14} {job.node_name:<20} {job.status:<12} "
            f"{job.dispatch_method:<8} {str(job.pid or '-'):<8} {started:<22} {cmd_display}"
        )


def cmd_kill(args: argparse.Namespace) -> None:
    """Handle the 'kill' subcommand."""
    config = load_config(args.config)
    nodes = load_nodes(config)
    name_map = {n.name: n for n in nodes}
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
        if target.dispatch_method == "jupyter":
            node = name_map.get(target.node_name)
            if node and node.jupyter_url:
                try:
                    from utils.jupyter_exec import JupyterExecutor
                    code = f"import os; os.kill({target.pid}, 15)\nprint('ok')\n"
                    with JupyterExecutor(base_url=node.jupyter_url, password=node.jupyter_password) as jup:
                        result = jup.run(code)
                    if "ok" in result.stdout:
                        print(f"Sent kill signal to PID {target.pid} on {target.node_name} ({target.hostname}, via Jupyter)")
                    else:
                        print(f"Warning: unexpected Jupyter kill response: {result.stdout}")
                except Exception as exc:
                    print(f"Warning: Jupyter kill failed: {exc}")
            else:
                print(f"Warning: cannot kill jupyter job — node '{target.node_name}' has no jupyter_url")
        else:
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


def cmd_sync_jupyter(args: argparse.Namespace) -> None:
    """Handle the 'sync-jupyter' subcommand."""
    config_path = Path(args.config) if args.config else _default_config_path()
    summary = sync_jupyter_nodes(config_path, dry_run=args.dry_run)

    allocs = summary["allocations"]
    print(f"Found {len(allocs)} live jupyter_*_<port> allocations from squeue --me")
    for a in allocs:
        print(f"  - {a['hostname']}:{a['port']}  job={a['job_id']}  time_left={a['time_left']}")

    if summary["added"]:
        print(f"\nAdded ({len(summary['added'])}):")
        for n in summary["added"]:
            print(f"  + {n}")
    if summary["removed"]:
        print(f"\nRemoved ({len(summary['removed'])}) — allocation no longer running:")
        for n in summary["removed"]:
            print(f"  - {n}")
    if summary["kept"]:
        print(f"\nRefreshed ({len(summary['kept'])}) — same name, slurm metadata updated:")
        for n in summary["kept"]:
            print(f"  ~ {n}")

    if not summary["added"] and not summary["removed"] and not summary["kept"]:
        print("\nNo squeue-managed Jupyter nodes (none currently allocated).")

    if args.dry_run:
        print(f"\n[dry-run] {config_path} NOT written.")
    else:
        print(f"\nWrote {config_path}")


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
    status_parser = subparsers.add_parser("status", help="Check GPU node health")
    status_parser.add_argument(
        "--ssh",
        action="store_true",
        help="Force SSH probes instead of the default Jupyter probes.",
    )

    # run
    run_parser = subparsers.add_parser("run", help="Dispatch a job to a GPU node")
    run_parser.add_argument("--node", help="Target specific node")
    run_parser.add_argument("--desc", help="Job description")
    run_parser.add_argument("--min-vram", type=float, help="Minimum free VRAM in GB")
    run_parser.add_argument(
        "--ssh",
        action="store_true",
        help=(
            "Dispatch via SSH (legacy). The default is Jupyter, which goes "
            "directly to the GPU node's already-allocated kernel and avoids "
            "the login node entirely."
        ),
    )
    run_parser.add_argument(
        "--jupyter",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated no-op (Jupyter is now the default)
    )
    run_parser.add_argument(
        "--force-concurrent",
        action="store_true",
        help=(
            "Bypass the max_concurrent_jobs busy guard for this dispatch. "
            "Use only when you know the new job won't contend with the "
            "running job (e.g. dispatching a CPU-only side job onto a node "
            "whose current occupant is GPU-bound)."
        ),
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

    # sync-jupyter
    sync_parser = subparsers.add_parser(
        "sync-jupyter",
        help="Reconcile squeue-managed Jupyter node entries in nodes.json",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the diff without writing nodes.json",
    )

    args = parser.parse_args()

    handlers = {
        "status": cmd_status,
        "run": cmd_run,
        "jobs": cmd_jobs,
        "kill": cmd_kill,
        "sync-jupyter": cmd_sync_jupyter,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
