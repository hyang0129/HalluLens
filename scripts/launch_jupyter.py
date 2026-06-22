#!/usr/bin/env python3
"""Guarded launcher for Jupyter Lab SLURM allocations on Empire AI.

This is the ONLY sanctioned way for an agent to start a Jupyter node. It
counts live SLURM jobs *before* submitting and refuses past two hard caps,
so the limits hold regardless of which agent runs it:

    MAX_ACTIVE_JUPYTER (4)  RUNNING jupyter_* allocations
    MAX_TOTAL_JOBS     (6)  all of the user's jobs, any state (incl. PENDING)

By design it has NO cancel/kill path. It submits exactly the approved
sbatch line and nothing else.

Run it ON the Empire AI login node (a SLURM submit host) — it shells out to
`squeue` and `sbatch` locally, so counting and submitting happen in one
process (no cross-SSH race). Typical invocation from an agent:

    ssh empire-ai 'cd ~/LLM_research/HalluLens && python scripts/launch_jupyter.py 8870'
    ssh empire-ai 'cd ~/LLM_research/HalluLens && python scripts/launch_jupyter.py 8870 --dry-run'

Stdlib only. Mirrors the squeue parsing convention in scripts/gpu_dispatch.py.

Exit codes:
    0   submitted (or --dry-run would-submit)
    10  a cap would be exceeded — not submitted
    11  requested port already serves a running jupyter job — not submitted
    12  environment error (squeue/sbatch missing, or squeue failed)
    13  invalid port argument
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --- Policy (hard caps) ------------------------------------------------------
# These encode the autonomy grant: future agents may launch Jupyter nodes
# without asking, but never past these limits. Change them here, in one place.
MAX_ACTIVE_JUPYTER = 4   # RUNNING jupyter_* jobs
MAX_TOTAL_JOBS = 6       # all jobs in any state, including PENDING/queued

# --- Submission shape (matches the approved sbatch line) ---------------------
JUPYTER_SCRIPT = Path.home() / "rit_rc_scripts" / "empire_jupyter_lab.sh"
# Worker variant: same allocation, but also drains a cell-worker queue on launch
# (see scripts/dispatch/jupyter_with_worker.sbatch). Submitted only when
# --dispatch-root is given; the caps below are enforced identically either way.
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = REPO_ROOT / "scripts" / "dispatch" / "jupyter_with_worker.sbatch"
SBATCH_FLAGS = ["--cpus-per-task=16", "--mem-per-cpu=24g", "--time=0-72:00:00", "--qos=rit"]
PORT_MIN, PORT_MAX = 8800, 8899  # 88xx convention (see gpu_dispatch._port_is_88xx)

# A jupyter allocation's name is "jupyter_empire_<port>" once the job's own
# `scontrol update` rename runs (at RUNNING). While PENDING it is still the
# literal "jupyter_empire_PORT" placeholder. Match on the "jupyter_" prefix so
# we never *undercount* running jupyter jobs (errs toward refusing).
_JUPYTER_NAME = re.compile(r"^jupyter_")
# Trailing digits = the live port, used only for the port-collision check.
_JUPYTER_PORT = re.compile(r"_(\d+)$")

SQUEUE_TIMEOUT = 20


def query_jobs():
    """Return one dict {job_id, name, state, nodelist} per `squeue --me` row."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%N"],
            capture_output=True, text=True, timeout=SQUEUE_TIMEOUT,
        )
    except FileNotFoundError:
        sys.exit(_fail(12, "squeue not found in PATH — run this on the Empire AI "
                           "login node (a SLURM submit host), not in the dev container."))
    except subprocess.TimeoutExpired:
        sys.exit(_fail(12, f"squeue timed out after {SQUEUE_TIMEOUT}s."))
    if result.returncode != 0:
        sys.exit(_fail(12, f"squeue failed (exit {result.returncode}): "
                           f"{result.stderr.strip()}"))

    jobs = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        job_id, name, state, nodelist = parts[:4]
        jobs.append({"job_id": job_id, "name": name, "state": state,
                     "nodelist": nodelist})
    return jobs


def running_jupyter_ports(jobs):
    """Ports of currently RUNNING jupyter jobs (for collision detection)."""
    ports = set()
    for j in jobs:
        if j["state"] == "RUNNING" and _JUPYTER_NAME.match(j["name"]):
            m = _JUPYTER_PORT.search(j["name"])
            if m:
                ports.add(int(m.group(1)))
    return ports


def _fail(code, msg):
    print(f"REFUSED: {msg}", file=sys.stderr)
    return code


def main():
    parser = argparse.ArgumentParser(
        description="Guarded Jupyter Lab launcher (enforces job caps; no cancel path).")
    parser.add_argument("port", help=f"Jupyter port ({PORT_MIN}-{PORT_MAX})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the cap checks and print the sbatch command, but do not submit.")
    parser.add_argument("--dispatch-root", default=None,
                        help="If set, submit the jupyter+worker variant: the allocation also "
                             "drains this cell-worker dispatch queue on launch, then keeps "
                             "Jupyter running. Relative paths resolve against the repo root.")
    args = parser.parse_args()

    # Validate the port.
    try:
        port = int(args.port)
    except ValueError:
        sys.exit(_fail(13, f"port must be an integer, got {args.port!r}."))
    if not (PORT_MIN <= port <= PORT_MAX):
        sys.exit(_fail(13, f"port {port} outside the {PORT_MIN}-{PORT_MAX} range."))

    # Choose the allocation script: plain jupyter, or jupyter+worker.
    dispatch_root = None
    if args.dispatch_root is not None:
        dispatch_root = Path(args.dispatch_root)
        if not dispatch_root.is_absolute():
            dispatch_root = (REPO_ROOT / dispatch_root).resolve()
        if not dispatch_root.is_dir():
            sys.exit(_fail(12, f"dispatch root not found: {dispatch_root}"))
        if not WORKER_SCRIPT.is_file():
            sys.exit(_fail(12, f"worker launch script not found: {WORKER_SCRIPT}"))
    elif not JUPYTER_SCRIPT.is_file():
        sys.exit(_fail(12, f"launch script not found: {JUPYTER_SCRIPT}"))

    jobs = query_jobs()
    active_jupyter = sum(
        1 for j in jobs if j["state"] == "RUNNING" and _JUPYTER_NAME.match(j["name"]))
    total = len(jobs)

    print(f"Current state: {active_jupyter} running jupyter job(s) "
          f"(cap {MAX_ACTIVE_JUPYTER}), {total} total job(s) (cap {MAX_TOTAL_JOBS}).")

    # --- Enforce caps ---
    if active_jupyter >= MAX_ACTIVE_JUPYTER:
        sys.exit(_fail(10, f"{active_jupyter} jupyter job(s) already running "
                           f"(cap {MAX_ACTIVE_JUPYTER}). Not submitting."))
    if total >= MAX_TOTAL_JOBS:
        sys.exit(_fail(10, f"{total} job(s) already queued/running "
                           f"(cap {MAX_TOTAL_JOBS}). Not submitting."))
    if port in running_jupyter_ports(jobs):
        sys.exit(_fail(11, f"port {port} is already served by a running jupyter "
                           f"job. Pick another port."))

    if dispatch_root is not None:
        cmd = ["sbatch", *SBATCH_FLAGS, str(WORKER_SCRIPT), str(port),
               str(dispatch_root), str(REPO_ROOT)]
    else:
        cmd = ["sbatch", *SBATCH_FLAGS, str(JUPYTER_SCRIPT), str(port)]
    printable = " ".join(cmd)

    if args.dry_run:
        print(f"[dry-run] would submit: {printable}")
        return 0

    print(f"Submitting: {printable}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.home()))
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
