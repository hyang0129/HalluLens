# GPU Dispatch Status — How to Get a Complete Picture

Use this guide whenever the user asks about node status, GPU utilization, what's running, or dispatch job activity.

## Step 1 — Gather raw data (run in parallel)

**1a. Live SLURM allocations**
```bash
ssh empire-ai 'squeue --me --format="%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R %N"'
```
The job name encodes the port: `jupyter_empire_<port>`. Node is in the last column.

**1b. Dispatch job manifest**
```bash
ssh empire-ai 'cat ~/LLM_research/HalluLens/shared/gpu_jobs.json'
```
Filter `status == "running"`. Key fields: `node_name` (format `alphagpuNN-PPPP`), `command`, `pid`, `description`, `started_at`.

**1c. worker_79 claimed cells**
```bash
ssh empire-ai 'for dir in ~/LLM_research/HalluLens/shared/issue_79_dispatch/claimed/*/; do
  echo "=== $(basename $dir) ==="; ls $dir; done'
```
Directory names encode `node-port_pid_random`. A `.json` file inside = active work item. Heartbeat-only = idle or stale cell.

**1d. Pending cell count**
```bash
ssh empire-ai 'ls ~/LLM_research/HalluLens/shared/issue_79_dispatch/pending/ | wc -l'
```

**1e. Per-node GPU utilization + processes**
```bash
ssh empire-ai 'for node in <node list>; do
  echo "=== $node ==="
  ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no $node \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
     ps -u hyang1 -o pid,etime,args --no-headers | grep -vE \"jupyter|ipykernel|grep|ps\" | grep python | head -8" 2>/dev/null || echo "SSH blocked"
done'
```
Use the node list from step 1a. Some nodes will be SSH-blocked from the login node due to host key rotation — infer their activity from claimed cells (1c) and GPU utilization numbers.

## Step 2 — Correlate

| Signal | Meaning |
|--------|---------|
| `gpu_jobs.json` entry with `command = bash scripts/dispatch/worker_79.sh` | **worker_79 cell-driven** — find its claimed cell in 1c for the current work item |
| `gpu_jobs.json` entry with any other script | **Direct dispatch** (non-cell) |
| SLURM allocation with no matching running `gpu_jobs.json` entry | **Idle jupyter** |
| Claimed cell directory with only `heartbeat`, no `.json` | Stale / worker crashed between cells |

Cross-reference node_name from gpu_jobs.json with the SLURM job list. The port in the SLURM job name (`jupyter_empire_<port>`) matches the port in `node_name`.

## Step 3 — Output format

Produce three sections:

### worker_79 cells
| Node:Port | SLURM JOBID | Uptime | Claimed Cell | Work Item |
|-----------|-------------|--------|--------------|-----------|
| `alphagpuNN:PPPP` | ... | ... | `node-port_pid` → `dataset__model__seed.json` | eval_transfer or run_experiment — dataset/model/seed |

### Direct dispatch (non-cell scripts)
| Node:Port | SLURM JOBID | Uptime | Job ID | Script | Runtime |
|-----------|-------------|--------|--------|--------|---------|

### Idle (jupyter only)
| Node:Port | SLURM JOBID | Uptime | Note |
|-----------|-------------|--------|------|

Footer line: pending cell count and total active worker count.
