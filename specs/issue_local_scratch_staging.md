# perf: stage memmap caches to per-node local NVMe to remove NFS contention

## Problem

When 8 training runs launch concurrently against memmap caches under `shared/*/activations.zarr/_memmap_cache/<fp>/`, the underlying NFS server (`10.0.4.155:/home`) saturates. The symptom is twofold:

1. **Training I/O slows** — every page fault on the mmap'd cache files goes over NFS; under contention this caps the data-loader throughput well below what local NVMe could deliver.
2. **Login-node experience degrades for everyone sharing `/mnt/home`** — measured 2026-05-18 on `alpha1.empire-ai.org`: `ls /mnt/home/hyang1/.ssh/authorized_keys` took **1000 ms**, `stat /mnt/home/hyang1` took **400 ms**, and two consecutive `date +%s.%N` invocations were 400–500 ms apart. SSH session-open took ~32 s for a trivial `ssh empire-ai echo OK`. See [`specs/empire_ai_ssh_slowdown.md`](empire_ai_ssh_slowdown.md) for the full diagnostic trail (including the disproof of the earlier reverse-DNS hypothesis).

The diagnosis is **two stacked causes**: contended `/mnt/home` NFS (driven by our own jobs) layered on top of an already-busy `alpha1` login node (37 active sessions, 8 GiB swap in use). NFS is the part we can actually do something about.

## Proposed fix

Stage the canonical memmap cache for each `(dataset × model × fingerprint)` to per-node local NVMe **before** the training process starts, then point the trainer at the local copy. The cache is read-mostly during training, so a one-time rsync amortizes well over the training duration.

## Evidence the design works

### Per-node scratch is consistent across the cluster

Probed 4 currently-allocated nodes (alphagpu01/03/07/17) on 2026-05-18 by piping an inspector script through `ssh empire-ai → ssh $NODE bash -s`. Every node returned the identical layout:

| Resource | Per node | Notes |
|---|---|---|
| CPUs | 96 | |
| RAM | 2016 GB | |
| `/local`, `/tmp`, `/var/tmp` (shared root XFS on `nvme0n1p3`) | ~878 GB total, **~837 GB free** | Sticky perm 1777 — world-writable but per-uid files survive |
| `/dev/shm` (tmpfs) | 1008 GB total, ~1006 GB free | RAM-backed — fastest but consumes node RAM |
| Second NVMe `nvme1n1` (Samsung 894 GB) | unmounted | Available if we ask Empire AI to mount it — would double effective scratch |
| `dd` write throughput to `/local` (100 MB, fdatasync) | 890 MB/s – 1.2 GB/s | Healthy NVMe |
| `TMPDIR` | unset | No SLURM-imposed per-job tmpdir scrubbing observed |

Raw sweep output is in [`/tmp/ssh_diag/gpu_sweep.log`](/tmp/ssh_diag/gpu_sweep.log) on the workstation.

Two nodes (alphagpu19, alphagpu23) failed with a host-key mismatch — `~/.ssh/known_hosts` lines 11 and 13 are stale. **Tracked separately**: re-add the current host keys with `ssh-keygen -R` + a fresh connect.

### Cache sizes fit

For the canonical params (layers 14–29, `pad_length=63`, `include_logprobs=True`, `top_k=20`, fp16), `activations.npy` dominates. Estimated sizes by train-split N:

| Dataset | Train N | `activations.npy` | Fits in 837 GB `/local`? |
|---|---|---|---|
| HotpotQA train | 90 K | ~740 GB | Tight (≈97 GB margin) |
| NQ / PopQA / SciQ / SearchQA / MMLU train | 10–20 K | 80–160 GB | Comfortable |
| Test splits (any dataset) | ~1 K | <10 GB | Trivial |

So one HotpotQA-scale cache fits per node; smaller datasets fit several at a time. `max_concurrent_jobs: 1` per node in [`configs/nodes.json`](../configs/nodes.json) means we only ever need one cache hot per node.

## Design constraints discovered

### Cache fingerprint includes the zarr's absolute path

[`activation_logging/activation_parser.py:1173`](../activation_logging/activation_parser.py#L1173):

```python
zarr_path_resolved = str(Path(self.activations_path).resolve())
key_parts: list = [zarr_path_resolved, ...]
```

A naive rsync of `_memmap_cache/<fp>/` from `/mnt/home/.../activations.zarr/` to `/local/.../activations.zarr/` will **not** be picked up by the consumer — the consumer computes a different fingerprint against the new path and misses the cache. Two acceptable resolutions:

- **Pattern A — caller-side path swap** (preferred, no parser changes): the dispatcher rsyncs `shared/<ds>/activations.zarr/` → `/local/cache/<ds>/activations.zarr/` and passes the local path as `--activations-path` to the trainer. The trainer recomputes the fingerprint against the local path and finds its cache there.
- **Pattern B — env-var override** (~10 LoC): add `HALLULENS_CACHE_DIR` that overrides `_memmap_cache_dir(fp)`. Keeps the trainer CLI unchanged but decouples cache location from path-based fingerprinting.

## Implementation sketch

Hook into `scripts/gpu_dispatch.py run` so every training launch flows through this:

```python
def stage_cache_if_room(zarr_path: Path, fingerprint: str, node: str) -> Path:
    """Return the path to use as --activations-path (local if staged, NFS if not)."""
    src = zarr_path                                  # /mnt/home/.../activations.zarr
    dst = Path("/local/hyang1_cache") / zarr_path.name  # /local/hyang1_cache/<ds>.zarr

    src_size = remote_du(node, src / "_memmap_cache" / fingerprint)
    free     = remote_free(node, "/local")
    if src_size > 0.7 * free:
        log.info("skip stage: cache=%s GB, /local free=%s GB", src_size, free)
        return src

    # idempotent: rsync only changed files
    remote_rsync(node, src, dst)
    return dst
```

Subsequent runs on the same node hit a no-op rsync. A `--no-stage` escape hatch falls back to direct NFS for debugging.

## Tasks

- [ ] Confirm with Empire AI whether `/local` persists across SLURM job boundaries on the same physical node (if not, we pay the rsync penalty per allocation, not per launch — still likely worth it for >1 h jobs).
- [ ] Ask Empire AI whether `nvme1n1` (the dark second NVMe) can be mounted as `/local2` for users — doubles scratch capacity, eliminates the "HotpotQA cache leaves only 97 GB margin" issue.
- [ ] Implement Pattern A in `scripts/gpu_dispatch.py`:
  - [ ] Helper `stage_cache_if_room()` with `src_size > 0.7 * free` guard
  - [ ] `--stage / --no-stage` CLI flag (default: stage)
  - [ ] Use rsync with `--info=progress2` and partial-resume so a killed/preempted job doesn't have to start the copy from scratch
- [ ] Add stage timing to `gpu_dispatch.py jobs` output so we can see actual rsync cost vs. training cost.
- [ ] (Optional) Pattern B as a follow-up if multiple consumers of `ActivationParser` outside the dispatcher path need staging without CLI changes.
- [ ] Re-measure: re-run [intra-session NFS probe](empire_ai_ssh_slowdown.md) during a multi-job training burst with staging enabled. Target: `ls /mnt/home/hyang1/.ssh/authorized_keys` back under 100 ms.

## Out of scope

- Fixing `alpha1`'s independent CPU/memory/swap pressure (37 active sessions, 8 GiB swap). That requires Empire AI admin action — separate ticket.
- A general-purpose SSH ControlMaster workaround on Git Bash. Confirmed not viable on this Windows client; see memory `feedback_ssh_controlmaster.md`.

## Cross-references

- [`specs/empire_ai_ssh_slowdown.md`](empire_ai_ssh_slowdown.md) — the diagnostic chain that landed us on NFS as the load-bearing cause
- [`scripts/build_memmap_cache.py`](../scripts/build_memmap_cache.py) — the existing prebuild helper this proposal complements
- [`activation_logging/activation_parser.py:1157-1195`](../activation_logging/activation_parser.py#L1157-L1195) — fingerprint and cache-dir definitions
- [`configs/nodes.json`](../configs/nodes.json) — registered GPU nodes (all 96-core / 2016 GB / 837 GB-local boxes per this sweep)
