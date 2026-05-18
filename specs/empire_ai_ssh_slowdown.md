# Empire AI SSH Slowdown Investigation
_Date: 2026-05-18 (updated with second diagnostic round)_

## Summary

SSH connections to `alpha1.empire-ai.org` **intermittently** stall for ~3 minutes despite 8ms network RTT. A second diagnostic battery (4 parallel `ssh -vvv` probes with per-line wall-clock timestamps) on 2026-05-18 07:21 ET found **no stall at all** — every probe completed in 25–45s. The slowdown is therefore **load- or state-dependent on the server**, not a constant misconfiguration. The original spec attributed the stall to a server-side reverse DNS lookup right after `SSH2_MSG_SERVICE_ACCEPT`; the new evidence **refutes** that placement and points instead to **slow server-side processing during the publickey auth step itself** (most plausibly an NFS read of `~/.ssh/authorized_keys`, or a stalled PAM session-open / `pam_systemd`).

---

## Measurements

| Test | Time | Result |
|------|------|--------|
| `ping alpha1.empire-ai.org` (5 packets) | 8–9ms avg | Normal |
| `ssh ... echo ok` (default options) | 3m 26s | Succeeded |
| `ssh -o GSSAPIAuthentication=no ... echo ok` | 3m 20s | Succeeded |
| `ssh -v ... echo ok` (verbose timing) | 2m 55s | Stalled at auth phase |

---

## Original Hypothesis (Refuted): Server-Side Reverse DNS Lookup

The first diagnostic round (single `ssh -v` run, no per-line timestamps) attributed the stall to `UseDNS yes` on the server, on the grounds that the 3-minute silence sat "right after `SSH2_MSG_SERVICE_ACCEPT received`". The second diagnostic round invalidates this:

- A probe with `PreferredAuthentications=none` — which forces the client to send a `userauth-request` with method `none` and then disconnect — completed in **25 seconds total**, with `SSH2_MSG_SERVICE_ACCEPT` arriving at the 23-second mark and the rejection arriving 2 seconds later. If the reverse-DNS stall existed on every connection, it would have hit here too.
- Even in the original verbose trace, the silence sat *between* `SERVICE_ACCEPT` and the next observable client→server message, which is the publickey `userauth-request`. The server only reads `~/.ssh/authorized_keys` and does NSS lookups **after** receiving that publickey request — so a stall in that window points at the auth pipeline, not at the connection accept path where `UseDNS` would fire.
- A real `UseDNS` stall manifests before the server sends its banner / first KEX message, not three round trips into the protocol.

So reverse DNS is unlikely to be the cause.

---

## Revised Root-Cause Hypothesis: Slow Server-Side Publickey Verification

The intermittent 3-minute stall most likely sits inside the server's handling of the `publickey` `userauth-request`, which involves:

1. **NSS lookup** of `hyang1` (UID, home dir, shell) via SSSD — fast in the probes here, so probably not the bottleneck.
2. **NFS read** of `/mnt/home/hyang1/.ssh/authorized_keys`. The home directory is on a shared filesystem (`/mnt/home/`). If the NFS server is under load, has a stale handle, or is in the middle of a `umount`/automount cycle, this read can block for the NFS default timeout (typically 60–600s). Stacking on top of TCP and `nfs` retransmits, a 3-minute total is consistent.
3. **`pam_systemd` session-open via D-Bus to `systemd-logind`.** On a node with 37 active sessions and 8 GiB swap-in-use, `logind` can be slow or temporarily wedged; D-Bus calls default to a 25-second timeout but PAM stacks may retry.
4. **`pam_mkhomedir` / quota / automount probe** on first login per boot — usually one-shot but can block when the underlying FS is unhealthy.

The `prefauth_none` probe (25s clean exit, no auth attempt) shows that **the protocol-negotiation and NSS-lookup paths are healthy when measured**; whatever fires for 3 minutes happens specifically when `publickey` auth or session-open is initiated, and only on certain server states we did not catch in this round.

### How to confirm next time the stall reproduces

When `ssh empire-ai` next hangs, in a second terminal run:

```bash
# 1. Timestamped trace of the hanging session — repeat the connect attempt
ssh -vvv -o BatchMode=yes -i ~/.ssh/empireai-openssh \
    hyang1@alpha1.empire-ai.org echo OK 2>&1 \
  | while IFS= read -r line; do printf '[%s] %s\n' "$(date +%T.%3N)" "$line"; done \
  | tee /tmp/ssh_slow.log
```

Note the exact wall-clock gap. The line **immediately preceding the gap** identifies the phase: if the gap follows `Offering public key:` it's authorized_keys read; if it follows `Authenticated to ...` it's PAM session-open; if it follows `Entering interactive session` it's shell init / motd / NFS home traversal.

If a fast second login is possible during the stall (e.g. open a second terminal and connect successfully), run on the server:

```bash
# Identify the stuck sshd worker
ps -ef | grep sshd | grep hyang1
# Then for that PID:
sudo strace -p <PID> -tt -e trace=network,file,read,write 2>&1 | head -200
sudo journalctl -u sshd -u systemd-logind --since '5 min ago'
```

A real reverse-DNS stall would show repeated `recvfrom`/`sendto` on UDP port 53 to a nameserver. A real NFS stall would show blocked `read()`/`getdents()` on the `/mnt/home` fd. A real `pam_systemd` stall would show blocked `recvmsg()` on a D-Bus AF_UNIX socket. These three look entirely different under `strace`.

## Contributing Factor: Congested Login Node

```
 01:49:26 up 55 days, 5:12, 37 users, load average: 2.21, 3.91, 3.78
Swap:  15Gi total,  8.3Gi used,  7.3Gi free
```

- **37 simultaneous sessions**, many of them zombie/stale (sessions from 2026-03-24, 2026-04-16, 2026-04-20, etc.)
- Swap usage is significant (55% of 15 GiB swap in use)
- This slows PAM and shell startup after the DNS delay resolves

---

## Secondary Issue: Broken SSH Config Include

`~/.ssh/config` contains:

```
Include C:\Users\HongM\projects\HalluLens\.ssh\config
```

Git Bash's `ssh` binary uses POSIX paths. The Windows backslash path is silently skipped:

```
debug1: /c/Users/HongM/.ssh/config line 2: include ~/.ssh/C:\\Users\\HongM\\projects\\HalluLens\\.ssh\\config matched no files
```

This means the `empire-ai` Host block (defined in `.ssh/config`) is **invisible** when using Git Bash `ssh`, forcing the use of explicit `-i key -l user host` flags instead of `ssh empire-ai`.

---

## Diagnostic Battery — 2026-05-18 07:21 ET

Four parallel `ssh -vvv` probes, each with per-line wall-clock timestamps from a bash `while read` wrapper. All four were issued within the same minute and went to the same login node IP (67.99.173.2):

| Probe | Total | KEX done | SERVICE_ACCEPT | Auth done | Command sent | Exit | Result |
|------|------|----------|---------------|-----------|--------------|------|--------|
| A: baseline (full auth) | 39s | 07:21:15 (~14s in) | 07:21:21 | 07:21:26 | 07:21:33 | 07:21:40 | Clean, no stall |
| B: `PreferredAuthentications=none` | 25s | 07:21:19 | 07:21:25 | n/a (rejected at 07:21:26) | n/a | 07:21:27 | Clean reject |
| C: pubkey-only, no GSS/PAM/password | 40s | 07:21:25 | 07:21:29 | 07:21:32 | 07:21:38 | 07:21:44 | Clean, no stall |
| D: connect by IP literal (skip client DNS) | 37s | 07:21:30 | 07:21:34 | 07:21:37 | 07:21:42 | 07:21:46 | Clean, no stall |

Logs preserved at `/tmp/ssh_diag/*.log` on the local machine for the duration of the bash session.

**Key observations:**
- All four probes succeeded with no multi-minute gap anywhere — the stall **does not reproduce** at this time of day.
- The slowest segment in every probe is the 5–7s gap between `Authenticated to ...` and `Sending command: echo OK`, which is PAM session-open + login shell init. That's elevated but not the headline issue.
- Probe B (`PreferredAuthentications=none`) being the **fastest** at 25s — and finishing cleanly *without* invoking publickey verification — is the strongest piece of evidence against the reverse-DNS hypothesis: if reverse-DNS were the cause, B would have stalled identically to A.

## Fixes

### Fix 1: File a more specific Empire AI support ticket

Do **not** ask for `UseDNS no` alone — that was the wrong diagnosis. Instead, ask Empire AI support to investigate:

1. **NFS health on `/mnt/home`** during 3-minute SSH stalls. Specifically request `nfsiostat` / `nfsstat` samples from `alpha1.empire-ai.org` at moments when users report slow logins, and `dmesg` for `nfs: server ... not responding` lines.
2. **`systemd-logind` health** under high session count. Check whether the unit is degraded or whether D-Bus call latency to `logind` spikes when ~40 users are logged in.
3. **Stale session cleanup.** The node shows sessions from 2026-03-24 onward (55-day uptime, 37 users). Ask whether `loginctl terminate-user` of idle sessions or a tighter `IdleAction=` in `logind.conf` could be applied.
4. Optionally, `UseDNS no` is still a worthwhile hardening change but is no longer the load-bearing ask.

A reproduction trace from the "How to confirm" section above is the single most useful artifact to attach to the ticket.

### Fix 2: Client-Side Mitigation — SSH ControlMaster Multiplexing

> **2026-05-18 — Tested and does not work on Git Bash for Windows.** The master socket file is created at `~/.ssh/cm-%C` (the `:` in the original `%h:%p` template is invalid on NTFS), but every client attempt fails with `mux_client_request_session: read from master failed: Connection reset by peer`. Root cause: Git Bash's OpenSSH uses MSYS2/Cygwin pseudo-sockets, which are not real AF_UNIX sockets and do not interoperate with OpenSSH's multiplexing protocol on this platform. Reverted. Only viable on a real POSIX client (WSL, Linux, macOS) or a Windows-native ssh that supports named-pipe ControlPath.

Once connected, reuse the authenticated socket for subsequent connections (avoids paying the ~3min penalty every time):

Add to `.ssh/config` (project config at `HalluLens/.ssh/config`):

```ssh-config
Host empire-ai
    HostName alpha1.empire-ai.org
    User hyang1
    IdentityFile C:\Users\HongM\.ssh\empireai-openssh
    ForwardAgent yes
    LocalForward 8889 alphagpu18:8889
    # Multiplexing — pay the slow auth only once per session
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

After the first connection, subsequent `ssh empire-ai` calls return in <1s.

### Fix 3: Fix the SSH Config Include Path

Change `~/.ssh/config` to use a POSIX path so Git Bash resolves it:

```diff
-Include C:\Users\HongM\projects\HalluLens\.ssh\config
+Include /c/Users/HongM/projects/HalluLens/.ssh/config
```

---

## Current SSH Config Status

| Host | Defined In | Reachable From Git Bash |
|------|-----------|------------------------|
| `empire-ai` | `HalluLens/.ssh/config` | No (Include broken) |
| `empire-ai-gpu` | `HalluLens/.ssh/config` | No (Include broken) |
| `rit-submit` | `HalluLens/.ssh/config` | No (Include broken) |
| `rit-gpu` | `HalluLens/.ssh/config` | No (Include broken) |
| `rit` | `~/.ssh/config` (inline) | Yes |

---

## Recommended Immediate Actions

1. **Apply Fix 3** (POSIX Include path) — done in current branch (`.ssh/config` diff).
2. **Reproduce-when-slow workflow** — keep the timestamped `ssh -vvv` one-liner from "How to confirm next time the stall reproduces" handy; the next 3-minute hang is the data we actually need.
3. **File a refined Empire AI ticket** asking about NFS-home and `systemd-logind` behavior under load (Fix 1, revised).
4. **Defer the ControlMaster workaround** — it cannot be made to work on Git Bash for Windows (see 2026-05-18 note inside the old Fix 2 block). The right path is either WSL or Windows-native OpenSSH with a named-pipe `ControlPath`.

## What Changed in This Round

- The "always 3 minutes" framing was wrong; the slowdown is intermittent.
- The "reverse-DNS at `SERVICE_ACCEPT`" placement was wrong; the stall is later in the auth pipeline.
- The most likely real cause is server-side: NFS read of `authorized_keys` and/or PAM session-open under contention.
- The `prefauth_none` probe is the cheapest reproducer to keep in the toolbox — it isolates whether the protocol stack itself is healthy in <30s.

