"""
Persistent SSH shell to the Empire AI login node.

Empire AI's login node spends ~60s on session setup for each new SSH
channel (PAM / NFS home-mount / MOTD overhead). ControlMaster does not
help because the cost is per-session, not per-connection. This module
keeps ONE interactive shell open in the background and feeds commands
to it — the same trick MobaXterm uses to stay responsive.

After the first command pays the ~60s warm-up, every subsequent call is
just (network RTT + real command time).

Usage:
    # One-shot CLI
    python utils/empire_shell.py 'squeue --me'

    # Programmatic
    from utils.empire_shell import run
    res = run('squeue --me')
    print(res.stdout)
    print(res.exit_code)

    # Daemon control
    python utils/empire_shell.py --status
    python utils/empire_shell.py --kill
    python utils/empire_shell.py --logs        # tail daemon log

Limitations:
    - Each call runs in the SAME shell, so `cd` persists between calls.
      (If you want isolation, chain commands with `&&` or wrap in `bash -c`.)
    - Commands are serialized (one at a time). Concurrent callers queue.
    - The daemon shuts itself down after IDLE_TIMEOUT (default 4h) of
      inactivity. The next call eats the 60s warm-up once.

Periodic reap:
    Empire's user.slice has TasksMax=512. Stale orphaned shells from prior
    SSH sessions accumulate and starve fork(). Every REAP_INTERVAL the
    daemon runs a small `ps + kill` script in its own SSH session to clean
    up anything of yours older than REAP_MAX_AGE that isn't in the session
    of an active sshd (your live shells, including this daemon's, are
    safe) and isn't on the allowlist (sshd, ssh-agent, gpg-agent,
    systemd*, tmux*, screen*). See [reap] entries in the daemon log.

    Bootstrap caveat: if your user.slice is *already* at 512/512, the
    daemon can't even ssh in to launch. One-time manual cleanup needed
    first (e.g. ssh and kill orphans with PPID=1). After that the daemon
    keeps the count low.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "default"
SOCKET_PATH = Path(f"/tmp/empire-shell-{USER}.sock")
PID_PATH = Path(f"/tmp/empire-shell-{USER}.pid")
LOG_PATH = Path(f"/tmp/empire-shell-{USER}.log")

SSH_HOST = "empire-ai"
DEFAULT_TIMEOUT = 120
IDLE_TIMEOUT = 4 * 3600          # daemon self-exits after this many seconds idle
STARTUP_TIMEOUT = 360            # how long to wait for the initial ssh login
                                 # — Empire's login node frequently runs 2-3 min
SOCKET_BACKLOG = 8

# Periodic reap: the user.slice on alpha1 has TasksMax=512. Orphaned
# bashes/ssh helpers from prior sessions accumulate and starve fork().
# Every REAP_INTERVAL the daemon runs a small shell script in its own
# warm SSH session to kill anything of ours older than REAP_MAX_AGE,
# excluding an allowlist of intentional long-runners and anything in
# the session of an active sshd (= our live shells, including this one).
REAP_INTERVAL = 300              # seconds between reaps
REAP_MAX_AGE = 1800              # kill non-allowlisted procs older than this
_REAP_SCRIPT = """\
__MAX_AGE__={max_age}
__ME__=$(id -un)
__ACTIVE__=$(ps -u "$__ME__" -o sid=,comm= | awk '$2 ~ /sshd/ {{print $1}}' | sort -u | tr '\\n' ' ')
__TS__=$(date +%FT%T)
ps -u "$__ME__" -o pid=,sid=,etimes=,comm= | while read -r __p __s __a __c; do
  [ "$__a" -lt "$__MAX_AGE__" ] && continue
  case "$__c" in sshd|ssh-agent|gpg-agent|systemd*|tmux*|screen*) continue ;; esac
  case " $__ACTIVE__ " in *" $__s "*) continue ;; esac
  echo "$__TS__ reap pid=$__p sid=$__s age=${{__a}}s cmd=$__c"
  kill "$__p" 2>/dev/null
done
"""

# Strip ANSI color/control sequences from PTY output.
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


@dataclass
class Result:
    stdout: str
    exit_code: int
    duration_ms: int

    def __str__(self) -> str:
        return self.stdout


# ----------------------------- client side -----------------------------

def run(cmd: str, timeout: int = DEFAULT_TIMEOUT) -> Result:
    """Run a shell command on the Empire AI login node. Auto-starts daemon."""
    cold_start = False
    if not SOCKET_PATH.exists():
        _spawn_daemon()
        _wait_for_socket(timeout=30)
        cold_start = True

    # First call after a cold start has to wait for ssh login (~60s) before
    # the daemon will accept. Give it enough headroom for that PLUS the
    # command's own timeout.
    read_timeout = timeout + (STARTUP_TIMEOUT if cold_start else 30)

    req = json.dumps({"cmd": cmd, "timeout": timeout}).encode() + b"\n"
    try:
        resp = _send(req, read_timeout=read_timeout)
    except (ConnectionRefusedError, FileNotFoundError):
        # Stale socket — daemon died. Clean up and retry once.
        SOCKET_PATH.unlink(missing_ok=True)
        _spawn_daemon()
        _wait_for_socket(timeout=30)
        resp = _send(req, read_timeout=timeout + STARTUP_TIMEOUT)

    data = json.loads(resp.decode())
    if "error" in data:
        raise RuntimeError(f"empire-shell: {data['error']}")
    return Result(
        stdout=data["stdout"],
        exit_code=data["exit"],
        duration_ms=data.get("duration_ms", 0),
    )


def _send(req: bytes, read_timeout: float) -> bytes:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(read_timeout)
    s.connect(str(SOCKET_PATH))
    s.sendall(req)
    chunks = []
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if chunk.endswith(b"\n"):
            break
    s.close()
    return b"".join(chunks)


def _wait_for_socket(timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if SOCKET_PATH.exists():
            # Probe — pexpect-side may still be opening it.
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect(str(SOCKET_PATH))
                s.close()
                return
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass
        time.sleep(0.2)
    raise RuntimeError("daemon did not come up within timeout")


def _spawn_daemon() -> None:
    """Double-fork a detached daemon process."""
    if os.fork() != 0:
        return  # parent returns immediately

    # First child
    os.setsid()
    if os.fork() != 0:
        os._exit(0)

    # Grandchild: become the daemon
    os.chdir("/")
    sys.stdin = open("/dev/null")
    sys.stdout = open(LOG_PATH, "a", buffering=1)
    sys.stderr = sys.stdout

    try:
        _run_daemon()
    except Exception as e:
        print(f"[{time.strftime('%F %T')}] daemon crashed: {e!r}", flush=True)
    finally:
        SOCKET_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)
        os._exit(0)


# ----------------------------- daemon side -----------------------------

def _run_daemon() -> None:
    import pexpect  # imported here so client paths don't require it

    PID_PATH.write_text(str(os.getpid()))
    print(f"[{time.strftime('%F %T')}] daemon starting pid={os.getpid()}", flush=True)

    # Bind unix socket FIRST so clients can connect immediately. The shell
    # init takes 60s+ — we don't want clients giving up before the socket
    # even appears.
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCKET_PATH))
    srv.listen(SOCKET_BACKLOG)
    os.chmod(SOCKET_PATH, 0o600)
    srv.settimeout(60.0)  # wake up periodically to check idle

    # Now run the slow ssh login. Any client that connects in the meantime
    # will block on recv() until we start accepting.
    shell = _open_shell(pexpect)
    print(f"[{time.strftime('%F %T')}] shell ready", flush=True)

    lock = threading.Lock()
    last_activity = time.time()
    last_reap = time.time()

    while True:
        # Idle check
        if time.time() - last_activity > IDLE_TIMEOUT:
            print(f"[{time.strftime('%F %T')}] idle timeout — exiting", flush=True)
            break

        # Health check — if ssh died, exit so next client gets a fresh daemon
        if not shell.isalive():
            print(f"[{time.strftime('%F %T')}] ssh shell died — exiting", flush=True)
            break

        # Periodic reap of stale processes (see REAP_* constants).
        if time.time() - last_reap > REAP_INTERVAL:
            _reap(shell, lock, pexpect)
            last_reap = time.time()

        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue

        last_activity = time.time()
        # Handle inline (commands are serialized anyway — no need for threads)
        try:
            _handle_client(conn, shell, lock, pexpect)
        except Exception as e:
            print(f"[{time.strftime('%F %T')}] handler error: {e!r}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        shell.sendline("exit")
        shell.close(force=True)
    except Exception:
        pass


def _open_shell(pexpect):
    """Spawn ssh, wait for it to be ready, neutralize prompt + echo.

    We exec `bash --norc --noprofile` deliberately: Empire's user dotfiles
    fork-bomb on the login node when it's busy, leaving the session in a
    half-working state ("bash: fork: Resource temporarily unavailable"
    bursts during `.bashrc`). Skipping rc/profile sidesteps that entirely
    — at the cost of losing user aliases/functions, which we don't need
    for orchestration commands (git, gh, squeue, gpu_dispatch.py).
    """
    # -tt forces a PTY on the remote so bash runs in line-buffered mode.
    # -o ClearAllForwardings=yes is critical: the user's ssh config has
    # LocalForward 8889 → alphagpu18, already bound by the existing
    # ControlMaster. Without this, the new client tries to re-bind 8889,
    # fails, then disables multiplexing entirely — meaning we pay full
    # session setup cost on every spawn.
    shell = pexpect.spawn(
        f"ssh -tt -o ClearAllForwardings=yes {SSH_HOST} "
        "bash --norc --noprofile",
        encoding=None,           # raw bytes
        timeout=STARTUP_TIMEOUT,
        echo=False,
        dimensions=(40, 200),
    )
    # When EMPIRE_SHELL_TRACE=1, mirror everything read from the PTY into
    # /tmp/empire-trace-$USER.log for debugging.
    if os.environ.get("EMPIRE_SHELL_TRACE"):
        trace_path = Path(f"/tmp/empire-trace-{USER}.log")
        shell.logfile_read = open(trace_path, "ab")
        print(f"[{time.strftime('%F %T')}] tracing to {trace_path}", flush=True)

    # First-touch: disable remote PTY echo, mute prompts, and emit a boot
    # marker. We send the setup commands together with the marker so the
    # first thing we have to wait for is a string we know we asked for.
    boot = uuid.uuid4().hex[:12]
    boot_marker = f"EMPIRE_BOOT_{boot}_OK".encode()
    setup = (
        b'stty -echo -onlcr 2>/dev/null; '
        b'PS1=""; PS2=""; PROMPT_COMMAND=""; export TERM=dumb; '
        b'echo ' + boot_marker
    )
    shell.sendline(setup)
    try:
        shell.expect(boot_marker, timeout=STARTUP_TIMEOUT)
    except pexpect.TIMEOUT:
        tail = _decode((shell.before or b"") + (shell.after or b""))[-2000:]
        print(f"[{time.strftime('%F %T')}] BOOT timeout — last 2KB of buffer:\n{tail}",
              flush=True)
        raise
    # Drain a possible echoed copy of the marker (if PTY echo was on when
    # the input line was received).
    try:
        shell.expect(boot_marker, timeout=2)
    except pexpect.TIMEOUT:
        pass

    return shell


def _handle_client(conn, shell, lock, pexpect):
    data = b""
    conn.settimeout(10.0)
    while not data.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk

    # Probe connections (e.g. from _wait_for_socket) send nothing and close.
    # Don't treat that as an error — just drop silently.
    if not data:
        return

    try:
        req = json.loads(data.decode())
    except json.JSONDecodeError as e:
        try:
            conn.sendall((json.dumps({"error": f"bad request: {e}"}) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
        return

    cmd = req.get("cmd", "")
    timeout = int(req.get("timeout", DEFAULT_TIMEOUT))

    with lock:
        start = time.time()
        try:
            stdout, exit_code = _exec(shell, cmd, timeout, pexpect)
            resp = {
                "stdout": stdout,
                "exit": exit_code,
                "duration_ms": int((time.time() - start) * 1000),
            }
        except _Timeout as e:
            tail = e.partial[-500:] if e.partial else ""
            msg = f"command timed out after {timeout}s"
            if tail:
                msg += f" (last 500B of stdout: {tail!r})"
            resp = {"error": msg}
        except pexpect.EOF:
            tail = ""
            try:
                tail = _decode((shell.before or b"") + (shell.after or b""))[-500:]
            except Exception:
                pass
            print(f"[{time.strftime('%F %T')}] EOF on shell — "
                  f"isalive={shell.isalive()} exitstatus={shell.exitstatus} "
                  f"tail={tail!r}", flush=True)
            resp = {"error": "ssh shell died (see daemon log for details)"}

    try:
        conn.sendall((json.dumps(resp) + "\n").encode())
    except (BrokenPipeError, ConnectionResetError):
        pass  # client already gave up; we still finished the work


class _Timeout(Exception):
    def __init__(self, partial: str):
        self.partial = partial


def _exec(shell, cmd: str, timeout: int, pexpect) -> tuple[str, int]:
    """Run a command in the held-open shell and return (stdout, exit_code)."""
    nonce = uuid.uuid4().hex[:16]
    sentinel = f"<<<EMPIRE_DONE_{nonce}:".encode()
    pattern = re.compile(re.escape(sentinel) + rb"(\d+):END>>>")

    # Send the user's command verbatim — supports multi-line by relying on
    # pexpect.sendline to emit one line at a time.
    for line in cmd.split("\n"):
        shell.sendline(line.encode())
    # Capture last exit code, then emit sentinel.
    shell.sendline(b"__rc=$?; printf '%s%d:END>>>\\n' '<<<EMPIRE_DONE_" +
                   nonce.encode() + b"' \"$__rc\"")

    try:
        shell.expect(pattern, timeout=timeout)
    except pexpect.TIMEOUT:
        # Interrupt the running command. The queued sentinel printf will
        # then run, giving us a recovery point.
        partial = _decode(shell.before or b"")
        shell.sendcontrol("c")
        try:
            shell.expect(pattern, timeout=10)
        except pexpect.TIMEOUT:
            # Couldn't recover — bubble up so the daemon shuts down.
            raise pexpect.EOF("shell unresponsive after Ctrl-C")
        raise _Timeout(partial)

    exit_code = int(shell.match.group(1))
    text = _decode(shell.before or b"")
    if text.endswith("\n"):
        text = text[:-1]
    return text, exit_code


def _decode(raw: bytes) -> str:
    clean = _ANSI_RE.sub(b"", raw)
    return clean.decode("utf-8", errors="replace")


def _reap(shell, lock, pexpect) -> None:
    """Run the periodic reap script in the held-open SSH session."""
    script = _REAP_SCRIPT.format(max_age=REAP_MAX_AGE)
    try:
        with lock:
            stdout, exit_code = _exec(shell, script, timeout=30, pexpect=pexpect)
    except (_Timeout, pexpect.EOF) as e:
        print(f"[{time.strftime('%F %T')}] reap aborted: {type(e).__name__}",
              flush=True)
        return
    if stdout.strip():
        # The script echoes one line per reap target.
        for line in stdout.strip().splitlines():
            print(f"[reap] {line}", flush=True)


# ----------------------------- CLI ------------------------------------

def _read_pid() -> Optional[int]:
    try:
        return int(PID_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _cli_status() -> int:
    pid = _read_pid()
    if pid and _is_alive(pid):
        print(f"daemon running: pid={pid}")
        print(f"  socket: {SOCKET_PATH}")
        print(f"  log:    {LOG_PATH}")
        return 0
    print("daemon not running")
    return 1


def _cli_kill() -> int:
    pid = _read_pid()
    if not pid or not _is_alive(pid):
        print("daemon not running")
        SOCKET_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)
        return 0
    import signal as _sig
    os.kill(pid, _sig.SIGTERM)
    for _ in range(20):
        if not _is_alive(pid):
            break
        time.sleep(0.2)
    else:
        os.kill(pid, _sig.SIGKILL)
    SOCKET_PATH.unlink(missing_ok=True)
    PID_PATH.unlink(missing_ok=True)
    print(f"daemon (pid={pid}) terminated")
    return 0


def _cli_logs() -> int:
    if not LOG_PATH.exists():
        print(f"no log at {LOG_PATH}")
        return 1
    print(LOG_PATH.read_text())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Persistent SSH shell to Empire AI login node.")
    p.add_argument("cmd", nargs="?", help="Shell command to run on empire-ai")
    p.add_argument("-t", "--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Command timeout in seconds (default {DEFAULT_TIMEOUT})")
    p.add_argument("--status", action="store_true", help="Show daemon status")
    p.add_argument("--kill", action="store_true", help="Stop the daemon")
    p.add_argument("--logs", action="store_true", help="Print daemon log")
    args = p.parse_args()

    if args.status:
        return _cli_status()
    if args.kill:
        return _cli_kill()
    if args.logs:
        return _cli_logs()
    if not args.cmd:
        p.print_help()
        return 2

    try:
        res = run(args.cmd, timeout=args.timeout)
    except Exception as e:
        print(f"empire-shell: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(res.stdout)
    if not res.stdout.endswith("\n"):
        sys.stdout.write("\n")
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
