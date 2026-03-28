"""
Remote Jupyter kernel executor.

Allows Claude Code (running on alphacpu) to execute Python code on the
GPU-enabled node (configured via GPUNODE in .env) via the Jupyter REST + WebSocket API.

Usage:
    from utils.jupyter_exec import JupyterExecutor

    with JupyterExecutor() as jup:
        result = jup.run("import torch; print(torch.cuda.get_device_name(0))")
        print(result.stdout)

CLI:
    python utils/jupyter_exec.py "import torch; print(torch.cuda.is_available())"
    python utils/jupyter_exec.py --file myscript.py
    python utils/jupyter_exec.py --timeout 120 "import time; time.sleep(5); print('done')"
"""

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests
import websocket  # pip install websocket-client

def _default_base_url() -> str:
    """Read GPUNODE and GPUNODEPORT from .env in the repo root or environment."""
    import os
    from pathlib import Path

    env_path = Path(__file__).parent.parent / ".env"
    node, port = None, "8889"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GPUNODE="):
                node = line.split("=", 1)[1].strip()
            elif line.startswith("GPUNODEPORT="):
                port = line.split("=", 1)[1].strip()
    # Environment variables take precedence over .env
    node = os.environ.get("GPUNODE", node)
    port = os.environ.get("GPUNODEPORT", port)
    if not node:
        raise RuntimeError("GPUNODE is not set. Add GPUNODE=<hostname> to .env or set the environment variable.")
    return f"http://{node}:{port}"


DEFAULT_BASE_URL = _default_base_url()
DEFAULT_PASSWORD = "123"
DEFAULT_TIMEOUT = 60  # seconds


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    error: str = ""       # traceback text if an exception was raised
    status: str = "ok"    # "ok" or "error"
    outputs: list = field(default_factory=list)  # raw display_data / execute_result items

    def __str__(self):
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.error:
            parts.append(f"[error]\n{self.error}")
        return "".join(parts)


class JupyterExecutor:
    """Context manager that holds an authenticated session to the Jupyter server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        password: str = DEFAULT_PASSWORD,
    ):
        self.base_url = base_url.rstrip("/")
        self.ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.password = password
        self.session = requests.Session()
        self._cookie_str = ""
        self._xsrf_token = ""

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self):
        login_page = self.session.get(f"{self.base_url}/login")
        login_page.raise_for_status()
        xsrf = self.session.cookies.get("_xsrf", "")
        resp = self.session.post(
            f"{self.base_url}/login",
            data={"password": self.password, "_xsrf": xsrf},
            allow_redirects=True,
        )
        resp.raise_for_status()
        self._cookie_str = "; ".join(f"{k}={v}" for k, v in self.session.cookies.items())
        self._xsrf_token = self.session.cookies.get("_xsrf", "")

    # ------------------------------------------------------------------
    # Kernel management
    # ------------------------------------------------------------------

    def start_kernel(self, name: str = "p311") -> str:
        """Start a new kernel and return its ID."""
        resp = self.session.post(
            f"{self.base_url}/api/kernels",
            json={"name": name},
            headers={"X-XSRFToken": self._xsrf_token},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def stop_kernel(self, kernel_id: str):
        """Shut down a kernel by ID."""
        self.session.delete(
            f"{self.base_url}/api/kernels/{kernel_id}",
            headers={"X-XSRFToken": self._xsrf_token},
        )

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def run(
        self,
        code: str,
        kernel_id: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ExecResult:
        """Execute *code* on the remote kernel and return collected outputs.

        If kernel_id is None a fresh p311 kernel is started and stopped automatically.
        Pass an existing kernel_id to reuse a long-running kernel (caller manages lifecycle).
        """
        if not self._cookie_str:
            self.login()

        owned = kernel_id is None
        if owned:
            kernel_id = self.start_kernel("p311")

        ws = websocket.create_connection(
            f"{self.ws_base}/api/kernels/{kernel_id}/channels",
            header={
                "Cookie": self._cookie_str,
                "X-XSRFToken": self._xsrf_token,
            },
        )
        try:
            return self._execute_on_ws(ws, code, timeout)
        finally:
            ws.close()
            if owned:
                self.stop_kernel(kernel_id)

    def _execute_on_ws(self, ws, code: str, timeout: float) -> ExecResult:
        msg_id = str(uuid.uuid4())
        ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "",
                "session": str(uuid.uuid4()),
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
            },
            "channel": "shell",
        }))

        result = ExecResult()
        deadline = time.time() + timeout

        while time.time() < deadline:
            ws.settimeout(max(0.1, deadline - time.time()))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                result.status = "timeout"
                result.error = f"Timed out after {timeout}s waiting for execute_reply"
                break

            msg = json.loads(raw)
            mtype = msg.get("msg_type", "")
            content = msg.get("content", {})

            # Only process messages that are replies to our request
            parent_id = msg.get("parent_header", {}).get("msg_id", "")
            if parent_id != msg_id:
                continue

            if mtype == "stream":
                if content.get("name") == "stdout":
                    result.stdout += content.get("text", "")
                else:
                    result.stderr += content.get("text", "")

            elif mtype in ("display_data", "execute_result"):
                result.outputs.append(content.get("data", {}))
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    result.stdout += text + "\n"

            elif mtype == "error":
                result.status = "error"
                result.error = "\n".join(content.get("traceback", [content.get("evalue", "")]))

            elif mtype == "execute_reply":
                if result.status != "error":
                    result.status = content.get("status", "ok")
                break

        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *_):
        self.session.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Execute Python code on remote Jupyter GPU kernel.")
    parser.add_argument("code", nargs="?", help="Python code string to execute")
    parser.add_argument("--file", "-f", help="Python file to execute")
    parser.add_argument("--timeout", "-t", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--kernel", help="Specific kernel ID (default: auto-select idle)")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        code = sys.stdin.read()

    jup = JupyterExecutor(base_url=args.url, password=args.password)
    result = jup.run(code, kernel_id=args.kernel, timeout=args.timeout)
    print(str(result), end="")
    sys.exit(0 if result.status == "ok" else 1)


if __name__ == "__main__":
    _main()
