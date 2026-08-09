#!/usr/bin/env bash
set -euo pipefail

# Lightweight, orchestration-only environment. The heavy ML stack (torch,
# transformers, vllm, llama-cpp-python) is intentionally NOT installed here:
# per CLAUDE.md there is no local GPU and all training/inference runs on the
# Empire AI cluster. This box only dispatches jobs and inspects results.
#
# gpu_dispatch.py is pure stdlib (drives SLURM over ssh); jupyter_exec.py needs
# requests + websocket-client. The rest are light data-inspection deps.
pip install --no-cache-dir \
  requests \
  websocket-client \
  loguru \
  jsonlines \
  pandas \
  tqdm \
  gdown

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

cat <<'EOF'

HalluLens local (orchestration) container ready.
  - Real compute runs on the Empire AI cluster (see CLAUDE.md).
  - To do CPU-heavy work here instead, install the full stack minus vllm:
      grep -viE '^[[:space:]]*vllm' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt
EOF
