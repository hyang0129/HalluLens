"""Tests for scripts/gpu_dispatch.py — all SSH calls are mocked."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.gpu_dispatch import (
    JobRecord,
    NodeConfig,
    NodeStatus,
    check_node_health,
    dispatch_job,
    load_config,
    load_manifest,
    load_nodes,
    save_manifest,
    select_best_node,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "defaults": {
        "python": "/usr/bin/python3",
        "project_root": "/home/user/project",
        "job_manifest": "shared/gpu_jobs.json",
        "ssh_timeout": 10,
    },
    "nodes": [
        {
            "hostname": "gpu01",
            "python": "/opt/python/bin/python",
            "project_root": "/data/project",
            "max_concurrent_jobs": 2,
            "tags": ["fast"],
        },
        {
            "hostname": "gpu02",
        },
    ],
}

NVIDIA_SMI_OUTPUT = (
    "NVIDIA H100, 45, 20480, 81920\n"
    "---PROCS---\n"
    "hyang1  12345  0.5  1.0 python train.py\n"
)


def _make_node(hostname="gpu01", max_concurrent=1, project_root="/data/project"):
    return NodeConfig(
        hostname=hostname,
        python="/usr/bin/python3",
        project_root=project_root,
        max_concurrent_jobs=max_concurrent,
    )


def _make_status(hostname="gpu01", reachable=True, mem_used=20480, mem_total=81920):
    return NodeStatus(
        hostname=hostname,
        reachable=reachable,
        gpu_name="NVIDIA H100",
        gpu_util_pct=45.0,
        gpu_mem_used_mb=mem_used,
        gpu_mem_total_mb=mem_total,
    )


# ---------------------------------------------------------------------------
# 1. test_load_nodes
# ---------------------------------------------------------------------------

def test_load_nodes():
    nodes = load_nodes(SAMPLE_CONFIG)
    assert len(nodes) == 2

    n0 = nodes[0]
    assert n0.hostname == "gpu01"
    assert n0.python == "/opt/python/bin/python"
    assert n0.project_root == "/data/project"
    assert n0.max_concurrent_jobs == 2
    assert n0.tags == ["fast"]

    n1 = nodes[1]
    assert n1.hostname == "gpu02"


# ---------------------------------------------------------------------------
# 2. test_load_nodes_applies_defaults
# ---------------------------------------------------------------------------

def test_load_nodes_applies_defaults():
    nodes = load_nodes(SAMPLE_CONFIG)
    n1 = nodes[1]  # gpu02 has no overrides
    assert n1.python == "/usr/bin/python3"
    assert n1.project_root == "/home/user/project"
    assert n1.max_concurrent_jobs == 1
    assert n1.tags == []


# ---------------------------------------------------------------------------
# 3. test_parse_nvidia_smi_output
# ---------------------------------------------------------------------------

@patch("scripts.gpu_dispatch.ssh_run")
def test_parse_nvidia_smi_output(mock_ssh):
    mock_ssh.return_value = MagicMock(
        returncode=0, stdout=NVIDIA_SMI_OUTPUT, stderr=""
    )
    node = _make_node()
    status = check_node_health(node)

    assert status.reachable is True
    assert status.gpu_name == "NVIDIA H100"
    assert status.gpu_util_pct == 45.0
    assert status.gpu_mem_used_mb == 20480.0
    assert status.gpu_mem_total_mb == 81920.0
    assert len(status.our_procs) == 1
    assert "train.py" in status.our_procs[0]


# ---------------------------------------------------------------------------
# 4. test_check_node_health_unreachable
# ---------------------------------------------------------------------------

@patch("scripts.gpu_dispatch.ssh_run")
def test_check_node_health_unreachable(mock_ssh):
    mock_ssh.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)
    node = _make_node()
    status = check_node_health(node)

    assert status.reachable is False
    assert status.error == "SSH timeout"


# ---------------------------------------------------------------------------
# 5. test_select_best_node_picks_least_busy
# ---------------------------------------------------------------------------

def test_select_best_node_picks_least_busy():
    node_a = _make_node("gpu01", max_concurrent=2)
    node_b = _make_node("gpu02", max_concurrent=2)

    statuses = {
        "gpu01": _make_status("gpu01"),
        "gpu02": _make_status("gpu02"),
    }

    # gpu01 has 1 running job, gpu02 has 0
    manifest = [
        JobRecord(
            job_id="aaa",
            hostname="gpu01",
            command="train",
            pid=1234,
            started_at="2026-01-01T00:00:00",
            status="running",
            log_file="log.txt",
        ),
    ]

    best = select_best_node([node_a, node_b], manifest, statuses)
    assert best is not None
    assert best.hostname == "gpu02"


# ---------------------------------------------------------------------------
# 6. test_select_best_node_picks_most_vram
# ---------------------------------------------------------------------------

def test_select_best_node_picks_most_vram():
    node_a = _make_node("gpu01")
    node_b = _make_node("gpu02")

    statuses = {
        "gpu01": _make_status("gpu01", mem_used=60000, mem_total=81920),  # ~21 GB free
        "gpu02": _make_status("gpu02", mem_used=10000, mem_total=81920),  # ~71 GB free
    }

    best = select_best_node([node_a, node_b], [], statuses)
    assert best is not None
    assert best.hostname == "gpu02"


# ---------------------------------------------------------------------------
# 7. test_select_best_node_all_busy
# ---------------------------------------------------------------------------

def test_select_best_node_all_busy():
    node_a = _make_node("gpu01", max_concurrent=1)
    node_b = _make_node("gpu02", max_concurrent=1)

    statuses = {
        "gpu01": _make_status("gpu01"),
        "gpu02": _make_status("gpu02"),
    }

    manifest = [
        JobRecord(job_id="a", hostname="gpu01", command="x", pid=1,
                  started_at="t", status="running", log_file="l"),
        JobRecord(job_id="b", hostname="gpu02", command="x", pid=2,
                  started_at="t", status="running", log_file="l"),
    ]

    best = select_best_node([node_a, node_b], manifest, statuses)
    assert best is None


# ---------------------------------------------------------------------------
# 8. test_select_best_node_none_reachable
# ---------------------------------------------------------------------------

def test_select_best_node_none_reachable():
    node_a = _make_node("gpu01")
    node_b = _make_node("gpu02")

    statuses = {
        "gpu01": _make_status("gpu01", reachable=False),
        "gpu02": _make_status("gpu02", reachable=False),
    }

    best = select_best_node([node_a, node_b], [], statuses)
    assert best is None


# ---------------------------------------------------------------------------
# 9. test_select_best_node_min_vram_filter
# ---------------------------------------------------------------------------

def test_select_best_node_min_vram_filter():
    node_a = _make_node("gpu01")
    node_b = _make_node("gpu02")

    # gpu01: 1 GB free, gpu02: 60 GB free
    statuses = {
        "gpu01": _make_status("gpu01", mem_used=80896, mem_total=81920),
        "gpu02": _make_status("gpu02", mem_used=20480, mem_total=81920),
    }

    best = select_best_node([node_a, node_b], [], statuses, min_vram_gb=40.0)
    assert best is not None
    assert best.hostname == "gpu02"


# ---------------------------------------------------------------------------
# 10. test_manifest_roundtrip
# ---------------------------------------------------------------------------

def test_manifest_roundtrip(tmp_path):
    manifest_path = tmp_path / "jobs.json"
    jobs = [
        JobRecord(
            job_id="abc123",
            hostname="gpu01",
            command="python train.py",
            pid=9999,
            started_at="2026-01-01T00:00:00+00:00",
            status="running",
            log_file="shared/logs/abc123.log",
            description="training run",
        ),
    ]

    save_manifest(manifest_path, jobs)
    loaded = load_manifest(manifest_path)

    assert len(loaded) == 1
    j = loaded[0]
    assert j.job_id == "abc123"
    assert j.hostname == "gpu01"
    assert j.command == "python train.py"
    assert j.pid == 9999
    assert j.status == "running"
    assert j.description == "training run"


# ---------------------------------------------------------------------------
# 11. test_manifest_empty_file
# ---------------------------------------------------------------------------

def test_manifest_empty_file(tmp_path):
    manifest_path = tmp_path / "nonexistent.json"
    loaded = load_manifest(manifest_path)
    assert loaded == []


# ---------------------------------------------------------------------------
# 12. test_dispatch_job
# ---------------------------------------------------------------------------

@patch("scripts.gpu_dispatch.ssh_run")
def test_dispatch_job(mock_ssh, tmp_path):
    # First call: mkdir, second call: dispatch (returns PID)
    mock_ssh.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),           # mkdir
        MagicMock(returncode=0, stdout="42\n", stderr=""),       # dispatch
    ]

    node = _make_node("gpu01", project_root=str(tmp_path / "project"))
    config = {
        "defaults": {
            "project_root": str(tmp_path),
            "job_manifest": "jobs.json",
        }
    }

    job = dispatch_job(node, "python train.py", "test run", config)

    assert job.hostname == "gpu01"
    assert job.command == "python train.py"
    assert job.pid == 42
    assert job.status == "running"
    assert job.description == "test run"
    assert job.log_file.startswith("shared/logs/")
    assert len(job.job_id) == 12

    # Verify manifest was written
    manifest_path = tmp_path / "jobs.json"
    assert manifest_path.exists()
    loaded = load_manifest(manifest_path)
    assert len(loaded) == 1
    assert loaded[0].job_id == job.job_id


# ---------------------------------------------------------------------------
# 13. test_build_dispatch_command
# ---------------------------------------------------------------------------

@patch("scripts.gpu_dispatch.ssh_run")
def test_build_dispatch_command(mock_ssh, tmp_path):
    mock_ssh.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="99\n", stderr=""),
    ]

    node = _make_node("gpu01", project_root="/data/project")
    config = {
        "defaults": {
            "project_root": str(tmp_path),
            "job_manifest": "jobs.json",
        }
    }

    dispatch_job(node, "python train.py", "test", config)

    # The second ssh_run call is the actual dispatch
    dispatch_call = mock_ssh.call_args_list[1]
    cmd_arg = dispatch_call[0][1]  # positional arg: cmd string

    assert "nohup" in cmd_arg
    assert "cd /data/project" in cmd_arg
    assert "echo $!" in cmd_arg
