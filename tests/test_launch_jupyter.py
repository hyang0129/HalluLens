"""Policy tests for the guarded Empire Jupyter launcher."""
from __future__ import annotations

import sys

import pytest

from scripts import launch_jupyter


def _running_jobs(count: int) -> list[dict[str, str]]:
    return [
        {
            "job_id": str(index),
            "name": f"jupyter_empire_{8880 + index}",
            "state": "RUNNING",
            "nodelist": f"alphagpu{index:02d}",
        }
        for index in range(count)
    ]


def test_guarded_launcher_permits_an_eighth_running_allocation(
    tmp_path, monkeypatch, capsys
):
    launch_script = tmp_path / "empire_jupyter_lab.sh"
    launch_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(launch_jupyter, "JUPYTER_SCRIPT", launch_script)
    monkeypatch.setattr(launch_jupyter, "query_jobs", lambda: _running_jobs(7))
    monkeypatch.setattr(sys, "argv", ["launch_jupyter.py", "8893", "--dry-run"])

    assert launch_jupyter.MAX_ACTIVE_JUPYTER == 8
    assert launch_jupyter.MAX_TOTAL_JOBS == 8
    assert launch_jupyter.main() == 0
    assert "7 running jupyter job(s) (cap 8)" in capsys.readouterr().out


def test_guarded_launcher_refuses_a_ninth_running_allocation(
    tmp_path, monkeypatch
):
    launch_script = tmp_path / "empire_jupyter_lab.sh"
    launch_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(launch_jupyter, "JUPYTER_SCRIPT", launch_script)
    monkeypatch.setattr(launch_jupyter, "query_jobs", lambda: _running_jobs(8))
    monkeypatch.setattr(sys, "argv", ["launch_jupyter.py", "8893", "--dry-run"])

    with pytest.raises(SystemExit, match="10"):
        launch_jupyter.main()
