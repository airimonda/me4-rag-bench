"""Laptop-side remote helpers, with ssh/scp replaced by fakes (no network)."""
import json
import os
from pathlib import Path

import pytest

from crb import config, remote


def test_run_dir_from_log():
    log = "2026 Starting run in /home/pi/me4-rag-bench/runs/s1/20261127-234219  (mode official, sample class-v1)\nmore"
    assert remote.run_dir_from_log(log) == "/home/pi/me4-rag-bench/runs/s1/20261127-234219"
    assert remote.run_dir_from_log("nothing yet") is None


def test_quote_keeps_tilde():
    assert remote._q("~/me4 rag") == "~/'me4 rag'"
    assert remote._q("/abs/path") == "/abs/path"


def test_pull_wait_copies_and_opens(tmp_path, monkeypatch):
    statuses = iter(["running", "running", "done"])
    calls = []

    def fake_ssh(host, command, check=True):
        calls.append(command)
        if command.startswith("ls -dt"):
            return "/home/pi/me4-rag-bench/runs/s1/20261127-234219/\n"
        if command.startswith("cat"):
            return json.dumps({"status": next(statuses)})
        return ""

    def fake_copy(host, run_dir, local_root):
        d = local_root / "s1" / "20261127-234219"
        d.mkdir(parents=True)
        (d / "report.html").write_text("<html></html>")
        return d

    opened = []
    monkeypatch.setattr(remote, "ssh", fake_ssh)
    monkeypatch.setattr(remote, "copy_back", fake_copy)
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    monkeypatch.setattr(remote.webbrowser, "open", lambda url: opened.append(url))
    rc = remote.pull("pi@host", wait=True, local_root=tmp_path)
    assert rc == 0
    assert opened and opened[0].startswith("file://") and opened[0].endswith("report.html")
    assert sum(c.startswith("cat") for c in calls) == 3


def test_pull_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ssh", lambda h, c, check=True: "")
    assert remote.pull("pi@host", local_root=tmp_path, do_open=False) == 2


def test_load_env_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CRB_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "env").write_text("# comment\nexport CRB_TEST_KEY='abc'\nCRB_TEST_SET=keep\n")
    monkeypatch.delenv("CRB_TEST_KEY", raising=False)
    monkeypatch.setenv("CRB_TEST_SET", "already")
    read = config.load_env_files()
    assert Path(tmp_path / "env") in read
    assert os.environ["CRB_TEST_KEY"] == "abc"
    assert os.environ["CRB_TEST_SET"] == "already"
    monkeypatch.delenv("CRB_TEST_KEY", raising=False)
