"""Laptop side: run the benchmark on the Pi over SSH, then copy the run back and open the report.

`crb remote HOST`      start a run on the Pi (it keeps going if the laptop disconnects), follow its
                       progress, then copy the run folder here and open report.html in the browser.
`crb pull HOST`        copy the newest finished run (or `--run PATH`) and open its report.
`crb pull HOST --wait` wait for the run that is going on now to finish first.

Only the OpenSSH client (`ssh`, `scp`) is needed: built into macOS, Linux and Windows 10/11.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

DEFAULT_REMOTE_DIR = "~/me4-rag-bench"
LOG_NAME = ".remote-last.log"
START_MARK = "Starting run in "


def default_local_root() -> Path:
    return Path(os.environ.get("CRB_LOCAL_RUNS") or "~/crb-runs").expanduser()


def ssh_opts() -> list[str]:
    """Share one SSH connection between calls so a password is asked once (not on Windows)."""
    opts = ["-o", "ServerAliveInterval=30"]
    if os.name != "nt":
        cp = Path("~/.ssh").expanduser() / "crb-%r@%h-%p"
        opts += ["-o", "ControlMaster=auto", "-o", f"ControlPath={cp}", "-o", "ControlPersist=15m"]
    return opts


def ssh(host: str, command: str, check: bool = True) -> str:
    """Run one shell command on the Pi and return its stdout."""
    proc = subprocess.run(["ssh", *ssh_opts(), host, command], capture_output=True, text=True)
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        raise RemoteError(msg[-1] if msg else f"ssh exited with {proc.returncode}")
    return proc.stdout


class RemoteError(Exception):
    pass


def _q(path: str) -> str:
    """Quote a remote path but keep a leading ~ so the remote shell expands it."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def run_dir_from_log(text: str) -> str | None:
    """The run directory the remote `crb run` announced, if it has started."""
    for line in text.splitlines():
        i = line.find(START_MARK)
        if i >= 0:
            rest = line[i + len(START_MARK):].strip()
            return rest.split("  ")[0].strip()
    return None


def newest_run(host: str, remote_dir: str) -> str | None:
    out = ssh(host, f"ls -dt {_q(remote_dir)}/runs/*/*/ 2>/dev/null | head -1", check=False).strip()
    return out.rstrip("/") or None


def run_status(host: str, run_dir: str) -> dict:
    out = ssh(host, f"cat {shlex.quote(run_dir)}/run.json 2>/dev/null", check=False)
    try:
        return json.loads(out) if out.strip() else {}
    except ValueError:
        return {}


def wait_until_finished(host: str, run_dir: str, every: float = 20.0) -> dict:
    """Poll run.json until the run is no longer running."""
    shown = None
    while True:
        st = run_status(host, run_dir)
        status = st.get("status")
        if status and status != "running":
            return st
        if status != shown:
            print(f"Waiting for the run on {host} to finish (checks every {int(every)} s; Ctrl-C stops waiting, not the run)...")
            shown = status
        time.sleep(every)


def copy_back(host: str, run_dir: str, local_root: Path) -> Path:
    """scp the run folder to <local_root>/<student>/<time>/ and return the local path."""
    parts = run_dir.rstrip("/").split("/")
    dest = local_root / parts[-2] / parts[-1] if len(parts) >= 2 else local_root / parts[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-q", "-r", *ssh_opts(), f"{host}:{run_dir.rstrip('/')}", str(dest.parent)]
    if subprocess.run(cmd).returncode != 0:
        raise RemoteError(f"could not copy {run_dir} from {host}")
    return dest


def open_report(local_run: Path, do_open: bool = True) -> Path | None:
    report = local_run / "report.html"
    if not report.exists():
        print(f"Copied to {local_run}, but there is no report.html yet (run `crb score` on the Pi).")
        return None
    print(f"Report: {report}")
    if do_open:
        try:
            webbrowser.open(report.resolve().as_uri())
        except Exception:  # noqa: BLE001 - opening is a convenience only
            print("Could not open a browser here; open the file above.")
    return report


def pull(host: str, remote_dir: str = DEFAULT_REMOTE_DIR, run: str | None = None, wait: bool = False,
         local_root: Path | None = None, do_open: bool = True) -> int:
    local_root = local_root or default_local_root()
    try:
        run_dir = run or newest_run(host, remote_dir)
        if not run_dir:
            print(f"No runs found under {remote_dir}/runs on {host}. Pass the folder with --run, or --dir if the repo is elsewhere.")
            return 2
        if not run_dir.startswith(("/", "~")):
            run_dir = f"{remote_dir.rstrip('/')}/{run_dir}"
        if wait:
            st = wait_until_finished(host, run_dir)
            print(f"Run finished on {host}: {st.get('status')}.")
        local = copy_back(host, run_dir, local_root)
    except RemoteError as exc:
        print(f"{exc}\nfix: check that `ssh {host}` works from this computer.")
        return 2
    except KeyboardInterrupt:
        print("\nStopped waiting. The run keeps going on the Pi; `crb pull " + host + " --wait` picks it up later.")
        return 130
    return 0 if open_report(local, do_open) else 1


def remote_run(host: str, run_args: list[str], remote_dir: str = DEFAULT_REMOTE_DIR,
               local_root: Path | None = None, do_open: bool = True, every: float = 5.0) -> int:
    """Start `crb run` on the Pi in the background, follow its log, then pull and open the report."""
    rd = _q(remote_dir)
    args = " ".join(shlex.quote(a) for a in run_args)
    start = (f"cd {rd} && mkdir -p runs && "
             f"nohup .venv/bin/crb run {args} > runs/{LOG_NAME} 2>&1 < /dev/null & echo $!")
    try:
        ssh(host, f"test -x {rd}/.venv/bin/crb")
    except RemoteError:
        print(f"crb is not installed in {remote_dir} on {host}.\nfix: on the Pi run `bash scripts/install_pi.sh`, or pass --dir.")
        return 2
    try:
        pid = ssh(host, start).strip().splitlines()[-1]
    except (RemoteError, IndexError) as exc:
        print(f"Could not start the run on {host}: {exc}")
        return 2
    print(f"Started on {host} (pid {pid}). It keeps running if this computer disconnects.\n")
    offset, run_dir = 0, None
    try:
        while True:
            chunk = ssh(host, f"tail -c +{offset + 1} {rd}/runs/{LOG_NAME} 2>/dev/null", check=False)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                offset += len(chunk.encode("utf-8"))
                run_dir = run_dir or run_dir_from_log(chunk)
            alive = ssh(host, f"kill -0 {shlex.quote(pid)} 2>/dev/null && echo yes", check=False).strip() == "yes"
            if not alive:
                tail = ssh(host, f"tail -c +{offset + 1} {rd}/runs/{LOG_NAME} 2>/dev/null", check=False)
                if tail:
                    sys.stdout.write(tail)
                    run_dir = run_dir or run_dir_from_log(tail)
                break
            time.sleep(every)
    except KeyboardInterrupt:
        print(f"\nStopped following. The run keeps going on the Pi; `crb pull {host} --wait` opens the report when it ends.")
        return 130
    if not run_dir:
        print(f"\nThe run on {host} stopped before it started a run folder (see the messages above).")
        return 1
    if not run_dir.startswith("/"):
        run_dir = f"{remote_dir.rstrip('/')}/{run_dir}"
    print()
    try:
        local = copy_back(host, run_dir, local_root or default_local_root())
    except RemoteError as exc:
        print(f"{exc}\nfix: `crb pull {host} --run {run_dir}`")
        return 2
    return 0 if open_report(local, do_open) else 1
