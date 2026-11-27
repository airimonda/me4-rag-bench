"""`crb serve`: a read-only static server over runs/ so the report can be opened from a laptop."""
from __future__ import annotations

import functools
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config


class _ReadOnly(SimpleHTTPRequestHandler):
    """GET/HEAD only (the base class has no POST/PUT/DELETE), no access log noise."""

    def log_message(self, *args) -> None:
        pass


def lan_ip() -> str:
    """The address other machines on the network can use to reach this one."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; this just picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


def report_links(root: Path, host: str, port: int, run_dir: Path | None = None) -> list[str]:
    """http://host:port/... links to every report.html under root (the requested run first)."""
    reports = sorted(root.glob("*/*/report.html"), reverse=True)
    if run_dir is not None:
        want = (Path(run_dir).resolve() / "report.html")
        reports.sort(key=lambda p: p.resolve() != want)
    return [f"http://{host}:{port}/{p.relative_to(root).as_posix()}" for p in reports]


def make_server(root: Path, port: int) -> ThreadingHTTPServer:
    handler = functools.partial(_ReadOnly, directory=str(root))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(run_dir: str | Path | None = None, port: int = config.DEFAULT_SERVE_PORT) -> int:
    root = config.runs_root()
    if run_dir is not None:
        rd = Path(run_dir).expanduser().resolve()
        if not rd.is_dir():
            print(f"There is no run folder at {rd}. Check the path (runs/<student>/<time>).")
            return 2
        if root not in rd.parents:
            root = rd.parent
    if not root.is_dir():
        print(f"There is no runs folder at {root} yet. Do a run first with `crb`.")
        return 2
    try:
        httpd = make_server(root, port)
    except OSError:
        print(f"Port {port} is already in use. Pick another with `crb serve --port 8303`.")
        return 2
    ip = lan_ip()
    links = report_links(root, ip, port, Path(run_dir) if run_dir else None)
    print(f"Serving {root} (read-only) on port {port}.")
    if links:
        print("Open one of these in a browser on any computer on the same network:")
        for link in links[:10]:
            print(f"  {link}")
        if len(links) > 10:
            print(f"  ... and {len(links) - 10} more under http://{ip}:{port}/")
    else:
        print(f"No report.html yet. Browse http://{ip}:{port}/ once a run has finished.")
    print("Press Ctrl-C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0
