"""`crb fetch`: download / build the packs and the class-v1 sample into $CRB_DATA.

    crb fetch --dev            tiny offline dev packs (no network)
    crb fetch                  all three real packs + samples/class-v1.json
    crb fetch --pack ID        one pack (vidore_v3_hr, vidore_v3_computer_science, crag_task1)

Safe to re-run: finished packs are skipped, interrupted downloads resume where they stopped.
`main(args)` takes an argparse-like object with `.dev`, `.pack`, `.data_dir` (and optionally `.force`)
and returns a process exit code (0 = fine).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import requests

from crb.data import packs

HF = "https://huggingface.co"
UA = {"User-Agent": "crb-data/0.1 (class benchmark)"}

# Rough sizes in GB (measured on the real sources): what the finished pack takes, and the
# temporary download that is deleted again once the pack is built.
FINAL_GB = {"vidore_v3_hr": 0.5, "vidore_v3_computer_science": 0.55, "crag_task1": 0.25}
CACHE_GB = {"vidore_v3_hr": 0.45, "vidore_v3_computer_science": 0.52, "crag_task1": 0.74}
MARGIN_GB = 0.3


class DownloadError(Exception):
    pass


def _auth() -> dict:
    h = dict(UA)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


# ------------------------------------------------------------------ progress

class Progress:
    """Plain-English progress: one line every few seconds (carriage-return rewrite on a terminal)."""

    def __init__(self, say: Callable[[str], None] = print, every: float = 3.0):
        self.say = say
        self.every = every
        self._last = 0.0

    def tick(self, msg: str, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last >= self.every:
            self._last = now
            self.say(msg)


# ------------------------------------------------------------------ download

def download(url: str, dest: Path, expected_size: int | None = None,
             say: Callable[[str], None] = print, label: str = "", retries: int = 6,
             headers: dict | None = None, session: requests.Session | None = None) -> Path:
    """Download `url` to `dest`: streamed to disk, resumable (HTTP Range on `dest + '.part'`), retried.

    Returns `dest`. If `dest` already exists with the expected size it is reused. Raises
    DownloadError with a plain-English message when it gives up.
    """
    dest = Path(dest)
    label = label or dest.name
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        say(f"{label}: already downloaded ({_mb(dest.stat().st_size)}).")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sess = session or requests.Session()
    hdr = dict(UA)
    hdr.update(headers or {})
    prog = Progress(say)
    failures = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None and have == expected_size:
            part.replace(dest)
            return dest
        if expected_size is not None and have > expected_size:
            part.unlink()
            have = 0
        req_h = dict(hdr)
        if have:
            req_h["Range"] = f"bytes={have}-"
        start_have = have
        try:
            with sess.get(url, headers=req_h, stream=True, timeout=(15, 60), allow_redirects=True) as r:
                if r.status_code == 416:
                    # Asked to start past the end: the part file is complete (or junk).
                    if have and expected_size is None:
                        part.replace(dest)
                        return dest
                    part.unlink(missing_ok=True)
                    continue
                if r.status_code in (429,) or r.status_code >= 500:
                    raise requests.ConnectionError(f"server answered {r.status_code}")
                if r.status_code >= 400:
                    raise DownloadError(
                        f"{label}: the server refused the download (HTTP {r.status_code}) for {url}. "
                        f"Check your internet connection and try again later.")
                if have and r.status_code == 200:
                    have = 0  # server ignored Range: start over
                    part.unlink(missing_ok=True)
                elif have and r.status_code == 206:
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {have}-"):
                        part.unlink(missing_ok=True)
                        continue
                length = r.headers.get("Content-Length")
                total = expected_size
                if total is None and length is not None and r.headers.get("Content-Encoding") is None:
                    total = have + int(length)
                mode = "ab" if have else "wb"
                t0 = time.monotonic()
                got0 = have
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 18):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        el = max(time.monotonic() - t0, 1e-3)
                        pct = f" ({100 * have / total:.0f}%)" if total else ""
                        tot = f" of {_mb(total)}" if total else ""
                        prog.tick(f"{label}: {_mb(have)}{tot}{pct}, {(have - got0) / 1e6 / el:.1f} MB/s")
                if total is not None and have < total:
                    raise requests.ConnectionError("connection ended early")
        except DownloadError:
            raise
        except (requests.RequestException, OSError) as e:
            if have > start_have:
                failures = 0
            failures += 1
            if failures > retries:
                raise DownloadError(
                    f"{label}: gave up after {retries} retries ({e}). Check your internet connection, "
                    f"then run `crb fetch` again: it continues from where it stopped.") from e
            wait = min(2 ** failures, 30)
            say(f"{label}: connection problem ({e}); retrying in {wait}s (try {failures}/{retries}).")
            time.sleep(wait)
            continue
        if expected_size is not None and part.stat().st_size != expected_size:
            failures += 1
            if failures > retries:
                raise DownloadError(f"{label}: downloaded size does not match the expected size.")
            continue
        part.replace(dest)
        prog.tick(f"{label}: done ({_mb(dest.stat().st_size)}).", force=True)
        return dest


def hf_files(repo: str, config: str) -> list[dict]:
    """Files of one config folder of a Hugging Face dataset: [{"path", "size", "url"}], sorted by path.

    Uses the repo tree API (real file paths and byte sizes) and `resolve/main` download URLs, so
    the original parquet files are fetched, not the auto-converted copies.
    """
    url = f"{HF}/api/datasets/{repo}/tree/main/{config}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=_auth(), timeout=30)
            if r.status_code == 200:
                out = []
                for e in r.json():
                    if e.get("type") == "file" and e["path"].endswith(".parquet"):
                        size = (e.get("lfs") or {}).get("size") or e.get("size")
                        out.append({"path": e["path"], "size": int(size),
                                    "url": f"{HF}/datasets/{repo}/resolve/main/{e['path']}"})
                if not out:
                    raise DownloadError(f"No parquet files found for {repo}/{config}.")
                return sorted(out, key=lambda x: x["path"])
            last = DownloadError(f"HTTP {r.status_code}")
        except requests.RequestException as e:
            last = e
        time.sleep(2 ** attempt)
    raise DownloadError(
        f"Could not list the files of {repo} on Hugging Face ({last}). "
        f"Check your internet connection and try again.")


# ------------------------------------------------------------------ disk

def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def check_disk(data: Path, pack_ids: list[str], say: Callable[[str], None] = print) -> bool:
    """Is there room for `pack_ids` (final size + the biggest temporary download + a margin)?"""
    need = sum(FINAL_GB.get(p, 0.5) for p in pack_ids) + max(CACHE_GB.get(p, 0.5) for p in pack_ids) + MARGIN_GB
    have = free_gb(data)
    if have < need:
        say(f"Not enough free disk space: the download needs about {need:.1f} GB free in {data}, "
            f"but only {have:.1f} GB is free. Free up some space, or point CRB_DATA at a bigger disk "
            f"(for example an external drive) and run `crb fetch` again.")
        return False
    say(f"Disk space check: need about {need:.1f} GB, {have:.1f} GB free. OK.")
    return True


# ------------------------------------------------------------------ main

def _pack_ready(data: Path, pack: str) -> bool:
    return not packs.verify(pack, quick=True)  # main() has pointed $CRB_DATA at `data`


def main(args) -> int:
    say = print
    data = Path(getattr(args, "data_dir", None) or packs.data_dir()).expanduser()
    # Make the rest of crb.data (which reads $CRB_DATA through packs.data_dir) see the same folder.
    os.environ["CRB_DATA"] = str(data)
    try:
        if getattr(args, "dev", False):
            from crb.data import devpack
            say(f"Building the tiny offline dev packs in {data} ...")
            devpack.generate(data)
            say("Dev packs ready (dev_vidore, dev_crag, samples/dev.json). No internet needed.")
            return 0
        wanted = getattr(args, "pack", None)
        if wanted:
            if wanted not in packs.PACK_IDS:
                say(f"Unknown pack '{wanted}'. Choose one of: {', '.join(packs.PACK_IDS)}.")
                return 2
            todo = [wanted]
        else:
            todo = list(packs.PACK_IDS)
        force = getattr(args, "force", False)
        pending = [p for p in todo if force or not _pack_ready(data, p)]
        for p in todo:
            if p not in pending:
                say(f"{p}: already downloaded and complete, skipping.")
        if pending:
            data.mkdir(parents=True, exist_ok=True)
            if not check_disk(data, pending, say):
                return 2
            for i, p in enumerate(pending, 1):
                say(f"[{i}/{len(pending)}] Getting {p} ...")
                t0 = time.monotonic()
                if p == "crag_task1":
                    from crb.data import crag
                    crag.build(data, say=say)
                else:
                    from crb.data import vidore
                    vidore.build(p, data, say=say)
                say(f"{p}: finished in {time.monotonic() - t0:.0f}s.")
        from crb.data import sample
        out = sample.build(data)
        say(f"Sample written: {out}")
        bad = []
        for p in todo:
            bad += packs.verify(p, quick=True)
        if bad:
            say("Some problems were found:")
            for b in bad:
                say(f"  - {b}")
            return 1
        say("All done. Data is ready.")
        return 0
    except KeyboardInterrupt:
        say("Stopped. Everything downloaded so far is kept; run `crb fetch` again to continue.")
        return 130
    except DownloadError as e:
        say(str(e))
        return 1
    except ImportError as e:
        if "pyarrow" in str(e):
            say("The `pyarrow` package is needed to build the ViDoRe packs: run `pip install pyarrow`.")
            return 1
        raise


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crb.data.fetch", description=__doc__.split("\n")[0])
    ap.add_argument("--dev", action="store_true", help="generate the tiny offline dev packs")
    ap.add_argument("--pack", help="only this pack id")
    ap.add_argument("--data-dir", dest="data_dir", help="data folder (default $CRB_DATA or ~/crb-data)")
    ap.add_argument("--force", action="store_true", help="rebuild even when a pack looks complete")
    return main(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(_cli())
