"""Pi health sampler: one row per second in health.csv, plus a summary for results.json.

Every source fails soft: if a reading is not available (no vcgencmd on a laptop, no /proc on
macOS) the cell is left empty. The sampler uses the standard library only and stays well under
1% of one core (reads of a few small files per second; vcgencmd is called once per second for
get_throttled and every 5 s for the core voltage).
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

COLUMNS = [
    "t", "iso", "temp_c", "throttled_hex", "under_voltage_now", "freq_capped_now",
    "throttled_now", "soft_temp_now", "arm_mhz", "core_v", "cpu_pct", "cpu_max_core_pct",
    "load1", "mem_avail_mb", "swap_used_mb", "adapter_rss_mb", "disk_free_mb", "net_rtt_ms", "phase",
]

RTT_HOST = ("openrouter.ai", 443)
PID_REFRESH_S = 30.0
RTT_EVERY_S = 10.0
VOLTS_EVERY_S = 5.0


# ----------------------------------------------------------------------------- parsers

def decode_throttled(value: int) -> dict:
    """Decode the get_throttled bit field. Bits 0-3 are 'now', bits 16-19 are 'since boot'."""
    return {
        "under_voltage_now": bool(value & (1 << 0)),
        "freq_capped_now": bool(value & (1 << 1)),
        "throttled_now": bool(value & (1 << 2)),
        "soft_temp_now": bool(value & (1 << 3)),
        "under_voltage_boot": bool(value & (1 << 16)),
        "freq_capped_boot": bool(value & (1 << 17)),
        "throttled_boot": bool(value & (1 << 18)),
        "soft_temp_boot": bool(value & (1 << 19)),
    }


def parse_throttled(text: str) -> int | None:
    """'throttled=0x50005' -> 0x50005. None if it does not parse."""
    try:
        return int(text.strip().split("=", 1)[1], 16)
    except (IndexError, ValueError, AttributeError):
        return None


def parse_temp(text: str) -> float | None:
    """'temp=48.3'C' -> 48.3"""
    try:
        return float(text.strip().split("=", 1)[1].split("'", 1)[0])
    except (IndexError, ValueError, AttributeError):
        return None


def parse_clock_mhz(text: str) -> float | None:
    """'frequency(48)=1500398464' (Hz) -> 1500.4"""
    try:
        return round(int(text.strip().split("=", 1)[1]) / 1e6, 1)
    except (IndexError, ValueError, AttributeError):
        return None


def parse_volts(text: str) -> float | None:
    """'volt=0.8438V' -> 0.8438"""
    try:
        return float(text.strip().split("=", 1)[1].rstrip("V"))
    except (IndexError, ValueError, AttributeError):
        return None


def parse_proc_stat(text: str) -> dict[str, tuple[int, int]]:
    """/proc/stat -> {'cpu': (busy, total), 'cpu0': (busy, total), ...} in jiffies."""
    out: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        try:
            nums = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
        total = sum(nums[:8]) if len(nums) >= 8 else sum(nums)
        out[parts[0]] = (total - idle, total)
    return out


def cpu_percent(prev: dict[str, tuple[int, int]], cur: dict[str, tuple[int, int]]) -> tuple[float | None, float | None]:
    """(overall %, busiest single core %) between two parse_proc_stat snapshots."""
    def pct(a: tuple[int, int], b: tuple[int, int]) -> float | None:
        dt = b[1] - a[1]
        if dt <= 0:
            return None
        return round(100.0 * (b[0] - a[0]) / dt, 1)

    overall = pct(prev["cpu"], cur["cpu"]) if "cpu" in prev and "cpu" in cur else None
    cores = [pct(prev[k], cur[k]) for k in cur if k != "cpu" and k in prev]
    cores = [c for c in cores if c is not None]
    return overall, (max(cores) if cores else None)


def parse_loadavg(text: str) -> float | None:
    try:
        return float(text.split()[0])
    except (IndexError, ValueError):
        return None


def parse_meminfo(text: str) -> dict[str, float]:
    """/proc/meminfo -> {'mem_avail_mb', 'swap_used_mb'} (only keys that could be computed)."""
    kb: dict[str, int] = {}
    for line in text.splitlines():
        try:
            key, rest = line.split(":", 1)
            kb[key] = int(rest.split()[0])
        except (ValueError, IndexError):
            continue
    out: dict[str, float] = {}
    if "MemAvailable" in kb:
        out["mem_avail_mb"] = round(kb["MemAvailable"] / 1024, 1)
    if "SwapTotal" in kb and "SwapFree" in kb:
        out["swap_used_mb"] = round((kb["SwapTotal"] - kb["SwapFree"]) / 1024, 1)
    return out


def parse_listen_inodes(text: str, port: int) -> set[str]:
    """Socket inodes in LISTEN state (st == 0A) on `port`, from /proc/net/tcp or tcp6 text."""
    inodes: set[str] = set()
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 10 or parts[3] != "0A":
            continue
        try:
            local_port = int(parts[1].rsplit(":", 1)[1], 16)
        except (IndexError, ValueError):
            continue
        if local_port == port:
            inodes.add(parts[9])
    return inodes


def parse_status_rss_mb(text: str) -> float | None:
    """/proc/<pid>/status -> VmRSS in MB."""
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            try:
                return round(int(line.split()[1]) / 1024, 1)
            except (IndexError, ValueError):
                return None
    return None


def find_pid_for_port(port: int, proc: str = "/proc") -> int | None:
    """PID of the process listening on `port` (needs the same user, or root, for other users)."""
    inodes: set[str] = set()
    for name in ("tcp", "tcp6"):
        try:
            inodes |= parse_listen_inodes(Path(proc, "net", name).read_text(), port)
        except OSError:
            pass
    if not inodes:
        return None
    wanted = {f"socket:[{i}]" for i in inodes}
    try:
        entries = os.listdir(proc)
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        fd_dir = Path(proc, entry, "fd")
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(fd_dir / fd) in wanted:
                        return int(entry)
                except OSError:
                    continue
        except OSError:
            continue
    return None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _read(path: str) -> str | None:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError:
        return None


# ----------------------------------------------------------------------------- sampler

class HealthSampler:
    """1 Hz sampler thread. `state` is a crb.state.State (only `.phase` and `.t()` are read)."""

    def __init__(self, data_dir: Path, adapter_port: int | None, out_csv: Path, state, adapter_pid: int | None = None):
        self.data_dir = Path(data_dir)
        self.adapter_port = adapter_port
        self.out_csv = Path(out_csv)
        self.state = state
        self._fixed_pid = adapter_pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rtt_thread: threading.Thread | None = None
        self._rtt: float | None = None
        self._rows: list[dict] = []
        self._rows_lock = threading.Lock()
        self._vcgencmd = shutil.which("vcgencmd")
        self._pid: int | None = adapter_pid
        self._pid_at = -1e9
        self._volts: float | None = None
        self._volts_at = -1e9
        self._prev_stat: dict[str, tuple[int, int]] | None = None

    # -- public
    def start(self) -> None:
        if self._thread is not None:
            return
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="crb-health", daemon=True)
        self._rtt_thread = threading.Thread(target=self._rtt_loop, name="crb-rtt", daemon=True)
        self._thread.start()
        self._rtt_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for th in (self._thread, self._rtt_thread):
            if th is not None:
                th.join(timeout=5)
        self._thread = None
        self._rtt_thread = None

    def summary(self) -> dict:
        """Numbers for results.json metrics.health, computed from the samples taken so far."""
        with self._rows_lock:
            rows = list(self._rows)

        def col(name: str) -> list[float]:
            out = []
            for r in rows:
                v = r.get(name)
                if v not in (None, ""):
                    out.append(float(v))
            return out

        temps = col("temp_c")
        mem = col("mem_avail_mb")
        mhz = col("arm_mhz")
        cpu = col("cpu_pct")
        core = col("cpu_max_core_pct")
        rtt = col("net_rtt_ms")
        thr_samples = [r for r in rows if r.get("throttled_hex") not in (None, "")]
        bad = [r for r in thr_samples
               if str(r.get("freq_capped_now")) == "1" or str(r.get("throttled_now")) == "1"
               or str(r.get("soft_temp_now")) == "1"]
        uv = any(str(r.get("under_voltage_now")) == "1" for r in thr_samples)

        def r1(x: float | None) -> float | None:
            return None if x is None else round(x, 1)

        return {
            "temp_max": r1(max(temps)) if temps else None,
            "throttled_share": round(len(bad) / len(thr_samples), 4) if thr_samples else None,
            "throttled_seconds": len(bad) if thr_samples else None,
            "under_voltage": uv if thr_samples else None,
            "cpu_p95": r1(_percentile(cpu, 95)),
            "busiest_core_p95": r1(_percentile(core, 95)),
            "mem_avail_min_mb": r1(min(mem)) if mem else None,
            "arm_mhz_min": r1(min(mhz)) if mhz else None,
            "net_rtt_p50": r1(_percentile(rtt, 50)),
            "net_rtt_p90": r1(_percentile(rtt, 90)),
            "samples": len(rows),
            "available": bool(thr_samples or temps),
        }

    # -- internals
    def _run(self) -> None:
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            f.flush()
            next_tick = time.monotonic()
            while not self._stop.is_set():
                try:
                    row = self._sample()
                except Exception:  # never let one bad reading kill the thread
                    row = {"t": round(self.state.t(), 2),
                           "iso": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                           "phase": self._phase()}
                w.writerow(row)
                f.flush()
                with self._rows_lock:
                    self._rows.append(row)
                next_tick += 1.0
                delay = next_tick - time.monotonic()
                if delay < 0:  # fell behind, do not burst
                    next_tick = time.monotonic()
                    delay = 0
                self._stop.wait(delay)

    def _phase(self) -> str:
        try:
            return self.state.phase or ""
        except Exception:
            return ""

    def _vcg(self, *args: str) -> str | None:
        if not self._vcgencmd:
            return None
        try:
            r = subprocess.run([self._vcgencmd, *args], capture_output=True, text=True, timeout=2)
            return r.stdout if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def _sample(self) -> dict:
        now = time.monotonic()
        row: dict = {c: "" for c in COLUMNS}
        row["t"] = round(self.state.t(), 2)
        row["iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        row["phase"] = self._phase()

        # temperature: sysfs first (cheap), then vcgencmd
        temp = None
        raw = _read("/sys/class/thermal/thermal_zone0/temp")
        if raw:
            try:
                temp = round(int(raw.strip()) / 1000.0, 1)
            except ValueError:
                temp = None
        if temp is None:
            out = self._vcg("measure_temp")
            temp = parse_temp(out) if out else None
        if temp is not None:
            row["temp_c"] = temp

        # throttle flags
        out = self._vcg("get_throttled")
        val = parse_throttled(out) if out else None
        if val is not None:
            d = decode_throttled(val)
            row["throttled_hex"] = hex(val)
            for k in ("under_voltage_now", "freq_capped_now", "throttled_now", "soft_temp_now"):
                row[k] = int(d[k])

        # arm clock
        mhz = None
        raw = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if raw:
            try:
                mhz = round(int(raw.strip()) / 1000.0, 1)
            except ValueError:
                mhz = None
        if mhz is None:
            out = self._vcg("measure_clock", "arm")
            mhz = parse_clock_mhz(out) if out else None
        if mhz is not None:
            row["arm_mhz"] = mhz

        # core voltage (slow-moving, every 5 s)
        if now - self._volts_at >= VOLTS_EVERY_S:
            self._volts_at = now
            out = self._vcg("measure_volts", "core")
            self._volts = parse_volts(out) if out else None
        if self._volts is not None:
            row["core_v"] = self._volts

        # CPU
        txt = _read("/proc/stat")
        if txt:
            cur = parse_proc_stat(txt)
            if self._prev_stat is not None:
                overall, busiest = cpu_percent(self._prev_stat, cur)
                if overall is not None:
                    row["cpu_pct"] = overall
                if busiest is not None:
                    row["cpu_max_core_pct"] = busiest
            self._prev_stat = cur

        # load
        load = None
        txt = _read("/proc/loadavg")
        if txt:
            load = parse_loadavg(txt)
        if load is None:
            try:
                load = os.getloadavg()[0]
            except OSError:
                load = None
        if load is not None:
            row["load1"] = round(load, 2)

        # memory
        txt = _read("/proc/meminfo")
        if txt:
            m = parse_meminfo(txt)
            if "mem_avail_mb" in m:
                row["mem_avail_mb"] = m["mem_avail_mb"]
            if "swap_used_mb" in m:
                row["swap_used_mb"] = m["swap_used_mb"]

        # adapter RSS
        rss = self._adapter_rss(now)
        if rss is not None:
            row["adapter_rss_mb"] = rss

        # disk
        try:
            st = os.statvfs(self.data_dir if self.data_dir.exists() else self.data_dir.parent)
            row["disk_free_mb"] = round(st.f_bavail * st.f_frsize / 1048576, 0)
        except OSError:
            pass

        if self._rtt is not None:
            row["net_rtt_ms"] = self._rtt
        return row

    def _adapter_rss(self, now: float) -> float | None:
        if now - self._pid_at >= PID_REFRESH_S or (self._pid is None and now - self._pid_at >= 5):
            self._pid_at = now
            if self._fixed_pid:
                self._pid = self._fixed_pid
            elif self.adapter_port:
                self._pid = find_pid_for_port(self.adapter_port)
        if not self._pid:
            return None
        txt = _read(f"/proc/{self._pid}/status")
        if txt is None:
            self._pid_at = -1e9  # process gone or changed: look again next second
            return None
        return parse_status_rss_mb(txt)

    def _rtt_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with socket.create_connection(RTT_HOST, timeout=3):
                    pass
                self._rtt = round((time.perf_counter() - t0) * 1000.0, 1)
            except OSError:
                self._rtt = None
            self._stop.wait(RTT_EVERY_S)
