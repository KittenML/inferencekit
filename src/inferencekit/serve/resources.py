"""Bounded, read-only CPU/process telemetry for a launched server process group.

RSS is sampled concurrently across the group's live processes. Per-process VmHWM
is retained separately: adding those high-water marks does not give a group peak.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

from inferencekit.serve import servers

PROC_ROOT = Path("/proc")
MAX_PROCESSES = 128
MAX_RECORDED_PROCESSES = 1024
MAX_TASKS = 256
MAX_STATUS_BYTES = 64 * 1024
MAX_PRELOAD_BYTES = 64 * 1024 * 1024

_THREAD_VARS = frozenset(("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "GOTO_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                         "RAYON_NUM_THREADS", "TBB_NUM_THREADS"))
_CHOICES = {"OMP_DYNAMIC": {"true", "false", "1", "0"}, "MKL_DYNAMIC": {"true", "false", "1", "0"},
            "OMP_WAIT_POLICY": {"active", "passive"},
            "MKL_THREADING_LAYER": {"intel", "sequential", "gnu", "tbb", "pgi"},
            "MKL_INTERFACE_LAYER": {"lp64", "ilp64", "gnu,lp64", "gnu,ilp64"},
            "MKL_ENABLE_INSTRUCTIONS": {"sse4_2", "avx", "avx2", "avx512", "avx512_mic", "avx512_mic_e1", "avx512_e1"},
            "MKL_VERBOSE": {"0", "1"}, "DNNL_VERBOSE": {"0", "1", "2"}, "ONEDNN_VERBOSE": {"0", "1", "2"}}
_ISA_VALUES = {"default", "sse41", "avx", "avx2", "avx2_vnni", "avx2_vnni_2", "avx512_core",
               "avx512_core_vnni", "avx512_core_bf16", "avx512_core_fp16", "avx512_core_amx",
               "avx512_core_amx_fp16", "avx10_1_512", "avx10_1_512_amx"}
_CHOICES.update({"DNNL_MAX_CPU_ISA": _ISA_VALUES, "ONEDNN_MAX_CPU_ISA": _ISA_VALUES})
_CPU_LIST = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")


def _control_value(name: str, value: str) -> str | None:
    if len(value) > 512:
        return None
    value = value.strip()
    if name in _THREAD_VARS:
        if re.fullmatch(r"[1-9]\d{0,6}(?:,[1-9]\d{0,6})*", value):
            return value
    elif name in _CHOICES:
        if value.lower() in _CHOICES[name]:
            return value.lower()
    elif name == "OMP_PROC_BIND":
        if all(x in {"true", "false", "master", "primary", "close", "spread"}
               for x in value.lower().split(",")):
            return value.lower()
    elif name == "OMP_PLACES":
        if re.fullmatch(r"(?:threads|cores|sockets|ll_caches|numa_domains)(?:\(\d+\))?", value.lower()) or \
                re.fullmatch(r"[\d{},:!\s-]+", value):
            return value.lower()
    elif name == "GOMP_CPU_AFFINITY":
        if re.fullmatch(r"\d+(?:-\d+(?::\d+)?)?(?:\s+\d+(?:-\d+(?::\d+)?)?)*", value):
            return value
    elif name == "KMP_BLOCKTIME":
        if value.lower() == "infinite" or re.fullmatch(r"\d{1,9}(?:ms|us)?", value):
            return value.lower()
    elif name == "KMP_AFFINITY":
        # No arbitrary text or paths. These are the documented tuning tokens and numeric masks.
        words = re.findall(r"[A-Za-z_]+", value.lower())
        if re.fullmatch(r"[A-Za-z_0-9,={}\[\]:@.\s-]+", value) and all(
                w in {"verbose", "noverbose", "warnings", "nowarnings", "respect", "norespect", "granularity",
                      "fine", "thread", "core", "package", "node", "group", "compact", "scatter", "balanced",
                      "explicit", "none", "disabled", "proclist", "duplicates", "noduplicates"} for w in words):
            return value.lower()
    return None


def _preload(path: str, *, cwd: str | os.PathLike | None = None) -> dict:
    p = Path(path)
    name = p.name
    if not re.fullmatch(r"lib[A-Za-z0-9_.+-]{1,120}\.(?:so(?:\.\d+)*|dylib)", name):
        return {"name": None, "sha256": None, "status": "invalid_or_redacted"}
    result = {"name": name, "sha256": None, "status": "unavailable"}
    # A bare soname uses the dynamic loader's search, not the launch directory.
    # Do not fingerprint a same-named cwd file that the loader may never load.
    if not p.is_absolute() and "/" not in path:
        return result
    if not p.is_absolute() and cwd is not None:
        p = Path(cwd) / p
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                return {**result, "status": "not_regular"}
            if info.st_size > MAX_PRELOAD_BYTES:
                return {**result, "status": "size_limit"}
            digest, size = hashlib.sha256(), 0
            while chunk := source.read(64 * 1024):
                size += len(chunk)
                if size > MAX_PRELOAD_BYTES:
                    return {**result, "status": "size_limit"}
                digest.update(chunk)
        return {**result, "sha256": digest.hexdigest(), "status": "hashed"}
    except OSError:
        return result  # Never put exception text (which contains private paths) into evidence.


def runtime_env(env: dict, *, cwd: str | os.PathLike | None = None) -> dict:
    """Only recognized CPU controls and preload basenames/content hashes; never arbitrary env values.

    Controls are requested values, not evidence that a runtime honored them. Invalid
    values are null and named in ``omitted``. Relative preload paths use the server's
    launch directory (the caller's by default); loader-only sonames remain unavailable.
    """
    names = _THREAD_VARS | _CHOICES.keys() | {"OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
                                            "KMP_BLOCKTIME", "KMP_AFFINITY"}
    controls, omitted = {}, []
    for name in sorted(names & env.keys()):
        raw = env[name]
        value = _control_value(name, str(raw)) if isinstance(raw, (str, int)) and not isinstance(raw, bool) else None
        controls[name] = value
        if value is None:
            omitted.append(name)
    preload = []
    raw = env.get("LD_PRELOAD")
    if isinstance(raw, str) and raw:
        if len(raw) <= 8192:
            paths = [p for p in re.split(r"[:\s]+", raw) if p]
            preload = [_preload(p, cwd=cwd) for p in paths[:16]]
            if len(paths) > 16:
                omitted.append("LD_PRELOAD (entry limit)")
        else:
            omitted.append("LD_PRELOAD (length limit)")
    return {"controls": controls, "ld_preload": preload, "omitted": omitted}


def _status(path: Path) -> dict | None:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_STATUS_BYTES + 1)
        if len(raw) > MAX_STATUS_BYTES:
            return None
    except OSError:
        return None
    fields = {}
    for line in raw.decode("ascii", errors="replace").splitlines():
        name, sep, value = line.partition(":")
        if sep and name in {"VmRSS", "VmHWM", "Threads", "Cpus_allowed_list"}:
            fields[name] = value.strip()
    out = {}
    for source, target in (("VmRSS", "rss_mib"), ("VmHWM", "vm_hwm_mib")):
        match = re.fullmatch(r"(\d+)\s+kB", fields.get(source, ""))
        out[target] = int(match[1]) / 1024 if match else None
    threads = fields.get("Threads", "")
    out["threads"] = int(threads) if threads.isdigit() else None
    affinity = fields.get("Cpus_allowed_list", "")
    out["affinity"] = affinity if len(affinity) <= 4096 and _CPU_LIST.fullmatch(affinity) else None
    return out


class ProcessMonitor:
    """Sample only ``pgid``; no signals, mutation, or inclusion of the separate benchmark client.

    ``start`` captures launch, ``settled`` captures readiness/idle state when called,
    and ``stop`` returns the complete JSON-safe result. All three are idempotent.
    """

    def __init__(self, pgid: int, requested_env: dict, poll_s: float = 0.05,
                 *, cwd: str | os.PathLike | None = None):
        if type(pgid) is not int or pgid <= 0 or not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("pgid and finite poll_s must be positive")
        self.pgid, self.poll_s = pgid, max(0.001, poll_s)
        self.requested = runtime_env(requested_env, cwd=cwd)
        self._event, self._lock = threading.Event(), threading.Lock()
        self._thread = None
        self._started = self._stopped = False
        self._samples = 0
        self._peak_rss = self._peak_threads = None
        self._processes, self._observations, self._affinity = {}, {}, set()
        self._notes = set()
        self._partial = False
        self._available = sys.platform.startswith("linux") and PROC_ROOT.is_dir()

    def _sample(self, phase: str | None = None):
        with self._lock:
            if not self._available:
                return
            try:
                members = servers.group_members(self.pgid)
            except OSError:
                members = None
            if members is None:
                self._notes.add("Process-group discovery is unavailable.")
                self._partial = True
                return
            if len(members) > MAX_PROCESSES:
                self._notes.add("Process count exceeded the sampling limit; group measurements are partial.")
                self._partial = True
            rows = []
            for pid in members[:MAX_PROCESSES]:
                row = _status(PROC_ROOT / str(pid) / "status")
                if row is None:
                    self._notes.add("A process status was unreadable or exited during sampling; brief peaks may be missed.")
                    self._partial = True
                    continue
                rows.append({"pid": pid, **row})
                if pid not in self._processes:
                    if len(self._processes) >= MAX_RECORDED_PROCESSES:
                        self._notes.add("Per-process diagnostic limit reached.")
                        continue
                    self._processes[pid] = {"pid": pid, "peak_rss_mib": None, "vm_hwm_mib": None,
                                            "peak_threads": None, "affinity": []}
                saved = self._processes[pid]
                for source, target in (("rss_mib", "peak_rss_mib"), ("vm_hwm_mib", "vm_hwm_mib"),
                                       ("threads", "peak_threads")):
                    if row[source] is not None:
                        saved[target] = max(saved[target] or 0, row[source])
                if row["affinity"] is not None:
                    if len(self._affinity) < 32:
                        self._affinity.add(row["affinity"])
                    saved["affinity"] = sorted(set(saved["affinity"]) | {row["affinity"]})[:32]
            rss = sum(r["rss_mib"] for r in rows) if rows and all(r["rss_mib"] is not None for r in rows) else None
            threads = sum(r["threads"] for r in rows) if rows and all(r["threads"] is not None for r in rows) else None
            if rss is not None:
                self._peak_rss = max(self._peak_rss or 0, rss)
            if threads is not None:
                self._peak_threads = max(self._peak_threads or 0, threads)
            self._samples += 1
            if phase:
                # Thread affinities can differ from the process leader. Inspect them only at
                # lifecycle boundaries, under one total bound, to keep the timing sampler small.
                masks, count = set(), 0
                for row in rows:
                    try:
                        with os.scandir(PROC_ROOT / str(row["pid"]) / "task") as entries:
                            for entry in entries:
                                if not entry.name.isdigit():
                                    continue
                                if count >= MAX_TASKS:
                                    self._notes.add("Thread affinity inspection reached its limit; masks are partial.")
                                    break
                                count += 1
                                task = _status(Path(entry.path) / "status")
                                if task and task["affinity"]:
                                    masks.add(task["affinity"])
                    except OSError:
                        self._notes.add("Per-thread affinity was unavailable for one or more processes.")
                self._observations[phase] = {"t_observed": time.time(), "processes": len(rows), "rss_mib": rss,
                                             "threads": threads, "affinity": sorted(masks), "tasks_checked": count}

    def _run(self):
        while not self._event.wait(self.poll_s):
            self._sample()

    def start(self):
        if not self._started and not self._stopped:
            self._started = True
            self._sample("launch")
            if self._available:
                self._thread = threading.Thread(target=self._run, name="inferencekit-process-monitor", daemon=True)
                self._thread.start()
        return self

    def settled(self):
        if self._started and not self._stopped and "settled" not in self._observations:
            self._sample("settled")
        return self

    def stop(self) -> dict:
        if not self._stopped:
            self._event.set()
            if self._thread is not None:
                self._thread.join()
            if self._started:
                self._sample("final")
            self._stopped = True
        limitations = ["Sampled RSS can miss peaks between observations; it is not an allocation high-water mark.",
                       "Group RSS sums resident pages and can count shared pages more than once.",
                       "Per-process VmHWM values are individual lifetime peaks; their sum is not a simultaneous group peak.",
                       "Observed threads include idle and runtime helper threads; requested controls do not prove effective pool sizes.",
                       "Processes that leave the server process group are outside this measurement."]
        if not self._available:
            limitations.append("Linux /proc telemetry is unavailable on this host.")
        if "settled" not in self._observations:
            limitations.append("No settled observation was requested after readiness.")
        return {"status": ("partial" if self._partial else "observed") if self._peak_rss is not None else "unavailable",
                "pgid": self.pgid,
                "method": "sampled_process_group_rss", "peak_rss_mib": self._peak_rss,
                "peak_threads": self._peak_threads, "affinity": sorted(self._affinity)[:32],
                "requested": self.requested, "observations": dict(self._observations),
                "processes": [self._processes[p] for p in sorted(self._processes)],
                "samples": self._samples, "poll_s": self.poll_s,
                "limitations": limitations + sorted(self._notes)}
