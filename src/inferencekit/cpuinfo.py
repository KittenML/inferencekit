"""Selected CPU facts; no framework imports, counter probes, or workloads.

Do not retain raw /proc output, process names, host names, serials, or paths.
Visible ISA support and configuration are not proof of an engine's dispatch.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path

from inferencekit.proc import which

CPU_ENV_RULES = {
    "OMP_NUM_THREADS": r"[1-9][0-9]*", "MKL_NUM_THREADS": r"[1-9][0-9]*",
    "OPENBLAS_NUM_THREADS": r"[1-9][0-9]*", "BLIS_NUM_THREADS": r"[1-9][0-9]*",
    "NUMEXPR_NUM_THREADS": r"[1-9][0-9]*", "OMP_PLACES": r"(?:threads|cores|sockets)",
    "OMP_PROC_BIND": r"(?:true|false|master|primary|close|spread)",
    "OMP_DYNAMIC": r"(?:true|false)", "MKL_DYNAMIC": r"(?:true|false)",
    "MKL_ENABLE_INSTRUCTIONS": r"(?:SSE4_2|AVX|AVX2|AVX512|AVX512_MIC|AVX512_MIC_E1|AVX512_E1)",
    "DNNL_MAX_CPU_ISA": r"(?:DEFAULT|SSE41|AVX|AVX2|AVX2_VNNI|AVX512_CORE|AVX512_CORE_VNNI|AVX512_CORE_BF16|AVX512_CORE_AMX|AVX512_CORE_AMX_FP16)",
    "ONEDNN_MAX_CPU_ISA": r"(?:DEFAULT|SSE41|AVX|AVX2|AVX2_VNNI|AVX512_CORE|AVX512_CORE_VNNI|AVX512_CORE_BF16|AVX512_CORE_AMX|AVX512_CORE_AMX_FP16)",
    "MKL_VERBOSE": r"[01]", "DNNL_VERBOSE": r"[012]", "ONEDNN_VERBOSE": r"[012]",
}
ISA_FLAGS = {"avx2": ("avx2",), "avx512": ("avx512f",), "avx512_vnni": ("avx512_vnni",),
             "avx_vnni": ("avx_vnni",), "avx512_bf16": ("avx512_bf16",),
             "amx_bf16": ("amx_bf16",), "amx_int8": ("amx_int8",), "neon": ("asimd", "neon")}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip() or None
    except OSError:
        return None


def _integer(text) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _topology_id(text) -> int | None:
    value = _integer(text)
    return value if value is not None and value >= 0 else None


def cpu_list(text: str | None) -> list[int]:
    """Parse a Linux CPU list; malformed/absurd masks are unknown, not partial."""
    if not text:
        return []
    result = set()
    for part in text.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            return []
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if a > b or b > 1_000_000 or b - a > 100_000:
            return []
        result.update(range(a, b + 1))
    return sorted(result)


def cpu_tuning_env(environ=None) -> dict:
    env = os.environ if environ is None else environ
    return {key: value for key, pattern in CPU_ENV_RULES.items()
            if isinstance(value := env.get(key), str) and re.fullmatch(pattern, value, re.I)}


def cpu_facts(*, proc_root=Path("/proc"), cpu_root=Path("/sys/devices/system/cpu"),
              node_root=Path("/sys/devices/system/node"), allowed_cpus=None) -> dict:
    """Read visible host topology and process affinity separately (containers may differ)."""
    blocks = (_read(proc_root / "cpuinfo") or "").split("\n\n")
    info = []
    for block in blocks:
        fields = {k.strip(): v.strip() for line in block.splitlines() if ":" in line
                  for k, v in [line.split(":", 1)]}
        if fields:
            info.append(fields)
    model = next((r.get("model name") or r.get("Processor") for r in info
                  if r.get("model name") or r.get("Processor")), None)
    if allowed_cpus is None:
        try:
            allowed_cpus = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            allowed_cpus = None
    online = cpu_list(_read(cpu_root / "online"))
    if not online:
        online = sorted(n for r in info if (n := _integer(r.get("processor"))) is not None)
    topology, l3 = [], {}
    for n in online:
        root = cpu_root / f"cpu{n}"
        topology.append({"cpu": n, "package": _topology_id(_read(root / "topology/physical_package_id")),
                         "core": _topology_id(_read(root / "topology/core_id")),
                         "siblings": cpu_list(_read(root / "topology/thread_siblings_list")),
                         "core_type": _integer(_read(root / "topology/core_type"))})
        for cache in root.glob("cache/index*"):
            if _read(cache / "level") != "3":
                continue
            shared = cpu_list(_read(cache / "shared_cpu_list"))
            if shared:
                size = _read(cache / "size")
                l3[tuple(shared)] = {"cpus": shared, "size": size if size and re.fullmatch(r"\d+[KMG]?", size) else None}
    cores = {(r["package"], r["core"]) for r in topology if r["package"] is not None and r["core"] is not None}
    topology_complete = bool(topology) and all(r["package"] is not None and r["core"] is not None for r in topology)
    nodes = [{"node": int(p.name[4:]), "cpus": cpu_list(_read(p / "cpulist"))}
             for p in sorted(node_root.glob("node[0-9]*")) if re.fullmatch(r"node\d+", p.name)]
    flag_sets = [set(r.get("flags", r.get("Features", "")).split()) for r in info
                 if "flags" in r or "Features" in r]
    common = set.intersection(*flag_sets) if flag_sets else None
    present = set.union(*flag_sets) if flag_sets else None
    governors = sorted({v for p in cpu_root.glob("cpufreq/policy*/scaling_governor")
                        if (v := _read(p)) and re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", v)})
    status = _read(proc_root / "self/status") or ""
    cap_match = re.search(r"^CapEff:\s*([0-9a-fA-F]+)\s*$", status, re.M)
    caps = int(cap_match.group(1), 16) if cap_match else None
    paranoid = _integer(_read(proc_root / "sys/kernel/perf_event_paranoid"))
    cap_perf = bool(caps & (1 << 38)) if caps is not None else None
    cap_admin = bool(caps & (1 << 21)) if caps is not None else None
    perf_present = which("perf") is not None
    access = ("policy_allows_user" if cap_perf or cap_admin or (paranoid is not None and paranoid <= 2)
              else "restricted" if paranoid is not None and paranoid > 2 and caps is not None else "unknown")
    load_fields = (_read(proc_root / "loadavg") or "").split()
    try:
        loads = [float(v) for v in load_fields[:3]] if len(load_fields) >= 3 else None
        if loads is not None and any(not (0 <= v < float("inf")) for v in loads):
            loads = None
    except ValueError:
        loads = None
    logical = len(online) or os.cpu_count()
    warnings = []
    capacity = len(allowed_cpus) if allowed_cpus else logical
    if loads and capacity and loads[0] >= max(1, capacity / 2):
        warnings.append("Host one-minute load is at least half the available logical CPU count; contention may skew timings. "
                        "Load is host-wide, may include tasks outside this affinity, and includes uninterruptible tasks; "
                        "it is not per-core utilization.")
    if not topology_complete:
        warnings.append("Physical-core topology is incomplete; do not infer SMT or placement from CPU numbering.")
    if not any(r["core_type"] is not None for r in topology):
        warnings.append("Hybrid core types are unavailable; no P/E assignment is inferred from model or frequency.")
    return {"model": model[:200] if model else None, "architecture": platform.machine(),
            "logical_cores": logical, "physical_cores": len(cores) if topology_complete else None,
            "allowed_cpus": sorted(set(allowed_cpus)) if allowed_cpus is not None else None,
            "topology": topology, "l3_groups": list(l3.values()), "numa_nodes": nodes,
            "isa": {k: any(f in common for f in fs) if common is not None else None for k, fs in ISA_FLAGS.items()},
            "isa_any_cpu": {k: any(f in present for f in fs) if present is not None else None for k, fs in ISA_FLAGS.items()},
            "governors": governors, "load_average": loads,
            "perf": {"installed": perf_present, "perf_event_paranoid": paranoid,
                     "cap_perfmon": cap_perf, "cap_sys_admin": cap_admin, "access": access, "verified": False},
            "tuning_env": cpu_tuning_env(),
            "library_dispatch": {"status": "unverified", "reason": "ISA flags and environment pins do not establish "
                                 "loaded BLAS/oneDNN libraries or selected kernels; retain engine runtime dispatch logs."},
            "limitations": warnings}


def cpu_identity(facts=None) -> dict:
    facts = cpu_facts() if facts is None else facts
    result = {k: facts.get(k) for k in ("model", "architecture", "logical_cores", "physical_cores", "allowed_cpus")}
    topology = {k: facts.get(k) for k in ("topology", "l3_groups", "numa_nodes", "isa")}
    result["topology_sha256"] = (hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest()
                                  if facts.get("topology") else None)
    return result
