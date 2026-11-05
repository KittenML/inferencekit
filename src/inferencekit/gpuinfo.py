"""GPU facts from nvidia-smi, shared by doctor and by every GPU-touching operation."""

from __future__ import annotations

import re

from inferencekit.proc import run, which

# Asked for one by one if the combined query fails (older drivers lack some fields).
QUERY_FIELDS = [
    "index", "name", "uuid", "driver_version", "memory.total", "compute_cap", "pstate", "persistence_mode",
    "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem", "clocks.applications.graphics",
    "clocks.default_applications.graphics", "clocks.applications.memory", "clocks.default_applications.memory",
    "power.limit", "temperature.gpu",
]


def _query(fields: list[str]) -> list[dict] | None:
    smi = which("nvidia-smi")
    if not smi:
        return None
    r = run([smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout=30)
    if not r.ok:
        return None
    rows = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append({f: (None if v in ("[N/A]", "N/A", "[Not Supported]", "") else v) for f, v in zip(fields, parts)})
    return rows


def query_gpus() -> list[dict]:
    """One dict per visible GPU (empty list without nvidia-smi or a driver)."""
    rows = _query(QUERY_FIELDS)
    if rows is not None:
        return rows
    # Field-by-field fallback: keep whatever this driver supports.
    base = _query(["index", "name", "driver_version"])
    if not base:
        return []
    for f in QUERY_FIELDS:
        if f in ("index", "name", "driver_version"):
            continue
        extra = _query(["index", f])
        if extra:
            by_index = {e["index"]: e[f] for e in extra}
            for row in base:
                row[f] = by_index.get(row["index"])
    return base


def driver_cuda_version() -> str | None:
    """Highest CUDA version the driver supports, from the nvidia-smi banner."""
    smi = which("nvidia-smi")
    if not smi:
        return None
    r = run([smi], timeout=30)
    m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
    return m.group(1) if m else None


def clocks_state(gpu: dict) -> str | None:
    """'locked' when application clocks differ from their defaults, else 'default'.

    Clock locks matter because a benchmark at boost clocks and one at locked base
    clocks are different experiments. Locks set with ``nvidia-smi -lgc`` are not
    exposed by every driver; doctor records that caveat.
    """
    pairs = [("clocks.applications.graphics", "clocks.default_applications.graphics"),
             ("clocks.applications.memory", "clocks.default_applications.memory")]
    known = False
    for cur, default in pairs:
        if gpu.get(cur) and gpu.get(default):
            known = True
            if gpu[cur] != gpu[default]:
                return "locked"
    return "default" if known else None


def gpu_identity(gpus: list[dict] | None = None) -> dict:
    """The ``identity.gpu`` block of an evidence record (first visible GPU, plus count)."""
    gpus = query_gpus() if gpus is None else gpus
    if not gpus:
        from inferencekit.hardware import discover_non_nvidia
        devices = discover_non_nvidia()["devices"]
        if not devices:
            return {}
        return {"name": devices[0]["name"], "count": len(devices), "driver": None,
                "cuda": None, "clocks": None}
    g = gpus[0]
    return {"name": g.get("name"), "count": len(gpus), "driver": g.get("driver_version"),
            "cuda": driver_cuda_version(), "clocks": clocks_state(g)}
