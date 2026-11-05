"""Parse Nsight Compute CSV exports into one summary per profiled kernel launch.

Two CSV layouts, both from ``ncu --import report.ncu-rep --csv``:

  --page raw      wide: row 1 = metric names, row 2 = units, then one row per launch.
                  Exact metric names (dram__bytes_read.sum, sm__throughput...).
  --page details  long: one row per (launch, section, metric) with human names
                  ("DRAM Throughput", "Achieved Occupancy").

Raw is preferred; details fills whatever raw lacks. Values are converted to base
units (ns, bytes, bytes/s) using the unit row, because ncu picks units per value
(usecond for one kernel, msecond for another) and its spelling changes between
releases ("Gbyte/second" in older ones, "Gbyte/s" and "us" in newer ones). Units are
parsed by structure (SI prefix + byte + optional per-second), and every field says
which kind of quantity it must be. A value whose unit is unknown, or of the wrong kind
(a "%" where a byte rate is expected), becomes None and is listed in
``unparsed_units``; it is never silently read as base units.

The "bound" label is a speed-of-light rule of thumb: ncu reports each unit's
throughput as % of its peak. If DRAM or SM is >= 60% of peak, that unit is the
limiter. If both are < 40%, neither is busy: the kernel is waiting (latency-bound:
too little parallelism, dependencies, or launch/tail effects). In between: mixed.
"""

from __future__ import annotations

import csv
import io
import re

# Time units -> nanoseconds.
TIME_NS = {"nsecond": 1.0, "ns": 1.0, "usecond": 1e3, "us": 1e3, "msecond": 1e6, "ms": 1e6, "second": 1e9, "s": 1e9}
SI_PREFIX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
# Byte amounts and byte rates: "byte", "Kbyte", "GB", "Gbyte/second", "Gbyte/s", "TB/s"...
BYTES_RX = re.compile(r"^(?P<prefix>[KMGT]?)(?:byte|B)(?P<rate>/(?:second|s))?$")
# Units that carry a physical dimension we do not convert (cycle/nsecond, Gbyte/ms, Hz...).
UNKNOWN_DIMENSION_RX = re.compile(r"byte|second|^[KMGT]?B(/|$)|hz|watt|/(?:[num]?s)$", re.IGNORECASE)

# What each field must be. Fields not listed are counts (registers, grid size, waves...).
FIELD_KIND = {"duration_ns": "time", "dram_read_bytes": "bytes", "dram_write_bytes": "bytes",
              "dram_bytes_per_s": "byte_rate", "dram_pct": "percent", "sm_pct": "percent", "mem_pct": "percent",
              "occupancy_achieved_pct": "percent", "occupancy_theoretical_pct": "percent", "l2_hit_pct": "percent"}

# field -> raw-page metric names, first present wins
RAW_FIELDS = {
    "duration_ns": ["gpu__time_duration.sum"],
    # Metric names differ across Nsight Compute releases (2025.x uses the *_op_* spelling).
    "dram_read_bytes": ["dram__bytes_read.sum", "dram__bytes_op_read.sum"],
    "dram_write_bytes": ["dram__bytes_write.sum", "dram__bytes_op_write.sum"],
    "dram_pct": ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                 "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "sm_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "mem_pct": ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
    "occupancy_achieved_pct": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "occupancy_theoretical_pct": ["sm__maximum_warps_per_active_cycle_pct"],
    "registers": ["launch__registers_per_thread"],
    "waves": ["launch__waves_per_multiprocessor"],
    "grid": ["launch__grid_size"],
    "block": ["launch__block_size"],
    "limit_registers": ["launch__occupancy_limit_registers"],
    "limit_shared_mem": ["launch__occupancy_limit_shared_mem"],
    "limit_warps": ["launch__occupancy_limit_warps"],
    "limit_blocks": ["launch__occupancy_limit_blocks"],
    "l2_hit_pct": ["lts__t_sector_hit_rate.pct"],
}

# field -> (section, metric) in the details page; section None = any section
DETAIL_FIELDS = {
    "duration_ns": [("GPU Speed Of Light Throughput", "Duration")],
    "dram_pct": [("GPU Speed Of Light Throughput", "DRAM Throughput")],
    "sm_pct": [("GPU Speed Of Light Throughput", "Compute (SM) Throughput")],
    "mem_pct": [("GPU Speed Of Light Throughput", "Memory Throughput")],
    "dram_bytes_per_s": [("Memory Workload Analysis", "Memory Throughput")],
    "occupancy_achieved_pct": [("Occupancy", "Achieved Occupancy")],
    "occupancy_theoretical_pct": [("Occupancy", "Theoretical Occupancy")],
    "registers": [("Launch Statistics", "Registers Per Thread")],
    "waves": [("Launch Statistics", "Waves Per SM")],
    "grid": [("Launch Statistics", "Grid Size")],
    "block": [("Launch Statistics", "Block Size")],
    "limit_registers": [("Occupancy", "Block Limit Registers")],
    "limit_shared_mem": [("Occupancy", "Block Limit Shared Mem")],
    "limit_warps": [("Occupancy", "Block Limit Warps")],
    "limit_blocks": [("Occupancy", "Block Limit SM")],
    "l2_hit_pct": [("Memory Workload Analysis", "L2 Hit Rate")],
}

# Warp stall reasons: cycles stalled per issued instruction, or PC-sampling counts.
STALL_RATIO_RX = re.compile(r"^smsp__average_warps_issue_stalled_(.+)_per_issue_active\.ratio$")
STALL_PCSAMP_RX = re.compile(r"^smsp__pcsamp_warps_issue_stalled_(.+?)(?:\.sum)?$")
NOT_A_STALL = {"selected", "not_selected"}  # "selected" = the warp issued; not a stall


def unit_kind(unit: str) -> tuple[str, float] | None:
    """(kind, factor to base units) for an ncu unit string, or None when it cannot be converted.

    'usecond' -> ('time', 1e3)        'Gbyte/s' -> ('byte_rate', 1e9)     'Kbyte' -> ('bytes', 1e3)
    '%'       -> ('percent', 1.0)     'block', 'register/thread', '' -> ('count', 1.0)
    'cycle/nsecond', 'Gbyte/ms'      -> None (a dimension this parser does not handle)
    """
    u = (unit or "").strip()
    if u == "%":
        return "percent", 1.0
    if u in TIME_NS:
        return "time", TIME_NS[u]
    m = BYTES_RX.match(u)
    if m:
        return ("byte_rate" if m["rate"] else "bytes"), SI_PREFIX[m["prefix"]]
    if UNKNOWN_DIMENSION_RX.search(u):
        return None
    return "count", 1.0


def to_number(text, unit: str = "", kind: str | None = None) -> float | None:
    """'1,234.5' + 'usecond' -> 1234500.0 (ns). Non-numeric -> None.

    ``kind`` is what the caller needs ('time', 'bytes', 'byte_rate', 'percent', 'count').
    Returns None when the unit is unknown or of another kind: 763 'Gbyte/s' must never be
    taken as 763 bytes/s, and a '%' must never be taken as a byte rate. A percent field
    also accepts a bare number (no unit).
    """
    try:
        x = float(str(text).replace(",", "").strip())
    except ValueError:
        return None
    found = unit_kind(unit)
    if found is None:
        return None
    got, factor = found
    if kind is not None and got != kind and not (kind == "percent" and got == "count" and not unit.strip()):
        return None
    return x * factor


def _field_value(field: str, text: str, unit: str, problems: list[str]) -> float | None:
    """Convert one field's value; note it in ``problems`` if it was there but could not be read."""
    value = to_number(text, unit, FIELD_KIND.get(field, "count"))
    if value is None and str(text).replace(",", "").strip() not in ("", "n/a", "N/A"):
        problems.append(f"{field}: {text!r} {unit!r}")
    return value


def _csv_rows(text: str) -> list[list[str]]:
    # ncu may print "==PROF==" or warning lines around the CSV; keep the quoted CSV lines only.
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    return list(csv.reader(io.StringIO("\n".join(lines))))


def parse_raw(text: str) -> list[dict]:
    """--page raw --csv -> [{id, kernel, fields..., stalls}] in profiling order."""
    rows = _csv_rows(text)
    if len(rows) < 2:
        return []
    header, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(header)}
    out = []
    for k, row in enumerate(data):
        if len(row) < len(header):
            continue
        rec = {"index": k, "id": row[col["ID"]] if "ID" in col else str(k),
               "kernel": row[col["Kernel Name"]] if "Kernel Name" in col else "", "_problems": []}
        for field, names in RAW_FIELDS.items():
            name = next((n for n in names if n in col), None)
            rec[field] = _field_value(field, row[col[name]], units[col[name]], rec["_problems"]) if name else None
        stalls = {}
        for name, i in col.items():
            m = STALL_PCSAMP_RX.match(name)
            # each reason also has a "_not_issued" twin (a subset): skip it to avoid double counting
            if m and m.group(1) not in NOT_A_STALL and not m.group(1).endswith("not_issued"):
                stalls[m.group(1)] = ("pcsamp", to_number(row[i]))
        if not stalls:
            for name, i in col.items():
                m = STALL_RATIO_RX.match(name)
                if m and m.group(1) not in NOT_A_STALL:
                    stalls[m.group(1)] = ("ratio", to_number(row[i]))
        rec["_stalls"] = stalls
        out.append(rec)
    return out


def parse_details(text: str) -> dict[str, dict]:
    """--page details --csv -> {launch id: {kernel, fields...}}."""
    rows = _csv_rows(text)
    if len(rows) < 2:
        return {}
    header = rows[0]
    col = {h: i for i, h in enumerate(header)}
    need = ("ID", "Metric Name", "Metric Unit", "Metric Value")
    if any(n not in col for n in need):
        return {}
    values: dict[str, dict] = {}
    for row in rows[1:]:
        if len(row) <= col["Metric Value"]:
            continue
        kid = row[col["ID"]]
        d = values.setdefault(kid, {"kernel": row[col["Kernel Name"]] if "Kernel Name" in col else "", "_m": {}})
        section = row[col["Section Name"]] if "Section Name" in col else ""
        d["_m"][(section, row[col["Metric Name"]])] = (row[col["Metric Value"]], row[col["Metric Unit"]])
    out = {}
    for kid, d in values.items():
        rec = {"kernel": d["kernel"], "_problems": []}
        for field, keys in DETAIL_FIELDS.items():
            rec[field] = None
            for section, name in keys:
                hit = d["_m"].get((section, name))
                if hit is None:  # section renamed in this ncu version: accept the name if it is unambiguous
                    same = [v for (_s, n), v in d["_m"].items() if n == name]
                    hit = same[0] if len(same) == 1 else None
                if hit is not None:
                    # With --set basic there is no Memory Workload Analysis section, and the only
                    # "Memory Throughput" left is the Speed Of Light one, in %. The kind check
                    # rejects it for the byte-rate field instead of reading 43% as 43 bytes/s.
                    rec[field] = _field_value(field, *hit, rec["_problems"])
                    break
        out[kid] = rec
    return out


def bound_label(dram_pct: float | None, sm_pct: float | None) -> str:
    if dram_pct is None and sm_pct is None:
        return "unknown"
    d, s = dram_pct or 0.0, sm_pct or 0.0
    if max(d, s) >= 60:
        return "memory-bound" if d >= s else "compute-bound"
    if d < 40 and s < 40:
        return "latency-bound"
    return "mixed"


def occupancy_limiter(rec: dict) -> str | None:
    """Which resource caps blocks per SM: the smallest of the per-resource block limits."""
    limits = {"registers": rec.get("limit_registers"), "shared memory": rec.get("limit_shared_mem"),
              "warps": rec.get("limit_warps"), "blocks per SM": rec.get("limit_blocks")}
    limits = {k: v for k, v in limits.items() if v is not None}
    return min(limits, key=limits.get) if limits else None


def summarize(raw_text: str = "", details_text: str = "", peak_gbs: float | None = None) -> list[dict]:
    """Merge both pages and derive GB/s, % of measured peak, limiter, stalls, bound label."""
    raw = parse_raw(raw_text) if raw_text else []
    details = parse_details(details_text) if details_text else {}
    if not raw:  # details only: build records from it, in id order
        raw = [{"index": i, "id": kid, **rec, "_stalls": {}}
               for i, (kid, rec) in enumerate(sorted(details.items(), key=lambda kv: _int_or(kv[0])))]
        details = {}  # every field is in raw now; nothing left to merge
    out = []
    for rec in raw:
        extra = details.get(rec["id"], {})
        problems = rec.pop("_problems", []) + extra.get("_problems", [])
        for k, v in extra.items():
            if k != "_problems" and rec.get(k) is None and v is not None:
                rec[k] = v
        dur = rec.get("duration_ns")
        moved = (rec.get("dram_read_bytes") or 0) + (rec.get("dram_write_bytes") or 0)
        gbs = moved / dur if dur and moved else None  # bytes per ns == GB/s
        if gbs is None and rec.get("dram_bytes_per_s") is not None:
            gbs = rec["dram_bytes_per_s"] / 1e9
        stalls = rec.pop("_stalls", {})
        total = sum(v for _, v in stalls.values() if v) or 0.0
        top = sorted(((v or 0.0, name, how) for name, (how, v) in stalls.items()), reverse=True)[:4]
        out.append({
            "index": rec["index"], "id": rec["id"], "kernel": rec.get("kernel", "")[:200],
            "duration_us": round(dur / 1e3, 3) if dur else None,
            "dram_gbs": round(gbs, 1) if gbs is not None else None,
            "dram_pct_of_measured_peak": round(100 * gbs / peak_gbs, 1) if gbs is not None and peak_gbs else None,
            "dram_pct": rec.get("dram_pct"), "sm_pct": rec.get("sm_pct"), "mem_pct": rec.get("mem_pct"),
            "occupancy_achieved_pct": rec.get("occupancy_achieved_pct"),
            "occupancy_theoretical_pct": rec.get("occupancy_theoretical_pct"),
            "occupancy_limiter": occupancy_limiter(rec),
            "registers": rec.get("registers"), "waves": rec.get("waves"), "grid": rec.get("grid"),
            "block": rec.get("block"), "l2_hit_pct": rec.get("l2_hit_pct"),
            "top_stalls": [{"reason": name, "value": round(v, 3), "share_pct": round(100 * v / total, 1) if total
                            else None, "source": how} for v, name, how in top if v > 0],
            "bound": bound_label(rec.get("dram_pct") if rec.get("dram_pct") is not None else rec.get("mem_pct"),
                                 rec.get("sm_pct")),
            "unparsed_units": problems,
        })
    return out


def _int_or(text: str):
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)
