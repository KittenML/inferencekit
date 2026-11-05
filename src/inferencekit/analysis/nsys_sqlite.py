"""Where decode time goes, from an Nsight Systems sqlite export.

Tables used (all optional except the kernel table):
  CUPTI_ACTIVITY_KIND_KERNEL   one row per kernel execution on the GPU (start/end in ns)
  CUPTI_ACTIVITY_KIND_MEMCPY   copies;  CUPTI_ACTIVITY_KIND_MEMSET   memsets
  CUPTI_ACTIVITY_KIND_RUNTIME  CUDA API calls on the CPU (cudaLaunchKernel, cudaGraphLaunch...)
  NVTX_EVENTS                  annotation ranges pushed by the application
  StringIds                    id -> text for kernel / API / NVTX names

Every GPU activity has a ``correlationId`` that matches the CPU API call that
launched it. Correlation ids are unique only *within one process*, so every join here
is on (process, correlation id): ``globalPid`` on the GPU side, and on the CPU side
``globalTid`` with its low 24 thread bits cleared (the same encoding nsys's own reports
use). With ``--cuda-graph-trace=node`` the kernels inside a CUDA graph carry the
correlation id of their ``cudaGraphLaunch``; that is how a graph replay is reassembled
from individual kernels.

What comes out:
  * GPU busy vs idle over the trace window (union of all GPU intervals);
  * a histogram of idle gaps (many 10-100 us gaps = launch overhead; ms gaps = CPU stalls);
  * top kernels and kernel families (gemm/gemv, attention, norm, ...);
  * cudaGraphLaunch count and the CPU time each call takes;
  * per decode step: span (first activity start to last activity end), busy, idle inside
    the step, the gap to the next step, and per family the critical-path time and share.
    The family shares plus the idle share add up to 100% of the step.
  * the names (never the values) of the environment variables the trace file stores.
"""

from __future__ import annotations

import bisect
import collections
import re
import sqlite3
from dataclasses import dataclass

from inferencekit.stats import quantile

NS_PER_US = 1e3
NS_PER_MS = 1e6

# Ordered: the first match wins. Norms go first because fused norm kernels can carry library
# names that look like attention ("flashinfer...fused_add_rmsnorm"). Attention is tested before
# gemm because fused attention kernels are often built from GEMM templates ("gemm", "cutlass").
FAMILY_RULES = [
    ("norm", r"rms_?norm|layer_?norm"),
    ("attention", r"attn|attention|flash|fmha|mha|paged|flashinfer|xformers|mla_"),
    ("gemm/gemv", r"gemm|gemv|matmul|mul_mat|mat_?vec|mmvq|mmq|cutlass|cublas|xmma|nvjet|wgmma|splitk|"
                  r"sm\d+_|_tn_|_nt_|_nn_|_tt_|dot"),
    # Activation quantization before a quantized matmul (e.g. llama.cpp quantize_q8_1, FP8 per-token
    # activation scaling). Its own family because it is pure overhead around the GEMV.
    ("quantize", r"quantize|quant_|_quant|scaled_fp8|fp8_quant|per_token_group"),
    ("norm", r"rms_?norm|layer_?norm|norm_kernel|_norm"),
    ("rope", r"rope|rotary|pos_enc"),
    ("sampling", r"argmax|topk|top_k|top_p|topp|sampl|multinomial|softmax|logsumexp|gumbel"),
    ("elementwise", r"elementwise|act_and_mul|silu|gelu|swiglu|copy|fill|cast|convert|scatter|gather|index|"
                    r"reduce|cat_|_cat|add|mul"),
]
_FAMILY_RX = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in FAMILY_RULES]
GAP_BUCKETS = [("<10us", 0, 10e3), ("10-100us", 10e3, 100e3), ("0.1-1ms", 100e3, 1e6), ("1-10ms", 1e6, 10e6),
               (">=10ms", 10e6, float("inf"))]
GRAPH_LAUNCH_RX = re.compile(r"^cu(da)?GraphLaunch")
DEFAULT_STEP_REGEX = r"^(step|decode)\b"
# Two decode steps further apart than this are in different requests: the pause is request
# setup or host work, not part of a step, and it is left out of the step-gap statistics.
BETWEEN_REQUESTS_NS = 20 * NS_PER_MS


def family(name: str) -> str:
    for fam, rx in _FAMILY_RX:
        if rx.search(name):
            return fam
    return "other"


@dataclass
class Activity:
    start: int
    end: int
    kind: str  # kernel | memcpy | memset
    name: str
    family: str
    corr: int | None
    stream: int | None
    device: int | None
    pid: int | None = None  # globalPid of the launching process (None in exports without that column)


@dataclass
class ApiCall:
    start: int
    end: int
    tid: int
    corr: int
    name: str
    pid: int | None = None  # the launching thread's process, from its globalTid


def pid_of_tid(global_tid: int | None) -> int | None:
    """nsys packs process and thread into one 64-bit globalTid; the low 24 bits are the thread.

    Clearing them gives the process's globalPid, the value the GPU activity tables carry.
    """
    return None if global_tid is None else global_tid >> 24 << 24


# --------------------------------------------------------------------------- interval helpers

def merge(intervals) -> list[list[int]]:
    """Sorted, non-overlapping [start, end] list covering the same time as the input."""
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered(merged: list[list[int]], starts: list[int], a: int, b: int) -> int:
    """How much of [a, b] is covered by the merged intervals (``starts`` = their start times)."""
    if b <= a:
        return 0
    i = max(0, bisect.bisect_right(starts, a) - 1)
    total = 0
    while i < len(merged) and merged[i][0] < b:
        lo, hi = max(merged[i][0], a), min(merged[i][1], b)
        if hi > lo:
            total += hi - lo
        i += 1
    return total


def dist(values: list[float], digits: int = 3) -> dict:
    """n / p50 / p95 / max / total of a list of durations."""
    if not values:
        return {"n": 0}
    r = lambda x: round(x, digits)  # noqa: E731
    return {"n": len(values), "p50": r(quantile(values, 50)), "p95": r(quantile(values, 95)), "max": r(max(values)),
            "total": r(sum(values))}


# --------------------------------------------------------------------------- loading

def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type in ('table', 'view')")}


def _columns(con, table) -> set[str]:
    return {r[1] for r in con.execute(f"pragma table_info({table})")}


def load(con) -> dict:
    tables = _tables(con)
    strings = dict(con.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    name_of = lambda v: strings.get(v, str(v)) if isinstance(v, int) else (v or "?")  # noqa: E731

    def optional(table, column):  # a column that older nsys exports may lack
        return column if column in _columns(con, table) else "NULL"

    acts: list[Activity] = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        t = "CUPTI_ACTIVITY_KIND_KERNEL"
        cols = _columns(con, t)
        short = "shortName" if "shortName" in cols else "demangledName"
        full = "demangledName" if "demangledName" in cols else short
        for s, e, st, c, d, pid, sn, fn in con.execute(
                f"select start, end, streamId, correlationId, {optional(t, 'deviceId')}, {optional(t, 'globalPid')}, "
                f"{short}, {full} from {t}"):
            label, long = name_of(sn), name_of(fn)
            acts.append(Activity(s, e, "kernel", label, family(long + " " + label), c, st, d, pid))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        t = "CUPTI_ACTIVITY_KIND_MEMCPY"
        for s, e, st, c, d, pid, nbytes in con.execute(
                f"select start, end, streamId, correlationId, {optional(t, 'deviceId')}, {optional(t, 'globalPid')}, "
                f"bytes from {t}"):
            acts.append(Activity(s, e, "memcpy", f"memcpy {nbytes} B", "memcpy", c, st, d, pid))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        t = "CUPTI_ACTIVITY_KIND_MEMSET"
        for s, e, st, c, d, pid in con.execute(
                f"select start, end, streamId, correlationId, {optional(t, 'deviceId')}, {optional(t, 'globalPid')} "
                f"from {t}"):
            acts.append(Activity(s, e, "memset", "memset", "memset", c, st, d, pid))

    calls: list[ApiCall] = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for s, e, tid, c, nid in con.execute(
                "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
            # API names carry version suffixes (cudaGraphLaunch_v10000): strip them.
            calls.append(ApiCall(s, e, tid, c, re.sub(r"_v\d+$", "", name_of(nid)), pid_of_tid(tid)))

    ranges = []
    if "NVTX_EVENTS" in tables:
        cols = _columns(con, "NVTX_EVENTS")
        text_id = "textId" if "textId" in cols else "NULL"
        # eventType 59 = push/pop range, 60 = start/end range
        for s, e, text, tid_, tid in con.execute(
                f"select start, end, text, {text_id}, globalTid from NVTX_EVENTS "
                f"where eventType in (59, 60) and end is not null and end > start"):
            label = text if text is not None else strings.get(tid_, "")
            ranges.append((s, e, label, tid))

    gpu_name = None
    if "TARGET_INFO_GPU" in tables and "name" in _columns(con, "TARGET_INFO_GPU"):
        row = con.execute("select name from TARGET_INFO_GPU limit 1").fetchone()
        gpu_name = row[0] if row else None
    return {"activities": acts, "calls": calls, "ranges": ranges, "gpu_name": gpu_name,
            "capture_env": capture_env(con, tables)}


# --------------------------------------------------------------------------- what the file stores about the launch

# META_DATA_CAPTURE rows in which nsys stores how the program was launched.
LAUNCH_ROWS = ("PROCESS_%:COMMAND", "PROCESS_%:ARGUMENT_%", "PROCESS_%:WORKING_DIR", "PROCESS_%:USER_NAME")
ENV_ROW = "PROCESS_%:ENVIRONMENT_VARIABLE"  # one row per variable, value "NAME=value"
HOST_ROWS = ("Hostname", "SystemUID", "NicInfo")  # in TARGET_INFO_SYSTEM_ENV
REDACTED = "<redacted>"


def capture_env(con, tables=None) -> dict:
    """Names (never values) of the variables the trace stores, and whether the launch details are still there."""
    tables = tables if tables is not None else _tables(con)
    if "META_DATA_CAPTURE" not in tables:
        return {"env_names": [], "values_present": False}
    rows = [v or "" for (v,) in con.execute("select value from META_DATA_CAPTURE where name like ?", (ENV_ROW,))]
    names = sorted({v.split("=", 1)[0] for v in rows})
    present = any(v.split("=", 1)[1:] not in ([], [REDACTED]) for v in rows)
    args = con.execute("select count(*) from META_DATA_CAPTURE where (name like ? or name like ?) and value <> ?",
                       ("PROCESS_%:ARGUMENT_%", "PROCESS_%:COMMAND", REDACTED)).fetchone()[0]
    return {"env_names": names, "values_present": present or args > 0}


def scrub_metadata(sqlite_path: str) -> dict:
    """Blank what nsys stores about the launch and the host, in an exported sqlite file (in place).

    nsys records the profiled program's command line, working directory, user name and
    ``NAME=value`` of every environment variable it received, plus the host name and
    every process running on the host. None of it is needed for the timeline analysis.
    Variable names are kept (useful to check the allowlist); values become <redacted>.
    The .nsys-rep this file was exported from still holds all of it and cannot be edited.
    """
    con = sqlite3.connect(sqlite_path)
    try:
        tables = _tables(con)
        before = capture_env(con, tables)
        blanked = 0
        if "META_DATA_CAPTURE" in tables:
            where = " or ".join("name like ?" for _ in LAUNCH_ROWS)
            blanked += con.execute(f"update META_DATA_CAPTURE set value = ? where {where}",
                                   (REDACTED, *LAUNCH_ROWS)).rowcount
            # "NAME=value" -> "NAME=<redacted>"
            blanked += con.execute("update META_DATA_CAPTURE set value = substr(value, 1, instr(value, '=')) || ? "
                                   "where name like ? and instr(value, '=') > 0", (REDACTED, ENV_ROW)).rowcount
        if "TARGET_INFO_SYSTEM_ENV" in tables:
            marks = ", ".join("?" for _ in HOST_ROWS)
            blanked += con.execute(f"update TARGET_INFO_SYSTEM_ENV set value = ? where name in ({marks})",
                                   (REDACTED, *HOST_ROWS)).rowcount
        dropped = 0
        kernels = "CUPTI_ACTIVITY_KIND_KERNEL"
        if "PROCESSES" in tables and kernels in tables and "globalPid" in _columns(con, kernels):
            # keep only the processes that ran GPU work
            dropped = con.execute("delete from PROCESSES where globalPid not in "
                                  "(select distinct globalPid from CUPTI_ACTIVITY_KIND_KERNEL)").rowcount
        con.commit()
    finally:
        con.close()
    return {"env_names": before["env_names"], "rows_blanked": blanked, "other_processes_dropped": dropped}


# --------------------------------------------------------------------------- step segmentation

# Each segmenter returns (groups, method, replay_ns) or None when it does not apply.
#   groups     {step key: [activities of that step]}
#   method     how steps were found ("nvtx", "graph_launch", "gap>20us")
#   replay_ns  {step key: span of the graph replay alone} for graph segmentation, else None

def _segment_by_nvtx(acts, calls, ranges, step_rx) -> tuple[dict, str, None] | None:
    """Group activities by the outermost NVTX step range open on the launching thread at launch time."""
    steps = [(s, e, tid) for s, e, label, tid in ranges if step_rx.search(label or "")]
    if not steps:
        return None
    by_thread: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for s, e, tid in sorted(steps):
        outer = by_thread[tid]
        if outer and s < outer[-1][1]:  # nested inside the previous step range: keep the outer one
            continue
        outer.append((s, e))
    call_of = {(c.pid, c.corr): c for c in calls}  # correlation ids repeat across processes
    groups: dict[tuple, list[Activity]] = collections.defaultdict(list)
    for a in acts:
        c = call_of.get((a.pid, a.corr))
        if c is None:
            continue
        rs = by_thread.get(c.tid, [])
        i = bisect.bisect_right([r[0] for r in rs], c.start) - 1
        if i >= 0 and c.start <= rs[i][1]:
            groups[(c.tid, rs[i][0])].append(a)
    return (groups, "nvtx", None) if groups else None


def _segment_by_graph(acts, calls) -> tuple[dict, str, dict] | None:
    """One step per cudaGraphLaunch: from the first kernel of replay i to the first kernel of replay i+1.

    Graph-node kernels carry the correlation id of their cudaGraphLaunch, which finds each
    replay. But a decode step is more than the graph: engines also run eager kernels
    between replays (vLLM: the lm_head GEMM and sampling, about 0.9 ms of a 6.7 ms token).
    So every activity that starts after replay i began and before replay i+1 begins is
    part of step i, unless the two replays are BETWEEN_REQUESTS_NS or more apart (then
    the work in between belongs to request setup, not to decoding). Eager work after the
    last replay has no next replay to bound it and is left out.
    """
    launch_start = {(c.pid, c.corr): c.start for c in calls if GRAPH_LAUNCH_RX.match(c.name)}
    if not launch_start:
        return None
    groups: dict[tuple, list[Activity]] = collections.defaultdict(list)
    eager = []
    for a in acts:
        key = (a.pid, a.corr)
        if key in launch_start:
            groups[(launch_start[key],) + key].append(a)
        else:
            eager.append(a)
    if not groups:
        return None
    order = sorted(groups, key=lambda k: min(a.start for a in groups[k]))
    firsts = [min(a.start for a in groups[k]) for k in order]
    replay_ns = {k: max(a.end for a in groups[k]) - first for k, first in zip(order, firsts)}
    for a in eager:
        i = bisect.bisect_right(firsts, a.start) - 1  # the last replay that began at or before this activity
        if 0 <= i < len(order) - 1 and firsts[i + 1] - firsts[i] < BETWEEN_REQUESTS_NS:
            groups[order[i]].append(a)
    return groups, "graph_launch", replay_ns


def _segment_by_gaps(acts, gap_us) -> tuple[dict, str, None]:
    """Fallback: a host-side pause longer than gap_us between kernels starts a new step."""
    groups: dict[tuple, list[Activity]] = {}
    current: list[Activity] = []
    running_end = 0
    for a in sorted(acts, key=lambda x: x.start):
        if current and a.start - running_end > gap_us * NS_PER_US:
            groups[(current[0].start,)] = current
            current = []
        if not current:
            running_end = a.end
        current.append(a)
        running_end = max(running_end, a.end)
    if current:
        groups[(current[0].start,)] = current
    return groups, f"gap>{gap_us:g}us", None


def step_stats(groups: dict, merged_all, starts_all, replay_ns: dict | None = None) -> dict:
    """Span / busy / idle per step, gaps between steps, and per-family critical path."""
    ordered = sorted(((sorted(v, key=lambda a: a.start), k) for k, v in groups.items() if v),
                     key=lambda sk: sk[0][0].start)
    steps = [s for s, _ in ordered]
    if not steps:
        return {"steps": 0}
    counts = collections.Counter(len(s) for s in steps)
    typical = counts.most_common(1)[0][0]
    # Steps with the most common activity count are the steady-state decode steps
    # (prefill or the first, cold step usually look different); drop the very first one.
    typical_idx = [i for i, s in enumerate(steps) if len(s) == typical]
    steady_idx = typical_idx[1:] or typical_idx
    steady = [steps[i] for i in steady_idx]
    spans, busys, idles, sums, idle_pcts = [], [], [], [], []
    crit = collections.defaultdict(list)
    crit_pct_total = collections.defaultdict(float)
    dur = collections.defaultdict(list)
    per_step_launches = collections.Counter(a.family for a in steady[0])
    for s in steady:
        span = max(a.end for a in s) - s[0].start
        busy = sum(e - b for b, e in merge((a.start, a.end) for a in s))
        spans.append(span / NS_PER_US)
        busys.append(busy / NS_PER_US)
        idles.append((span - busy) / NS_PER_US)
        idle_pcts.append(100 * (span - busy) / span if span else 0.0)
        sums.append(sum(a.end - a.start for a in s) / NS_PER_US)
        # Critical path: walk the step in start order and keep the running end time of
        # everything seen so far. Each activity is charged only for how far it pushes that
        # end time; a stretch with nothing running before it is idle, not the activity's
        # time (idle is reported on its own as idle_in_step_us). So the family times plus
        # idle add up exactly to the step span. With overlapping kernels (programmatic
        # dependent launch, several streams) summed durations exceed the span; this counts
        # each nanosecond of the step once.
        step_crit, step_dur = collections.defaultdict(float), collections.defaultdict(float)
        running_end = s[0].start
        for a in s:
            running_end = max(running_end, a.start)  # skip [running_end, a.start): nothing ran, it is idle
            new_end = max(running_end, a.end)
            step_crit[a.family] += (new_end - running_end) / NS_PER_US
            running_end = new_end
            step_dur[a.family] += (a.end - a.start) / NS_PER_US
        for f, v in step_crit.items():
            crit[f].append(v)
            crit_pct_total[f] += 100 * v * NS_PER_US / span if span else 0.0
        for f, v in step_dur.items():
            dur[f].append(v)
    gaps, fully_idle = [], []
    for prev, nxt in zip(steps, steps[1:]):
        end_prev = max(a.end for a in prev)
        gap = nxt[0].start - end_prev
        if gap < BETWEEN_REQUESTS_NS:  # longer pauses are between requests, not between decode steps
            gaps.append(gap / NS_PER_US)
            idle = gap - covered(merged_all, starts_all, end_prev, nxt[0].start) if gap > 0 else 0
            fully_idle.append(idle / NS_PER_US)
    # critical_pct_of_step is the mean over steady steps of (family critical time / step span).
    # Means, unlike medians, add up: the families' shares plus idle_pct_of_step are 100%.
    families = {}
    for f in sorted(crit, key=lambda f: -quantile(crit[f], 50)):
        families[f] = {"launches_per_step": per_step_launches.get(f, 0),
                       "critical_us_p50": round(quantile(crit[f], 50), 2),
                       "critical_pct_of_step": round(crit_pct_total[f] / len(steady), 2),
                       "duration_us_p50": round(quantile(dur[f], 50), 2)}
    out = {"steps": len(steps), "steady_steps": len(steady), "kernels_per_step": typical,
           "span_us": dist(spans, 2), "busy_us": dist(busys, 2), "idle_in_step_us": dist(idles, 2),
           "idle_pct_of_step": round(sum(idle_pcts) / len(idle_pcts), 2),
           "kernel_sum_us": dist(sums, 2), "gap_to_next_step_us": dist(gaps, 2),
           "gap_fully_idle_us": dist(fully_idle, 2), "families": families}
    if replay_ns:  # graph segmentation: the replay alone, without the eager work after it
        out["graph_replay_us"] = dist([replay_ns[ordered[i][1]] / NS_PER_US for i in steady_idx], 2)
    return out


# --------------------------------------------------------------------------- top level

def analyze(sqlite_path: str, step_regex: str = DEFAULT_STEP_REGEX, gap_us: float = 20.0, top: int = 25) -> dict:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        d = load(con)
    finally:
        con.close()
    acts: list[Activity] = d["activities"]
    if not acts:
        return {"error": "no GPU activity in this trace (was CUDA tracing on?)", "gpu_name": d["gpu_name"]}
    # Several GPUs: analyse the one with the most kernel time.
    per_dev = collections.Counter()
    for a in acts:
        per_dev[a.device] += a.end - a.start
    device = per_dev.most_common(1)[0][0]
    notes = []
    if len(per_dev) > 1:
        notes.append(f"{len(per_dev)} devices in trace; analysed device {device} (most GPU time)")
    acts = [a for a in acts if a.device == device]
    # Keep only the API calls of the processes that ran work on this device. Correlation ids
    # are per process, so another process's calls would otherwise be joined to these kernels.
    calls: list[ApiCall] = d["calls"]
    pids = {a.pid for a in acts}
    if None in pids:  # an export without globalPid: fall back to joining on correlation id alone
        for x in [*acts, *calls]:
            x.pid = None
        notes.append("trace has no globalPid column: kernels joined to API calls by correlation id alone "
                     "(wrong if several processes launch work)")
    else:
        calls = [c for c in calls if c.pid in pids]

    merged = merge((a.start, a.end) for a in acts)
    starts = [m[0] for m in merged]
    t0, t1 = merged[0][0], merged[-1][1]
    window = t1 - t0
    busy = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    histogram = {label: {"count": sum(1 for g in gaps if lo <= g < hi),
                         "idle_ms": round(sum(g for g in gaps if lo <= g < hi) / NS_PER_MS, 3)}
                 for label, lo, hi in GAP_BUCKETS}

    kernels = [a for a in acts if a.kind == "kernel"]
    kernel_ns = sum(a.end - a.start for a in kernels) or 1
    by_name: dict[str, list] = collections.defaultdict(lambda: [0, 0, ""])
    by_family: dict[str, list] = collections.defaultdict(lambda: [0, 0])
    for a in kernels:
        x = by_name[a.name]
        x[0] += 1
        x[1] += a.end - a.start
        x[2] = a.family
        f = by_family[a.family]
        f[0] += 1
        f[1] += a.end - a.start
    top_kernels = [{"name": n[:160], "family": v[2], "count": v[0], "total_ms": round(v[1] / NS_PER_MS, 3),
                    "mean_us": round(v[1] / v[0] / NS_PER_US, 2), "pct_kernel_time": round(100 * v[1] / kernel_ns, 2)}
                   for n, v in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top]]
    families = {f: {"count": v[0], "total_ms": round(v[1] / NS_PER_MS, 3),
                    "pct_kernel_time": round(100 * v[1] / kernel_ns, 2)}
                for f, v in sorted(by_family.items(), key=lambda kv: -kv[1][1])}

    launches = [c for c in calls if GRAPH_LAUNCH_RX.match(c.name)]
    graph = {"launches": len(launches), "host_us": dist([(c.end - c.start) / NS_PER_US for c in launches], 2),
             "host_ms_total": round(sum(c.end - c.start for c in launches) / NS_PER_MS, 3)}

    step_rx = re.compile(step_regex)
    seg = (_segment_by_nvtx(acts, calls, d["ranges"], step_rx) or _segment_by_graph(acts, calls)
           or _segment_by_gaps(acts, gap_us))
    groups, method, replay_ns = seg
    steps = step_stats(groups, merged, starts, replay_ns)
    steps["segmentation"] = method
    if method.startswith("gap"):
        notes.append("no NVTX step ranges and no graph launches: steps split at host gaps (approximate)")
    if d["capture_env"]["values_present"]:
        notes.append("this trace file stores the profiled command line and NAME=value of every environment "
                     f"variable passed ({', '.join(d['capture_env']['env_names']) or 'none'}): treat it as sensitive")

    return {
        "gpu_name": d["gpu_name"],
        "device": device,
        "window_ms": round(window / NS_PER_MS, 3),
        "gpu_busy_ms": round(busy / NS_PER_MS, 3),
        "gpu_idle_ms": round((window - busy) / NS_PER_MS, 3),
        "gpu_busy_pct": round(100 * busy / window, 2) if window else None,
        "activity_counts": dict(collections.Counter(a.kind for a in acts)),
        "idle_gap_histogram": histogram,
        "families": families,
        "top_kernels": top_kernels,
        "graph_launch": graph,
        "steps": steps,
        "capture_env_names": d["capture_env"]["env_names"],
        "notes": notes,
    }
