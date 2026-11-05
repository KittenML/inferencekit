"""nsys capture / nsys analyze: a timeline of every kernel, copy and CUDA API call.

Capture command (environment reduced to an allowlist, see proc.clean_env):

    nsys profile -t cuda,nvtx --cuda-graph-trace=node -s none --cpuctxsw=none -o <out> <argv>
    nsys export --type sqlite -o <out>.sqlite <out>.nsys-rep

  -t cuda,nvtx                trace CUDA API + GPU activity, and NVTX annotation ranges
  --cuda-graph-trace=node     record each kernel inside a CUDA graph, not one opaque block
                              (without it, a decode step under graphs is a single bar)
  -s none / --cpuctxsw=none   no CPU sampling or context-switch tracing: less overhead,
                              smaller files; this capture is about the GPU timeline

What the report file holds: nsys stores the profiled program's full command line,
working directory, user name and NAME=value of every environment variable it received,
plus the host name and the host's process list. The evidence record keeps only a digest
of the command and the variable names. The exported .sqlite is scrubbed of those values
(nsys_sqlite.scrub_metadata); the .nsys-rep cannot be edited and keeps them, so treat it
as sensitive. That is also why secret-looking names are refused for --keep-env.
"""

from __future__ import annotations

from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.analysis import nsys_sqlite
from inferencekit.gpuinfo import gpu_identity
from inferencekit.proc import clean_env, dropped_config_env, require, run, tool_version
from inferencekit.registry import Arg, OpError, operation

REP_IS_SENSITIVE = ("the .nsys-rep stores the profiled command line, working directory, user and host name, and "
                    "NAME=value of every passed variable ({names}): treat it as sensitive and do not share it; the "
                    "exported .sqlite has these values blanked")


def export_sqlite(nsys: str, rep: Path, out: Path) -> None:
    r = run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(out), str(rep)],
            timeout=3600)
    if not r.ok or not out.is_file():
        raise OpError(f"nsys export failed (exit {r.returncode}): {r.stderr[-800:]}")


def redact_outputs(staging: Path) -> None:
    """The program's own output may echo secrets: redact it as soon as the run ends."""
    for name in ("stdout.txt", "stderr.txt"):
        p = staging / name
        if p.is_file():
            p.write_text(ev.redact_text(p.read_text(errors="replace")))


@operation("capture_nsys", ("nsys", "capture"), "Run a command under 'nsys profile' with an allowlisted "
           "environment (env -i equivalent) and export the trace to sqlite.",
           [Arg("keep_env", "array", "Extra environment variable NAMES to pass through. The evidence record stores "
                "names only, but nsys writes NAME=value of every passed variable (and the command line) into the "
                ".nsys-rep, so secret-looking names (*TOKEN*, *KEY*, *SECRET*...) are refused.", default=[]),
            Arg("trace", "string", "nsys -t value.", default="cuda,nvtx"),
            Arg("capture_range", "string", "Limit capture to cudaProfilerStart/Stop or an NVTX range.",
                default="none", choices=("none", "cudaProfilerApi", "nvtx")),
            Arg("nvtx_capture", "string", "NVTX range name for capture_range=nvtx."),
            Arg("delay", "number", "Seconds to wait before collecting."),
            Arg("duration", "number", "Seconds to collect."),
            Arg("timeout", "integer", "Seconds before the run (and every process it started) is killed.",
                default=3600)],
           passthrough=True)
def capture_nsys(call, keep_env, trace, capture_range, nvtx_capture, delay, duration, timeout, argv):
    nsys = require("nsys", "Install Nsight Systems (it ships with the CUDA toolkit).")
    if capture_range == "nvtx" and not nvtx_capture:
        raise OpError("capture_range=nvtx needs nvtx_capture=<range name>")
    env, env_names = clean_env(keep_env)  # raises on secret-looking names, before anything runs
    dropped = dropped_config_env(env)
    store = ev.Store()
    with store.staging_dir() as staging:  # removed again if anything below fails
        base = staging / "profile"
        cmd = [nsys, "profile", "-t", trace, "--cuda-graph-trace=node", "-s", "none", "--cpuctxsw=none",
               "--force-overwrite", "true", "-o", str(base)]
        if capture_range != "none":
            cmd += ["--capture-range", capture_range]
        if nvtx_capture:
            cmd += ["--nvtx-capture", nvtx_capture]
        if delay:
            cmd += ["--delay", str(delay)]
        if duration:
            cmd += ["--duration", str(duration)]
        r = run(cmd + list(argv), timeout=timeout, env=env, stdout_path=staging / "stdout.txt",
                stderr_path=staging / "stderr.txt")
        redact_outputs(staging)
        rep = base.with_suffix(".nsys-rep")
        if not rep.is_file():
            raise OpError(f"nsys produced no report (exit {r.returncode}): {ev.redact_text(r.stderr[-800:])}")
        sqlite = base.with_suffix(".sqlite")
        export_sqlite(nsys, rep, sqlite)
        scrubbed = nsys_sqlite.scrub_metadata(str(sqlite))
        limitations, missing = [], []
        if r.returncode != 0:
            limitations.append(f"profiled command exited with code {r.returncode}: the trace may be partial")
            missing.append("clean exit")
        limitations.append(REP_IS_SENSITIVE.format(names=", ".join(env_names)))
        if dropped:
            limitations.append("set here but not passed to the profiled run, so the configuration may differ from a "
                               "benchmark run with the full environment: " + ", ".join(dropped)
                               + " (pass each one that matters with --keep-env NAME)")
        summary = f"nsys trace ({rep.stat().st_size / 1e6:.1f} MB), exit {r.returncode}; run 'nsys analyze' on it"
        record = ev.build(call, kind="nsys_capture", claim_class="observed", summary=summary,
                          identity_=ev.identity(gpu=gpu_identity()),
                          artifacts=[ev.artifact(rep, staging, "nsys_rep"), ev.artifact(sqlite, staging, "sqlite"),
                                     ev.artifact(staging / "stderr.txt", staging, "stderr")],
                          missing=missing, limitations=limitations,
                          data={"nsys_version": tool_version("nsys"), "exit_code": r.returncode,
                                "env_names": env_names, "env_dropped": dropped, "sqlite_scrubbed": scrubbed,
                                "trace": trace, "cuda_graph_trace": "node", "capture_range": capture_range})
        return store.save(record, staging)


def _resolve_input(store: ev.Store, source: str, staging: Path) -> tuple[Path, bool, dict | None]:
    """(sqlite path, whether it was exported into ``staging``, source record if an evidence id was given)."""
    p = Path(source)
    if p.suffix == ".sqlite" and p.is_file():
        return p, False, None
    if p.suffix == ".nsys-rep" and p.is_file():
        out = staging / (p.stem + ".sqlite")
        export_sqlite(require("nsys", "nsys is needed to export .nsys-rep files"), p, out)
        nsys_sqlite.scrub_metadata(str(out))  # this copy is stored as an artifact
        return out, True, None
    rec = store.load(source)  # an evidence id of an nsys capture
    if rec.get("kind") != "nsys_capture":
        raise OpError(f"{source}: expected a .sqlite, a .nsys-rep or an nsys_capture evidence id")
    art = next((a for a in rec["artifacts"] if a.get("role") == "sqlite"), None)
    if art is None:
        raise OpError(f"{source}: capture has no sqlite artifact")
    return store.run_dir(rec["evidence_id"]) / art["path"], False, rec


def timeline_metrics(result: dict) -> tuple[list[dict], list[str]]:
    """Evidence metrics for an analysed trace, and any limitations they need.

    Families are ranked by critical-path share of the decode step (``critical_path_pct``),
    with GPU idle inside the step as its own row, so the rows add up to 100%. Summed
    kernel durations are not used for ranking: when kernels overlap (programmatic
    dependent launch, several streams) they over-count; ``kernel_sum_over_span`` > 1
    says by how much.
    """
    steps = result["steps"]
    metrics = [ev.metric("gpu_busy_pct", result["gpu_busy_pct"], "%", "window"),
               ev.metric("gpu_idle_ms", result["gpu_idle_ms"], "ms", "window"),
               ev.metric("graph_launches", result["graph_launch"]["launches"], "count", "total")]
    limitations = []
    host = result["graph_launch"]["host_us"]
    if host.get("n"):
        metrics.append(ev.metric("graph_launch_host_us", host["p50"], "us", "median", n=host["n"]))
    # step_gap_us is time with *nothing* running on the GPU between two steps (not just the
    # gap between the steps' kernels, which can contain other GPU work).
    for key, name in (("span_us", "step_span_us"), ("idle_in_step_us", "step_idle_us"),
                      ("gap_fully_idle_us", "step_gap_us")):
        d = steps.get(key) or {}
        if d.get("n"):
            metrics.append(ev.metric(name, d["p50"], "us", "median", n=d["n"]))
    span = (steps.get("span_us") or {}).get("p50")
    kernel_sum = (steps.get("kernel_sum_us") or {}).get("p50")
    families = steps.get("families") or {}
    n = steps.get("steady_steps")
    if span and kernel_sum:
        ratio = kernel_sum / span
        metrics.append(ev.metric("kernel_sum_over_span", round(ratio, 3), "ratio", "median", n=n))
        if ratio > 1.05:
            limitations.append(f"kernels overlap: summed kernel time is {ratio:.2f}x the step span, so shares of "
                               "summed duration over-count; rank by critical_path_pct")
    if families:
        metrics.append(ev.metric("critical_path_pct", steps.get("idle_pct_of_step"), "%", "mean_step_share", n=n,
                                 subject="GPU idle inside the step"))
        for f, v in list(families.items())[:6]:  # already sorted by critical-path time
            metrics.append(ev.metric("critical_path_pct", v["critical_pct_of_step"], "%", "mean_step_share", n=n,
                                     subject=f))
    else:  # no decode steps found: only whole-window shares of summed kernel time are available
        for f, v in list(result["families"].items())[:6]:
            metrics.append(ev.metric("kernel_time_pct", v["pct_kernel_time"], "%", "summed_duration_share",
                                     subject=f))
        limitations.append("no decode steps found: family shares are of summed kernel time over the whole window, "
                           "which over-counts overlapping kernels")
    return metrics, limitations


@operation("analyze_nsys", ("nsys", "analyze"), "Per-step GPU busy/idle, idle-gap histogram, kernel families "
           "ranked by critical-path share of the step, graph launches and top kernels from an nsys trace.",
           [Arg("source", "string", "A .nsys-rep, a .sqlite export, or an nsys_capture evidence id.", required=True,
                positional=True),
            Arg("step_regex", "string", "NVTX range names that mark one decode step.",
                default=nsys_sqlite.DEFAULT_STEP_REGEX),
            Arg("gap_us", "number", "Fallback segmentation: a host gap longer than this starts a new step.",
                default=20.0),
            Arg("top", "integer", "How many top kernels to keep.", default=25)])
def analyze_nsys(call, source, step_regex, gap_us, top):
    store = ev.Store()
    with store.staging_dir() as staging:
        sqlite, exported, src_rec = _resolve_input(store, source, staging)
        result = nsys_sqlite.analyze(str(sqlite), step_regex=step_regex, gap_us=gap_us, top=top)
        if "error" in result:
            raise OpError(result["error"])
        steps = result["steps"]
        metrics, limitations = timeline_metrics(result)
        limitations = list(result["notes"]) + limitations
        limitations.append("nsys adds a small per-launch overhead; absolute step times run slightly long under the "
                           "profiler (compare to bench numbers, not across tools)")
        gpu = {"name": result.get("gpu_name")}
        if src_rec:
            gpu = src_rec["identity"]["gpu"]
            problem = ev.integrity_problem(src_rec)
            if problem:
                limitations.append(f"source capture {problem}")
        span = steps.get("span_us", {})
        summary = (f"GPU busy {result['gpu_busy_pct']}% of {result['window_ms']:.1f} ms; "
                   f"{steps.get('steps', 0)} steps ({steps.get('segmentation')})")
        if span.get("n"):
            summary += f", step span p50 {span['p50']} us"
        top_family = next(iter(steps.get("families") or {}), None)
        if top_family:
            summary += (f"; top family {top_family} {steps['families'][top_family]['critical_pct_of_step']}% "
                        f"of the step's critical path")
        artifacts = [ev.artifact(sqlite, staging if exported else None, "sqlite" if exported else "input")]
        record = ev.build(call, kind="nsys_timeline", claim_class="observed", summary=summary,
                          identity_=ev.identity(gpu=gpu), metrics=metrics, artifacts=artifacts,
                          limitations=limitations,
                          data={**result, "source_evidence": src_rec["evidence_id"] if src_rec else None})
        return store.save(record, staging)
