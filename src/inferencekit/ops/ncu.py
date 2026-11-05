"""ncu capture / ncu summarize: hardware counters for a few chosen kernel launches.

Capture command (allowlisted environment):

    ncu --clock-control none --kernel-name-base function -k regex:^(?:R)$
        --launch-skip N --launch-count C --set detailed -o <out> <argv>

  --clock-control none     keep the GPU's normal clocks. By default ncu locks clocks to
                           base, which makes memory-bound kernels look different from
                           how they run in production.
  --kernel-name-base function + an anchored regex
                           match the plain function name exactly, so "gemv" does not
                           also catch "gemv_splitk_reduce" by accident.
  --launch-skip / --launch-count
                           skip warm-up launches, then profile C launches. ncu replays
                           each kernel many times to collect all counters, so profile
                           as few as possible.

The .ncu-rep stores the full profiler command line (so the profiled command too), the
host name and the host's IP addresses: treat it as sensitive. The evidence record keeps
only a digest of the command and the names of the variables passed.
"""

from __future__ import annotations

from pathlib import Path

from inferencekit import evidence as ev
from inferencekit.analysis import ncu_csv
from inferencekit.gpuinfo import gpu_identity
from inferencekit.proc import clean_env, dropped_config_env, require, run, tool_version
from inferencekit.registry import Arg, OpError, operation


def anchor(regex: str) -> str:
    """Anchor a kernel-name regex at both ends unless the caller already anchored it."""
    if regex.startswith("^") or regex.endswith("$"):
        return regex
    return f"^(?:{regex})$"


@operation("capture_ncu", ("ncu", "capture"), "Targeted Nsight Compute run (--clock-control none, anchored kernel "
           "regex, launch skip/count) under an allowlisted environment.",
           [Arg("kernel_regex", "string", "Kernel function-name regex (anchored at both ends unless it has ^ or $).",
                required=True),
            Arg("launch_skip", "integer", "Matching launches to skip first (warm-up).", default=0),
            Arg("count", "integer", "Matching launches to profile.", default=1),
            Arg("set", "string", "ncu section set.", default="detailed", choices=("basic", "detailed", "full")),
            Arg("keep_env", "array", "Extra environment variable NAMES to pass through (names only in the record; "
                "secret-looking names are refused).", default=[]),
            Arg("timeout", "integer", "Seconds before the run (and every process it started) is killed.",
                default=3600)],
           passthrough=True)
def capture_ncu(call, kernel_regex, launch_skip, count, set, keep_env, timeout, argv):  # noqa: A002
    ncu = require("ncu", "Install Nsight Compute (it ships with the CUDA toolkit).")
    env, env_names = clean_env(keep_env)  # raises on secret-looking names, before anything runs
    dropped = dropped_config_env(env)
    store = ev.Store()
    with store.staging_dir() as staging:  # removed again if anything below fails
        base = staging / "kernels"
        cmd = [ncu, "--clock-control", "none", "--kernel-name-base", "function", "-k",
               "regex:" + anchor(kernel_regex), "--launch-skip", str(launch_skip), "--launch-count", str(count),
               "--set", set, "-f", "-o", str(base)]
        r = run(cmd + list(argv), timeout=timeout, env=env, stdout_path=staging / "stdout.txt",
                stderr_path=staging / "stderr.txt")
        text = r.stdout + r.stderr
        for name in ("stdout.txt", "stderr.txt"):  # the program's own output may echo secrets
            p = staging / name
            p.write_text(ev.redact_text(p.read_text(errors="replace")))
        rep = base.with_suffix(".ncu-rep")
        limitations, missing = [], []
        if "ERR_NVGPUCTRPERM" in text:
            limitations.append("no permission for GPU performance counters (run as root or set "
                               "NVreg_RestrictProfilingToAdminUsers=0)")
            missing.append("counters")
        if "No kernels were profiled" in text or not rep.is_file():
            limitations.append("no kernel matched: check the regex against 'nsys analyze' top_kernels names")
            missing.append("profiled kernels")
        if r.returncode != 0:
            limitations.append(f"profiled command exited with code {r.returncode}")
        if rep.is_file():
            limitations.append("the .ncu-rep stores the full profiler command line (the profiled command "
                               "included), the host name and the host's IP addresses: treat it as sensitive")
        if dropped:
            limitations.append("set here but not passed to the profiled run, so the configuration may differ from a "
                               "benchmark run with the full environment: " + ", ".join(dropped)
                               + " (pass each one that matters with --keep-env NAME)")
        artifacts = [ev.artifact(staging / "stderr.txt", staging, "stderr")]
        if rep.is_file():
            artifacts.insert(0, ev.artifact(rep, staging, "ncu_rep"))
        summary = (f"ncu {set} on /{anchor(kernel_regex)}/ skip {launch_skip} count {count}: "
                   + ("report written" if rep.is_file() else "NO REPORT"))
        record = ev.build(call, kind="ncu_capture", claim_class="observed", summary=summary,
                          identity_=ev.identity(gpu=gpu_identity()), artifacts=artifacts, missing=missing,
                          limitations=limitations,
                          data={"ncu_version": tool_version("ncu"), "exit_code": r.returncode,
                                "env_names": env_names, "env_dropped": dropped,
                                "kernel_regex": anchor(kernel_regex), "launch_skip": launch_skip, "count": count,
                                "set": set, "clock_control": "none"})
        return store.save(record, staging)


def _peak(peak_gbs: str | None) -> tuple[float | None, str | None]:
    if not peak_gbs:
        return None, None
    try:
        return float(peak_gbs), "user"
    except ValueError:
        pass
    from inferencekit.ops.roofline import _bw_from_record

    value, info, _ = _bw_from_record(ev.Store().load(peak_gbs))
    return value, info.get("evidence_id")


@operation("summarize_ncu", ("ncu", "summarize"), "Per kernel: duration, DRAM GB/s and % of measured peak, SM %, "
           "occupancy and its limiter, top stall reasons, bound label.",
           [Arg("source", "string", "A .ncu-rep, an ncu_capture evidence id, or a raw-page CSV file.",
                required=True, positional=True),
            Arg("details_csv", "string", "Details-page CSV (when source is a CSV file)."),
            Arg("peak_gbs", "string", "Measured DRAM peak: GB/s number or a kind=ceiling evidence id.")])
def summarize_ncu(call, source, details_csv, peak_gbs):
    store = ev.Store()
    src_rec, inputs = None, []
    p = Path(source)
    if p.suffix == ".csv" and p.is_file():
        raw_text = p.read_text(errors="replace")
        details_text = Path(details_csv).read_text(errors="replace") if details_csv else ""
        inputs = [p] + ([Path(details_csv)] if details_csv else [])
        ncu_version = None
    else:
        if p.suffix == ".ncu-rep" and p.is_file():
            rep = p
        else:
            src_rec = store.load(source)
            if src_rec.get("kind") != "ncu_capture":
                raise OpError(f"{source}: expected a .ncu-rep, a raw-page CSV or an ncu_capture evidence id")
            art = next((a for a in src_rec.get("artifacts", []) if a.get("role") == "ncu_rep"), None)
            if art is None:
                raise OpError(f"{source}: no .ncu-rep artifact in that record")
            rep = store.run_dir(src_rec["evidence_id"]) / art["path"]
        ncu = require("ncu", "ncu is needed to read .ncu-rep files")
        raw = run([ncu, "--import", str(rep), "--page", "raw", "--csv"], timeout=600)
        det = run([ncu, "--import", str(rep), "--page", "details", "--csv"], timeout=600)
        if not raw.ok and not det.ok:
            raise OpError(f"ncu --import failed: {raw.stderr[-500:]}")
        raw_text, details_text = raw.stdout, det.stdout
        inputs = [rep]
        ncu_version = tool_version("ncu")
    peak, peak_src = _peak(peak_gbs)
    kernels = ncu_csv.summarize(raw_text, details_text, peak)
    if not kernels:
        raise OpError("no kernel rows found in the ncu output")
    metrics, limitations = [], []
    for k in kernels:
        subject = f"#{k['index']} {k['kernel'][:80]}"
        metrics.append(ev.metric("duration_us", k["duration_us"], "us", "single_launch", n=1, subject=subject))
        metrics.append(ev.metric("dram_gbs", k["dram_gbs"], "GB/s", "single_launch", n=1, subject=subject))
        if k["dram_pct_of_measured_peak"] is not None:
            metrics.append(ev.metric("dram_pct_of_measured_peak", k["dram_pct_of_measured_peak"], "%",
                                     "single_launch", n=1, subject=subject))
    if peak is None:
        limitations.append("no measured peak given: % of peak is ncu's nominal DRAM peak only (pass --peak-gbs)")
    unparsed = [f"#{k['index']} {p}" for k in kernels for p in k["unparsed_units"]]
    if unparsed:
        limitations.append("values with an unknown or unexpected unit were left empty, not guessed: "
                           + "; ".join(unparsed[:8]) + (" ..." if len(unparsed) > 8 else ""))
    if src_rec and ev.integrity_problem(src_rec):
        limitations.append(f"source capture {ev.integrity_problem(src_rec)}")
    limitations.append("bound labels are speed-of-light rules of thumb (>=60% of a unit's peak = bound by it; "
                       "both <40% = latency-bound), not a full analysis")
    if any(k["duration_us"] and k["duration_us"] < 5 for k in kernels):
        limitations.append("kernels under 5 us: launch and tail effects dominate, counters are noisy")
    labels = ", ".join(f"#{k['index']} {k['kernel'][:30]}: {k['bound']}" for k in kernels[:4])
    summary = f"{len(kernels)} kernel launch(es): {labels}"
    gpu = src_rec["identity"]["gpu"] if src_rec else {}
    record = ev.build(call, kind="ncu_kernels", claim_class="observed", summary=summary,
                      identity_=ev.identity(gpu=gpu), metrics=metrics,
                      artifacts=[ev.artifact(i, None, "input") for i in inputs], limitations=limitations,
                      data={"kernels": kernels, "peak_gbs": peak, "peak_source": peak_src,
                            "ncu_version": ncu_version,
                            "source_evidence": src_rec["evidence_id"] if src_rec else None})
    return store.save(record)
