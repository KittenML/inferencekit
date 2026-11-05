"""The nsys analyzer on a tiny hand-built sqlite with the same tables nsys exports.

Timeline (ns). Three decode steps, each one cudaGraphLaunch replaying 3 kernels:
  gemv 20 us, 2 us launch gap, attention 10 us, norm 3 us   (span 35 us, busy 33 us)
Steps start at 10, 60 and 110 us; then one 1 us memcpy at 150 us.

Process ids use nsys's encoding: globalPid = pid << 24, and a thread's globalTid is
globalPid | tid (the low 24 bits). Kernels carry globalPid; API calls carry globalTid.
"""

import sqlite3

import pytest

from inferencekit import registry
from inferencekit.analysis import nsys_sqlite

STRINGS = {1: "gemv_kernel", 2: "void gemv_kernel<128>(float const*, float*)", 3: "flash_decode_attn",
           4: "void flash_decode_attn<64>()", 5: "rms_norm_kernel", 6: "void ops::rms_norm_kernel<float>()",
           7: "cudaGraphLaunch_v10000", 8: "cudaMemcpyAsync_v3020", 9: "argmax_sampling_kernel",
           10: "void argmax_sampling_kernel<float>()", 11: "cudaLaunchKernel_v7000"}
PID = 4242 << 24  # globalPid of the profiled process
TID = PID | 4242  # globalTid of its main thread
STEP_BASES = (10_000, 60_000, 110_000)


def make_trace(path, with_pid=True):
    con = sqlite3.connect(path)
    con.execute("create table StringIds (id integer primary key, value text)")
    con.executemany("insert into StringIds values (?, ?)", STRINGS.items())
    pid_col = ", globalPid int" if with_pid else ""
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, deviceId int, streamId int, "
                f"correlationId int{pid_col}, shortName int, demangledName int)")
    con.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, globalTid int, correlationId int, "
                "nameId int)")
    con.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, deviceId int, streamId int, "
                f"correlationId int{pid_col}, bytes int, copyKind int)")
    host_us = [5, 4, 6]
    for k, base in enumerate(STEP_BASES):
        corr = 100 * (k + 1)
        launch = base - 10_000
        add_call(con, launch, launch + host_us[k] * 1000, corr, 7)
        for start, end, short, full in ((base, base + 20_000, 1, 2), (base + 22_000, base + 32_000, 3, 4),
                                        (base + 32_000, base + 35_000, 5, 6)):
            add_kernel(con, start, end, corr, short, full, with_pid=with_pid)
    add_call(con, 146_000, 147_000, 400, 8)
    pid_val = (PID,) if with_pid else ()
    con.execute(f"insert into CUPTI_ACTIVITY_KIND_MEMCPY values (150000, 151000, 0, 7, 400{', ?' if with_pid else ''}, "
                "4096, 1)", pid_val)
    con.commit()
    con.close()


def add_call(con, start, end, corr, name_id, tid=TID):
    con.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, ?, ?)", (start, end, tid, corr, name_id))


def add_kernel(con, start, end, corr, short, full, device=0, pid=PID, with_pid=True):
    if with_pid:
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 7, ?, ?, ?, ?)",
                    (start, end, device, corr, pid, short, full))
    else:
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 7, ?, ?, ?)",
                    (start, end, device, corr, short, full))


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / "trace.sqlite"
    make_trace(path)
    return path


def test_pid_of_tid_clears_the_thread_bits():
    assert nsys_sqlite.pid_of_tid(TID) == PID
    assert nsys_sqlite.pid_of_tid(None) is None


def test_busy_idle_and_gap_histogram(trace):
    r = nsys_sqlite.analyze(str(trace))
    assert r["window_ms"] == pytest.approx(0.141)  # 10 us .. 151 us
    assert r["gpu_busy_ms"] == pytest.approx(0.100)  # 3 x 33 us + 1 us memcpy
    assert r["gpu_busy_pct"] == pytest.approx(70.92, abs=0.01)
    h = r["idle_gap_histogram"]
    assert h["<10us"]["count"] == 4  # three 2 us launch gaps + 5 us before the memcpy
    assert h["10-100us"]["count"] == 2  # 15 us between steps
    assert r["activity_counts"] == {"kernel": 9, "memcpy": 1}


def test_families_and_top_kernels(trace):
    r = nsys_sqlite.analyze(str(trace))
    assert list(r["families"]) == ["gemm/gemv", "attention", "norm"]
    assert r["families"]["gemm/gemv"]["pct_kernel_time"] == pytest.approx(100 * 60 / 99, abs=0.01)
    top = r["top_kernels"][0]
    assert top["name"] == "gemv_kernel" and top["count"] == 3 and top["mean_us"] == pytest.approx(20.0)


def test_graph_launches_and_steps(trace):
    r = nsys_sqlite.analyze(str(trace))
    assert r["graph_launch"]["launches"] == 3
    assert r["graph_launch"]["host_us"]["p50"] == pytest.approx(5.0)
    s = r["steps"]
    assert s["segmentation"] == "graph_launch"
    assert s["steps"] == 3 and s["steady_steps"] == 2 and s["kernels_per_step"] == 3
    assert s["span_us"]["p50"] == pytest.approx(35.0)
    assert s["graph_replay_us"]["p50"] == pytest.approx(35.0)  # no eager work in this trace
    assert s["busy_us"]["p50"] == pytest.approx(33.0)
    assert s["idle_in_step_us"]["p50"] == pytest.approx(2.0)
    assert s["gap_to_next_step_us"]["p50"] == pytest.approx(15.0)
    fam = s["families"]
    # Critical path: each kernel is charged only for its own time; the 2 us launch gap
    # before attention is idle, reported once as idle_in_step_us, not added to attention.
    assert fam["gemm/gemv"]["critical_us_p50"] == pytest.approx(20.0)
    assert fam["attention"]["critical_us_p50"] == pytest.approx(10.0)
    assert fam["norm"]["critical_us_p50"] == pytest.approx(3.0)
    assert fam["attention"]["duration_us_p50"] == pytest.approx(10.0)
    critical = sum(f["critical_us_p50"] for f in fam.values())
    assert critical + s["idle_in_step_us"]["p50"] == pytest.approx(s["span_us"]["p50"])
    # Shares of the step: families plus idle add up to 100%.
    assert fam["attention"]["critical_pct_of_step"] == pytest.approx(100 * 10 / 35, abs=0.01)
    assert s["idle_pct_of_step"] == pytest.approx(100 * 2 / 35, abs=0.01)
    shares = sum(f["critical_pct_of_step"] for f in fam.values()) + s["idle_pct_of_step"]
    assert shares == pytest.approx(100.0, abs=0.02)


def test_eager_work_between_graph_replays_belongs_to_the_step(tmp_path):
    # vLLM-style: after each graph replay, an eager sampling kernel (launched with
    # cudaLaunchKernel, so its own correlation id) runs at base+40..45 us.
    path = tmp_path / "eager.sqlite"
    make_trace(path)
    con = sqlite3.connect(path)
    for k, base in enumerate(STEP_BASES):
        corr = 1000 + k
        add_call(con, base + 30_000, base + 31_000, corr, 11)
        add_kernel(con, base + 40_000, base + 45_000, corr, 9, 10)
    con.commit()
    con.close()
    s = nsys_sqlite.analyze(str(path))["steps"]
    assert s["segmentation"] == "graph_launch" and s["steps"] == 3
    # Steps 1 and 2 end with their eager kernel; the last replay has no next replay to bound it.
    assert s["kernels_per_step"] == 4
    assert s["span_us"]["p50"] == pytest.approx(45.0)  # whole token: graph 35 us + 5 us idle + 5 us sampling
    assert s["graph_replay_us"]["p50"] == pytest.approx(35.0)  # the graph alone
    assert s["gap_fully_idle_us"]["p50"] == pytest.approx(5.0)
    assert s["gap_to_next_step_us"]["p50"] == pytest.approx(5.0)
    assert s["families"]["sampling"]["critical_us_p50"] == pytest.approx(5.0)
    assert s["idle_in_step_us"]["p50"] == pytest.approx(7.0)  # 2 us launch gap + 5 us before sampling


def test_correlation_ids_are_joined_per_process(tmp_path):
    # A second process (another GPU) reuses correlation id 400, the id of the first
    # process's memcpy, for a graph launch. Joined on correlation id alone, that memcpy
    # would become a fourth "decode step" and the launch would be counted.
    path = tmp_path / "two_procs.sqlite"
    make_trace(path)
    other_pid = 777 << 24
    con = sqlite3.connect(path)
    add_call(con, 140_000, 141_000, 400, 7, tid=other_pid | 777)
    add_kernel(con, 141_000, 142_000, 400, 1, 2, device=1, pid=other_pid)
    con.commit()
    con.close()
    r = nsys_sqlite.analyze(str(path))
    assert r["device"] == 0 and any("2 devices" in n for n in r["notes"])
    assert r["graph_launch"]["launches"] == 3
    assert r["steps"]["steps"] == 3


def test_trace_without_global_pid_falls_back_to_correlation_id(tmp_path):
    path = tmp_path / "old.sqlite"
    make_trace(path, with_pid=False)
    r = nsys_sqlite.analyze(str(path))
    assert r["steps"]["steps"] == 3 and r["steps"]["segmentation"] == "graph_launch"
    assert any("no globalPid" in n for n in r["notes"])


def test_nvtx_ranges_take_precedence(tmp_path):
    path = tmp_path / "nvtx.sqlite"
    make_trace(path)
    con = sqlite3.connect(path)
    con.execute("create table NVTX_EVENTS (start int, end int, eventType int, text text, textId int, globalTid int)")
    # "decode" ranges around the first two graph launches (on the launching thread), plus a
    # nested range that must not count as a step of its own; launch 3 has no range.
    con.executemany("insert into NVTX_EVENTS values (?, ?, 59, ?, NULL, ?)", [
        (0, 45_000, "decode step", TID), (1_000, 2_000, "decode inner", TID), (50_000, 95_000, "decode step", TID)])
    con.commit()
    con.close()
    s = nsys_sqlite.analyze(str(path))["steps"]
    assert s["segmentation"] == "nvtx"
    assert s["steps"] == 2 and s["kernels_per_step"] == 3


def add_capture_metadata(path):
    """What nsys stores about the launch: command line, environment, host and process list."""
    con = sqlite3.connect(path)
    con.execute("create table META_DATA_CAPTURE (name text, value text)")
    con.executemany("insert into META_DATA_CAPTURE values (?, ?)", [
        ("PROCESS_0:COMMAND", "python"), ("PROCESS_0:ARGUMENT_0", "run.py"),
        ("PROCESS_0:ARGUMENT_1", "--api-key=sk-not-a-real-key"), ("PROCESS_0:WORKING_DIR", "/home/someone/proj"),
        ("PROCESS_0:USER_NAME", "someone"), ("PROCESS_0:ENVIRONMENT_VARIABLE", "PATH=/usr/bin"),
        ("PROCESS_0:ENVIRONMENT_VARIABLE", "HF_TOKEN=hf_not_a_real_token"), ("DEVICE_ID", "0")])
    con.execute("create table TARGET_INFO_SYSTEM_ENV (globalVid int, devStateName text, name text, nameEnum int, "
                "value text)")
    con.executemany("insert into TARGET_INFO_SYSTEM_ENV values (0, '', ?, 0, ?)",
                    [("Hostname", "secret-host"), ("CpuCores", "32")])
    con.execute("create table PROCESSES (globalPid int, pid int, name text)")
    con.executemany("insert into PROCESSES values (?, ?, ?)", [(PID, 4242, "python"), (99 << 24, 99, "sshd")])
    con.commit()
    con.close()


def test_scrub_metadata_blanks_launch_details_but_keeps_names(trace):
    add_capture_metadata(trace)
    before = nsys_sqlite.analyze(str(trace))
    assert before["capture_env_names"] == ["HF_TOKEN", "PATH"]
    assert any("treat it as sensitive" in n for n in before["notes"])

    out = nsys_sqlite.scrub_metadata(str(trace))
    assert out["env_names"] == ["HF_TOKEN", "PATH"] and out["other_processes_dropped"] == 1
    con = sqlite3.connect(trace)
    dump = "\n".join(str(row) for t in ("META_DATA_CAPTURE", "TARGET_INFO_SYSTEM_ENV", "PROCESSES")
                     for row in con.execute(f"select * from {t}"))
    con.close()
    for secret in ("sk-not-a-real-key", "hf_not_a_real_token", "/home/someone", "secret-host", "sshd", "/usr/bin"):
        assert secret not in dump
    assert "HF_TOKEN=<redacted>" in dump and "CpuCores" in dump
    after = nsys_sqlite.analyze(str(trace))
    assert after["capture_env_names"] == ["HF_TOKEN", "PATH"]
    assert not any("treat it as sensitive" in n for n in after["notes"])
    assert after["steps"] == before["steps"]  # the timeline itself is untouched


def test_analyze_operation_writes_evidence(trace, store):
    rec = registry.call("analyze_nsys", {"source": str(trace)})
    assert rec["kind"] == "nsys_timeline"
    names = {m["name"] for m in rec["metrics"]}
    assert {"gpu_busy_pct", "graph_launches", "step_span_us", "step_gap_us", "kernel_sum_over_span"} <= names
    assert rec["artifacts"][0]["role"] == "input"
    # Families are ranked by critical-path share of the step, with idle as its own row: 100% in total.
    shares = {m["subject"]: m["value"] for m in rec["metrics"] if m["name"] == "critical_path_pct"}
    assert set(shares) == {"GPU idle inside the step", "gemm/gemv", "attention", "norm"}
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.05)
    assert "kernel_time_pct" not in names  # summed durations over-count overlapping kernels
    gap = next(m for m in rec["metrics"] if m["name"] == "step_gap_us")
    assert gap["value"] == pytest.approx(15.0)  # fully idle time between steps
    assert "top family gemm/gemv 57.14% of the step's critical path" in rec["summary"]


def test_overlapping_kernels_are_flagged(tmp_path, store):
    # Two kernels on different streams that overlap: summed duration exceeds the span.
    path = tmp_path / "overlap.sqlite"
    make_trace(path)
    con = sqlite3.connect(path)
    for k, base in enumerate(STEP_BASES):
        add_kernel(con, base + 2_000, base + 18_000, 100 * (k + 1), 9, 10)  # inside the gemv, same replay
    con.commit()
    con.close()
    rec = registry.call("analyze_nsys", {"source": str(path)})
    ratio = next(m["value"] for m in rec["metrics"] if m["name"] == "kernel_sum_over_span")
    assert ratio == pytest.approx((33 + 16) / 35, abs=0.001)
    assert any("kernels overlap" in lim for lim in rec["limitations"])
    shares = {m["subject"]: m["value"] for m in rec["metrics"] if m["name"] == "critical_path_pct"}
    assert shares.get("sampling", 0.0) == pytest.approx(0.0)  # hidden under the gemv: no critical-path time
