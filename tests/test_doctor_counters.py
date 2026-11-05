"""doctor: profiling-counter access from the driver setting and CAP_SYS_ADMIN (never from root alone), and the
optional --probe-profilers test capture. No GPU needed: /proc files are temp files and the profilers are fakes.
One test runs the real driver-API kernel; it is opt-in (INFERENCEKIT_TEST_GPU=1) because it uses a GPU."""

import ctypes
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from inferencekit import cli, registry
from inferencekit.evidence import set_default_root
from inferencekit.ops import doctor as dr
from inferencekit.proc import Result

DOCKER_DEFAULT_CAPS = "00000000a80425fb"  # what a default container gets: no CAP_SYS_ADMIN
ALL_CAPS = "000001ffffffffff"  # privileged root


def proc_files(tmp_path, admin_only=None, cap_eff=None):
    params, status = tmp_path / "params", tmp_path / "status"
    if admin_only is not None:
        params.write_text(f"ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: {admin_only}\nEnableMSI: 1\n")
    if cap_eff is not None:
        status.write_text(f"Name:\tpython\nCapInh:\t0000000000000000\nCapPrm:\t{cap_eff}\nCapEff:\t{cap_eff}\n")
    return params, status


@pytest.fixture
def no_ncu(monkeypatch):
    monkeypatch.setattr(dr, "which", lambda name: None)


def test_cap_sys_admin_is_bit_21_of_capeff(tmp_path):
    _, status = proc_files(tmp_path, cap_eff=ALL_CAPS)
    assert dr.has_cap_sys_admin(status) is True
    _, status = proc_files(tmp_path, cap_eff=DOCKER_DEFAULT_CAPS)
    assert dr.has_cap_sys_admin(status) is False
    _, status = proc_files(tmp_path, cap_eff=f"{1 << 21:016x}")
    assert dr.has_cap_sys_admin(status) is True
    assert dr.has_cap_sys_admin(tmp_path / "absent") is None


@pytest.mark.parametrize("euid,caps", [(0, ALL_CAPS), (0, DOCKER_DEFAULT_CAPS), (1000, "0" * 16)])
def test_driver_setting_zero_allows_everyone(tmp_path, no_ncu, euid, caps):
    params, status = proc_files(tmp_path, admin_only=0, cap_eff=caps)
    perm = dr.counter_permission(params, status, euid=euid)
    assert perm["status"] == "allowed" and "RmProfilingAdminOnly=0" in perm["reason"]
    assert perm["rm_profiling_admin_only"] == 0 and perm["euid"] == euid


def test_root_is_never_allowed_just_for_being_root(tmp_path, no_ncu):
    # a real false positive: root in a container, RmProfilingAdminOnly=1, no CAP_SYS_ADMIN
    params, status = proc_files(tmp_path, admin_only=1, cap_eff=DOCKER_DEFAULT_CAPS)
    perm = dr.counter_permission(params, status, euid=0)
    assert perm["status"] == "admin_only" and "lacks CAP_SYS_ADMIN" in perm["reason"]
    assert perm["cap_sys_admin"] is False
    # no readable driver parameter, root without CAP_SYS_ADMIN: expected to fail
    perm = dr.counter_permission(tmp_path / "absent", status, euid=0)
    assert perm["status"] == "admin_only" and perm["rm_profiling_admin_only"] is None
    # nothing readable at all: unknown, with the way to find out
    perm = dr.counter_permission(tmp_path / "none", tmp_path / "none2", euid=0)
    assert perm["status"] == "unknown" and "probe needed" in perm["reason"] and "--probe-profilers" in perm["reason"]


def test_admin_only_setting_follows_cap_sys_admin_not_euid(tmp_path, no_ncu):
    # root on a VM or bare-metal host: the driver's check (CAP_SYS_ADMIN) passes, so a probe decides, never
    # admin_only (a plugin agent would skip ncu) and never allowed (a container or VM can still refuse)
    params, status = proc_files(tmp_path, admin_only=1, cap_eff=ALL_CAPS)
    perm = dr.counter_permission(params, status, euid=0)
    assert perm["status"] == "unknown" and perm["cap_sys_admin"] is True
    assert "probe needed" in perm["reason"] and "--probe-profilers" in perm["reason"]
    # a non-root process granted CAP_SYS_ADMIN is treated the same: the driver checks the capability, not uid 0
    perm = dr.counter_permission(params, status, euid=1000)
    assert perm["status"] == "unknown" and "non-root process has CAP_SYS_ADMIN" in perm["reason"]
    # no CAP_SYS_ADMIN: admin_only, root or not
    params, status = proc_files(tmp_path, admin_only=1, cap_eff="0" * 16)
    perm = dr.counter_permission(params, status, euid=1000)
    assert perm["status"] == "admin_only" and "lacks CAP_SYS_ADMIN" in perm["reason"]
    assert "NVreg_RestrictProfilingToAdminUsers=0" in perm["reason"]
    # capabilities unreadable: root needs a probe, a non-root process is expected to be refused
    perm = dr.counter_permission(params, tmp_path / "absent", euid=0)
    assert perm["status"] == "unknown" and "probe needed" in perm["reason"] and perm["cap_sys_admin"] is None
    perm = dr.counter_permission(params, tmp_path / "absent", euid=1000)
    assert perm["status"] == "admin_only"


@pytest.mark.parametrize("caps", [ALL_CAPS, DOCKER_DEFAULT_CAPS, "0" * 16, None])
def test_root_under_the_admin_only_setting_is_never_allowed(tmp_path, no_ncu, caps):
    params, status = proc_files(tmp_path, admin_only=1, cap_eff=caps)
    assert dr.counter_permission(params, status, euid=0)["status"] != "allowed"


def test_root_with_cap_and_no_driver_parameter_needs_a_probe(tmp_path, no_ncu):
    _, status = proc_files(tmp_path, cap_eff=ALL_CAPS)
    perm = dr.counter_permission(tmp_path / "absent", status, euid=0)
    assert perm["status"] == "unknown" and "probe needed" in perm["reason"]


def test_ncu_query_metrics_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "which", lambda name: "/fake/ncu" if name == "ncu" else None)
    monkeypatch.setattr(dr, "run", lambda argv, timeout=None: Result(1, "", "==ERROR== ERR_NVGPUCTRPERM - no perm"))
    perm = dr.counter_permission(tmp_path / "absent", tmp_path / "absent", euid=1000)
    assert perm["status"] == "admin_only" and "ERR_NVGPUCTRPERM" in perm["reason"]
    monkeypatch.setattr(dr, "run", lambda argv, timeout=None: Result(0, "metrics...", ""))
    perm = dr.counter_permission(tmp_path / "absent", tmp_path / "absent", euid=1000)
    assert perm["status"] == "unknown" and "probe needed" in perm["reason"]


@pytest.fixture(autouse=True)
def isolate_nvidia_permission_tests(monkeypatch):
    # These fixtures model NVIDIA permission/probe paths, independent of the host's
    # real AMD/Apple/WSL device interfaces. Vendor routing has its own tests.
    monkeypatch.setattr(dr, "discover_non_nvidia", lambda: {"platform": "cpu", "devices": []})


# --------------------------------------------------------------------------- --probe-profilers (fakes)

class FakeTools:
    """Stands in for proc.run / proc.which: the driver-API program, ncu and nsys, each scriptable."""

    def __init__(self, tools=("ncu", "nsys"), program_ok=True, ncu="ok", nsys="ok"):
        self.tools, self.program_ok, self.ncu, self.nsys = set(tools), program_ok, ncu, nsys
        self.calls: list[list[str]] = []

    def which(self, name):
        return f"/fake/bin/{name}" if name in self.tools else None

    def run(self, argv, timeout=None, **kw):
        self.calls.append(list(argv))
        tool = Path(argv[0]).name
        if tool == "ncu":
            if self.ncu == "ok":
                return Result(0, "==PROF== Connected to process\n  twice (4096, 1, 1)x(256, 1, 1), Device 0, CC 12.0\n"
                                 "    sm__cycles_elapsed.avg       cycle     5,123.50\n", "")
            if self.ncu == "perm":
                return Result(1, "ok\n", "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access "
                                         "NVIDIA GPU Performance Counters on the target device 0.")
            return Result(1, "", "==ERROR== the profiler crashed")
        if tool == "nsys":
            if self.nsys == "ok":
                out = Path(argv[argv.index("-o") + 1])
                out.with_suffix(".nsys-rep").write_bytes(b"x" * 64)
                return Result(0, "Generated: probe.nsys-rep", "")
            if self.nsys == "silent":
                return Result(0, "", "")
            return Result(1, "", "Error: failed to load the CUDA injection library")
        if tool == "nvcc":
            return Result(1, "", "nvcc: error")
        # the program itself, run without a profiler
        return Result(0, "ok\n", "") if self.program_ok else Result(1, "", "cuInit: CUDA_ERROR_NO_DEVICE")


def use(monkeypatch, fake):
    monkeypatch.setattr(dr, "which", fake.which)
    monkeypatch.setattr(dr, "run", fake.run)


def test_probe_records_a_working_capture_per_tool(tmp_path, monkeypatch, store):
    fake = FakeTools()
    use(monkeypatch, fake)
    probe = dr.profiler_probe(None)
    assert probe["ncu"]["capture_ok"] is True and probe["ncu"]["value"] == pytest.approx(5123.5)
    assert probe["nsys"]["capture_ok"] is True and probe["nsys"]["report_bytes"] == 64
    assert "driver API" in probe["program"]
    ncu_call = next(c for c in fake.calls if Path(c[0]).name == "ncu")
    assert ncu_call[1:5] == ["--launch-count", "1", "--metrics", "sm__cycles_elapsed.avg"]
    assert ncu_call[5] == sys.executable and ncu_call[6] == "-I"
    assert not list((store.root / ".staging").iterdir())  # the probe's files are removed


def test_probe_failures_are_findings_not_errors(tmp_path, monkeypatch, store):
    use(monkeypatch, FakeTools(ncu="perm", nsys="fail"))
    probe = dr.profiler_probe(None)
    assert probe["ncu"] == {**probe["ncu"], "capture_ok": False, "ran": True} and "ERR_NVGPUCTRPERM" in probe["ncu"]["error"]
    assert probe["nsys"]["capture_ok"] is False and "injection" in probe["nsys"]["error"]
    use(monkeypatch, FakeTools(ncu="crash", nsys="silent"))
    probe = dr.profiler_probe(None)
    assert probe["ncu"]["capture_ok"] is False and "crashed" in probe["ncu"]["error"]
    assert probe["nsys"]["capture_ok"] is False and "no .nsys-rep" in probe["nsys"]["error"]


def test_probe_without_a_program_or_tools_tries_nothing(monkeypatch, store):
    fake = FakeTools(program_ok=False)
    use(monkeypatch, fake)
    probe = dr.profiler_probe(None)
    assert probe["program"] is None and "CUDA_ERROR_NO_DEVICE" in probe["program_error"]
    assert probe["ncu"]["capture_ok"] is None and probe["nsys"]["ran"] is False
    assert not any(Path(c[0]).name in ("ncu", "nsys") for c in fake.calls)  # no profiler run on a dead program
    use(monkeypatch, FakeTools(tools=()))
    probe = dr.profiler_probe(None)
    assert probe["ncu"]["error"] == "ncu not found" and probe["nsys"]["error"] == "nsys not found"


def test_probe_never_raises(monkeypatch, store):
    use(monkeypatch, FakeTools())

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(dr, "_probe_ncu", boom)
    probe = dr.profiler_probe(None)
    assert probe["ncu"]["capture_ok"] is False and "RuntimeError" in probe["ncu"]["error"]
    assert probe["nsys"]["capture_ok"] is True
    monkeypatch.setattr(dr, "_probe_program", boom)
    assert "RuntimeError" in dr.profiler_probe(None)["error"]

    def no_space(self):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dr.ev.Store, "staging", no_space)
    probe = dr.profiler_probe(None)
    assert probe["ncu"]["capture_ok"] is None and "No space left" in probe["nsys"]["error"]


def _doctor(monkeypatch, store, fake, perm_status="unknown"):
    use(monkeypatch, fake)
    monkeypatch.setattr(dr, "counter_permission", lambda: {"status": perm_status, "reason": "static", "euid": 0})
    monkeypatch.setattr(dr, "query_gpus", lambda: [])
    monkeypatch.setattr(dr, "tool_version", lambda name: "2026.1" if name in fake.tools else None)
    return registry.call("doctor", {"probe_profilers": True})


def test_doctor_probe_decides_the_counter_status(monkeypatch, store):
    rec = _doctor(monkeypatch, store, FakeTools())
    perm = rec["data"]["profiling_counters"]
    assert perm["status"] == "allowed" and "--probe-profilers" in perm["reason"] and perm["static_reason"] == "static"
    assert "test capture: ncu ok, nsys ok" in rec["summary"] and rec["data"]["profiler_probe"]["nsys"]["capture_ok"]
    rec = _doctor(monkeypatch, store, FakeTools(ncu="perm"), perm_status="allowed")
    assert rec["data"]["profiling_counters"]["status"] == "admin_only"
    assert any(x.startswith("ncu test capture failed") for x in rec["limitations"])
    assert any("admin-only" in x for x in rec["limitations"]) and "ncu FAILED" in rec["summary"]


def test_doctor_without_probe_says_probe_needed(monkeypatch, store):
    monkeypatch.setattr(dr, "counter_permission", lambda: {"status": "unknown", "reason": "no params (probe needed)"})
    monkeypatch.setattr(dr, "query_gpus", lambda: [])
    rec = registry.call("doctor", {})
    assert "counters unknown (probe needed)" in rec["summary"] and "profiler_probe" not in rec["data"]
    assert any(x.startswith("profiling counter access unknown") for x in rec["limitations"])


def test_cli_flag(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(dr, "profiler_probe", lambda python: calls.append(python) or {"ncu": None, "nsys": None})
    monkeypatch.setattr(dr, "query_gpus", lambda: [])
    try:
        assert cli.main(["--runs-dir", str(tmp_path / "runs"), "doctor", "--probe-profilers"]) == 0
    finally:
        set_default_root(None)
    assert calls == [None] and json.loads(capsys.readouterr().out)["kind"] == "environment"


# --------------------------------------------------------------------------- the real program (needs a driver)

def _driver_loads() -> bool:
    try:
        ctypes.CDLL("libcuda.so.1")
        return shutil.which("nvidia-smi") is not None
    except OSError:
        return False


@pytest.mark.skipif(os.environ.get("INFERENCEKIT_TEST_GPU") != "1" or not _driver_loads(),
                    reason="uses a GPU: opt in with INFERENCEKIT_TEST_GPU=1 on a host with an NVIDIA driver")
def test_driver_api_program_runs_on_this_gpu(tmp_path):
    prog = tmp_path / "probe_kernel.py"
    prog.write_text(dr._DRIVER_PROGRAM)
    r = dr.run([sys.executable, "-I", str(prog)], timeout=120)
    assert r.ok and r.stdout.strip() == "ok", r.stderr
