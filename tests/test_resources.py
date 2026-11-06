"""CPU telemetry uses the server group, bounded proc reads, and safe tuning metadata."""

import hashlib
import json

import pytest

from inferencekit.serve import resources


def write_process(root, pid, *, rss=1024, hwm=2048, threads=2, affinity="0-3", task_affinity=None):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    text = f"Name:\ttest\nVmRSS:\t{rss} kB\nVmHWM:\t{hwm} kB\nThreads:\t{threads}\nCpus_allowed_list:\t{affinity}\n"
    (directory / "status").write_text(text)
    task = directory / "task" / str(pid)
    task.mkdir(parents=True, exist_ok=True)
    (task / "status").write_text(text.replace(affinity, task_affinity or affinity))


def test_runtime_env_only_persists_valid_controls_and_library_identity(tmp_path):
    library = tmp_path / "private-directory" / "libcpu.so.1"
    library.parent.mkdir()
    library.write_bytes(b"synthetic library")
    result = resources.runtime_env({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "4", "OMP_PROC_BIND": "close",
                                    "GOMP_CPU_AFFINITY": "0-7:2 9", "OMP_PLACES": "{0:4},{8:4}",
                                    "LD_PRELOAD": str(library), "HF_TOKEN": "secret-value",
                                    "PRIVATE_TOKEN": "secret-value", "LD_LIBRARY_PATH": "/private/library/path",
                                    "KMP_AFFINITY": "granularity=fine,compact,1,0"})
    assert result["controls"]["OMP_NUM_THREADS"] == "8"
    assert result["ld_preload"] == [{"name": "libcpu.so.1", "status": "hashed",
                                      "sha256": hashlib.sha256(b"synthetic library").hexdigest()}]
    encoded = json.dumps(result)
    assert "secret-value" not in encoded and "private-directory" not in encoded and "/private" not in encoded
    assert "HF_TOKEN" not in encoded and "LD_LIBRARY_PATH" not in encoded


def test_bad_control_values_and_preload_paths_are_not_retained():
    result = resources.runtime_env({"OMP_NUM_THREADS": "some-secret-value", "KMP_AFFINITY": "/private/key",
                                    "LD_PRELOAD": "/private/key.txt", "OMP_DYNAMIC": "FALSE"})
    assert result["controls"] == {"KMP_AFFINITY": None, "OMP_DYNAMIC": "false", "OMP_NUM_THREADS": None}
    assert result["omitted"] == ["KMP_AFFINITY", "OMP_NUM_THREADS"]
    assert result["ld_preload"][0]["name"] is None
    assert "some-secret-value" not in json.dumps(result) and "/private" not in json.dumps(result)


def test_preload_file_read_is_bounded(tmp_path, monkeypatch):
    library = tmp_path / "liblarge.so"
    library.write_bytes(b"large")
    monkeypatch.setattr(resources, "MAX_PRELOAD_BYTES", 2)
    assert resources.runtime_env({"LD_PRELOAD": str(library)})["ld_preload"][0]["status"] == "size_limit"


def test_relative_preload_uses_actual_server_cwd_without_exposing_paths(tmp_path, monkeypatch):
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "libcpu.so").write_bytes(name.encode())
    monkeypatch.chdir(tmp_path)
    first = resources.runtime_env({"LD_PRELOAD": "./libcpu.so"}, cwd="first")
    second = resources.runtime_env({"LD_PRELOAD": "./libcpu.so"}, cwd=tmp_path / "second")
    assert first["ld_preload"][0]["sha256"] == hashlib.sha256(b"first").hexdigest()
    assert second["ld_preload"][0]["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert first["ld_preload"][0]["name"] == second["ld_preload"][0]["name"] == "libcpu.so"
    assert "first" not in json.dumps(first) and "second" not in json.dumps(second)
    result = resources.ProcessMonitor(11, {"LD_PRELOAD": "./libcpu.so"}, cwd="first").stop()
    assert result["requested"] == first
    # A bare library name follows loader search paths, not cwd; stay unknown.
    soname = resources.runtime_env({"LD_PRELOAD": "libcpu.so"}, cwd="first")
    assert soname["ld_preload"][0]["sha256"] is None


def test_group_peak_is_not_sum_of_individual_high_water_marks(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "PROC_ROOT", tmp_path)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources.servers, "group_members", lambda pgid: [11, 12])
    write_process(tmp_path, 11, rss=10 * 1024, hwm=15 * 1024, threads=3)
    write_process(tmp_path, 12, rss=20 * 1024, hwm=24 * 1024, threads=4, affinity="4-7")
    monitor = resources.ProcessMonitor(11, {"OMP_NUM_THREADS": "3"}, poll_s=60).start()
    write_process(tmp_path, 11, rss=12 * 1024, hwm=15 * 1024, threads=5, affinity="1-2", task_affinity="2")
    write_process(tmp_path, 12, rss=1024, hwm=24 * 1024, threads=4, affinity="4-7")
    monitor.settled()
    result = monitor.stop()
    assert result["peak_rss_mib"] == 30
    assert result["peak_threads"] == 9
    assert result["observations"]["launch"]["rss_mib"] == 30
    assert result["observations"]["settled"]["rss_mib"] == 13
    assert result["observations"]["settled"]["affinity"] == ["2", "4-7"]
    assert [p["vm_hwm_mib"] for p in result["processes"]] == [15, 24]
    assert result["requested"]["controls"]["OMP_NUM_THREADS"] == "3"
    assert monitor.stop() == result  # no additional final sample on repeated stop
    assert any("not a simultaneous" in note for note in result["limitations"])


def test_unavailable_proc_never_reports_zero_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    monkeypatch.setattr(resources.servers, "group_members", lambda _: pytest.fail("non-Linux must not read proc"))
    result = resources.ProcessMonitor(11, {}).start().settled().stop()
    assert result["status"] == "unavailable"
    assert result["peak_rss_mib"] is None and result["peak_threads"] is None
    assert result["samples"] == 0


def test_process_limit_and_unreadable_status_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "PROC_ROOT", tmp_path)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources, "MAX_PROCESSES", 2)
    monkeypatch.setattr(resources.servers, "group_members", lambda _: [11, 12, 13])
    write_process(tmp_path, 11)
    result = resources.ProcessMonitor(11, {}, poll_s=60).start().stop()
    assert result["peak_rss_mib"] == 1
    assert any("sampling limit" in note for note in result["limitations"])
    assert any("unreadable" in note for note in result["limitations"])


@pytest.mark.parametrize("pgid,poll_s", [(0, .05), (True, .05), (10, 0), (10, float("nan"))])
def test_monitor_rejects_invalid_parameters(pgid, poll_s):
    with pytest.raises(ValueError):
        resources.ProcessMonitor(pgid, {}, poll_s)
