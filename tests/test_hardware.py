"""Discovery and routing use recorded OS facts, never a GPU workload."""
import json
from pathlib import Path

import pytest

from inferencekit import gpuinfo, hardware, registry
from inferencekit.ops import doctor as dr
from inferencekit.proc import Result


def test_amd_sysfs_ignores_connectors_other_vendors_and_non_display_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "which", lambda _: None)
    for card, vendor, cls in (("card0", "0x1002", "0x030000"), ("card1", "0x10de", "0x030000"),
                              ("card2", "0x1002", "0x040300"), ("card0-DP-1", "0x1002", "0x030000")):
        device = tmp_path / card / "device"
        device.mkdir(parents=True)
        for name, value in (("vendor", vendor), ("class", cls), ("device", "0x7550"),
                            ("product_name", "AMD Radeon fixture"), ("mem_info_vram_total", str(16 << 30))):
            (device / name).write_text(value)
    devices = hardware.amd_devices(tmp_path)
    assert len(devices) == 1
    assert devices[0]["memory_bytes"] == 16 << 30
    assert devices[0]["memory_kind"] == "vram" and devices[0]["backend"] == "rocm"
    assert devices[0]["name"] == "AMD Radeon fixture"


def test_amd_unknown_product_is_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "which", lambda _: None)
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002")
    (device / "device").write_text("0xabcd")
    row = hardware.amd_devices(tmp_path)[0]
    assert row["memory_bytes"] is None
    assert row["name"] == "AMD GPU (0xabcd)"
    assert "unverified" in row["name_source"]


def test_apple_selected_fields_preserve_unified_memory_and_exclude_serials(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_sysctl", lambda key: "Apple M4" if key.startswith("machdep") else str(24 << 30))
    output = {"SPDisplaysDataType": [{"sppci_vendor": "sppci_vendor_Apple", "sppci_model": "Apple M4",
                                     "serial_number": "SECRET_SERIAL", "hostname": "SECRET_HOST"}]}
    monkeypatch.setattr(hardware, "run", lambda *a, **k: Result(0, json.dumps(output), ""))
    rows = hardware.apple_devices()
    assert rows[0]["memory_kind"] == "unified" and rows[0]["memory_bytes"] == 24 << 30
    assert "SECRET" not in json.dumps(rows)


@pytest.mark.parametrize("backend", ["rocm", "metal"])
def test_vendor_doctor_never_queries_cuda_permissions_or_runs_cuda_probe(backend, monkeypatch, store):
    facts = {"platform": backend, "devices": [{"name": "fixture accelerator", "backend": backend,
              "memory_bytes": 24 << 30, "memory_kind": "unified" if backend == "metal" else "vram"}],
             "limitations": []}
    monkeypatch.setattr(dr, "query_gpus", lambda: [])
    monkeypatch.setattr(dr, "discover_non_nvidia", lambda: facts)
    monkeypatch.setattr(dr, "tool_version", lambda _: None)
    monkeypatch.setattr(dr, "engine_versions", lambda _: {})
    monkeypatch.setattr(dr, "llamacpp_version", lambda: None)
    def forbidden(*a, **k):
        pytest.fail("NVIDIA-only operation called for a detected vendor GPU")
    monkeypatch.setattr(dr, "counter_permission", forbidden)
    monkeypatch.setattr(dr, "profiler_probe", forbidden)
    record = registry.call("doctor", {"probe_profilers": True})
    assert record["identity"]["gpu"]["name"] == "fixture accelerator"
    assert record["identity"]["gpu"]["cuda"] is None
    assert record["data"]["profiling_route"] == hardware.profiler_route(backend)
    assert record["data"]["profiler_probe"]["ncu"]["ran"] is False
    assert record["data"]["profiling_counters"]["status"] == "not_applicable"
    assert "gpu processes" in record["coverage"]["missing"]


def test_evidence_gpu_identity_uses_vendor_facts_without_cuda_version(monkeypatch):
    monkeypatch.setattr(hardware, "discover_non_nvidia", lambda: {"devices": [{"name": "Apple M4"}]})
    monkeypatch.setattr(gpuinfo, "driver_cuda_version", lambda: pytest.fail("CUDA version queried"))
    assert gpuinfo.gpu_identity([]) == {"name": "Apple M4", "count": 1, "cuda": None, "driver": None, "clocks": None}


def test_rocm_fallback_identifies_wsl_radeon_and_ignores_cpu_agents(monkeypatch):
    output = """Agent 1
  Name: CPU marketing name
  Marketing Name: Test CPU
  Device Type: CPU
Agent 2
  Name: gfx1201
  Marketing Name: AMD Radeon RX 9070 XT
  Device Type: GPU
  Uuid: SECRET_UUID
  Segment: GLOBAL
  Size: 16760832(0xffc000) KB
"""
    monkeypatch.setattr(hardware, "which", lambda _: "/opt/rocm/bin/rocminfo")
    monkeypatch.setattr(hardware, "run", lambda *a, **k: Result(0, output, ""))
    rows = hardware.rocm_devices()
    assert len(rows) == 1 and rows[0]["name"] == "AMD Radeon RX 9070 XT"
    assert rows[0]["architecture"] == "gfx1201" and rows[0]["memory_bytes"] is None
    assert "SECRET" not in json.dumps(rows)
    monkeypatch.setattr(hardware, "apple_devices", lambda: [])
    monkeypatch.setattr(hardware, "amd_devices", lambda: [])
    assert hardware.discover_non_nvidia()["platform"] == "rocm"


def test_rocm_installed_without_gpu_does_not_imply_rocm_hardware(monkeypatch):
    monkeypatch.setattr(hardware, "which", lambda _: "/opt/rocm/bin/rocminfo")
    monkeypatch.setattr(hardware, "run", lambda *a, **k: Result(1, "ROCk module not loaded", ""))
    assert hardware.rocm_devices() == []
    monkeypatch.setattr(hardware, "which", lambda _: None)
    assert hardware.rocm_devices() == []


def test_dxg_without_vendor_facts_is_unknown_accelerator_not_cpu(monkeypatch, store):
    monkeypatch.setattr(hardware, "apple_devices", lambda: [])
    monkeypatch.setattr(hardware, "amd_devices", lambda: [])
    monkeypatch.setattr(hardware, "rocm_devices", lambda: [])
    monkeypatch.setattr(hardware.Path, "exists", lambda p: str(p) == "/dev/dxg")
    facts = hardware.discover_non_nvidia()
    assert facts["platform"] == "unknown" and facts["devices"] == []
    monkeypatch.setattr(dr, "query_gpus", lambda: [])
    monkeypatch.setattr(dr, "discover_non_nvidia", lambda: facts)
    monkeypatch.setattr(dr, "engine_versions", lambda _: {})
    monkeypatch.setattr(dr, "llamacpp_version", lambda: None)
    monkeypatch.setattr(dr, "counter_permission", lambda: pytest.fail("CUDA permissions queried"))
    record = registry.call("doctor", {})
    assert record["data"]["hardware"]["platform"] == "unknown"
    assert record["metrics"] == [] and "unknown accelerator" in record["summary"]
