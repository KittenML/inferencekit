"""CPU discovery uses fixture OS files; no workloads or counter probes."""
import json

import pytest

from inferencekit import cpuinfo, evidence, hardware, registry
from inferencekit.ops import doctor


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


@pytest.fixture
def cpu_card(tmp_path, monkeypatch):
    proc, cpu, node = tmp_path / "proc", tmp_path / "cpu", tmp_path / "node"
    put(proc / "cpuinfo", "\n\n".join(f"processor : {i}\nmodel name : Fixture Hybrid\n"
                                      "flags : avx2 avx512f avx512_vnni avx_vnni avx512_bf16\n"
                                      "Serial : SECRET_SERIAL\n" for i in range(8)))
    put(cpu / "online", "0-7")
    for i in range(8):
        root = cpu / f"cpu{i}"
        put(root / "topology/physical_package_id", 0)
        put(root / "topology/core_id", i % 4)
        put(root / "topology/thread_siblings_list", f"{i % 4},{i % 4 + 4}")
        put(root / "topology/core_type", 1 if i % 4 < 2 else 2)
        put(root / "cache/index3/level", 3)
        put(root / "cache/index3/shared_cpu_list", "0-1,4-5" if i % 4 < 2 else "2-3,6-7")
        put(root / "cache/index3/size", "16384K")
    put(node / "node0/cpulist", "0-1,4-5")
    put(node / "node1/cpulist", "2-3,6-7")
    put(cpu / "cpufreq/policy0/scaling_governor", "powersave")
    put(proc / "sys/kernel/perf_event_paranoid", 4)
    put(proc / "self/status", "Name:\tSECRET_PROCESS\nCapEff:\t0000000000000000\n")
    put(proc / "loadavg", "30.00 20.00 10.00 1/2 12345")
    monkeypatch.setattr(cpuinfo, "which", lambda _: "/fixture/perf")
    monkeypatch.setattr(cpuinfo.os, "environ", {"HF_TOKEN": "SECRET_TOKEN", "LD_LIBRARY_PATH": "/SECRET_PATH",
                                              "OMP_NUM_THREADS": "4", "MKL_ENABLE_INSTRUCTIONS": "AVX2",
                                              "DNNL_MAX_CPU_ISA": "SECRET_VALUE"})
    return cpuinfo.cpu_facts(proc_root=proc, cpu_root=cpu, node_root=node, allowed_cpus=[0, 1, 2, 3])


def test_cpu_card_has_observed_topology_and_no_raw_or_secret_data(cpu_card):
    assert cpu_card["physical_cores"] == 4 and cpu_card["logical_cores"] == 8
    assert cpu_card["allowed_cpus"] == [0, 1, 2, 3]
    assert len(cpu_card["l3_groups"]) == len(cpu_card["numa_nodes"]) == 2
    assert {r["core_type"] for r in cpu_card["topology"]} == {1, 2}
    assert all(cpu_card["isa"][k] for k in ("avx2", "avx512", "avx512_vnni", "avx_vnni", "avx512_bf16"))
    assert cpu_card["governors"] == ["powersave"]
    assert cpu_card["perf"]["access"] == "restricted" and not cpu_card["perf"]["verified"]
    assert any("contention" in w for w in cpu_card["limitations"])
    assert cpu_card["tuning_env"] == {"OMP_NUM_THREADS": "4", "MKL_ENABLE_INSTRUCTIONS": "AVX2"}
    assert cpu_card["library_dispatch"]["status"] == "unverified"
    assert "SECRET" not in json.dumps(cpu_card)


def test_identity_excludes_volatile_load_and_dispatch(cpu_card):
    before = cpuinfo.cpu_identity(cpu_card)
    cpu_card["load_average"] = [0, 0, 0]
    cpu_card["governors"] = ["performance"]
    cpu_card["tuning_env"] = {}
    assert cpuinfo.cpu_identity(cpu_card) == before
    assert "cpu" not in evidence.identity()
    assert evidence.identity(cpu=before)["cpu"] == before
    cpu_card["topology"][0]["core"] = 99
    assert cpuinfo.cpu_identity(cpu_card)["topology_sha256"] != before["topology_sha256"]


def test_missing_topology_and_isa_stay_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(cpuinfo, "which", lambda _: None)
    card = cpuinfo.cpu_facts(proc_root=tmp_path, cpu_root=tmp_path, node_root=tmp_path, allowed_cpus=[])
    assert card["physical_cores"] is None and card["topology"] == []
    assert all(v is None for v in card["isa"].values())
    assert card["load_average"] is None and card["perf"]["access"] == "unknown"


@pytest.mark.parametrize("paranoid,caps,expected", [(2, 0, "policy_allows_user"), (4, 0, "restricted"),
                                                    (4, 1 << 38, "policy_allows_user")])
def test_perf_policy_is_not_a_permission_probe(tmp_path, monkeypatch, paranoid, caps, expected):
    put(tmp_path / "sys/kernel/perf_event_paranoid", paranoid)
    put(tmp_path / "self/status", f"CapEff:\t{caps:016x}\n")
    monkeypatch.setattr(cpuinfo, "which", lambda _: "/fixture/perf")
    card = cpuinfo.cpu_facts(proc_root=tmp_path, cpu_root=tmp_path, node_root=tmp_path, allowed_cpus=[])
    assert card["perf"]["access"] == expected and card["perf"]["verified"] is False
    route = hardware.profiler_route("cpu", card)
    assert ("perf (" in route["timeline"]) == (expected == "policy_allows_user")
    assert "torch.profiler" in route["fallback"]


def test_cpu_doctor_never_calls_cuda_permissions_or_probes(cpu_card, monkeypatch, store):
    monkeypatch.setattr(doctor, "query_gpus", lambda: [])
    monkeypatch.setattr(doctor, "discover_non_nvidia", lambda: {"platform": "cpu", "devices": [], "cpu": cpu_card})
    monkeypatch.setattr(doctor, "counter_permission", lambda: pytest.fail("NVIDIA permission queried"))
    monkeypatch.setattr(doctor, "profiler_probe", lambda *_: pytest.fail("CUDA workload launched"))
    monkeypatch.setattr(doctor, "engine_versions", lambda _: {})
    monkeypatch.setattr(doctor, "llamacpp_version", lambda: None)
    monkeypatch.setattr(doctor, "tool_version", lambda _: "fixture")
    result = registry.call("doctor", {"probe_profilers": True})
    assert result["identity"]["cpu"] == cpuinfo.cpu_identity(cpu_card)
    assert result["identity"]["gpu"]["name"] is None
    assert result["data"]["profiler_probe"]["ncu"]["ran"] is False
    assert "torch.profiler" in result["data"]["profiling_route"]["timeline"]
    assert evidence.validate(result) == []
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("value", ["1-0", "1-x", "1000001", "0-100001", "1,,3"])
def test_invalid_cpu_lists_are_not_partially_used(value):
    assert cpuinfo.cpu_list(value) == []
