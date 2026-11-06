"""Manifest preparation is GPU/CPU-workload free and never changes the source."""
import copy
import json

import pytest

from inferencekit import cli
from inferencekit.ops import cpu
from inferencekit.registry import OpError
from test_manifest import good_manifest


def facts():
    return {"model": "Fixture CPU", "architecture": "x86_64", "logical_cores": 8, "physical_cores": 4,
            "allowed_cpus": list(range(8)),
            "topology": [{"cpu": n, "core": n % 4, "package": 0, "core_type": None} for n in range(8)],
            "l3_groups": [{"cpus": [0, 1, 4, 5]}, {"cpus": [2, 3, 6, 7]}], "numa_nodes": []}


def baseline():
    m = good_manifest()
    e = m["engines"][0]
    e["launch"][-1] = "{port}"
    e["launch"] += ["--threads", "{threads}"]
    m["engines"] = [e]
    return m


def test_grid_is_deterministic_explicit_and_does_not_reuse_resolved_config():
    source = baseline()
    before = copy.deepcopy(source)
    result, plan = cpu.screen_manifest(source, facts=facts(), threads=[1, 2], pinning=["none", "compact", "spread"])
    assert source == before
    assert result["model"] == source["model"] and result["workload"] == source["workload"]
    assert result["hardware"]["backend"] == "cpu" and "gpu" not in result["hardware"]
    assert len(result["engines"]) == 6
    assert [p["cpus"] for p in plan if p["threads"] == 2] == [list(range(8)), [0, 1], [0, 2]]
    assert len({e["port"] for e in result["engines"]}) == 6
    assert all("resolved_config" not in e and "{" not in " ".join(e["launch"]) for e in result["engines"])
    assert result["engines"][-1]["env"]["OMP_PROC_BIND"] == "spread"


@pytest.mark.parametrize("kwargs,needle", [
    ({"threads": [0]}, "positive"), ({"threads": [5]}, "physical"),
    ({"cpus": [9]}, "allowed"), ({"thread_env": ["HF_TOKEN"]}, "thread_env"),
    ({"pinning": ["invented"]}, "pinning"),
])
def test_invalid_screen_is_rejected(kwargs, needle):
    args = {"facts": facts(), "threads": [2], "pinning": ["compact"], **kwargs}
    with pytest.raises(OpError, match=needle):
        cpu.screen_manifest(baseline(), **args)


def test_smt_is_explicit_and_client_cores_remain_outside_pool():
    _, plan = cpu.screen_manifest(baseline(), facts=facts(), threads=[4], pinning=["compact"],
                                  cpus=[0, 1, 4, 5], include_smt=True)
    assert plan[0]["cpus"] == [0, 1, 4, 5]


def test_placeholders_or_explicit_count_env_required():
    m = baseline()
    m["engines"][0]["launch"] = m["engines"][0]["launch"][:-2]
    with pytest.raises(OpError, match="threads"):
        cpu.screen_manifest(m, facts=facts(), threads=[2], pinning=["none"])
    result, _ = cpu.screen_manifest(m, facts=facts(), threads=[2], pinning=["none"], thread_env=["OMP_NUM_THREADS"])
    assert result["engines"][0]["env"]["OMP_NUM_THREADS"] == "2"


def test_cli_writes_only_a_manifest_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    source, out = tmp_path / "base.json", tmp_path / "screen.json"
    source.write_text(json.dumps(baseline()))
    monkeypatch.setattr(cpu, "cpu_facts", facts)
    args = ["cpu", "screen-manifest", str(source), "--out", str(out), "--threads", "1", "--threads", "2"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ran"] is False and len(result["variants"]) == 6
    assert out.is_file() and json.loads(source.read_text()) == baseline()
    assert cli.main(args) == 1


def test_secret_literal_env_fails_before_output():
    m = baseline()
    m["engines"][0]["env"]["HF_TOKEN"] = "secret-value"
    with pytest.raises(OpError, match="secret"):
        cpu.screen_manifest(m, facts=facts(), threads=[1], pinning=["none"])
