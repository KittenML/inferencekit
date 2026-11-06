"""Evidence v2: new records are v2, v1 records still validate, serving workloads are checked."""

import copy
import json
from pathlib import Path

import pytest

from inferencekit import evidence as ev

from conftest import fake_call, needs_repo_file

REPO = Path(__file__).resolve().parents[1]
# A v1 evidence record from a worked example outside the plugin repository: checked when present.
V1_RECORD = REPO / "examples/llamacpp-q8-reuse/results/nsys/analyze_base.json"
needs_v1_record = needs_repo_file(V1_RECORD, "the v1 example record")


def _serve_record(**workload):
    w = {"prompt_len": None, "gen_len": 64, "batch": 1, "seed": 0, "mode": "load", "concurrency": [1, 2, 4],
         "rate_rps": None, "duration_s": 30.0, "bucket": None,
         "prompt_set": {"name": "dev", "sha256": "a" * 64},
         "slo": {"ttft_p95_ms": 500.0, "tpot_p95_ms": 50.0, "max_fail": 0}, "endpoint_kind": "openai_chat"}
    w.update(workload)
    return ev.build(fake_call("bench_serve", ("bench", "serve")), kind="capacity", claim_class="derived",
                    summary="n* 4", identity_=ev.identity(workload=w),
                    metrics=[ev.metric("n_star", 4, "streams", "threshold")])


def test_new_records_are_v2_and_validate():
    rec = ev.finalize(_serve_record())
    assert rec["schema"] == "inferencekit.evidence.v2"
    assert ev.validate(rec) == []
    w = rec["identity"]["workload"]
    assert w["mode"] == "load" and w["concurrency"] == [1, 2, 4] and w["prompt_set"]["name"] == "dev"
    assert "rate_rps" not in w  # v2 fields are written only when known


def test_every_contract_kind_is_accepted():
    for kind in ("serve_benchmark", "capacity", "gates", "quality_trajectory", "quality_task", "manifest"):
        rec = ev.build(fake_call(), kind=kind, claim_class="observed", summary=kind)
        assert ev.validate(ev.finalize(rec)) == [], kind


def test_v2_rejects_bad_serving_fields():
    rec = ev.finalize(_serve_record(mode="burst"))
    assert any("mode" in e for e in ev.validate(rec))
    bad = ev.finalize(_serve_record(prompt_set={"name": "dev", "sha256": "not-a-hash"}))
    assert ev.validate(bad)


@needs_v1_record
def test_v1_records_still_validate_and_verify():
    rec = json.loads(V1_RECORD.read_text())
    assert rec["schema"] == "inferkit.evidence.v1"
    assert ev.validate(rec) == []
    assert ev.verify(rec)["id_ok"]
    # v1 does not know the v2 kinds or workload fields
    v1_bad = copy.deepcopy(rec)
    v1_bad["kind"] = "serve_benchmark"
    assert ev.validate(v1_bad)
    v1_bad = copy.deepcopy(rec)
    v1_bad["identity"]["workload"]["mode"] = "single"
    assert ev.validate(v1_bad)


def test_unknown_schema_is_an_error():
    rec = ev.finalize(_serve_record())
    rec["schema"] = "something.else"
    assert ev.validate(rec) and "not a known evidence schema" in ev.validate(rec)[0]


@needs_v1_record
def test_store_loads_v1_and_v2_side_by_side(store):
    v1 = json.loads(V1_RECORD.read_text())
    d = store.root / v1["evidence_id"]
    d.mkdir(parents=True)
    (d / "evidence.json").write_text(json.dumps(v1))
    v2 = store.save(_serve_record())
    schemas = sorted(r["schema"] for r in store.records())
    assert schemas == ["inferencekit.evidence.v2", "inferkit.evidence.v1"]
    assert store.load(v2["evidence_id"][:12])["kind"] == "capacity"


@pytest.mark.parametrize("name", ["evidence.v1.json", "evidence.v2.json"])
def test_schema_files_are_valid_json_schema(name):
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(json.loads((REPO / "schemas" / name).read_text()))


def test_local_cpu_scheduler_identity_does_not_probe_gpu(monkeypatch):
    from inferencekit.ops import bench_serve
    from inferencekit.serve.client import Endpoint
    monkeypatch.setenv("INFERENCEKIT_CPU_LOCK_HELD", "/test/cpu.lock")
    monkeypatch.setattr(bench_serve, "gpu_identity", lambda: pytest.fail("CPU child probed GPU identity"))
    monkeypatch.setattr(bench_serve, "cpu_identity", lambda: {"model": "test CPU", "architecture": "test"})
    identity, missing = bench_serve._identity(Endpoint.parse("http://127.0.0.1:1"), None, 1, "custom", "v1", "m", {})
    assert identity["cpu"]["model"] == "test CPU" and identity["gpu"]["name"] is None
    assert missing == []
