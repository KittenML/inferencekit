"""Speech operation evidence checks using synthetic HTTP and PCM only."""

import hashlib
import json

import pytest

from inferencekit import evidence as ev
from inferencekit.ops import bench_audio as ba
from inferencekit.registry import Call, OpError, REGISTRY
from test_audio_client import FakeAudioServer


def fixture_file(tmp_path, reference=False):
    doc = {"id": "one", "text": "A previously unseen speech fixture.", "unseen": True}
    if reference:
        (tmp_path / "reference.wav").write_bytes(b"fake reference bytes")
        doc.update(reference_audio="reference.wav", reference_text="Reference transcript")
    path = tmp_path / "fixtures.jsonl"
    path.write_text(json.dumps(doc) + "\n")
    return path


def invoke(server, path, **kwargs):
    op = REGISTRY["bench_audio"]
    args = op.validate({"base_url": server.url, "model": "fake/model", "fixtures": str(path), "sample_rate": 8000,
                        "rounds": 2, "warmup": 1, "resamples": 20, "hardware_name": "Fake CPU", **kwargs})
    return op.handler(Call(op, args), **args)


def test_evidence_has_structural_gates_but_quality_remains_unproven(tmp_path, store, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    with FakeAudioServer() as server:
        record = invoke(server, fixture_file(tmp_path, True), settings='{"temperature":0.7}')
        assert len(server.requests) == 3
        assert server.requests[0]["input"] != server.requests[1]["input"]
    assert ev.validate(record) == []
    assert record["kind"] == "audio_benchmark"
    assert record["data"]["total_requests"] == 2
    assert record["data"]["structural_verdict"] == "PASS"
    assert record["data"]["quality_verdict"] == "UNPROVEN"
    assert record["data"]["verdict"] == "UNPROVEN"
    assert record["data"]["memory"]["verdict"] == "NOT RUN"
    assert record["data"]["rounds"][0]["audio_ms"] > 0
    assert record["identity"]["workload"]["request_set"]["sha256"]
    assert server.url not in json.dumps(record)
    root = store.run_dir(record["evidence_id"])
    rows = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["reference_sha256"] == hashlib.sha256(b"fake reference bytes").hexdigest()
    assert rows[0]["first_audio_ms"] >= rows[0]["first_pcm_ms"] + 19
    assert rows[0]["rtf"] == pytest.approx(rows[0]["complete_audio_ms"] / 1020)
    assert any(a["role"] == "audio_pcm" for a in record["artifacts"])


def test_bad_pcm_invalidates_entire_timing_verdict(tmp_path, store):
    with FakeAudioServer("wrong_count") as server:
        record = invoke(server, fixture_file(tmp_path), rounds=1)
    assert record["data"]["structural_verdict"] == "FAIL"
    assert record["data"]["timing_verdict"] == "FAIL"
    assert record["data"]["verdict"] == "FAIL"
    assert all(metric["value"] is None for metric in record["metrics"])


@pytest.mark.parametrize("mode", ["missing_final", "wrong_count"])
def test_failed_stream_retains_delivery_without_claiming_complete_audio(tmp_path, store, mode):
    with FakeAudioServer(mode) as server:
        record = invoke(server, fixture_file(tmp_path), rounds=1, warmup=0)
    row = json.loads((store.run_dir(record["evidence_id"]) / "requests.jsonl").read_text())
    assert not row["ok"]
    assert row["sample_count"] is None
    assert row["received_pcm_frames"] == 8160 == sum(chunk["sample_count"] for chunk in row["chunks"])
    assert row["first_pcm_ms"] is None and row["audio_duration_s"] is None
    assert all(metric["value"] is None for metric in record["metrics"])
    assert not any(artifact["role"] == "audio_pcm" for artifact in record["artifacts"])


def test_underruns_fail_even_when_complete_rtf_is_fast(tmp_path, store):
    with FakeAudioServer("gap") as server:
        record = invoke(server, fixture_file(tmp_path), rounds=1, warmup=0)
    assert record["data"]["per_request"]["rtf"]["max"] < 1
    assert record["data"]["structural_verdict"] == "PASS"
    assert record["data"]["timing_verdict"] == "FAIL"


def test_initial_silence_cannot_buy_a_first_audio_slo_pass(tmp_path, store):
    with FakeAudioServer("leading_silence") as server:
        record = invoke(server, fixture_file(tmp_path), rounds=1, warmup=0)
    assert record["data"]["per_request"]["first_pcm_ms"]["p95"] < 2000
    assert record["data"]["per_request"]["first_signal_ms"]["p95"] < 2000
    assert record["data"]["per_request"]["first_audio_ms"]["p95"] > 3000
    assert record["data"]["timing_verdict"] == "FAIL"


def test_fixture_hash_binds_reference_contents_and_text(tmp_path):
    path = fixture_file(tmp_path, True)
    _, original = ba.load_fixtures(path)
    (tmp_path / "reference.wav").write_bytes(b"different bytes")
    _, changed = ba.load_fixtures(path)
    assert changed != original
    doc = json.loads(path.read_text())
    doc["text"] += " Another sentence."
    path.write_text(json.dumps(doc))
    assert ba.load_fixtures(path)[1] != changed


@pytest.mark.parametrize("doc", [{"id": "../escape", "text": "x"}, {"id": "one", "text": ""},
                                 {"id": "one", "text": "x", "reference_text": "orphan"},
                                 {"id": "one", "text": "x", "unseen": "true"}])
def test_bad_fixture_rejected(tmp_path, doc):
    path = tmp_path / "fixtures.jsonl"
    path.write_text(json.dumps(doc))
    with pytest.raises(OpError):
        ba.load_fixtures(path)


def test_extend_keeps_rows_outputs_and_configuration(tmp_path, store, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    path = fixture_file(tmp_path)
    with FakeAudioServer() as server:
        first = invoke(server, path, rounds=1, warmup=0)
        second = invoke(server, path, rounds=1, warmup=0, extend=first["evidence_id"])
        with pytest.raises(OpError, match="same workload"):
            invoke(server, path, rounds=1, extend=second["evidence_id"], settings='{"temperature":0.8}')
        with pytest.raises(OpError, match="conflicts"):
            invoke(server, path, rounds=1, extend=second["evidence_id"], first_round=1)
        with pytest.raises(OpError, match="skipped rounds"):
            invoke(server, path, rounds=1, extend=second["evidence_id"], first_round=3)
    assert [r["index"] for r in second["data"]["rounds"]] == [0, 1]
    assert second["data"]["total_requests"] == 2
    assert len([a for a in second["artifacts"] if a["role"] == "audio_pcm"]) == 2
    assert second["identity"]["workload"]["request_set"] == first["identity"]["workload"]["request_set"]


def test_request_hash_changes_for_precision_and_settings(tmp_path, store):
    path = fixture_file(tmp_path)
    with FakeAudioServer() as server:
        first = invoke(server, path, rounds=1, warmup=0, model_quant="3/8/8")
        second = invoke(server, path, rounds=1, warmup=0, model_quant="4/8/8")
    assert first["identity"]["workload"]["request_set"] != second["identity"]["workload"]["request_set"]


def test_extend_rejects_changed_request_artifact(tmp_path, store, monkeypatch):
    monkeypatch.setenv("INFERENCEKIT_KEY_FILE", str(tmp_path / "endpoint.key"))
    path = fixture_file(tmp_path)
    with FakeAudioServer() as server:
        first = invoke(server, path, rounds=1, warmup=0)
        (store.run_dir(first["evidence_id"]) / "requests.jsonl").write_text("{}\n")
        with pytest.raises(OpError, match="missing or changed"):
            invoke(server, path, rounds=1, warmup=0, extend=first["evidence_id"])


def test_endpoint_with_inline_auth_is_rejected_without_echo(tmp_path, store):
    with FakeAudioServer() as server:
        with pytest.raises(OpError) as error:
            invoke(server, fixture_file(tmp_path), base_url="http://user:private-token@secret.example:9000")
    assert "private-token" not in str(error.value)


@pytest.mark.parametrize("flag,wire_key", [("model_revision", "model_revision"), ("model_dtype", "dtype"),
                                         ("model_quant", "quant"), ("engine_version", "engine_version")])
def test_declared_pins_are_sent_and_must_match_loaded_effective_settings(tmp_path, store, flag, wire_key):
    path = fixture_file(tmp_path)
    with FakeAudioServer("wrong_pin:" + wire_key) as server:
        record = invoke(server, path, rounds=1, warmup=0, **{flag: "expected-loaded-value"})
    assert server.requests[0]["settings"][wire_key] == "expected-loaded-value"
    assert record["data"]["structural_verdict"] == "FAIL"
    assert record["data"]["verdict"] == "FAIL"


def test_pin_conflicts_fail_before_request(tmp_path, store):
    with FakeAudioServer() as server:
        with pytest.raises(OpError, match="conflict"):
            invoke(server, fixture_file(tmp_path), model_revision="expected", settings='{"model_revision":"wrong"}')
    assert not server.requests


def test_unsolicited_metadata_cannot_leak_into_request_artifact(tmp_path, store):
    with FakeAudioServer("extra_metadata") as server:
        record = invoke(server, fixture_file(tmp_path), rounds=1, warmup=0)
    text = (store.run_dir(record["evidence_id"]) / "requests.jsonl").read_text()
    assert "private-token" not in text
    assert "private.example" not in text


def test_different_engine_versions_share_workload_but_cannot_extend_each_other(tmp_path, store):
    from inferencekit import registry

    path = fixture_file(tmp_path)
    pins = {"model_revision": "checkpoint-r1", "model_dtype": "bf16", "model_quant": "none"}
    previous = {"base": None, "cand": None}
    with FakeAudioServer() as server:
        for _ in range(3):
            for label, version in (("base", "engine-v1"), ("cand", "engine-v2")):
                previous[label] = invoke(server, path, label=label, engine_version=version,
                                         rounds=1, warmup=0,
                                         extend=previous[label]["evidence_id"] if previous[label] else None, **pins)
        with pytest.raises(OpError, match="same workload"):
            invoke(server, path, label="base", engine_version="engine-v2", rounds=1, warmup=0,
                   extend=previous["base"]["evidence_id"], **pins)
    base, candidate = previous["base"], previous["cand"]
    assert base["identity"]["workload"]["request_set"] == candidate["identity"]["workload"]["request_set"]
    comparison = registry.call("compare", {"baseline": base["evidence_id"], "candidate": candidate["evidence_id"],
                                          "metric": "audio_ms", "resamples": 30})
    assert "comparable inputs" not in comparison["coverage"]["missing"]
    assert comparison["data"]["interleaved"] is True
