import hashlib
import json

import pytest

from inferencekit import evidence as ev
from inferencekit.registry import OpError, call


def capture(tmp_path, change=lambda rows: rows):
    fixtures = tmp_path / "fixtures.tsv"
    fixtures.write_text("a\tFirst.\tSecond.\nb\tThird.\tFourth.\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "levels": [1, 2], "rounds": 3, "waves": 1, "period_ms": 2000, "sample_rate": 24000,
        "mode": "stream", "chunks_per_request": 2, "fixture_ids": ["a", "b"],
        "fixture_sha256": hashlib.sha256(fixtures.read_bytes()).hexdigest(),
    }))
    rows = []
    for index in range(3):
        for level in (1, 2):
            for stream in range(level):
                first = 100 if stream == 0 else 450
                rows.append({"round": index, "concurrency": level, "wave": 0, "stream": stream,
                             "id": "ab"[stream], "complete_ms": first + 350,
                             "chunks": [[first, 12000, 480], [first + 300, 12000, 0]]})
    recorded = tmp_path / "timings.jsonl"
    recorded.write_text("".join(json.dumps(row) + "\n" for row in change(rows)))
    args = {"timings": str(recorded), "plan": str(plan), "fixtures": str(fixtures), "label": "candidate",
            "model": "example", "model_revision": "abc", "engine_version": "0.1", "hardware_name": "phone"}
    return args


def test_capacity_checks_every_stream_and_window(tmp_path, store):
    record = call("import_audio_capacity", capture(tmp_path))
    assert ev.validate(record) == []
    data = record["data"]
    assert data["n_star"] == 1
    assert data["quality_verdict"] == "UNPROVEN"
    assert record["identity"]["device"]["name"] == "phone"
    trailers = call("ledger_trailers", {"ids": [record["evidence_id"]], "format": "json"})
    assert any(item["key"] == "Perf-Speech-Capacity-N" for item in trailers["trailers"])
    passing, failing = data["levels"]
    assert passing["verdict"] == "PASS"
    assert passing["rounds"][0]["audio_s_per_s"] == 0.5
    assert failing["verdict"] == "FAIL"
    assert failing["rounds"][0]["streams"][1]["first_audio_p95_ms"] == 470


def test_first_audio_can_be_measured_without_a_capacity_gate(tmp_path, store):
    args = capture(tmp_path)
    record = call("import_audio_capacity", {**args, "first_audio_as_metric": True})
    assert ev.validate(record) == []
    data = record["data"]
    assert data["slo"] == {"rtf_p95": 1.0, "underruns": 0}
    assert data["highest_passing_tested"] == 2
    assert data["n_star"] is None
    assert data["levels"][1]["rounds"][0]["streams"][1]["first_audio_p95_ms"] == 470

    def overload(rows):
        for row in rows:
            if row["concurrency"] == 2:
                row["complete_ms"] = 1200
        return rows

    args = capture(tmp_path, overload)
    record = call("import_audio_capacity", {**args, "first_audio_as_metric": True})
    assert record["data"]["n_star"] == 1
    trailers = call("ledger_trailers", {"ids": [record["evidence_id"]], "format": "json"})
    capacity = next(item for item in trailers["trailers"] if item["key"] == "Perf-Speech-Capacity-N")
    assert "first audio" not in str(capacity)
    assert "p95 RTF <= 1" in str(capacity)


def test_missing_rows_cannot_establish_capacity(tmp_path, store):
    record = call("import_audio_capacity", capture(tmp_path, lambda rows: rows[:-1]))
    assert record["data"]["n_star"] is None
    assert record["data"]["highest_passing_tested"] == 1
    assert record["data"]["levels"][1]["verdict"] == "UNPROVEN"


def test_detects_underruns_despite_good_real_time_factor(tmp_path, store):
    def delay(rows):
        rows[0]["chunks"][1][0] = 650
        rows[0]["complete_ms"] = 700
        return rows

    record = call("import_audio_capacity", {**capture(tmp_path, delay), "first_audio_as_metric": True})
    assert record["data"]["n_star"] == 0
    assert record["data"]["levels"][0]["rounds"][0]["underruns"] == 1


def test_rejects_tampered_fixture_and_malformed_failed_row(tmp_path, store):
    args = capture(tmp_path)
    args["fixtures"] = args["plan"]
    with pytest.raises(OpError, match="fixture file differs"):
        call("import_audio_capacity", args)

    def malformed(rows):
        rows[0].update(complete_ms=None, chunks=[["invalid", 100, 0]])
        return rows

    with pytest.raises(OpError, match="timing row 1"):
        call("import_audio_capacity", capture(tmp_path, malformed))
