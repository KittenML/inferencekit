"""Real synthetic HTTP and retained evidence, with no model installation."""
import io
import json
import socket
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from inferencekit import evidence as ev, registry
from inferencekit.ops.bench_stt import load_fixtures
from inferencekit.ops.quality_stt import normalize
from inferencekit.registry import OpError
from inferencekit.review.rundir import load_run
from inferencekit.serve.client import Endpoint
from inferencekit.serve.stt_client import transcribe
from fakes_rounds.fake_stt_server import server

PINS = {"model": "synthetic", "model_revision": "fixture-rev", "model_dtype": "fp32",
        "model_quant": "none", "engine_version": "v1"}


def wav(sample):
    raw = io.BytesIO()
    with wave.open(raw, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(struct.pack("<h", sample) * 1600)
    return raw.getvalue()


@pytest.fixture
def fixture_set(tmp_path):
    doc = tmp_path / "fixtures.jsonl"
    rows = []
    for sample, name, text, lang in ((100, "first", "hello world", "en"), (900, "second", "second phrase", "hi")):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(wav(sample))
        rows.append({"id": name, "audio": path.name, "reference_text": text, "language": lang,
                     "unseen": name == "second"})
    doc.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return doc


@pytest.fixture
def endpoint():
    with server() as httpd:
        worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield httpd, f"http://127.0.0.1:{httpd.server_port}"
        finally:
            httpd.shutdown()
            worker.join()


def call_bench(url, fixtures, **kwargs):
    return registry.call("bench_stt", {"base_url": url, "fixtures": str(fixtures), "hardware_name": "test CPU",
                                      "device": "cpu", "rounds": 1, **PINS, **kwargs})


def register(tmp, fixture_set, acceptance="exact", **kwargs):
    policy = tmp / "policy.json"
    policy.write_text(json.dumps({"acceptance": acceptance, "min_fixtures": 2, "resamples": 100, **kwargs}))
    return registry.call("quality_stt_policy", {"policy": str(policy), "fixtures": str(fixture_set),
                                                "run_dir": str(tmp)})


@pytest.mark.parametrize("mode,stream,error", [("ok", False, None), ("ok", True, None),
                                               ("empty", False, "empty"), ("truncated", True, "terminal")])
def test_multipart_client_and_timing(endpoint, mode, stream, error):
    httpd, url = endpoint
    httpd.mode = mode
    result = transcribe(Endpoint.parse(url + "/v1"), wav(100), model="synthetic", language="en",
                        settings={"temperature": 0.0}, stream=stream)
    assert result.complete_ms is not None and result.complete_ms > 0
    assert result.elapsed_ms >= result.complete_ms
    path, request_id, parts = httpd.requests[0]
    assert path == "/v1/audio/transcriptions" and request_id
    assert parts["file"] == wav(100) and parts["response_format"] == b"json"
    assert set(parts) == ({"file", "model", "language", "temperature", "response_format"} |
                          ({"stream"} if stream else set()))
    if stream:
        assert 0 < result.first_partial_ms < result.complete_ms
    else:
        assert result.first_partial_ms is None
    if error:
        assert error in result.error
        if stream:
            assert result.text == "hello world"
    else:
        assert result.error is None and result.text == "hello world"


def test_metadata_failure_keeps_times_and_transcripts(endpoint, fixture_set, store):
    httpd, url = endpoint
    httpd.mode = "missing"
    evidence = call_bench(url, fixture_set)
    assert evidence["data"]["failed_requests"] == 2
    assert evidence["data"]["rounds"][0]["stt_ms"] > 0
    assert evidence["data"]["claim_validity"]["valid"] is False
    capture = store.run_dir(evidence["evidence_id"]) / "transcripts.jsonl"
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    assert all(row["complete_ms"] > 0 and row["completed"] and not row["ok"] for row in rows)
    assert [row["transcript"] for row in rows] == ["hello world", "second phrase"]


@pytest.mark.parametrize("mode,late,expected", [("ok", False, "PASS"), ("wrong", False, "FAIL"),
                                              ("ok", True, "UNPROVEN"), ("empty", False, "FAIL")])
def test_quality_wer_and_review(endpoint, fixture_set, tmp_path, store, mode, late, expected):
    httpd, url = endpoint
    if not late:
        policy = register(tmp_path, fixture_set, "wer_noninferiority", wer_increase_max=0.01)
    base = call_bench(url, fixture_set, label="base", run_dir=str(tmp_path))
    httpd.mode = mode
    candidate = call_bench(url, fixture_set, label="candidate", run_dir=str(tmp_path))
    if late:
        policy = register(tmp_path, fixture_set, "wer_noninferiority", wer_increase_max=0.01)
    quality = registry.call("quality_stt", {"baseline": base["evidence_id"], "candidate": candidate["evidence_id"],
                                           "protocol": policy["evidence_id"], "fixtures": str(fixture_set),
                                           "run_dir": str(tmp_path)})
    assert quality["data"]["verdict"] == expected
    assert quality["data"]["checks"]["wer"]["baseline"] == 0
    assert quality["data"]["checks"]["wer"]["candidate"] == (0 if mode == "ok" else 1)
    report = registry.call("report", {"ids": [quality["evidence_id"]], "format": "json"})
    assert report["verdict"] == expected
    run = load_run(tmp_path, reference="base")
    assert run.modality == "stt"
    assert run.cells[("base", "r0/first")].text == "hello world"
    assert run.cells[("base", "r0/first")].metrics["final_ms"] > 0
    assert run.items[0].ground_truth == "hello world"
    assert ev.integrity_problem(quality) is None


def test_exact_and_changed_decoding_are_distinct(endpoint, fixture_set, tmp_path, store):
    _, url = endpoint
    policy = register(tmp_path, fixture_set)
    base = call_bench(url, fixture_set)
    candidate = call_bench(url, fixture_set, settings='{"beam_size":1}')
    with pytest.raises(OpError, match="decoding settings"):
        registry.call("quality_stt", {"baseline": base["evidence_id"], "candidate": candidate["evidence_id"],
                                      "protocol": policy["evidence_id"], "fixtures": str(fixture_set)})
    rec = registry.call("compare", {"baseline": base["evidence_id"], "candidate": candidate["evidence_id"],
                                    "metric": "stt_ms", "resamples": 100})
    assert "STT decoding settings or streaming mode differ" in rec["data"]["decision"]["reason"]
    exact = registry.call("quality_stt", {"baseline": base["evidence_id"], "candidate": base["evidence_id"],
                                        "protocol": policy["evidence_id"], "fixtures": str(fixture_set)})
    assert exact["data"]["checks"]["exact"]["verdict"] == "PASS"


@pytest.mark.parametrize("mode,failed", [("cache", "no_answer_cache"), ("duration", "duration_honesty"),
                                        ("perturbation", "perturbation"), ("heldout", "held_out_fixtures")])
def test_gates_time_all_attempts(endpoint, fixture_set, store, mode, failed):
    httpd, url = endpoint
    httpd.mode = mode
    httpd.originals = {wav(100), wav(900)}
    record = registry.call("gates_stt", {"base_url": url, "fixtures": str(fixture_set),
                                         "repeat_min_ratio": 0.01, "noise_max_timing_ratio": 100,
                                         "locked_language": "en", **PINS})
    assert record["data"]["verdict"] == "FAIL"
    assert record["data"]["gates"][failed]["verdict"] == "FAIL"
    assert record["data"]["gates"]["language_lock"]["verdict"] == "WARN"
    assert len(httpd.requests) == 7 and len({x[1] for x in httpd.requests}) == 7
    for row in record["data"]["gates"]["perturbation"]["observations"]:
        assert all(m["complete_ms"] > 0 for m in row["timing"])


def test_import_device_and_reference_hash(fixture_set, tmp_path, store):
    fixtures, _ = load_fixtures(str(fixture_set))
    now = time.time()
    rows = [{"id": f.id, "round": 0, "audio_sha256": f.audio_sha, "text": f.reference, "elapsed_ms": 100,
             "complete_ms": 100, "effective": {"model": PINS["model"], "model_revision": PINS["model_revision"],
             "dtype": "fp32", "quant": "none", "engine_version": "v1", "temperature": 0.0,
             "language": f.language, "cache": False, "audio_duration_s": f.duration}}
            for f in fixtures]
    path = tmp_path / "device.json"
    doc = {"format": "inferencekit.import.stt.v1", "capture_start_unix_s": now - 2,
           "capture_end_unix_s": now - 1, "timer": "synthetic monotonic clock", "rows": rows}
    path.write_text(json.dumps(doc))
    args = {"timings": str(path), "fixtures": str(fixture_set), "hardware_name": "test SoC", **PINS}
    result = registry.call("import_stt", args)
    assert result["identity"]["device"] == {"class": "mobile", "name": "test SoC"}
    assert result["data"]["rounds"][0]["rtf"] == 1
    assert result["data"]["claim_validity"]["valid"] is True
    doc["rows"][0]["audio_sha256"] = "wrong"
    path.write_text(json.dumps(doc))
    with pytest.raises(OpError, match="fixture identity"):
        registry.call("import_stt", args)


def test_normalizers_and_fixture_validation(fixture_set):
    assert normalize("It'll cost twenty one dollars.", "whisper_english_v1") == normalize(
        "It will cost $21.", "whisper_english_v1")
    assert normalize("क़\u200dल", "indic_cer_v1") == normalize("कल", "indic_cer_v1")
    fixture_set.write_text(fixture_set.read_text().replace("first.wav", "../first.wav"))
    with pytest.raises(OpError, match="escapes"):
        load_fixtures(str(fixture_set))


@pytest.mark.parametrize("bad", [False, True])
def test_stt_fresh_process_rounds(tmp_path, fixture_set, store, bad, monkeypatch):
    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    script = Path(__file__).parent / "fakes_rounds/fake_stt_server.py"
    engines = []
    for label in ("base", "candidate"):
        p = port()
        mode = "empty" if bad and label == "candidate" else "ok"
        engines.append({"label": label, "name": "custom", "version": "v1", "port": p, "endpoint_kind": "stt_http",
                        "launch": [sys.executable, str(script), "--port", str(p), "--mode", mode]})
    manifest = {"model": {"id": "synthetic", "revision": "fixture-rev", "precision": "fp32"},
                "hardware": {"backend": "cpu", "cpu": {"model": "test CPU"}, "count": 1},
                "engines": engines, "workload": {"modes": ["stt"], "modality": "stt", "quant": "none"}}
    plan = tmp_path / "manifest.json"
    plan.write_text(json.dumps(manifest))
    monkeypatch.setenv("INFERENCEKIT_ENDPOINT_HMAC_KEY", "synthetic-test-key")
    result = registry.call("bench_rounds", {"manifest": str(plan), "prompts": str(fixture_set), "rounds": 4,
                                          "warmup": 0, "startup_timeout_s": 5, "stop_timeout_s": 1,
                                          "cpu_lock": str(tmp_path / "cpu.lock")})
    final = result["data"]["plan"]["stt_evidence_final"]
    candidate = store.load(final["candidate"])
    assert len(candidate["data"]["rounds"]) == 4
    assert all(row["stt_ms"] > 0 for row in candidate["data"]["rounds"])
    assert candidate["data"]["lifecycle"]["verdict"] == ("UNPROVEN" if bad else "PASS")
    cells = result["data"]["cells"]
    assert [cell["status"] for cell in cells if cell["label"] == "candidate"] == [
        "serve_failed" if bad else "ok"] * 4
