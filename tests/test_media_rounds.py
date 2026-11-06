"""Real local server/client subprocesses exercise CPU media orchestration, never a model."""
import json
import socket
import sys
from pathlib import Path

import pytest

from inferencekit import evidence as ev, registry
from inferencekit.registry import OpError
from inferencekit.serve import media_rounds as mr, servers


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_run(tmp_path, mode, fail=False):
    script = Path(__file__).parent / "fakes_rounds" / "fake_media_server.py"
    engines = []
    for label, threads in (("base", "1"), ("candidate", "2")):
        port = free_port()
        engines.append({"label": label, "name": "custom", "version": "pinned", "port": port,
                        "env": {"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads},
                        "launch": [sys.executable, str(script), "--port", str(port), "--mode", mode]
                        + (["--fail"] if fail and label == "candidate" else [])})
    manifest = {"model": {"id": "fake/model", "revision": "a" * 40, "precision": "fp32"},
                "hardware": {"backend": "cpu", "cpu": {"model": "local CPU"}, "count": 1},
                "engines": engines, "workload": {"modes": [mode], "sample_rate": 8000, "quant": "none"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    fixtures = tmp_path / "fixtures.jsonl"
    item = {"id": "one", "text": "Synthetic test."} if mode == "audio" else {
        "id": "one", "prompt": "a fox", "width": 4, "height": 3, "steps": 5, "guidance_scale": 3.0}
    fixtures.write_text(json.dumps(item) + "\n")
    return path, fixtures


@pytest.mark.parametrize("mode", ["audio", "image"])
def test_fresh_process_cpu_media_evidence(tmp_path, store, monkeypatch, mode):
    manifest, fixtures = write_run(tmp_path, mode)
    monkeypatch.setenv("INFERENCEKIT_ENDPOINT_HMAC_KEY", "synthetic-test-key")
    monkeypatch.setattr(servers, "gpu_memory_used_mib", lambda: pytest.fail("CPU route probed GPU"))
    result = mr.run_media_rounds(str(manifest), str(fixtures), mode, rounds=4, warmup=1,
                                startup_timeout_s=5, stop_timeout_s=1, poll_s=.02,
                                cpu_lock=str(tmp_path / "cpu.lock"), serve_args="--resamples 10")
    assert all(c["status"] == "ok" for c in result["cells"]), result["cells"]
    logs = [(tmp_path / c["log"]).read_text() for c in result["cells"]]
    assert len({text.split("pid=")[1].splitlines()[0] for text in logs}) == 8
    assert all(c["stop"]["group_gone"] for c in result["cells"])
    final = result["plan"][f"{mode}_evidence_final"]
    assert len(final) == 2
    for label, reference in final.items():
        record = store.load(reference)
        assert ev.integrity_problem(record) is None
        assert record["identity"]["cpu"]["model"]
        assert not record["identity"]["gpu"]["name"]
        assert record["data"]["round_coverage"]["complete"] is True
        assert record["data"]["lifecycle"]["verdict"] == "PASS"
        assert record["data"]["memory"]["peak_rss_mib"] > 0
        assert record["identity"]["engine"]["flags"]["server_env"]["controls"]["OMP_NUM_THREADS"] == (
            "1" if label == "base" else "2")
        assert len(record["data"]["rounds"]) == 4
        assert all((store.run_dir(reference) / a["path"]).is_file() for a in record["artifacts"])
        assert (tmp_path / "artifacts" / reference / "evidence.json").is_file()
    compared = registry.call("compare", {"baseline": final["base"], "candidate": final["candidate"],
                                        "metric": "audio_ms" if mode == "audio" else "image_ms", "resamples": 20})
    assert compared["data"]["interleaved"] is True
    review = registry.call("build_review", {"run_dir": str(tmp_path), "prompts": str(fixtures), "reference": "base"})
    assert review["engines"] == ["base", "candidate"]
    assert review["inputs"] == 4


def test_failed_media_cells_remain_in_plan_and_are_not_ranked(tmp_path, store):
    manifest, fixtures = write_run(tmp_path, "audio", fail=True)
    result = mr.run_media_rounds(str(manifest), str(fixtures), "audio", rounds=3, warmup=0,
                                startup_timeout_s=5, stop_timeout_s=1, poll_s=.02,
                                cpu_lock=str(tmp_path / "cpu.lock"), serve_args="--resamples 10")
    bad = [c for c in result["cells"] if c["label"] == "candidate"]
    assert bad[0]["status"] == "serve_failed"
    assert [c["status"] for c in bad[1:]] == ["skipped_after_failure"] * 2
    assert all(r["best"] != "candidate" for r in result["summary"]["rows"])
    record = store.load(result["plan"]["audio_evidence_final"]["candidate"])
    assert record["data"]["round_coverage"]["missing_indices"] == [1, 2]
    assert record["data"]["lifecycle"]["verdict"] == "UNPROVEN"


def test_auto_routing_and_protected_arguments(tmp_path):
    assert mr.mode_for({"workload": {"modality": "tts"}}, "auto", str(tmp_path / "m.json")) == "audio"
    (tmp_path / "intake.normalized.json").write_text('{"modality":"image"}')
    assert mr.mode_for({}, "auto", str(tmp_path / "m.json")) == "image"
    (tmp_path / "intake.normalized.json").write_text('{"modality":"llm"}')
    assert mr.mode_for({}, "auto", str(tmp_path / "m.json")) == "single"
    with pytest.raises(OpError, match="multiple"):
        mr.mode_for({"workload": {"modes": ["single", "load"]}}, "auto", "m.json")
    with pytest.raises(OpError, match="override"):
        mr.media_argv("audio", {}, None, "--rounds=2")


def test_operation_auto_routes_cpu_audio(tmp_path, store, monkeypatch):
    from inferencekit.ops import bench_rounds
    manifest, fixtures = write_run(tmp_path, "audio")
    monkeypatch.setattr(bench_rounds, "gpu_processes", lambda: pytest.fail("CPU mode probed GPU"))
    monkeypatch.setattr(bench_rounds, "gpu_identity", lambda: pytest.fail("CPU mode probed GPU"))
    rec = registry.call("bench_rounds", {"manifest": str(manifest), "prompts": str(fixtures), "rounds": 1,
                                        "warmup": 0, "startup_timeout_s": 5, "stop_timeout_s": 1,
                                        "cpu_lock": str(tmp_path / "cpu.lock"), "serve_args": "--resamples 10"})
    assert rec["kind"] == "audio_benchmark"
    assert rec["data"]["plan"]["resource_mode"] == "cpu"
    assert rec["data"]["plan"]["audio_evidence_final"]


def test_cleanup_failure_stops_all_later_cells_and_retains_plan(tmp_path, store, monkeypatch):
    manifest, fixtures = write_run(tmp_path, "audio")
    calls = []
    def failed_cleanup(spec, k, position, cfg):
        assert (tmp_path / "audio_rounds" / "plan.json").is_file()
        calls.append(spec["label"])
        return {"label": spec["label"], "round": k, "position": position, "status": "stop_failed",
                "stop": {"group_gone": False}}
    monkeypatch.setattr(mr, "run_cell", failed_cleanup)
    result = mr.run_media_rounds(str(manifest), str(fixtures), "audio", rounds=3,
                                cpu_lock=str(tmp_path / "cpu.lock"))
    assert calls == ["base"]
    assert all(c["status"] == "skipped_after_cleanup_failure" for c in result["cells"][1:])


def test_conflicting_hardware_refused_and_gpu_count_forwarded(tmp_path):
    from inferencekit.serve.round_evidence import resource_mode
    with pytest.raises(OpError, match="conflicting"):
        resource_mode({"hardware": {"backend": "cpu", "gpu": "GPU"}})
    ctx = {"base_url": "http://127.0.0.1:1", "prompts": "p", "model": "m", "label": "base", "first_round": 0,
           "warmup": 0, "seed": 0, "device": "gpu", "gpu_count": 4, "hardware": "GPU"}
    argv = mr.media_argv("image", ctx, None, None)
    assert argv[argv.index("--gpu-count") + 1] == "4"
    with pytest.raises(OpError, match="override"):
        mr.media_argv("image", ctx, None, "--gpu-count 2")
